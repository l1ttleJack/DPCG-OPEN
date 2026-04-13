"""Minimal spconv model definitions for the lightweight DPCG release."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from dpcg.utils import ensure_spconv_input, inverse_sqrt_diagonal, sync_cuda

try:
    import spconv.pytorch as spconv
except Exception:  # pragma: no cover - optional dependency
    spconv = None


def _require_spconv() -> None:
    if spconv is None:
        raise RuntimeError(
            "spconv is required for SUNet0 training and benchmark. "
            "Install a CUDA-matched spconv wheel before using this release."
        )


def _ensure_spconv_cuda_input(x):
    _require_spconv()
    x_sp = ensure_spconv_input(x)
    if x_sp.features.device.type != "cuda":
        raise RuntimeError(
            f"SUNet0 requires CUDA sparse tensors; got {x_sp.features.device.type!r}"
        )
    return x_sp


def _empty_sparse_model_timings() -> dict[str, float]:
    return {
        "dense_input_time_sec": 0.0,
        "diag_build_time_sec": 0.0,
        "encoder_decoder_time_sec": 0.0,
        "final_dense_time_sec": 0.0,
        "diag_fill_time_sec": 0.0,
        "final_matmul_time_sec": 0.0,
        "sparse_head_time_sec": 0.0,
    }


def _maybe_sync_cuda(device, *, enabled: bool) -> None:
    if enabled:
        sync_cuda(device)


class SparseLayerNorm(nn.Module):
    """Apply LayerNorm to sparse features or plain tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(int(channels))

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.ln(x)
        return x.replace_feature(self.ln(x.features))


def _sparse_norm(channels: int, use_layer_norm: bool = True) -> nn.Module:
    return SparseLayerNorm(channels) if use_layer_norm else nn.Identity()


@dataclass
class SparseFactorPrediction:
    """Sparse factor prediction on active coordinates."""

    values: torch.Tensor
    coords: torch.Tensor
    shape: tuple[int, int]
    timings: dict[str, float]


def sparse_prediction_to_dense(
    prediction: SparseFactorPrediction,
    matrix,
    diag_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize a sparse prediction into a dense lower-triangular factor."""
    if diag_inv is None:
        diag_inv = inverse_sqrt_diagonal(matrix, dtype=prediction.values.dtype)
    else:
        diag_inv = diag_inv.to(device=prediction.values.device, dtype=prediction.values.dtype)
    n = int(diag_inv.shape[0])
    dense = torch.zeros((n, n), dtype=prediction.values.dtype, device=prediction.values.device)
    coords = prediction.coords.to(device=prediction.values.device, dtype=torch.long)
    values = prediction.values.reshape(-1).to(device=prediction.values.device)
    dense[coords[:, 0], coords[:, 1]] = values
    diag = torch.arange(n, device=prediction.values.device)
    dense[diag, diag] = 1.0
    dense = torch.tril(dense)
    return dense * diag_inv.unsqueeze(0)


class SUNet0(nn.Module):
    """Sparse SUNet0 with optional tail expansion layers."""

    def __init__(
        self,
        channels: tuple[int, int, int, int, int] = (8, 16, 32, 48, 64),
        block_depth: int = 1,
        use_sparse_head: bool = True,
        use_layer_norm: bool = True,
        tail_expansion_layers: int = 1,
        tail_kernel_size: int = 3,
    ) -> None:
        _require_spconv()
        if not torch.cuda.is_available():
            raise RuntimeError("SUNet0 requires CUDA; CPU execution is not supported")
        super().__init__()
        self.input_kind = "spconv"
        self.channels = tuple(int(v) for v in channels)
        if len(self.channels) != 5:
            raise ValueError("SUNet0 expects exactly 5 body channel stages")
        self.block_depth = int(block_depth)
        if self.block_depth < 1:
            raise ValueError("SUNet0 block_depth must be >= 1")
        self.use_sparse_head = bool(use_sparse_head)
        self.use_layer_norm = bool(use_layer_norm)
        self.tail_expansion_layers = int(tail_expansion_layers)
        if self.tail_expansion_layers < 0:
            raise ValueError("SUNet0 tail_expansion_layers must be non-negative")
        self.tail_kernel_size = int(tail_kernel_size)
        if self.tail_kernel_size <= 0 or self.tail_kernel_size % 2 == 0:
            raise ValueError("SUNet0 tail_kernel_size must be a positive odd integer")
        c0, c1, c2, c3, c4 = self.channels

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(1, c0, kernel_size=1, stride=1, padding=0, indice_key="sunet0_in"),
            _sparse_norm(c0, self.use_layer_norm),
            nn.PReLU(),
        )
        self.encoder1 = self._down_block(c0, c1, stride=2, indice_key="sunet0_enc1")
        self.encoder2 = self._down_block(c1, c2, stride=2, indice_key="sunet0_enc2")
        self.encoder3 = self._down_block(c2, c3, stride=2, indice_key="sunet0_enc3")
        self.midlayer = spconv.SparseSequential(
            spconv.SubMConv2d(
                c3,
                c4,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key="sunet0_mid",
            ),
            _sparse_norm(c4, self.use_layer_norm),
            nn.PReLU(),
        )
        self.inv_mid = spconv.SparseSequential(
            spconv.SubMConv2d(
                c4,
                c3,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key="sunet0_mid_inv",
            ),
            _sparse_norm(c3, self.use_layer_norm),
            nn.PReLU(),
        )
        self.decoder3 = self._up_block(c3, c2, indice_key="sunet0_enc3")
        self.decoder2 = self._up_block(c2, c1, indice_key="sunet0_enc2")
        self.decoder1 = self._up_block(c1, c0, indice_key="sunet0_enc1")
        if self.tail_expansion_layers > 0:
            padding = self.tail_kernel_size // 2
            tail_layers: list[nn.Module] = []
            for idx in range(self.tail_expansion_layers):
                tail_layers.extend(
                    [
                        spconv.SparseConv2d(
                            c0,
                            c0,
                            kernel_size=self.tail_kernel_size,
                            stride=1,
                            padding=padding,
                            indice_key=f"sunet0_tail_{idx}",
                        ),
                        _sparse_norm(c0, self.use_layer_norm),
                        nn.PReLU(),
                    ]
                )
            self.tail_expansion = spconv.SparseSequential(*tail_layers)
        else:
            self.tail_expansion = None
        self.head = (
            spconv.SubMConv2d(
                c0,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                indice_key="sunet0_head",
            )
            if self.use_sparse_head
            else spconv.SparseConv2d(
                c0,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                indice_key="sunet0_head",
            )
        )

    def _block_rulebook_key(self, indice_key: str, stage_name: str, depth_idx: int) -> str:
        if self.block_depth == 1:
            return f"{indice_key}_{stage_name}"
        return f"{indice_key}_{stage_name}_{depth_idx}"

    def _down_block(self, in_channels: int, out_channels: int, stride: int, indice_key: str):
        layers: list[nn.Module] = []
        current_in = in_channels
        current_out = out_channels
        for depth_idx in range(self.block_depth):
            subm1_key = self._block_rulebook_key(indice_key, "subm1", depth_idx)
            down_key = self._block_rulebook_key(indice_key, "down", depth_idx)
            subm2_key = self._block_rulebook_key(indice_key, "subm2", depth_idx)
            layers.extend(
                [
                    spconv.SubMConv2d(
                        current_in,
                        current_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        indice_key=subm1_key,
                    ),
                    _sparse_norm(current_out, self.use_layer_norm),
                    nn.PReLU(),
                    spconv.SparseConv2d(
                        current_out,
                        current_out,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        indice_key=down_key,
                    ),
                    _sparse_norm(current_out, self.use_layer_norm),
                    nn.PReLU(),
                    spconv.SubMConv2d(
                        current_out,
                        current_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        indice_key=subm2_key,
                    ),
                    _sparse_norm(current_out, self.use_layer_norm),
                    nn.PReLU(),
                ]
            )
            current_in = current_out
        return spconv.SparseSequential(*layers)

    def _up_block(self, in_channels: int, out_channels: int, indice_key: str):
        layers: list[nn.Module] = []
        current_channels = in_channels
        for depth_idx in reversed(range(self.block_depth)):
            subm2_key = self._block_rulebook_key(indice_key, "subm2", depth_idx)
            down_key = self._block_rulebook_key(indice_key, "down", depth_idx)
            subm1_key = self._block_rulebook_key(indice_key, "subm1", depth_idx)
            final_out = out_channels if depth_idx == 0 else current_channels
            layers.extend(
                [
                    spconv.SubMConv2d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        indice_key=subm2_key,
                    ),
                    _sparse_norm(current_channels, self.use_layer_norm),
                    nn.PReLU(),
                    spconv.SparseInverseConv2d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        indice_key=down_key,
                    ),
                    _sparse_norm(current_channels, self.use_layer_norm),
                    nn.PReLU(),
                    spconv.SubMConv2d(
                        current_channels,
                        final_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        indice_key=subm1_key,
                    ),
                    _sparse_norm(final_out, self.use_layer_norm),
                    nn.PReLU(),
                ]
            )
            current_channels = final_out
        return spconv.SparseSequential(*layers)

    def forward(self, x, profile: bool = False):
        x_sp = _ensure_spconv_cuda_input(x)
        device = x_sp.features.device
        timings = _empty_sparse_model_timings()
        profile_enabled = bool(profile)

        _maybe_sync_cuda(device, enabled=profile_enabled)
        t0 = perf_counter()
        input1 = self.conv1(x_sp)
        e1 = self.encoder1(input1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        mid = self.midlayer(e3)
        inv_mid = self.inv_mid(mid)
        d3 = self.decoder3(inv_mid + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)
        if self.tail_expansion is not None:
            d1 = self.tail_expansion(d1)
        _maybe_sync_cuda(device, enabled=profile_enabled)
        if profile_enabled:
            timings["encoder_decoder_time_sec"] = perf_counter() - t0

        _maybe_sync_cuda(device, enabled=profile_enabled)
        t0 = perf_counter()
        head = self.head(d1)
        values = head.features.squeeze(-1)
        coords = head.indices[:, 1:3].to(torch.int64)
        _maybe_sync_cuda(device, enabled=profile_enabled)
        if profile_enabled:
            timings["sparse_head_time_sec"] = perf_counter() - t0
            timings["forward_total_time_sec"] = sum(timings.values())

        prediction = SparseFactorPrediction(
            values=values,
            coords=coords,
            shape=tuple(int(v) for v in head.spatial_shape),
            timings=timings,
        )
        if profile:
            return prediction, timings
        return prediction


SUNet0Tail = SUNet0


def normalize_model_kwargs(raw_kwargs: dict[str, object] | None) -> dict[str, object]:
    if raw_kwargs is None:
        return {}
    if not isinstance(raw_kwargs, dict):
        raise TypeError("MODEL_KWARGS must be a mapping")
    kwargs = dict(raw_kwargs)
    if "channels" in kwargs and kwargs["channels"] is not None:
        kwargs["channels"] = tuple(int(v) for v in kwargs["channels"])
    if "block_depth" in kwargs and kwargs["block_depth"] is not None:
        kwargs["block_depth"] = int(kwargs["block_depth"])
    if "tail_expansion_layers" in kwargs and kwargs["tail_expansion_layers"] is not None:
        kwargs["tail_expansion_layers"] = int(kwargs["tail_expansion_layers"])
    if "tail_kernel_size" in kwargs and kwargs["tail_kernel_size"] is not None:
        kwargs["tail_kernel_size"] = int(kwargs["tail_kernel_size"])
    if "use_sparse_head" in kwargs and kwargs["use_sparse_head"] is not None:
        kwargs["use_sparse_head"] = bool(kwargs["use_sparse_head"])
    if "use_layer_norm" in kwargs and kwargs["use_layer_norm"] is not None:
        kwargs["use_layer_norm"] = bool(kwargs["use_layer_norm"])
    return kwargs


def _load_checkpoint(path: str, *, map_location: str | torch.device | None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions
        return torch.load(path, map_location=map_location)


def load_model(
    name: str,
    *,
    model_kwargs: dict[str, object] | None = None,
    device: str | torch.device | None = None,
    checkpoint_path: str | None = None,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    normalized_name = str(name).strip().lower()
    if normalized_name not in {"sunet0", "sunet0_tail", "sunet0tail"}:
        raise ValueError(f"Unsupported lightweight model: {name!r}")
    model = SUNet0(**normalize_model_kwargs(model_kwargs))
    if device is not None:
        model = model.to(device)
    if checkpoint_path:
        checkpoint = _load_checkpoint(
            checkpoint_path,
            map_location=map_location if map_location is not None else device,
        )
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
    return model
