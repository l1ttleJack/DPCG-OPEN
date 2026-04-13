"""Condition-number training losses and spectral estimators."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from torch import nn

from dpcg.models import SparseFactorPrediction
from dpcg.utils import (
    assemble_sparse_factor_from_prediction_torch,
    ensure_sparse_square,
    normalize_diag_strategy,
    squeeze_batch_matrix,
    torch_sparse_to_scipy_csr,
)

_SCIPY_EIGSH_TOL = 1.0e-3
_SCIPY_EIGSH_MAXITER = 4000
_SCIPY_EIGSH_FALLBACK_MAXITER = 8000
_MIN_EIG_EPS = 1.0e-12
_RELATIVE_RESIDUAL_TOL = 1.0e-2
_LOGCOND_EPS = 1.0e-12
_LOSS_STEP_DIAG_KEYS = (
    "factor_assembly_time_sec",
    "l_sparse_build_time_sec",
    "pm_build_time_sec",
    "gpu_to_cpu_time_sec",
    "lambda_min_solve_time_sec",
    "lambda_max_solve_time_sec",
    "cpu_to_gpu_time_sec",
    "spectral_loss_forward_time_sec",
    "spectral_backward_grad_pm_time_sec",
    "spectral_backward_grad_factor_time_sec",
    "spectral_backward_total_time_sec",
    "matrix_size",
    "factor_nnz",
    "pm_nnz",
    "active_pred_nnz",
    "exact_cond",
)


@dataclass
class SpectralCacheEntry:
    """Per-sample cached state for the smallest eigenpair warm start."""

    lambda_min: float
    vec_min: np.ndarray
    matrix_size: int
    last_seen_step: int
    residual_norm: float | None = None


def _empty_cache_stats() -> dict[str, float | int]:
    return {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_evictions": 0,
        "cache_reuse_hits": 0,
        "eigsh_sigma0_hits": 0,
        "eigsh_sigma_hits": 0,
        "eigsh_sa_fallbacks": 0,
        "lambda_min_solve_time_sec_sum": 0.0,
        "lambda_max_solve_time_sec_sum": 0.0,
        "lambda_min_solve_time_sec_hit_sum": 0.0,
        "lambda_min_solve_time_sec_miss_sum": 0.0,
        "lambda_max_solve_time_sec_when_hit_sum": 0.0,
        "lambda_max_solve_time_sec_when_miss_sum": 0.0,
    }


def _relative_residual_norm(matrix: csr_matrix, eigenvalue: float, vector: np.ndarray) -> float:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    residual = matrix @ vec - eigenvalue * vec
    denom = max(abs(float(eigenvalue)) * float(np.linalg.norm(vec)), _MIN_EIG_EPS)
    return float(np.linalg.norm(residual) / denom)


def _normalize_eigenvector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("eigenvector norm must be finite and positive")
    return vec / norm


def _eigsh_smallest(
    matrix: csr_matrix,
    *,
    sigma: float | None,
    v0: np.ndarray | None,
    tol: float,
    maxiter: int,
) -> tuple[float, np.ndarray]:
    which = "LM" if sigma is not None else "SA"
    eigenvalues_np, eigenvectors_np = eigsh(
        matrix,
        k=1,
        which=which,
        sigma=sigma,
        v0=None if v0 is None else np.asarray(v0, dtype=np.float64).reshape(-1),
        tol=tol,
        maxiter=maxiter,
    )
    return float(np.real(eigenvalues_np[0])), np.asarray(eigenvectors_np[:, 0], dtype=np.float64)


def _dense_smallest(matrix: csr_matrix) -> tuple[float, np.ndarray]:
    dense = np.asarray(matrix.toarray(), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=np.float64)


def _dense_largest(matrix: csr_matrix) -> tuple[float, np.ndarray]:
    dense = np.asarray(matrix.toarray(), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return float(eigenvalues[-1]), np.asarray(eigenvectors[:, -1], dtype=np.float64)


def _eigsh_largest(matrix: csr_matrix, *, tol: float, maxiter: int) -> tuple[float, np.ndarray]:
    eigenvalues_np, eigenvectors_np = eigsh(matrix, k=1, which="LA", tol=tol, maxiter=maxiter)
    return float(np.real(eigenvalues_np[0])), np.asarray(eigenvectors_np[:, 0], dtype=np.float64)


def _mask_support_from_mask(mask, *, n: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if getattr(mask, "is_sparse", False):
        coords = mask.coalesce().indices().to(device=device, dtype=torch.long)
    else:
        coords = torch.nonzero(mask, as_tuple=False).t().to(device=device, dtype=torch.long)
    rows = coords[0].reshape(-1)
    cols = coords[1].reshape(-1)
    return rows, cols, rows * int(n) + cols


def _diag_sync(device, *, enabled: bool) -> None:
    if not enabled or torch is None or device is None:
        return
    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        torch.cuda.synchronize(device=device)


def _diag_timer_start(device, *, enabled: bool) -> float | None:
    if not enabled:
        return None
    _diag_sync(device, enabled=True)
    return perf_counter()


def _diag_timer_stop(
    step_diag: dict[str, Any] | None,
    key: str,
    start_time: float | None,
    *,
    device,
    enabled: bool,
) -> None:
    if step_diag is None or start_time is None or not enabled:
        return
    _diag_sync(device, enabled=True)
    step_diag[key] = float(perf_counter() - start_time)


def _new_loss_step_diag(matrix_size: int) -> dict[str, Any]:
    diag: dict[str, Any] = {key: 0.0 for key in _LOSS_STEP_DIAG_KEYS}
    diag["matrix_size"] = int(matrix_size)
    diag["factor_nnz"] = 0
    diag["pm_nnz"] = 0
    diag["active_pred_nnz"] = 0
    diag["cache_hit"] = False
    diag["lambda_min_solver"] = None
    return diag


class _ConditionNumberSpectralLoss(torch.autograd.Function):  # type: ignore[misc]
    @staticmethod
    def forward(
        ctx,
        A_sparse,
        pred_values,
        factor_rows,
        factor_cols,
        factor_values,
        active_pred_input_indices,
        active_rows,
        active_cols,
        active_scales,
        eig_min,
        eig_max,
        vec_min,
        vec_max,
        step_diag,
        diag_enabled,
        logcond_eps,
    ):
        safe_eig_max = torch.clamp_min(eig_max, logcond_eps)
        safe_eig_min = torch.clamp_min(eig_min, logcond_eps)
        loss = torch.log(safe_eig_max) - torch.log(safe_eig_min)
        ctx.A_sparse = A_sparse
        ctx.original_dtype = pred_values.dtype
        ctx.matrix_size = int(A_sparse.shape[0])
        ctx.num_pred_values = int(pred_values.numel())
        ctx.step_diag = step_diag
        ctx.diag_enabled = bool(diag_enabled)
        ctx.logcond_eps = float(logcond_eps)
        ctx.save_for_backward(
            factor_rows,
            factor_cols,
            factor_values,
            active_pred_input_indices,
            active_rows,
            active_cols,
            active_scales,
            eig_min.reshape(1),
            eig_max.reshape(1),
            vec_min,
            vec_max,
        )
        return loss.reshape(())

    @staticmethod
    def backward(ctx, grad_output):
        (
            factor_rows,
            factor_cols,
            factor_values,
            active_pred_input_indices,
            active_rows,
            active_cols,
            active_scales,
            eig_min,
            eig_max,
            vec_min,
            vec_max,
        ) = ctx.saved_tensors
        step_diag = getattr(ctx, "step_diag", None)
        diag_enabled = bool(getattr(ctx, "diag_enabled", False))
        device = factor_values.device
        timer_total = _diag_timer_start(device, enabled=diag_enabled)
        eps = float(getattr(ctx, "logcond_eps", _LOGCOND_EPS))
        v_min = vec_min[:, :1]
        v_max = vec_max[:, :1]
        d_lambda_max = torch.where(
            eig_max[0] > eps,
            grad_output / torch.clamp_min(eig_max[0], eps),
            torch.zeros_like(grad_output),
        )
        d_lambda_min = torch.where(
            eig_min[0] > eps,
            -grad_output / torch.clamp_min(eig_min[0], eps),
            torch.zeros_like(grad_output),
        )
        timer_grad_pm = _diag_timer_start(device, enabled=diag_enabled)
        L_sparse = torch.sparse_coo_tensor(
            torch.stack([factor_rows, factor_cols], dim=0),
            factor_values,
            size=(ctx.matrix_size, ctx.matrix_size),
            dtype=factor_values.dtype,
            device=device,
        ).coalesce()
        proj_max = torch.sparse.mm(
            ctx.A_sparse,
            torch.sparse.mm(L_sparse.transpose(0, 1), v_max),
        ).reshape(-1)
        proj_min = torch.sparse.mm(
            ctx.A_sparse,
            torch.sparse.mm(L_sparse.transpose(0, 1), v_min),
        ).reshape(-1)
        _diag_timer_stop(
            step_diag,
            "spectral_backward_grad_pm_time_sec",
            timer_grad_pm,
            device=device,
            enabled=diag_enabled,
        )
        timer_grad_factor = _diag_timer_start(device, enabled=diag_enabled)
        grad_values = torch.zeros(
            ctx.num_pred_values,
            dtype=factor_values.dtype,
            device=device,
        )
        if active_pred_input_indices.numel() > 0:
            grad_entries = (
                2.0
                * d_lambda_max
                * v_max.reshape(-1)[active_rows]
                * proj_max[active_cols]
                + 2.0
                * d_lambda_min
                * v_min.reshape(-1)[active_rows]
                * proj_min[active_cols]
            ) * active_scales
            grad_values.scatter_add_(0, active_pred_input_indices, grad_entries)
        _diag_timer_stop(
            step_diag,
            "spectral_backward_grad_factor_time_sec",
            timer_grad_factor,
            device=device,
            enabled=diag_enabled,
        )
        _diag_timer_stop(
            step_diag,
            "spectral_backward_total_time_sec",
            timer_total,
            device=device,
            enabled=diag_enabled,
        )
        return (
            None,
            grad_values.to(dtype=ctx.original_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CachedConditionNumberLoss(nn.Module):
    """Condition-number loss with per-sample smallest-eigenpair caching."""

    def __init__(
        self,
        *,
        max_entries: int = 4096,
        eigsh_tol: float = _SCIPY_EIGSH_TOL,
        eigsh_maxiter: int = _SCIPY_EIGSH_MAXITER,
        eigsh_fallback_maxiter: int = _SCIPY_EIGSH_FALLBACK_MAXITER,
        enable_train_cache: bool = True,
        enable_val_cache: bool = False,
        relative_residual_tol: float = _RELATIVE_RESIDUAL_TOL,
        min_eigenvalue_eps: float = _MIN_EIG_EPS,
        logcond_eps: float = _LOGCOND_EPS,
        diag_strategy: str = "learned_exp",
    ) -> None:
        super().__init__()
        self.loss_name = "condition_number_loss_eigs"
        self.objective_variant = "logcond_sigma0"
        self.max_entries = int(max_entries)
        self.eigsh_tol = float(eigsh_tol)
        self.eigsh_maxiter = int(eigsh_maxiter)
        self.eigsh_fallback_maxiter = int(eigsh_fallback_maxiter)
        self.enable_train_cache = bool(enable_train_cache)
        self.enable_val_cache = bool(enable_val_cache)
        self.relative_residual_tol = float(relative_residual_tol)
        self.min_eigenvalue_eps = float(min_eigenvalue_eps)
        self.logcond_eps = float(logcond_eps)
        self.diag_strategy = normalize_diag_strategy(diag_strategy)
        self.mask_projection_enabled = True
        self._current_split = "train"
        self._step_counter = 0
        self._caches: dict[str, OrderedDict[str, SpectralCacheEntry]] = {
            "train": OrderedDict(),
            "val": OrderedDict(),
        }
        self._stats = _empty_cache_stats()
        self._last_call_info: dict[str, Any] = {}
        self._diag_enabled = False
        self._active_step_diag: dict[str, Any] | None = None
        self._last_step_diag: dict[str, Any] | None = None

    def clear_cache(self, split: str | None = None) -> None:
        if split is None:
            for cache in self._caches.values():
                cache.clear()
            return
        self._caches.setdefault(split, OrderedDict()).clear()

    def reset_stats(self) -> None:
        self._stats = _empty_cache_stats()
        self._last_call_info = {}
        self._active_step_diag = None
        self._last_step_diag = None

    def start_epoch(self, split: str = "train") -> None:
        self._current_split = str(split)
        self._last_call_info = {}
        self._active_step_diag = None
        self._last_step_diag = None

    def set_diag_enabled(self, enabled: bool) -> None:
        self._diag_enabled = bool(enabled)
        self._active_step_diag = None
        self._last_step_diag = None

    def set_diag_strategy(self, diag_strategy: str) -> None:
        normalized = normalize_diag_strategy(diag_strategy)
        if normalized == "raw_pred":
            raise ValueError("diag_strategy='raw_pred' is reserved for offline analysis only")
        self.diag_strategy = normalized

    def set_mask_projection_enabled(self, enabled: bool) -> None:
        self.mask_projection_enabled = bool(enabled)

    def pop_last_step_diag(self) -> dict[str, Any] | None:
        if self._last_step_diag is None:
            return None
        step_diag = dict(self._last_step_diag)
        self._last_step_diag = None
        self._active_step_diag = None
        return step_diag

    def get_cache_stats(self) -> dict[str, Any]:
        total = dict(self._stats)
        total["current_split"] = self._current_split
        total["train_cache_size"] = len(self._caches["train"])
        total["val_cache_size"] = len(self._caches["val"])
        attempts = int(total["cache_hits"]) + int(total["cache_misses"])
        total["cache_hit_rate"] = (
            float(total["cache_hits"]) / float(attempts) if attempts > 0 else None
        )
        total["objective_variant"] = self.objective_variant
        total["mask_projection_enabled"] = bool(self.mask_projection_enabled)
        total["last_call_info"] = dict(self._last_call_info)
        return total

    def _cache_enabled_for_current_split(self) -> bool:
        if self._current_split == "train":
            return self.enable_train_cache
        if self._current_split == "val":
            return self.enable_val_cache
        return False

    def _active_cache(self) -> OrderedDict[str, SpectralCacheEntry]:
        return self._caches.setdefault(self._current_split, OrderedDict())

    def _lookup_cache(
        self, sample_id: str | None, matrix_size: int
    ) -> tuple[SpectralCacheEntry | None, bool]:
        if sample_id is None or not self._cache_enabled_for_current_split():
            return None, False
        cache = self._active_cache()
        entry = cache.get(sample_id)
        if entry is None:
            self._stats["cache_misses"] += 1
            return None, False
        if entry.matrix_size != matrix_size or entry.vec_min.shape[0] != matrix_size:
            cache.pop(sample_id, None)
            self._stats["cache_misses"] += 1
            return None, False
        cache.move_to_end(sample_id)
        self._stats["cache_hits"] += 1
        return entry, True

    def _update_cache(
        self,
        sample_id: str | None,
        *,
        lambda_min: float,
        vec_min: np.ndarray,
        matrix_size: int,
        residual_norm: float,
    ) -> None:
        if sample_id is None or not self._cache_enabled_for_current_split():
            return
        cache = self._active_cache()
        cache[sample_id] = SpectralCacheEntry(
            lambda_min=float(lambda_min),
            vec_min=_normalize_eigenvector(vec_min),
            matrix_size=int(matrix_size),
            last_seen_step=int(self._step_counter),
            residual_norm=float(residual_norm),
        )
        cache.move_to_end(sample_id)
        while len(cache) > self.max_entries:
            cache.popitem(last=False)
            self._stats["cache_evictions"] += 1

    def _record_last_call(self, **kwargs: Any) -> None:
        self._last_call_info = dict(kwargs)

    def _begin_step_diag(self, *, matrix_size: int) -> dict[str, Any] | None:
        if not self._diag_enabled:
            self._active_step_diag = None
            self._last_step_diag = None
            return None
        step_diag = _new_loss_step_diag(matrix_size)
        self._active_step_diag = step_diag
        self._last_step_diag = step_diag
        return step_diag

    def _warm_start_vector(
        self, cache_entry: SpectralCacheEntry | None, matrix_size: int
    ) -> np.ndarray | None:
        if cache_entry is None:
            return None
        vec = np.asarray(cache_entry.vec_min, dtype=np.float64).reshape(-1)
        if vec.shape[0] != matrix_size or not np.isfinite(vec).all():
            return None
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return vec / norm

    def _solve_lambda_min(
        self,
        matrix: csr_matrix,
        *,
        sample_id: str | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, str]:
        del seed
        matrix_size = int(matrix.shape[0])
        cache_entry, cache_hit = self._lookup_cache(sample_id, matrix_size)
        note_parts: list[str] = []
        v0_hint = self._warm_start_vector(cache_entry, matrix_size)
        self._stats["eigsh_sigma0_hits"] += 1
        self._stats["eigsh_sigma_hits"] += 1
        try:
            eig_min, vec_min = _eigsh_smallest(
                matrix,
                sigma=0.0,
                v0=v0_hint,
                tol=self.eigsh_tol,
                maxiter=self.eigsh_maxiter,
            )
            lambda_min_solver = "eigsh_sigma0"
        except Exception as exc:  # pragma: no cover - rare SciPy failures
            note_parts.append(f"eigsh_sigma0_error={exc}")
            self._stats["eigsh_sa_fallbacks"] += 1
            try:
                eig_min, vec_min = _eigsh_smallest(
                    matrix,
                    sigma=None,
                    v0=v0_hint,
                    tol=self.eigsh_tol,
                    maxiter=self.eigsh_fallback_maxiter,
                )
                lambda_min_solver = "eigsh_sa"
            except Exception as inner_exc:  # pragma: no cover - rare SciPy failures
                note_parts.append(f"eigsh_sa_error={inner_exc}")
                eig_min, vec_min = _dense_smallest(matrix)
                lambda_min_solver = "dense_eigh"
        rel_residual = _relative_residual_norm(matrix, eig_min, vec_min)
        self._update_cache(
            sample_id,
            lambda_min=eig_min,
            vec_min=vec_min,
            matrix_size=matrix_size,
            residual_norm=rel_residual,
        )
        self._record_last_call(
            sample_id=sample_id,
            cache_hit=cache_hit,
            lambda_min_solver=lambda_min_solver,
            lambda_min=float(eig_min),
            lambda_min_relative_residual=float(rel_residual),
            note=";".join(note_parts) if note_parts else None,
        )
        return (
            torch.tensor(eig_min, dtype=torch.float32),
            torch.from_numpy(_normalize_eigenvector(vec_min)).to(torch.float32).reshape(-1, 1),
            cache_hit,
            lambda_min_solver,
        )

    def _solve_lambda_max(self, matrix: csr_matrix) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            eig_max, vec_max = _eigsh_largest(
                matrix,
                tol=self.eigsh_tol,
                maxiter=self.eigsh_maxiter,
            )
            lambda_max_solver = "eigsh_la"
        except Exception:  # pragma: no cover - rare SciPy failures
            eig_max, vec_max = _dense_largest(matrix)
            lambda_max_solver = "dense_eigh"
        rel_residual = _relative_residual_norm(matrix, eig_max, vec_max)
        self._last_call_info["lambda_max_solver"] = lambda_max_solver
        self._last_call_info["lambda_max"] = float(eig_max)
        self._last_call_info["lambda_max_relative_residual"] = float(rel_residual)
        return (
            torch.tensor(eig_max, dtype=torch.float32),
            torch.from_numpy(_normalize_eigenvector(vec_max)).to(torch.float32).reshape(-1, 1),
        )

    def _apply_spectral_loss(
        self,
        *,
        A,
        pred_values: torch.Tensor,
        factor_rows: torch.Tensor,
        factor_cols: torch.Tensor,
        factor_values: torch.Tensor,
        active_pred_input_indices: torch.Tensor,
        active_rows: torch.Tensor,
        active_cols: torch.Tensor,
        active_scales: torch.Tensor,
        sample_id: str | None,
        step_diag: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        compute_dtype = pred_values.dtype
        device = factor_values.device
        timer_forward_total = _diag_timer_start(device, enabled=step_diag is not None)
        A_sparse = ensure_sparse_square(A, compute_dtype)
        spectral_dtype = torch.float64
        A_sparse_spectral = ensure_sparse_square(A, spectral_dtype)
        timer_l_sparse = _diag_timer_start(device, enabled=step_diag is not None)
        L_sparse = torch.sparse_coo_tensor(
            torch.stack([factor_rows, factor_cols], dim=0),
            factor_values,
            size=tuple(int(v) for v in A_sparse.shape),
            dtype=compute_dtype,
            device=factor_values.device,
        ).coalesce()
        L_sparse_spectral = torch.sparse_coo_tensor(
            torch.stack([factor_rows, factor_cols], dim=0),
            factor_values.to(dtype=spectral_dtype),
            size=tuple(int(v) for v in A_sparse.shape),
            dtype=spectral_dtype,
            device=factor_values.device,
        ).coalesce()
        _diag_timer_stop(
            step_diag,
            "l_sparse_build_time_sec",
            timer_l_sparse,
            device=device,
            enabled=step_diag is not None,
        )
        timer_pm = _diag_timer_start(device, enabled=step_diag is not None)
        pm_sparse = torch.sparse.mm(
            torch.sparse.mm(L_sparse_spectral, A_sparse_spectral),
            L_sparse_spectral.transpose(0, 1),
        ).coalesce()
        _diag_timer_stop(
            step_diag,
            "pm_build_time_sec",
            timer_pm,
            device=device,
            enabled=step_diag is not None,
        )
        self._step_counter += 1
        timer_gpu_to_cpu = _diag_timer_start(device, enabled=step_diag is not None)
        pm_matrix = torch_sparse_to_scipy_csr(pm_sparse).astype(np.float64, copy=False)
        # Numerical roundoff in float32 sparse matmuls can break symmetry enough to
        # pollute the smallest-eigenpair solve; enforce the intended SPD operator.
        pm_matrix = (0.5 * (pm_matrix + pm_matrix.transpose())).tocsr()
        _diag_timer_stop(
            step_diag,
            "gpu_to_cpu_time_sec",
            timer_gpu_to_cpu,
            device=device,
            enabled=step_diag is not None,
        )
        t_min = perf_counter()
        eig_min, vec_min, cache_hit, lambda_min_solver = self._solve_lambda_min(
            pm_matrix,
            sample_id=sample_id,
            seed=self._step_counter,
        )
        lambda_min_solve_time = float(perf_counter() - t_min)
        self._stats["lambda_min_solve_time_sec_sum"] += lambda_min_solve_time
        if cache_hit:
            self._stats["lambda_min_solve_time_sec_hit_sum"] += lambda_min_solve_time
        else:
            self._stats["lambda_min_solve_time_sec_miss_sum"] += lambda_min_solve_time
        if step_diag is not None:
            step_diag["lambda_min_solve_time_sec"] = lambda_min_solve_time
            step_diag["cache_hit"] = bool(cache_hit)
            step_diag["lambda_min_solver"] = lambda_min_solver

        t_max = perf_counter()
        eig_max, vec_max = self._solve_lambda_max(pm_matrix)
        lambda_max_solve_time = float(perf_counter() - t_max)
        self._stats["lambda_max_solve_time_sec_sum"] += lambda_max_solve_time
        if cache_hit:
            self._stats["lambda_max_solve_time_sec_when_hit_sum"] += lambda_max_solve_time
        else:
            self._stats["lambda_max_solve_time_sec_when_miss_sum"] += lambda_max_solve_time
        if step_diag is not None:
            step_diag["lambda_max_solve_time_sec"] = lambda_max_solve_time

        eig_min_value = float(eig_min.detach().cpu().item())
        eig_max_value = float(eig_max.detach().cpu().item())
        exact_cond = float(eig_max_value / max(eig_min_value, self.min_eigenvalue_eps))
        safe_eig_min_value = max(eig_min_value, self.logcond_eps)
        safe_eig_max_value = max(eig_max_value, self.logcond_eps)
        objective_value = float(np.log(safe_eig_max_value) - np.log(safe_eig_min_value))
        self._last_call_info["lambda_min"] = eig_min_value
        self._last_call_info["lambda_max"] = eig_max_value
        self._last_call_info["exact_cond"] = exact_cond
        self._last_call_info["objective_value"] = objective_value
        self._last_call_info["objective_variant"] = self.objective_variant

        target_device = factor_values.device
        target_dtype = factor_values.dtype
        timer_cpu_to_gpu = _diag_timer_start(target_device, enabled=step_diag is not None)
        eig_min = eig_min.to(device=target_device, dtype=target_dtype)
        eig_max = eig_max.to(device=target_device, dtype=target_dtype)
        vec_min = vec_min.to(device=target_device, dtype=target_dtype)
        vec_max = vec_max.to(device=target_device, dtype=target_dtype)
        _diag_timer_stop(
            step_diag,
            "cpu_to_gpu_time_sec",
            timer_cpu_to_gpu,
            device=target_device,
            enabled=step_diag is not None,
        )
        if step_diag is not None:
            step_diag["factor_nnz"] = int(L_sparse._nnz())
            step_diag["pm_nnz"] = int(pm_sparse._nnz())
            step_diag["active_pred_nnz"] = int(active_pred_input_indices.numel())
            step_diag["exact_cond"] = exact_cond
        loss = _ConditionNumberSpectralLoss.apply(
            A_sparse,
            pred_values,
            factor_rows,
            factor_cols,
            factor_values,
            active_pred_input_indices,
            active_rows,
            active_cols,
            active_scales,
            eig_min,
            eig_max,
            vec_min,
            vec_max,
            step_diag,
            step_diag is not None,
            self.logcond_eps,
        )
        _diag_timer_stop(
            step_diag,
            "spectral_loss_forward_time_sec",
            timer_forward_total,
            device=target_device,
            enabled=step_diag is not None,
        )
        return loss

    def _forward_sparse_prediction(
        self,
        A,
        prediction: SparseFactorPrediction,
        *,
        sample_id: str | None,
        mask_rows: torch.Tensor,
        mask_cols: torch.Tensor,
        mask_key: torch.Tensor,
        diag_inv: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = (
            torch.float32
            if prediction.values.dtype in (torch.float16, torch.bfloat16)
            else prediction.values.dtype
        )
        pred_values = prediction.values.to(dtype=compute_dtype)
        step_diag = self._begin_step_diag(matrix_size=int(prediction.shape[0]))
        timer_assembly = _diag_timer_start(pred_values.device, enabled=step_diag is not None)
        assembled = assemble_sparse_factor_from_prediction_torch(
            coords=prediction.coords,
            values=pred_values,
            mask_rows=mask_rows,
            mask_cols=mask_cols,
            mask_key=mask_key,
            diag_inv=diag_inv,
            shape=prediction.shape,
            diag_strategy=self.diag_strategy,
            force_unit_diag=True,
            use_mask_projection=bool(self.mask_projection_enabled),
        )
        _diag_timer_stop(
            step_diag,
            "factor_assembly_time_sec",
            timer_assembly,
            device=pred_values.device,
            enabled=step_diag is not None,
        )
        return self._apply_spectral_loss(
            A=A,
            pred_values=pred_values,
            factor_rows=assembled["sparse_rows"],
            factor_cols=assembled["sparse_cols"],
            factor_values=assembled["sparse_values"],
            active_pred_input_indices=assembled["active_pred_input_indices"],
            active_rows=assembled["active_rows"],
            active_cols=assembled["active_cols"],
            active_scales=assembled["active_scales"],
            sample_id=sample_id,
            step_diag=step_diag,
        )

    def _forward_dense_compat(self, A, L, mask, *, sample_id: str | None) -> torch.Tensor:
        compute_dtype = torch.float32 if L.dtype in (torch.float16, torch.bfloat16) else L.dtype
        n = int(L.shape[-1])
        step_diag = self._begin_step_diag(matrix_size=n)
        mask_rows, mask_cols, mask_key = _mask_support_from_mask(mask, n=n, device=L.device)
        gathered_values = L[mask_rows, mask_cols].to(dtype=compute_dtype)
        timer_assembly = _diag_timer_start(gathered_values.device, enabled=step_diag is not None)
        assembled = assemble_sparse_factor_from_prediction_torch(
            coords=torch.stack([mask_rows, mask_cols], dim=1),
            values=gathered_values,
            mask_rows=mask_rows,
            mask_cols=mask_cols,
            mask_key=mask_key,
            diag_inv=torch.ones(n, dtype=compute_dtype, device=L.device),
            shape=tuple(int(v) for v in L.shape),
            force_unit_diag=False,
        )
        _diag_timer_stop(
            step_diag,
            "factor_assembly_time_sec",
            timer_assembly,
            device=gathered_values.device,
            enabled=step_diag is not None,
        )
        return self._apply_spectral_loss(
            A=A,
            pred_values=gathered_values,
            factor_rows=assembled["sparse_rows"],
            factor_cols=assembled["sparse_cols"],
            factor_values=assembled["sparse_values"],
            active_pred_input_indices=assembled["active_pred_input_indices"],
            active_rows=assembled["active_rows"],
            active_cols=assembled["active_cols"],
            active_scales=assembled["active_scales"],
            sample_id=sample_id,
            step_diag=step_diag,
        )

    def forward(  # type: ignore[override]
        self,
        A,
        factor_or_prediction,
        mask=None,
        sample_id: str | None = None,
        *,
        mask_rows: torch.Tensor | None = None,
        mask_cols: torch.Tensor | None = None,
        mask_key: torch.Tensor | None = None,
        diag_inv: torch.Tensor | None = None,
    ):
        if isinstance(sample_id, (list, tuple)):
            raise NotImplementedError("batch>1 sample_id sequences are not yet implemented")
        if isinstance(factor_or_prediction, SparseFactorPrediction):
            if mask_rows is None or mask_cols is None or mask_key is None or diag_inv is None:
                raise ValueError(
                    "sparse prediction loss requires mask_rows, mask_cols, mask_key, and diag_inv"
                )
            return self._forward_sparse_prediction(
                A,
                factor_or_prediction,
                sample_id=sample_id,
                mask_rows=mask_rows,
                mask_cols=mask_cols,
                mask_key=mask_key,
                diag_inv=diag_inv,
            )
        if mask is None:
            raise ValueError("dense compatibility path requires mask")
        return self._forward_dense_compat(A, factor_or_prediction, mask, sample_id=sample_id)


def reset_condition_number_loss_eigs_cache() -> None:
    """Reset cached spectral states for the default condition-number loss."""
    condition_number_loss_eigs.clear_cache()
    condition_number_loss_eigs.reset_stats()


def condition_number_loss_masked(A, L, mask, sample_id: str | None = None):
    """Dense masked condition-number loss used as a correctness reference."""
    del sample_id
    mask = mask.to_dense() if getattr(mask, "is_sparse", False) else mask
    A_matrix = squeeze_batch_matrix(A) if isinstance(A, torch.Tensor) else A
    compute_dtype = torch.float32 if L.dtype in (torch.float16, torch.bfloat16) else L.dtype
    if isinstance(A_matrix, torch.Tensor) and getattr(A_matrix, "is_sparse", False):
        A_matrix = A_matrix.to_dense()
    if isinstance(A_matrix, torch.Tensor):
        A_matrix = A_matrix.to(dtype=compute_dtype)
    L_masked = (L * mask).to(dtype=compute_dtype)
    pm = L_masked @ A_matrix @ L_masked.t()
    return torch.linalg.cond(pm)


def torch_cond_metric(matrix: torch.Tensor) -> torch.Tensor:
    """Compute `torch.linalg.cond` in float64 for validation and smoke tests."""
    if matrix.ndim < 2:
        raise ValueError("torch_cond_metric expects a matrix or a batch of matrices")
    return torch.linalg.cond(matrix.to(dtype=torch.float64))


ConditionNumberLoss = CachedConditionNumberLoss
condition_number_loss_eigs = CachedConditionNumberLoss()
cond_loss = condition_number_loss_eigs
