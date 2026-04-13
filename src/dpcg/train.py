"""训练入口与训练循环（面向 NN 预处理器）。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

try:
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional
    torch = None
    optim = None
    SummaryWriter = None

import numpy as np
import yaml

import dpcg.data as data_mod
import dpcg.losses as losses_mod
import dpcg.models as models_mod
from dpcg.utils import (
    inverse_sqrt_diagonal,
    normalize_diag_strategy,
    squeeze_batch_matrix,
    torch_sparse_to_spconv,
)

_GRAPHNET_DEFAULT_LR = 5.0e-4
_GRAPHNET_DEFAULT_GRAD_CLIP_NORM = 1.0
_SUNET0_DEFAULT_LR = 1.0e-3
_SUNET0_DEFAULT_GRAD_CLIP_NORM = 1.0
_CONVNET_DEFAULT_LR = 1.0e-4
_CONVNET_DEFAULT_GRAD_CLIP_NORM = 1.0
_STEP_PROFILE_KEYS = (
    "prepare_batch_time_sec",
    "model_forward_time_sec",
    "sparse_prediction_to_dense_time_sec",
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
    "backward_time_sec",
    "optimizer_step_time_sec",
)


def _require_torch():
    if torch is None or optim is None:
        raise RuntimeError("torch is required for training")


def _capture_rng_state() -> dict[str, Any]:
    """采集当前 RNG 状态（便于断点恢复后完全复现）。"""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state() if torch else None,
    }
    if torch and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """恢复 RNG 状态（需在 torch 可用时调用）。"""
    if not state:
        return
    random.setstate(state.get("python", random.getstate()))
    np.random.set_state(state.get("numpy", np.random.get_state()))
    if torch and state.get("torch") is not None:
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state.to(device="cpu", dtype=torch.uint8))
    if torch and torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = []
        for cuda_state in state["torch_cuda"]:
            if not isinstance(cuda_state, torch.Tensor):
                cuda_state = torch.as_tensor(cuda_state, dtype=torch.uint8)
            cuda_states.append(cuda_state.to(device="cpu", dtype=torch.uint8))
        torch.cuda.set_rng_state_all(cuda_states)


def _grad_norm(model) -> float:
    """计算所有参数梯度的 2 范数（用于诊断训练稳定性）。"""
    if torch is None:
        return 0.0
    norms = []
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().data.norm(2)
        norms.append(param_norm)
    if not norms:
        return 0.0
    return torch.norm(torch.stack(norms), 2).item()


def _get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _count_parameters(model) -> tuple[int, int]:
    total = sum(int(param.numel()) for param in model.parameters())
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return total, trainable


def _checkpoint_size_mb(path: str | None) -> float | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return float(file_path.stat().st_size) / (1024.0 * 1024.0)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _model_metadata(model) -> dict[str, Any]:
    total_params, trainable_params = _count_parameters(model)
    metadata = {
        "model_class": type(model).__name__,
        "input_kind": getattr(model, "input_kind", None),
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
    }
    for key in (
        "channels",
        "block_depth",
        "use_sparse_head",
        "latent_dim",
        "hidden_dim",
        "num_message_passing",
        "conv_algo",
    ):
        value = getattr(model, key, None)
        if value is not None:
            metadata[key] = value
    return metadata


def _write_epoch_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [
            "epoch",
            "train_loss",
            "val_loss",
            "train_exact_cond_avg",
            "train_exact_cond_last",
            "val_exact_cond_avg",
            "val_exact_cond_last",
            "grad_norm",
            "lr",
            "train_epoch_time_sec",
            "train_step_time_sec_avg",
            "peak_gpu_memory_allocated_mb",
            "peak_gpu_memory_reserved_mb",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_training_summary(
    *,
    model,
    device: str,
    total_time_sec: float,
    best_epoch: int | None,
    best_val_loss: float | None,
    train_file: str | None,
    last_checkpoint_path: str | None,
    history_rows: list[dict[str, Any]],
    loss_name: str | None = None,
    amp_enabled: bool | None = None,
    diag_profile_enabled: bool = False,
    diagnostic_rows: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    loss_state: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = _model_metadata(model)
    summary.update(
        {
            "device": device,
            "total_training_time_sec": float(total_time_sec),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_checkpoint": train_file,
            "last_checkpoint": last_checkpoint_path,
            "best_checkpoint_size_mb": _checkpoint_size_mb(train_file),
            "last_checkpoint_size_mb": _checkpoint_size_mb(last_checkpoint_path),
            "num_logged_epochs": len(history_rows),
            "loss_name": loss_name,
            "amp_enabled": amp_enabled,
            "diag_profile_enabled": bool(diag_profile_enabled),
            "git_commit_hash": _git_commit_hash(),
            "torch_version": None if torch is None else torch.__version__,
            "config": config,
            "loss_state": loss_state,
            "split_manifest": split_manifest,
        }
    )
    if split_manifest is not None:
        summary["split_counts"] = split_manifest.get("split_counts")
        summary["family_counts_by_split"] = split_manifest.get("family_counts_by_split")
        summary["dataset_root"] = split_manifest.get("dataset_root")
        summary["split_source"] = split_manifest.get("split_source")
    if history_rows:
        epoch_times = [
            row["train_epoch_time_sec"]
            for row in history_rows
            if row["train_epoch_time_sec"] is not None
        ]
        step_times = [
            row["train_step_time_sec_avg"]
            for row in history_rows
            if row["train_step_time_sec_avg"] is not None
        ]
        alloc_peaks = [
            row["peak_gpu_memory_allocated_mb"]
            for row in history_rows
            if row["peak_gpu_memory_allocated_mb"] is not None
        ]
        reserve_peaks = [
            row["peak_gpu_memory_reserved_mb"]
            for row in history_rows
            if row["peak_gpu_memory_reserved_mb"] is not None
        ]
        summary["train_epoch_time_sec_avg"] = float(np.mean(epoch_times)) if epoch_times else None
        summary["train_step_time_sec_avg"] = float(np.mean(step_times)) if step_times else None
        summary["peak_gpu_memory_allocated_mb"] = (
            float(np.max(alloc_peaks)) if alloc_peaks else None
        )
        summary["peak_gpu_memory_reserved_mb"] = (
            float(np.max(reserve_peaks)) if reserve_peaks else None
        )
    else:
        summary["train_epoch_time_sec_avg"] = None
        summary["train_step_time_sec_avg"] = None
        summary["peak_gpu_memory_allocated_mb"] = None
        summary["peak_gpu_memory_reserved_mb"] = None
    if diagnostic_rows:
        diag_summary = {
            "num_profiled_epochs": len(diagnostic_rows),
            "last_epoch_profile": diagnostic_rows[-1],
        }
        for key in _STEP_PROFILE_KEYS:
            avg_key = f"{key}_avg"
            max_key = f"{key}_max"
            avg_values = [row[avg_key] for row in diagnostic_rows if row.get(avg_key) is not None]
            max_values = [row[max_key] for row in diagnostic_rows if row.get(max_key) is not None]
            diag_summary[avg_key] = float(np.mean(avg_values)) if avg_values else None
            diag_summary[max_key] = float(np.max(max_values)) if max_values else None
        summary["diag_profile"] = diag_summary
    else:
        summary["diag_profile"] = {"num_profiled_epochs": 0, "last_epoch_profile": None}
    return summary


def _sync_training_device(device: str) -> None:
    if torch is None or not torch.cuda.is_available():
        return
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device=device)


def _loss_name(loss_fn) -> str:
    if hasattr(loss_fn, "loss_name"):
        return str(getattr(loss_fn, "loss_name"))
    return str(getattr(loss_fn, "__name__", type(loss_fn).__name__))


def _is_condition_number_loss_eigs(loss_fn) -> bool:
    return _loss_name(loss_fn) == "condition_number_loss_eigs"


def _reset_loss_state(loss_fn, *, split: str) -> None:
    if hasattr(loss_fn, "start_epoch"):
        loss_fn.start_epoch(split=split)
        return
    if _is_condition_number_loss_eigs(loss_fn):
        losses_mod.reset_condition_number_loss_eigs_cache()


def _loss_state_summary(loss_fn) -> dict[str, Any] | None:
    summary = loss_fn.get_cache_stats() if hasattr(loss_fn, "get_cache_stats") else None
    if hasattr(loss_fn, "diag_strategy"):
        if summary is None:
            summary = {}
        summary["diag_strategy"] = str(getattr(loss_fn, "diag_strategy"))
    if hasattr(loss_fn, "mask_projection_enabled"):
        if summary is None:
            summary = {}
        summary["mask_projection_enabled"] = bool(getattr(loss_fn, "mask_projection_enabled"))
    return summary


def _loss_last_exact_cond(loss_fn) -> float | None:
    if not hasattr(loss_fn, "get_cache_stats"):
        return None
    try:
        stats = loss_fn.get_cache_stats()
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    last_call_info = stats.get("last_call_info")
    if not isinstance(last_call_info, dict):
        return None
    exact_cond = last_call_info.get("exact_cond")
    if exact_cond is None:
        return None
    return float(exact_cond)


def _resolve_loss(loss_name: str):
    loss_fn = getattr(losses_mod, loss_name)
    if isinstance(loss_fn, losses_mod.CachedConditionNumberLoss):
        return losses_mod.CachedConditionNumberLoss(
            max_entries=loss_fn.max_entries,
            eigsh_tol=loss_fn.eigsh_tol,
            eigsh_maxiter=loss_fn.eigsh_maxiter,
            eigsh_fallback_maxiter=loss_fn.eigsh_fallback_maxiter,
            enable_train_cache=loss_fn.enable_train_cache,
            enable_val_cache=loss_fn.enable_val_cache,
            relative_residual_tol=loss_fn.relative_residual_tol,
            min_eigenvalue_eps=loss_fn.min_eigenvalue_eps,
            logcond_eps=loss_fn.logcond_eps,
            diag_strategy=getattr(loss_fn, "diag_strategy", "learned_exp"),
        )
    return loss_fn


def _compute_loss(loss_fn, prepared: dict[str, Any], conditioner) -> Any:
    sample_id = prepared.get("sample_id")
    if _is_condition_number_loss_eigs(loss_fn) and isinstance(
        conditioner, models_mod.SparseFactorPrediction
    ):
        return loss_fn(
            prepared["matrix_sparse_2d"],
            conditioner,
            sample_id=sample_id,
            mask_rows=prepared["mask_rows"],
            mask_cols=prepared["mask_cols"],
            mask_key=prepared["mask_key"],
            diag_inv=prepared["diag_inv"],
        )
    dense_conditioner = _materialize_model_output(
        conditioner,
        prepared["matrix_sparse"],
        diag_inv=prepared["diag_inv"],
    )
    return loss_fn(
        prepared["matrix_sparse_2d"],
        dense_conditioner,
        prepared["mask_tensor"],
        sample_id=sample_id,
    )


def _summarize_step_profiles(step_profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not step_profiles:
        return None
    summary: dict[str, Any] = {"num_profiled_steps": len(step_profiles)}
    for key in ("mask_nnz", "mask_diag_nnz", "mask_offdiag_nnz"):
        values = [row[key] for row in step_profiles if row.get(key) is not None]
        summary[key] = int(values[-1]) if values else 0
    for key in _STEP_PROFILE_KEYS:
        values = [row[key] for row in step_profiles if row.get(key) is not None]
        summary[f"{key}_avg"] = float(np.mean(values)) if values else None
        summary[f"{key}_max"] = float(np.max(values)) if values else None
    return summary


def _summarize_exact_cond(exact_conds: list[float]) -> dict[str, float | None]:
    if not exact_conds:
        return {"avg": None, "last": None}
    return {
        "avg": float(np.mean(exact_conds)),
        "last": float(exact_conds[-1]),
    }


def _mask_profile(mask_rows, mask_cols) -> dict[str, int]:
    if torch is None:
        return {"mask_nnz": 0, "mask_diag_nnz": 0, "mask_offdiag_nnz": 0}
    rows = mask_rows.to(dtype=torch.int64)
    cols = mask_cols.to(dtype=torch.int64)
    total_nnz = int(rows.numel())
    diag_nnz = int(torch.count_nonzero(rows == cols).item())
    return {
        "mask_nnz": total_nnz,
        "mask_diag_nnz": diag_nnz,
        "mask_offdiag_nnz": total_nnz - diag_nnz,
    }


def _make_scheduler(optimizer, name: str | None, kwargs: dict[str, Any] | None):
    """按配置构造学习率调度器。"""
    if not name:
        return None
    if torch is None:
        raise RuntimeError("torch is required for scheduler")
    kwargs = kwargs or {}
    name_lower = name.lower()
    if name_lower == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name_lower == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name_lower == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    raise ValueError(f"Unsupported scheduler: {name}")


def _make_grad_scaler(enabled: bool):
    if torch is None:
        raise RuntimeError("torch is required for AMP training")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if torch is None:
        raise RuntimeError("torch is required for AMP training")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _resolve_training_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        requested = "cuda"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA is required but not available in this environment")
    return requested


def _build_model_input(model, batch, device: str):
    _m_tensor, _solutions, _rhs, l_tensor, _ref, _mask = batch
    input_kind = getattr(model, "input_kind", "spconv")
    if input_kind == "graph":
        return data_mod.sample_to_half_graph(batch).to(device)
    return torch_sparse_to_spconv(l_tensor.to(device))


def _materialize_model_output(model_output, matrix, diag_inv=None):
    if isinstance(model_output, models_mod.SparseFactorPrediction):
        return models_mod.sparse_prediction_to_dense(model_output, matrix, diag_inv=diag_inv)
    return model_output


def _prepare_training_batch(model, batch, device: str) -> dict[str, Any]:
    m_tensor, _solutions, _rhs, _l_tensor, _ref, mask = batch
    m_tensor = m_tensor.to(device)
    matrix_sparse_2d = squeeze_batch_matrix(m_tensor)
    n = int(matrix_sparse_2d.shape[-1])
    if hasattr(batch, "mask_rows") and getattr(batch, "mask_rows") is not None:
        mask_rows = batch.mask_rows.to(device=device, dtype=torch.int64)
        mask_cols = batch.mask_cols.to(device=device, dtype=torch.int64)
        mask_key = batch.mask_key.to(device=device, dtype=torch.int64)
    else:
        mask_sparse = mask.to(device)
        if getattr(mask_sparse, "is_sparse", False):
            coords = mask_sparse.coalesce().indices().to(dtype=torch.int64)
        else:
            coords = torch.nonzero(mask_sparse, as_tuple=False).t().to(dtype=torch.int64)
        mask_rows = coords[0].reshape(-1)
        mask_cols = coords[1].reshape(-1)
        mask_key = mask_rows * n + mask_cols
    if hasattr(batch, "diag_inv") and getattr(batch, "diag_inv") is not None:
        diag_inv = batch.diag_inv.to(device)
    else:
        diag_inv = inverse_sqrt_diagonal(matrix_sparse_2d, dtype=torch.float32).to(device)
    return {
        "sample_id": getattr(batch, "sample_id", None),
        "matrix_sparse": m_tensor,
        "matrix_sparse_2d": matrix_sparse_2d,
        "mask_tensor": mask.to(device),
        "mask_rows": mask_rows,
        "mask_cols": mask_cols,
        "mask_key": mask_key,
        "diag_inv": diag_inv,
        "model_input": _build_model_input(model, batch, device),
    }


def train_one_epoch(
    model,
    loss_fn,
    loader,
    optimizer,
    device: str,
    scaler=None,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    capture_grads: bool = False,
    diag_profile: bool = False,
):
    """训练一个 epoch，支持 AMP、梯度累积和裁剪。"""
    model.train()
    if hasattr(loss_fn, "set_diag_enabled"):
        loss_fn.set_diag_enabled(bool(diag_profile))
    epoch_loss = 0.0
    grad_norms = []
    last_grads = None
    step_profiles: list[dict[str, Any]] = []
    exact_conds: list[float] = []
    warned_zero_offdiag_mask = False
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        step_profile = {key: None for key in _STEP_PROFILE_KEYS} if diag_profile else None
        if diag_profile:
            _sync_training_device(device)
            t_prepare = time.perf_counter()
        prepared = _prepare_training_batch(model, batch, device)
        mask_stats = _mask_profile(prepared["mask_rows"], prepared["mask_cols"])
        if not warned_zero_offdiag_mask and mask_stats["mask_offdiag_nnz"] == 0:
            print(
                "Warning: training mask contains only diagonal entries; "
                "sparse-factor predictions are masked out and gradients may be zero."
            )
            warned_zero_offdiag_mask = True
        if diag_profile and step_profile is not None:
            _sync_training_device(device)
            step_profile["prepare_batch_time_sec"] = float(time.perf_counter() - t_prepare)
            step_profile.update(mask_stats)
        with _autocast_context(use_amp):
            if diag_profile:
                _sync_training_device(device)
                t_forward = time.perf_counter()
            model_output = model(prepared["model_input"])
            if diag_profile and step_profile is not None:
                _sync_training_device(device)
                step_profile["model_forward_time_sec"] = float(time.perf_counter() - t_forward)
                step_profile["sparse_prediction_to_dense_time_sec"] = 0.0
        loss_raw = _compute_loss(loss_fn, prepared, model_output)
        exact_cond = _loss_last_exact_cond(loss_fn)
        if exact_cond is not None:
            exact_conds.append(exact_cond)
        loss = loss_raw / max(grad_accum_steps, 1)
        if diag_profile:
            _sync_training_device(device)
            t_backward = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if diag_profile and step_profile is not None and hasattr(loss_fn, "pop_last_step_diag"):
            loss_step_diag = loss_fn.pop_last_step_diag()
            if loss_step_diag:
                step_profile.update(loss_step_diag)
        if diag_profile and step_profile is not None:
            _sync_training_device(device)
            step_profile["backward_time_sec"] = float(time.perf_counter() - t_backward)
        epoch_loss += float(loss_raw.detach().cpu().item())

        if (step + 1) % max(grad_accum_steps, 1) == 0 or (step + 1) == len(loader):
            if diag_profile:
                _sync_training_device(device)
                t_step = time.perf_counter()
            if scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if capture_grads:
                    last_grads = {
                        name: param.grad.detach().cpu().clone() if param.grad is not None else None
                        for name, param in model.named_parameters()
                    }
                grad_norms.append(_grad_norm(model))
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if capture_grads:
                    last_grads = {
                        name: param.grad.detach().cpu().clone() if param.grad is not None else None
                        for name, param in model.named_parameters()
                    }
                grad_norms.append(_grad_norm(model))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if diag_profile and step_profile is not None:
                _sync_training_device(device)
                step_profile["optimizer_step_time_sec"] = float(time.perf_counter() - t_step)
        if diag_profile and step_profile is not None:
            step_profiles.append(step_profile)

    grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    return (
        epoch_loss / max(len(loader), 1),
        grad_norm,
        last_grads,
        _summarize_step_profiles(step_profiles),
        _summarize_exact_cond(exact_conds),
    )


def validate_one_epoch(model, loss_fn, loader, device: str):
    """验证一个 epoch（无梯度）。"""
    model.eval()
    val_loss = 0.0
    exact_conds: list[float] = []
    if hasattr(loss_fn, "set_diag_enabled"):
        loss_fn.set_diag_enabled(False)
    _reset_loss_state(loss_fn, split="val")
    with torch.no_grad():
        for batch in loader:
            prepared = _prepare_training_batch(model, batch, device)
            loss = _compute_loss(loss_fn, prepared, model(prepared["model_input"]))
            val_loss += float(loss.detach().cpu().item())
            exact_cond = _loss_last_exact_cond(loss_fn)
            if exact_cond is not None:
                exact_conds.append(exact_cond)
    return val_loss / max(len(loader), 1), _summarize_exact_cond(exact_conds)


def train_model(
    model,
    loss_fn: Callable,
    loaders: Tuple,
    total_epochs: int,
    log_dir: str | None,
    train_file: str | None,
    resume_path: str | None = None,
    patience: int = 1000,
    device: str = "cpu",
    val_epoch: int = 5,
    lr: float = 1e-5,
    save_every: int = 1,
    last_checkpoint_path: str | None = None,
    grad_clip_norm: float | None = None,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scheduler_name: str | None = None,
    scheduler_kwargs: dict[str, Any] | None = None,
    log_histograms: bool = False,
    log_every: int = 1,
    diag_profile: bool = False,
    split_manifest: dict[str, Any] | None = None,
    split_manifest_path: str | None = None,
):
    """训练主循环，支持断点恢复、诊断日志与多种训练策略。"""
    _require_torch()
    if save_every < 1:
        save_every = 1
    if grad_accum_steps < 1:
        grad_accum_steps = 1
    if val_epoch < 1:
        val_epoch = 1
    if train_file is None and log_dir:
        train_file = str(Path(log_dir) / "best.ckpt")
    if last_checkpoint_path is None and train_file is not None:
        last_checkpoint_path = str(Path(train_file).with_name("last.ckpt"))
    history_path = Path(log_dir) / "epoch_metrics.csv" if log_dir else None

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _make_scheduler(optimizer, scheduler_name, scheduler_kwargs)
    amp_enabled = bool(use_amp and torch.cuda.is_available() and str(device).startswith("cuda"))
    cuda_training = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    scaler = _make_grad_scaler(enabled=amp_enabled)
    initial_epoch = 0
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter and log_dir else None
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch: int | None = None
    no_improvement_epochs = 0
    history_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    train_loader, val_loader, _iid_loader, _ood_loader = loaders
    if hasattr(loss_fn, "clear_cache"):
        loss_fn.clear_cache()
    if hasattr(loss_fn, "reset_stats"):
        loss_fn.reset_stats()

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        initial_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        _restore_rng_state(checkpoint.get("rng_state", {}))

    model.train()
    start_time = time.time()
    if writer and split_manifest_path:
        writer.add_text("data/split_manifest_path", split_manifest_path, 0)
    if writer and split_manifest:
        split_counts = split_manifest.get("split_counts", {})
        writer.add_text(
            "data/split_manifest_summary",
            f"n_total={split_manifest.get('n_total')} "
            f"n_train={split_counts.get('train', 0)} "
            f"n_val={split_counts.get('val', 0)} "
            f"n_iid={split_counts.get('iid', 0)} "
            f"n_ood={split_counts.get('ood', 0)}",
            0,
        )

    for epoch in range(initial_epoch, total_epochs):
        epoch_start = time.time()
        if cuda_training:
            torch.cuda.reset_peak_memory_stats(device=device)
        _reset_loss_state(loss_fn, split="train")
        epoch_loss, grad_norm, last_grads, epoch_diag, train_exact_cond = train_one_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            device,
            scaler=scaler if amp_enabled else None,
            grad_accum_steps=grad_accum_steps,
            grad_clip_norm=grad_clip_norm,
            use_amp=amp_enabled,
            capture_grads=log_histograms and writer is not None and epoch % log_every == 0,
            diag_profile=diag_profile,
        )
        epoch_time = time.time() - epoch_start
        step_time_avg = epoch_time / max(len(train_loader), 1)
        lr_value = _get_lr(optimizer)
        peak_allocated_mb = None
        peak_reserved_mb = None
        if cuda_training:
            peak_allocated_mb = float(torch.cuda.max_memory_allocated(device=device)) / (
                1024.0 * 1024.0
            )
            peak_reserved_mb = float(torch.cuda.max_memory_reserved(device=device)) / (
                1024.0 * 1024.0
            )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(epoch_loss),
                "val_loss": None,
                "train_exact_cond_avg": train_exact_cond["avg"],
                "train_exact_cond_last": train_exact_cond["last"],
                "val_exact_cond_avg": None,
                "val_exact_cond_last": None,
                "grad_norm": float(grad_norm),
                "lr": float(lr_value),
                "train_epoch_time_sec": float(epoch_time),
                "train_step_time_sec_avg": float(step_time_avg),
                "peak_gpu_memory_allocated_mb": peak_allocated_mb,
                "peak_gpu_memory_reserved_mb": peak_reserved_mb,
            }
        )
        if epoch_diag is not None:
            diagnostic_rows.append({"epoch": epoch, **epoch_diag})
            for key, value in epoch_diag.items():
                history_rows[-1][key] = value
        if writer and epoch % log_every == 0:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            if train_exact_cond["avg"] is not None:
                writer.add_scalar("ExactCond/train", train_exact_cond["avg"], epoch)
            if train_exact_cond["last"] is not None:
                writer.add_scalar("ExactCond/train_last", train_exact_cond["last"], epoch)
            writer.add_scalar("Grad/norm", grad_norm, epoch)
            writer.add_scalar("LR", lr_value, epoch)
            writer.add_scalar("Time/train_epoch_sec", epoch_time, epoch)
            writer.add_scalar("Time/train_step_avg_sec", step_time_avg, epoch)
            if epoch_diag is not None:
                for key in _STEP_PROFILE_KEYS:
                    avg_key = f"{key}_avg"
                    if epoch_diag.get(avg_key) is not None:
                        writer.add_scalar(f"Diag/{avg_key}", epoch_diag[avg_key], epoch)
            if peak_allocated_mb is not None:
                writer.add_scalar("Memory/peak_allocated_mb", peak_allocated_mb, epoch)
            if peak_reserved_mb is not None:
                writer.add_scalar("Memory/peak_reserved_mb", peak_reserved_mb, epoch)
            if log_histograms:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Params/{name}", param.detach().cpu(), epoch)
                    grad = None if last_grads is None else last_grads.get(name)
                    if grad is not None:
                        writer.add_histogram(f"Grads/{name}", grad, epoch)

        val_loss = None
        if epoch % val_epoch == 0:
            _reset_loss_state(loss_fn, split="val")
            val_loss, val_exact_cond = validate_one_epoch(model, loss_fn, val_loader, device)
            history_rows[-1]["val_loss"] = float(val_loss)
            history_rows[-1]["val_exact_cond_avg"] = val_exact_cond["avg"]
            history_rows[-1]["val_exact_cond_last"] = val_exact_cond["last"]
            if writer and epoch % log_every == 0:
                writer.add_scalar("Loss/val", val_loss, epoch)
                if val_exact_cond["avg"] is not None:
                    writer.add_scalar("ExactCond/val", val_exact_cond["avg"], epoch)
                if val_exact_cond["last"] is not None:
                    writer.add_scalar("ExactCond/val_last", val_exact_cond["last"], epoch)
            print(
                f"Epoch {epoch}, train={epoch_loss:.6e}, val={val_loss:.6e}, "
                f"grad_norm={grad_norm:.6e}, lr={lr_value:.3e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                if train_file:
                    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                            "scaler_state_dict": scaler.state_dict() if scaler else None,
                            "loss": epoch_loss,
                            "val_loss": val_loss,
                            "best_val_loss": best_val_loss,
                            "rng_state": _capture_rng_state(),
                            "split_manifest_path": split_manifest_path,
                        },
                        train_file,
                    )
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        if last_checkpoint_path and (epoch + 1) % save_every == 0:
            Path(last_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "loss": epoch_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "rng_state": _capture_rng_state(),
                    "split_manifest_path": split_manifest_path,
                },
                last_checkpoint_path,
            )

        if history_path is not None:
            _write_epoch_metrics_csv(history_path, history_rows)

        if val_loss is not None and no_improvement_epochs > patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif train_file and Path(train_file).exists():
        checkpoint = torch.load(train_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    total_time = time.time() - start_time
    summary_path = Path(log_dir) / "training_summary.json" if log_dir else None
    if writer:
        writer.close()
    if history_path is not None:
        _write_epoch_metrics_csv(history_path, history_rows)
    summary = build_training_summary(
        model=model,
        device=device,
        total_time_sec=total_time,
        best_epoch=best_epoch,
        best_val_loss=None if not np.isfinite(best_val_loss) else float(best_val_loss),
        train_file=train_file,
        last_checkpoint_path=last_checkpoint_path,
        history_rows=history_rows,
        loss_name=_loss_name(loss_fn),
        amp_enabled=amp_enabled,
        diag_profile_enabled=diag_profile,
        diagnostic_rows=diagnostic_rows,
        config=None,
        loss_state=_loss_state_summary(loss_fn),
        split_manifest=split_manifest,
    )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return model, total_time


def _get(cfg: Dict, key: str, default=None):
    if "train" in cfg:
        return cfg["train"].get(key, default)
    return cfg.get(key, default)


def _resolve_optimization_defaults(
    model_name: str,
    lr: float | None,
    grad_clip_norm: float | None,
) -> tuple[float, float | None]:
    """Resolve model-specific optimization defaults."""
    if lr is None:
        if model_name == "graphnet":
            lr = _GRAPHNET_DEFAULT_LR
        elif model_name in {"sunet0", "sunet_aspp"}:
            lr = _SUNET0_DEFAULT_LR
        elif model_name == "convnet":
            lr = _CONVNET_DEFAULT_LR
        else:
            lr = 1e-5
    if grad_clip_norm is None:
        if model_name == "graphnet":
            grad_clip_norm = _GRAPHNET_DEFAULT_GRAD_CLIP_NORM
        elif model_name in {"sunet0", "sunet_aspp"}:
            grad_clip_norm = _SUNET0_DEFAULT_GRAD_CLIP_NORM
        elif model_name == "convnet":
            grad_clip_norm = _CONVNET_DEFAULT_GRAD_CLIP_NORM
    return float(lr), None if grad_clip_norm is None else float(grad_clip_norm)


def train_from_cfg(cfg: Dict, model_name: str, loss_name: str):
    _require_torch()
    seed = _get(cfg, "SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = _resolve_training_device(_get(cfg, "DEVICE", "cpu"))
    data_root = _get(cfg, "DATA_ROOT", "./data")
    n_train = int(_get(cfg, "PC_TRAIN", 200))
    n_val = int(_get(cfg, "PC_VAL", 10))
    n_test = int(_get(cfg, "PC_TEST", 2))
    mask_percentile = float(_get(cfg, "MASK_PERCENTILE", 50.0))
    split_source = str(_get(cfg, "DATASET_SPLIT_SOURCE", "meta_json"))
    train_max_samples_per_split = _get(cfg, "TRAIN_MAX_SAMPLES_PER_SPLIT", None)
    val_max_samples_per_split = _get(cfg, "VAL_MAX_SAMPLES_PER_SPLIT", None)
    iid_max_samples_per_split = _get(cfg, "IID_MAX_SAMPLES_PER_SPLIT", None)
    ood_max_samples_per_split = _get(cfg, "OOD_MAX_SAMPLES_PER_SPLIT", None)
    iid_family_quotas = _get(cfg, "IID_FAMILY_QUOTAS", None)
    ood_family_quotas = _get(cfg, "OOD_FAMILY_QUOTAS", None)
    train_max_samples_per_split = (
        int(train_max_samples_per_split) if train_max_samples_per_split is not None else None
    )
    val_max_samples_per_split = (
        int(val_max_samples_per_split) if val_max_samples_per_split is not None else None
    )
    iid_max_samples_per_split = (
        int(iid_max_samples_per_split) if iid_max_samples_per_split is not None else None
    )
    ood_max_samples_per_split = (
        int(ood_max_samples_per_split) if ood_max_samples_per_split is not None else None
    )
    epochs = int(_get(cfg, "EPOCHS", _get(cfg, "epochs", 10)))
    raw_lr = _get(cfg, "INI_LR", _get(cfg, "lr", None))
    val_epoch = int(_get(cfg, "VAL_EPOCH", _get(cfg, "val_epoch", 5)))
    log_dir = _get(cfg, "LOG-DIR_M", _get(cfg, "log_dir", None))
    train_file = _get(cfg, "TRAIN_FILE_M", _get(cfg, "train_file", None))
    resume_path = _get(cfg, "MODEL_STATE_M", _get(cfg, "resume", None))
    save_every = int(_get(cfg, "SAVE_EVERY", _get(cfg, "save_every", 1)))
    last_checkpoint = _get(cfg, "LAST_CHECKPOINT", _get(cfg, "last_checkpoint", None))
    raw_grad_clip_norm = _get(cfg, "GRAD_CLIP_NORM", _get(cfg, "grad_clip_norm", None))
    grad_accum_steps = int(_get(cfg, "GRAD_ACCUM_STEPS", _get(cfg, "grad_accum_steps", 1)))
    use_amp = bool(_get(cfg, "AMP", _get(cfg, "use_amp", False)))
    diag_profile = bool(_get(cfg, "DIAG_PROFILE", _get(cfg, "diag_profile", False)))
    log_histograms = bool(_get(cfg, "LOG_HISTOGRAMS", _get(cfg, "log_histograms", False)))
    log_every = int(_get(cfg, "LOG_EVERY", _get(cfg, "log_every", 1)))
    split_manifest_path = _get(cfg, "SPLIT_MANIFEST", _get(cfg, "split_manifest", None))
    if split_manifest_path is None and log_dir:
        split_manifest_path = str(Path(log_dir) / "split_manifest.json")

    num_workers = int(_get(cfg, "NUM_WORKERS", _get(cfg, "num_workers", 0)))
    pin_memory = bool(_get(cfg, "PIN_MEMORY", _get(cfg, "pin_memory", False)))
    persistent_workers = bool(
        _get(cfg, "PERSISTENT_WORKERS", _get(cfg, "persistent_workers", False))
    )
    prefetch_factor = _get(cfg, "PREFETCH_FACTOR", _get(cfg, "prefetch_factor", None))
    prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else None
    drop_last = bool(_get(cfg, "DROP_LAST", _get(cfg, "drop_last", False)))
    shuffle_train = bool(_get(cfg, "SHUFFLE_TRAIN", _get(cfg, "shuffle_train", True)))
    cache_samples = bool(_get(cfg, "CACHE_SAMPLES", _get(cfg, "cache_samples", True)))
    cache_graph = bool(_get(cfg, "CACHE_GRAPH", _get(cfg, "cache_graph", True)))
    diag_strategy = normalize_diag_strategy(
        str(_get(cfg, "DIAG_STRATEGY", _get(cfg, "diag_strategy", "learned_exp")))
    )

    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    scheduler_cfg = train_cfg.get("scheduler", {}) if isinstance(train_cfg, dict) else {}
    scheduler_name = scheduler_cfg.get("name", _get(cfg, "SCHEDULER", _get(cfg, "scheduler", None)))
    scheduler_kwargs = scheduler_cfg.get("kwargs", _get(cfg, "SCHEDULER_KWARGS", {}))
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    lr, grad_clip_norm = _resolve_optimization_defaults(model_name, raw_lr, raw_grad_clip_norm)
    model_kwargs = models_mod.normalize_model_kwargs(
        _get(cfg, "MODEL_KWARGS", _get(cfg, "model_kwargs", {}))
    )

    loaders = data_mod.build_dataloaders(
        data_root=data_root,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        mask_percentile=mask_percentile,
        split_manifest_path=split_manifest_path,
        return_manifest=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        shuffle_train=shuffle_train,
        cache_samples=cache_samples,
        cache_graph=cache_graph,
        split_source=split_source,
        train_max_samples_per_split=train_max_samples_per_split,
        val_max_samples_per_split=val_max_samples_per_split,
        iid_max_samples_per_split=iid_max_samples_per_split,
        ood_max_samples_per_split=ood_max_samples_per_split,
        iid_family_quotas=iid_family_quotas,
        ood_family_quotas=ood_family_quotas,
    )

    model = models_mod.load_model(
        model_name,
        model_kwargs=model_kwargs,
        device=device,
    )
    loss_fn = _resolve_loss(loss_name)
    if hasattr(loss_fn, "set_diag_strategy"):
        loss_fn.set_diag_strategy(diag_strategy)
    if hasattr(loss_fn, "set_mask_projection_enabled"):
        loss_fn.set_mask_projection_enabled(mask_percentile >= 0.0)

    train_loader, val_loader, iid_loader, ood_loader, manifest = loaders
    model, total_time = train_model(
        model,
        loss_fn,
        (train_loader, val_loader, iid_loader, ood_loader),
        total_epochs=epochs,
        log_dir=log_dir,
        train_file=train_file,
        resume_path=resume_path,
        patience=int(_get(cfg, "patience", 1000)),
        device=device,
        val_epoch=val_epoch,
        lr=lr,
        save_every=save_every,
        last_checkpoint_path=last_checkpoint,
        grad_clip_norm=grad_clip_norm,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        diag_profile=diag_profile,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        log_histograms=log_histograms,
        log_every=log_every,
        split_manifest=manifest,
        split_manifest_path=split_manifest_path,
    )
    summary_path = Path(log_dir) / "training_summary.json" if log_dir else None
    if summary_path is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["config"] = cfg
        summary["git_commit_hash"] = _git_commit_hash()
        summary["torch_version"] = None if torch is None else torch.__version__
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return model, total_time


def load_config(path: str) -> Dict:
    cfg_path = Path(path)
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(cfg_path.read_text())
    if cfg_path.suffix.lower() == ".py":
        import importlib.util

        spec = importlib.util.spec_from_file_location("cfg_module", str(cfg_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return getattr(module, "CFG")
    raise ValueError("Config must be .yaml/.yml or .py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/train_sunet0_tail_default.yaml")
    parser.add_argument("--model", type=str, default="sunet0")
    parser.add_argument("--loss", type=str, default="cond_loss")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_from_cfg(cfg, model_name=args.model, loss_name=args.loss)


if __name__ == "__main__":
    main()
