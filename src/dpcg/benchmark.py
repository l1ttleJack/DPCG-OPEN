"""Benchmark learned and classical preconditioners on a common SciPy CG path."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags, tril
from scipy.sparse.linalg import (
    LinearOperator,
    eigs,
    eigsh,
    spilu,
    spsolve_triangular,
)
from scipy.sparse.linalg import (
    cg as scipy_cg,
)

from dpcg.sample import BenchmarkSample
from dpcg.utils import assemble_sparse_factor_from_prediction_numpy, normalize_diag_strategy

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from pyamg.aggregation import smoothed_aggregation_solver
except Exception:  # pragma: no cover - optional dependency
    smoothed_aggregation_solver = None

if TYPE_CHECKING:
    from dpcg.models import SparseFactorPrediction

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None

DEFAULT_METHODS = ("amg", "learning")
PETSC_GPU_DEFAULT_METHODS = ("amg", "learning")
SCIPY_SUPPORTED_METHODS = ("none", "jacobi", "sgs", "ssor", "ic0", "ilu", "amg", "learning")
PETSC_GPU_SUPPORTED_METHODS = ("none", "jacobi", "sgs", "ssor", "ic0", "amg", "parasails", "learning")
SUPPORTED_METHODS = frozenset(SCIPY_SUPPORTED_METHODS + PETSC_GPU_SUPPORTED_METHODS)
SUPPORTED_METHODS_BY_BACKEND = {
    "scipy": frozenset(SCIPY_SUPPORTED_METHODS),
    "petsc_gpu": frozenset(PETSC_GPU_SUPPORTED_METHODS),
}
SUPPORTED_BENCHMARK_BACKENDS = frozenset(("scipy", "petsc_gpu"))
_FIXED_SETUP_WARMUP_RUNS_GPU_LEARNING = 1
_FIXED_SETUP_WARMUP_RUNS_OTHER = 0
_FIXED_APPLY_WARMUP_RUNS_CPU = 0
_FIXED_SOLVE_WARMUP_RUNS_CPU = 0
_FIXED_PETSC_APPLY_WARMUP_RUNS = 0
_SETUP_TIMING_POLICY = (1, 3, 2, 0.05)
_APPLY_TIMING_POLICY = (3, 10, 3, 0.03)
_SOLVE_TIMING_POLICY = (1, 5, 2, 0.05)
_SETUP_MEASURE_REPEATS = 3
_APPLY_MEASURE_REPEATS = 10
_SOLVE_MEASURE_REPEATS = 3


@dataclass
class PreparedPreconditioner:
    method: str
    operator: LinearOperator | None
    apply_kind: str
    setup_time_sec: float
    factor_matrix: csr_matrix | None = None
    factor_nnz: float | None = None
    factor_density: float | None = None
    explicit_operator: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningBenchmarkModel:
    model_name: str
    checkpoint_path: str | None
    model_kwargs: dict[str, Any]
    diag_strategy: str
    label: str
    model: Any
    mask_percentile: float | None = None
    use_mask_projection: bool | None = None


@dataclass(frozen=True)
class TimingPolicy:
    min_runs: int
    max_runs: int
    window: int
    rel_tol: float


@dataclass(frozen=True)
class _ResolvedMethodSpec:
    method: str
    model: Any
    model_name: str
    model_label: str
    is_learning_model: bool
    diag_strategy: str
    mask_percentile: float | None
    use_mask_projection: bool


@dataclass(frozen=True)
class _TimedCall:
    elapsed_sec: float
    value: Any = None


@dataclass
class MethodResult:
    sample_id: str
    split_name: str | None
    family: str | None
    sampling_mode: str | None
    method: str
    status: str
    setup_time_sec: float
    apply_time_sec: float | None
    solve_time_sec: float | None
    total_time_sec: float | None
    iterations: int | None
    info: int | None
    converged: bool
    relative_residual: float | None
    final_residual_norm: float | None
    ksp_reason: int | None = None
    ksp_converged: bool | None = None
    true_residual_converged: bool | None = None
    true_relative_residual: float | None = None
    true_final_residual_norm: float | None = None
    ksp_norm_type: str | None = None
    ksp_monitor_initial_residual: float | None = None
    ksp_monitor_final_residual: float | None = None
    factor_nnz: float | None = None
    factor_density: float | None = None
    operator_nnz: float | None = None
    operator_density: float | None = None
    apply_kind: str = ""
    wall_forward_time_sec: float | None = None
    steady_forward_time_sec: float | None = None
    forward_wall_time_sec: float | None = None
    forward_steady_time_sec: float | None = None
    transfer_time_sec: float | None = None
    postprocess_time_sec: float | None = None
    factor_assembly_time_sec: float | None = None
    factor_gpu_to_host_time_sec: float | None = None
    factor_petsc_build_time_sec: float | None = None
    a_petsc_build_time_sec: float | None = None
    solve_ready_build_time_sec: float | None = None
    excluded_materialization_time_sec: float | None = None
    operator_build_time_sec: float | None = None
    graph_build_time_sec: float | None = None
    factor_gpu_to_host_cold_time_sec: float | None = None
    factor_petsc_build_cold_time_sec: float | None = None
    a_petsc_build_cold_time_sec: float | None = None
    solve_ready_build_cold_time_sec: float | None = None
    excluded_materialization_cold_time_sec: float | None = None
    inference_peak_gpu_memory_mb: float | None = None
    cpu_rss_delta_mb: float | None = None
    residual_history_file: str | None = None
    cond_est_ref: float | None = None
    lambda_min_ref: float | None = None
    lambda_max_ref: float | None = None
    cond_est_method: str | None = None
    cond_est_approx: float | None = None
    sigma_max_approx: float | None = None
    sigma_min_approx: float | None = None
    spectral_solve_time_sec: float | None = None
    eig_cv: float | None = None
    fro_error: float | None = None
    message: str | None = None
    backend: str = "scipy"
    matrix_type: str | None = None
    vector_type: str | None = None
    operator_matrix_type: str | None = None
    factor_matrix_type: str | None = None
    preconditioner_impl: str | None = None
    resolved_pc_type: str | None = None
    resolved_factor_solver_type: str | None = None
    ic0_shift_type: str | None = None
    solve_operator_mode: str | None = None
    learning_transformed_internal_rtol: float | None = None
    learning_convergence_basis: str | None = None
    cond_ref_time_sec: float | None = None
    cond_ref_backend: str | None = None
    total_time_basis: str | None = None
    setup_cold_time_sec: float | None = None
    apply_cold_time_sec: float | None = None
    setup_warmup_runs: int | None = None
    apply_warmup_runs: int | None = None
    solve_warmup_runs: int | None = None
    setup_measure_repeats: int | None = None
    apply_measure_repeats: int | None = None
    solve_measure_repeats: int | None = None
    timing_stable: bool | None = None
    burnin_sample_id: str | None = None
    model_name: str = "baseline"
    model_label: str = "baseline"
    is_learning_model: bool = False
    diag_strategy: str = ""
    rtol: float | None = None
    rtol_label: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


_RAW_OUTPUT_FIELD_ORDER = (
    "sample_id",
    "split_name",
    "family",
    "sampling_mode",
    "burnin_sample_id",
    "rtol",
    "rtol_label",
    "method",
    "status",
    "converged",
    "ksp_converged",
    "true_residual_converged",
    "message",
    "setup_time_sec",
    "setup_cold_time_sec",
    "factor_gpu_to_host_cold_time_sec",
    "factor_petsc_build_cold_time_sec",
    "a_petsc_build_cold_time_sec",
    "solve_ready_build_cold_time_sec",
    "excluded_materialization_cold_time_sec",
    "apply_time_sec",
    "apply_cold_time_sec",
    "solve_time_sec",
    "total_time_sec",
    "setup_warmup_runs",
    "apply_warmup_runs",
    "solve_warmup_runs",
    "setup_measure_repeats",
    "apply_measure_repeats",
    "solve_measure_repeats",
    "timing_stable",
    "iterations",
    "relative_residual",
    "true_relative_residual",
    "final_residual_norm",
    "true_final_residual_norm",
    "ksp_reason",
    "ksp_monitor_initial_residual",
    "ksp_monitor_final_residual",
    "residual_history_file",
    "apply_kind",
    "factor_nnz",
    "factor_density",
    "operator_nnz",
    "operator_density",
    "cond_est_ref",
    "lambda_min_ref",
    "lambda_max_ref",
    "cond_est_method",
    "cond_est_approx",
    "sigma_max_approx",
    "sigma_min_approx",
    "spectral_solve_time_sec",
    "backend",
    "matrix_type",
    "vector_type",
    "operator_matrix_type",
    "factor_matrix_type",
    "preconditioner_impl",
    "resolved_pc_type",
    "resolved_factor_solver_type",
    "ic0_shift_type",
    "solve_operator_mode",
    "learning_transformed_internal_rtol",
    "learning_convergence_basis",
    "wall_forward_time_sec",
    "steady_forward_time_sec",
    "transfer_time_sec",
    "postprocess_time_sec",
    "factor_assembly_time_sec",
    "factor_gpu_to_host_time_sec",
    "factor_petsc_build_time_sec",
    "a_petsc_build_time_sec",
    "solve_ready_build_time_sec",
    "excluded_materialization_time_sec",
    "operator_build_time_sec",
    "graph_build_time_sec",
    "cond_ref_time_sec",
    "inference_peak_gpu_memory_mb",
    "cpu_rss_delta_mb",
)

_SUMMARY_FIELD_ORDER = (
    "method",
    "stat",
    "n_samples",
    "n_converged",
    "setup_time_sec",
    "apply_time_sec",
    "solve_time_sec",
    "total_time_sec",
    "iterations",
    "relative_residual",
    "factor_nnz",
    "factor_density",
    "operator_nnz",
    "operator_density",
    "cond_est_ref",
    "cond_est_approx",
)

_FAMILY_SUMMARY_FIELD_ORDER = (
    "family",
    "method",
    "stat",
    "n_samples",
    "n_converged",
    "setup_time_sec",
    "apply_time_sec",
    "solve_time_sec",
    "total_time_sec",
    "iterations",
    "relative_residual",
    "factor_nnz",
    "factor_density",
    "operator_nnz",
    "operator_density",
    "cond_est_ref",
    "cond_est_approx",
)

_SUMMARY_STATS = ("mean", "max", "min")


class _IterationCounter:
    def __init__(self, A, b: np.ndarray):
        self.A = A
        self.b = b
        self.residuals: list[float] = []

    def __call__(self, xk: np.ndarray) -> None:
        rk = self.b - self.A @ xk
        self.residuals.append(float(np.linalg.norm(rk)))


class _FactorSystemIterationCounter:
    def __init__(self, A: csr_matrix, b: np.ndarray, factor_t: csr_matrix):
        self.A = A
        self.b = b
        self.factor_t = factor_t
        self.residuals: list[float] = []

    def __call__(self, yk: np.ndarray) -> None:
        xk = self.factor_t @ yk
        rk = self.b - self.A @ xk
        self.residuals.append(float(np.linalg.norm(rk)))


def _cfg_get(cfg: dict[str, Any], key: str, default=None):
    if "train" in cfg and isinstance(cfg["train"], dict):
        return cfg["train"].get(key, default)
    return cfg.get(key, default)


def _normalize_rtol_label(value: float) -> str:
    text = format(float(value), ".15e").lower()
    mantissa, exponent = text.split("e", maxsplit=1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = "+"
    digits = exponent
    if exponent.startswith(("+", "-")):
        sign = exponent[0]
        digits = exponent[1:]
    return f"{mantissa}e{sign}{digits.zfill(2)}"


def _safe_rtol_label(label: str) -> str:
    return label.replace(".", "p").replace("+", "")


def _mask_projection_enabled_from_percentile(mask_percentile: float) -> bool:
    return float(mask_percentile) >= 0.0


def _parse_mask_percentile(value: Any, *, field_name: str) -> float:
    try:
        percentile = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float") from exc
    if not np.isfinite(percentile):
        raise ValueError(f"{field_name} must be a finite float")
    return percentile


def _normalize_rtol_key(value: Any) -> str:
    return _normalize_rtol_label(float(value))


def _flatten_rtol_values(value: Any) -> list[float]:
    if isinstance(value, (int, float, np.floating, np.integer)):
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"Invalid rtol value {value!r}; expected a positive finite float")
        return [number]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("rtol must not be empty")
        if text[0] in "[(":
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Unable to parse rtol spec {value!r}") from exc
            return _flatten_rtol_values(parsed)
        if "," in text:
            flattened: list[float] = []
            for item in text.split(","):
                stripped = item.strip()
                if stripped:
                    flattened.extend(_flatten_rtol_values(stripped))
            if not flattened:
                raise ValueError("rtol list must not be empty")
            return flattened
        return _flatten_rtol_values(float(text))
    if isinstance(value, Sequence):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_rtol_values(item))
        if not flattened:
            raise ValueError("rtol list must not be empty")
        return flattened
    raise ValueError(f"Unsupported rtol spec {value!r}")


def _parse_rtol_values(value: Any) -> list[float]:
    parsed = _flatten_rtol_values(value)
    unique_values: list[float] = []
    seen: set[float] = set()
    for item in parsed:
        if item in seen:
            continue
        seen.add(item)
        unique_values.append(item)
    if not unique_values:
        raise ValueError("rtol list must not be empty")
    return unique_values


def _resolve_requested_rtol(cfg: dict[str, Any], cli_rtol: Any) -> Any:
    if cli_rtol is not None:
        return cli_rtol
    for key in ("RTOLS", "rtols", "RTOL", "rtol"):
        resolved = _cfg_get(cfg, key, None)
        if resolved is not None:
            return resolved
    return 1e-5


def _parse_benchmark_maxiter_by_rtol(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("BENCHMARK_MAXITER_BY_RTOL must be a mapping from rtol to maxiter")
    parsed: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        maxiter = int(raw_value)
        if maxiter <= 0:
            raise ValueError("BENCHMARK_MAXITER_BY_RTOL values must be positive integers")
        parsed[_normalize_rtol_key(raw_key)] = maxiter
    return parsed


def _resolve_benchmark_maxiter(cfg: dict[str, Any], rtol_value: float) -> int:
    per_rtol = _parse_benchmark_maxiter_by_rtol(_cfg_get(cfg, "BENCHMARK_MAXITER_BY_RTOL", None))
    rtol_key = _normalize_rtol_key(rtol_value)
    if rtol_key in per_rtol:
        return per_rtol[rtol_key]
    if _cfg_get(cfg, "BENCHMARK_MAXITER_BASE", None) is not None:
        return int(_cfg_get(cfg, "BENCHMARK_MAXITER_BASE", 20_000))
    return int(_cfg_get(cfg, "BENCHMARK_MAXITER", 20_000))


def _resolve_runtime_maxiter(maxiter: Any, rtol_value: float) -> int | None:
    if maxiter is None:
        return None
    if isinstance(maxiter, dict):
        rtol_key = _normalize_rtol_key(rtol_value)
        if rtol_key in maxiter:
            return int(maxiter[rtol_key])
        if str(rtol_value) in maxiter:
            return int(maxiter[str(rtol_value)])
        return None
    return int(maxiter)


def _with_stem_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    return path.with_name(f"{path.name}_{suffix}")


def _setup_timing_policy() -> TimingPolicy:
    return TimingPolicy(*_SETUP_TIMING_POLICY)


def _apply_timing_policy() -> TimingPolicy:
    return TimingPolicy(*_APPLY_TIMING_POLICY)


def _solve_timing_policy() -> TimingPolicy:
    return TimingPolicy(*_SOLVE_TIMING_POLICY)


def _clone_benchmark_sample(sample: BenchmarkSample) -> BenchmarkSample:
    return copy.deepcopy(sample)


def _timed_call(runner, *, sync=None, elapsed_resolver=None) -> _TimedCall:
    if sync is not None:
        sync()
    start = time.perf_counter()
    value = runner()
    if sync is not None:
        sync()
    elapsed_sec = time.perf_counter() - start
    if elapsed_resolver is not None:
        elapsed_sec = float(elapsed_resolver(value))
    return _TimedCall(elapsed_sec=float(elapsed_sec), value=value)


def _recent_window_is_stable(elapsed: Sequence[float], *, window: int, rel_tol: float) -> bool:
    if window <= 0 or len(elapsed) < window:
        return False
    recent = np.asarray(elapsed[-window:], dtype=np.float64)
    median = float(np.median(recent))
    if not np.isfinite(median):
        return False
    scale = max(abs(median), 1.0e-12)
    spread = float(np.max(recent) - np.min(recent))
    return spread <= float(rel_tol) * scale


def _adaptive_warmup(
    runner,
    *,
    policy: TimingPolicy,
    sync=None,
    elapsed_resolver=None,
) -> tuple[int, bool]:
    elapsed: list[float] = []
    target_runs = max(int(policy.max_runs), 0)
    for _ in range(target_runs):
        observation = _timed_call(runner, sync=sync, elapsed_resolver=elapsed_resolver)
        elapsed.append(observation.elapsed_sec)
        if len(elapsed) < max(int(policy.min_runs), 0):
            continue
        if _recent_window_is_stable(
            elapsed,
            window=int(policy.window),
            rel_tol=float(policy.rel_tol),
        ):
            return len(elapsed), True
    if not elapsed:
        return 0, True
    return len(elapsed), _recent_window_is_stable(
        elapsed,
        window=min(int(policy.window), len(elapsed)),
        rel_tol=float(policy.rel_tol),
    )


def _measure_repeated_calls(
    runner,
    *,
    repeats: int,
    sync=None,
    elapsed_resolver=None,
) -> list[_TimedCall]:
    observations: list[_TimedCall] = []
    for _ in range(max(int(repeats), 1)):
        observations.append(_timed_call(runner, sync=sync, elapsed_resolver=elapsed_resolver))
    return observations


def _median_elapsed(observations: Sequence[_TimedCall]) -> float:
    if not observations:
        raise ValueError("timed observations must not be empty")
    return float(
        np.median(np.asarray([item.elapsed_sec for item in observations], dtype=np.float64))
    )


def _median_observation(observations: Sequence[_TimedCall]) -> _TimedCall:
    if not observations:
        raise ValueError("timed observations must not be empty")
    order = np.argsort(
        np.asarray([item.elapsed_sec for item in observations], dtype=np.float64),
        kind="mergesort",
    )
    return observations[int(order[(len(observations) - 1) // 2])]


def _prepared_metric(prepared: Any, field_name: str) -> Any:
    if hasattr(prepared, field_name):
        return getattr(prepared, field_name)
    metadata = getattr(prepared, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(field_name)
    return None


def _reset_learning_group_context(context: dict[str, Any] | None) -> None:
    if context is None:
        return
    if "base_spconv_input" in context:
        context["base_spconv_input"] = None


def _matrix_density(matrix: csr_matrix) -> float:
    n_rows, n_cols = matrix.shape
    total_entries = int(n_rows) * int(n_cols)
    if total_entries <= 0:
        return 0.0
    return float(matrix.nnz) / float(total_entries)


def _has_output_value(value: Any) -> bool:
    return value not in (None, "")


def _raw_output_row(row: MethodResult) -> dict[str, Any]:
    raw = {
        "sample_id": row.sample_id,
        "split_name": row.split_name,
        "family": row.family,
        "sampling_mode": row.sampling_mode,
        "burnin_sample_id": row.burnin_sample_id,
        "rtol": row.rtol,
        "rtol_label": row.rtol_label,
        "method": row.method,
        "status": row.status,
        "converged": row.converged,
        "ksp_converged": row.ksp_converged,
        "true_residual_converged": row.true_residual_converged,
        "message": row.message,
        "setup_time_sec": row.setup_time_sec,
        "setup_cold_time_sec": row.setup_cold_time_sec,
        "factor_gpu_to_host_cold_time_sec": row.factor_gpu_to_host_cold_time_sec,
        "factor_petsc_build_cold_time_sec": row.factor_petsc_build_cold_time_sec,
        "a_petsc_build_cold_time_sec": row.a_petsc_build_cold_time_sec,
        "solve_ready_build_cold_time_sec": row.solve_ready_build_cold_time_sec,
        "excluded_materialization_cold_time_sec": row.excluded_materialization_cold_time_sec,
        "apply_time_sec": row.apply_time_sec,
        "apply_cold_time_sec": row.apply_cold_time_sec,
        "solve_time_sec": row.solve_time_sec,
        "total_time_sec": row.total_time_sec,
        "setup_warmup_runs": row.setup_warmup_runs,
        "apply_warmup_runs": row.apply_warmup_runs,
        "solve_warmup_runs": row.solve_warmup_runs,
        "setup_measure_repeats": row.setup_measure_repeats,
        "apply_measure_repeats": row.apply_measure_repeats,
        "solve_measure_repeats": row.solve_measure_repeats,
        "timing_stable": row.timing_stable,
        "iterations": row.iterations,
        "relative_residual": row.relative_residual,
        "true_relative_residual": row.true_relative_residual,
        "final_residual_norm": row.final_residual_norm,
        "true_final_residual_norm": row.true_final_residual_norm,
        "ksp_reason": row.ksp_reason,
        "ksp_monitor_initial_residual": row.ksp_monitor_initial_residual,
        "ksp_monitor_final_residual": row.ksp_monitor_final_residual,
        "residual_history_file": row.residual_history_file,
        "apply_kind": row.apply_kind,
        "factor_nnz": row.factor_nnz,
        "factor_density": row.factor_density,
        "operator_nnz": row.operator_nnz,
        "operator_density": row.operator_density,
        "cond_est_ref": row.cond_est_ref,
        "lambda_min_ref": row.lambda_min_ref,
        "lambda_max_ref": row.lambda_max_ref,
        "cond_est_method": row.cond_est_method,
        "cond_est_approx": row.cond_est_approx,
        "sigma_max_approx": row.sigma_max_approx,
        "sigma_min_approx": row.sigma_min_approx,
        "spectral_solve_time_sec": row.spectral_solve_time_sec,
        "backend": row.backend,
        "matrix_type": row.matrix_type,
        "vector_type": row.vector_type,
        "operator_matrix_type": row.operator_matrix_type,
        "factor_matrix_type": row.factor_matrix_type,
        "preconditioner_impl": row.preconditioner_impl,
        "resolved_pc_type": row.resolved_pc_type,
        "resolved_factor_solver_type": row.resolved_factor_solver_type,
        "ic0_shift_type": row.ic0_shift_type,
        "solve_operator_mode": row.solve_operator_mode,
        "learning_transformed_internal_rtol": row.learning_transformed_internal_rtol,
        "learning_convergence_basis": row.learning_convergence_basis,
        "wall_forward_time_sec": row.wall_forward_time_sec,
        "steady_forward_time_sec": row.steady_forward_time_sec,
        "transfer_time_sec": row.transfer_time_sec,
        "postprocess_time_sec": row.postprocess_time_sec,
        "factor_assembly_time_sec": row.factor_assembly_time_sec,
        "factor_gpu_to_host_time_sec": row.factor_gpu_to_host_time_sec,
        "factor_petsc_build_time_sec": row.factor_petsc_build_time_sec,
        "a_petsc_build_time_sec": row.a_petsc_build_time_sec,
        "solve_ready_build_time_sec": row.solve_ready_build_time_sec,
        "excluded_materialization_time_sec": row.excluded_materialization_time_sec,
        "operator_build_time_sec": row.operator_build_time_sec,
        "graph_build_time_sec": row.graph_build_time_sec,
        "cond_ref_time_sec": row.cond_ref_time_sec,
        "inference_peak_gpu_memory_mb": row.inference_peak_gpu_memory_mb,
        "cpu_rss_delta_mb": row.cpu_rss_delta_mb,
    }
    return {field: raw[field] for field in _RAW_OUTPUT_FIELD_ORDER}


def _summary_method_name(
    row: MethodResult,
    *,
    include_learning_label: bool,
    include_split_name: bool,
) -> str:
    method_name = row.method
    if row.is_learning_model and include_learning_label:
        label = str(row.model_label or row.model_name or "learning")
        method_name = "learning" if label == "learning" else f"learning:{label}"
    if include_split_name:
        split_name = str(row.split_name or "unknown")
        method_name = f"{split_name}:{method_name}"
    return method_name


def _aggregate_stat(values: list[float], stat: str) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if stat == "mean":
        return float(np.mean(array))
    if stat == "max":
        return float(np.max(array))
    if stat == "min":
        return float(np.min(array))
    raise ValueError(f"Unsupported stat: {stat}")


def _load_training_summary_loss_state(checkpoint_path: str | None) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    summary_path = Path(checkpoint_path).with_name("training_summary.json")
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    loss_state = summary.get("loss_state")
    if not isinstance(loss_state, dict):
        return None
    return loss_state


def _load_training_summary_diag_strategy(checkpoint_path: str | None) -> str | None:
    loss_state = _load_training_summary_loss_state(checkpoint_path)
    if not isinstance(loss_state, dict):
        return None
    diag_strategy = loss_state.get("diag_strategy")
    if diag_strategy is None:
        return None
    return normalize_diag_strategy(str(diag_strategy))


def _load_training_summary_mask_projection_enabled(checkpoint_path: str | None) -> bool | None:
    loss_state = _load_training_summary_loss_state(checkpoint_path)
    if not isinstance(loss_state, dict):
        return None
    enabled = loss_state.get("mask_projection_enabled")
    if enabled is None:
        return None
    return bool(enabled)


def _load_learning_model_specs(
    cfg: dict[str, Any],
    *,
    cli_model_name: str,
    learning_device: str,
    global_diag_strategy: str,
    global_mask_percentile: float,
) -> list[LearningBenchmarkModel]:
    from dpcg.models import load_model

    entries = _cfg_get(cfg, "LEARNING_MODELS", None)
    if entries is None:
        checkpoint_path = _cfg_get(cfg, "MODEL_STATE_M", None)
        model_kwargs = _cfg_get(cfg, "MODEL_KWARGS", _cfg_get(cfg, "model_kwargs", {}))
        mask_percentile = _parse_mask_percentile(
            _cfg_get(cfg, "MASK_PERCENTILE", global_mask_percentile),
            field_name="MASK_PERCENTILE",
        )
        use_mask_projection = _mask_projection_enabled_from_percentile(mask_percentile)
        summary_diag_strategy = _load_training_summary_diag_strategy(
            None if checkpoint_path is None else str(checkpoint_path)
        )
        if summary_diag_strategy is not None and summary_diag_strategy != global_diag_strategy:
            raise ValueError(
                "learning model diag_strategy mismatch: "
                f"config={global_diag_strategy!r}, training_summary={summary_diag_strategy!r}"
            )
        summary_mask_projection_enabled = _load_training_summary_mask_projection_enabled(
            None if checkpoint_path is None else str(checkpoint_path)
        )
        if (
            summary_mask_projection_enabled is not None
            and summary_mask_projection_enabled != use_mask_projection
        ):
            raise ValueError(
                "learning model mask_projection_enabled mismatch: "
                f"config={use_mask_projection!r}, "
                f"training_summary={summary_mask_projection_enabled!r}"
            )
        model = load_model(
            cli_model_name,
            model_kwargs=model_kwargs,
            device=learning_device,
            checkpoint_path=checkpoint_path,
            map_location=learning_device,
        )
        return [
            LearningBenchmarkModel(
                model_name=str(cli_model_name),
                checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
                model_kwargs=dict(model_kwargs or {}),
                diag_strategy=global_diag_strategy,
                label=str(cli_model_name),
                model=model,
                mask_percentile=mask_percentile,
                use_mask_projection=use_mask_projection,
            )
        ]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("LEARNING_MODELS must be a list of model configs")
    specs: list[LearningBenchmarkModel] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError("Each LEARNING_MODELS entry must be a mapping")
        entry_model_name = str(raw_entry.get("model_name", "")).strip()
        if not entry_model_name:
            raise ValueError(f"LEARNING_MODELS[{index}].model_name is required")
        checkpoint_path = raw_entry.get("checkpoint_path", None)
        if checkpoint_path is None or not str(checkpoint_path).strip():
            raise ValueError(f"LEARNING_MODELS[{index}].checkpoint_path is required")
        entry_model_kwargs = raw_entry.get("model_kwargs", {})
        if not isinstance(entry_model_kwargs, dict):
            raise ValueError(f"LEARNING_MODELS[{index}].model_kwargs must be a mapping")
        raw_diag_strategy = raw_entry.get("diag_strategy", None)
        if raw_diag_strategy is None:
            raise ValueError(f"LEARNING_MODELS[{index}].diag_strategy is required")
        diag_strategy = normalize_diag_strategy(str(raw_diag_strategy))
        if diag_strategy not in {"learned_exp", "unit_diag"}:
            raise ValueError(
                f"LEARNING_MODELS[{index}].diag_strategy must be 'learned_exp' or 'unit_diag'"
            )
        mask_percentile = _parse_mask_percentile(
            raw_entry.get("mask_percentile", global_mask_percentile),
            field_name=f"LEARNING_MODELS[{index}].mask_percentile",
        )
        use_mask_projection = _mask_projection_enabled_from_percentile(mask_percentile)
        summary_diag_strategy = _load_training_summary_diag_strategy(str(checkpoint_path))
        if summary_diag_strategy is not None and summary_diag_strategy != diag_strategy:
            raise ValueError(
                f"LEARNING_MODELS[{index}] diag_strategy mismatch: "
                f"config={diag_strategy!r}, training_summary={summary_diag_strategy!r}"
            )
        summary_mask_projection_enabled = _load_training_summary_mask_projection_enabled(
            str(checkpoint_path)
        )
        if (
            summary_mask_projection_enabled is not None
            and summary_mask_projection_enabled != use_mask_projection
        ):
            raise ValueError(
                f"LEARNING_MODELS[{index}] mask_projection_enabled mismatch: "
                f"config={use_mask_projection!r}, "
                f"training_summary={summary_mask_projection_enabled!r}"
            )
        label = str(raw_entry.get("label", entry_model_name)).strip() or entry_model_name
        model = load_model(
            entry_model_name,
            model_kwargs=entry_model_kwargs,
            device=learning_device,
            checkpoint_path=str(checkpoint_path),
            map_location=learning_device,
        )
        specs.append(
            LearningBenchmarkModel(
                model_name=entry_model_name,
                checkpoint_path=str(checkpoint_path),
                model_kwargs=dict(entry_model_kwargs),
                diag_strategy=diag_strategy,
                label=label,
                model=model,
                mask_percentile=mask_percentile,
                use_mask_projection=use_mask_projection,
            )
        )
    return specs


def _parse_backend(value: str | None) -> str:
    backend = "scipy" if value is None else str(value).strip().lower()
    if backend not in SUPPORTED_BENCHMARK_BACKENDS:
        raise ValueError(
            f"Unsupported benchmark backend: {backend!r}. "
            f"Expected one of {sorted(SUPPORTED_BENCHMARK_BACKENDS)}"
        )
    return backend


def _parse_methods(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_METHODS)
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.split(",") if part.strip()]
    else:
        items = [str(part).strip().lower() for part in value if str(part).strip()]
    unknown = [item for item in items if item not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported benchmark methods: {unknown}")
    return items


def _validate_methods_for_backend(methods: Sequence[str], backend: str) -> list[str]:
    resolved_backend = _parse_backend(backend)
    allowed = SUPPORTED_METHODS_BY_BACKEND[resolved_backend]
    unknown = [item for item in methods if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unsupported benchmark methods for backend {resolved_backend!r}: {unknown}. "
            f"Expected methods from {sorted(allowed)}"
        )
    return list(methods)


def _default_methods_for_backend(backend: str) -> list[str]:
    if _parse_backend(backend) == "petsc_gpu":
        return list(PETSC_GPU_DEFAULT_METHODS)
    return list(DEFAULT_METHODS)


def _to_numpy_vector(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _cpu_rss_mb() -> float:
    if resource is None:
        return 0.0
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_kb / 1024.0


def _mask_to_indices(mask: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n = int(shape[0])
    if mask is None:
        diag = np.arange(n, dtype=np.int64)
        return diag, diag
    if isinstance(mask, torch.Tensor):
        if mask.is_sparse:
            idx = mask.coalesce().indices()
            rows = idx[-2].detach().cpu().numpy().astype(np.int64, copy=False)
            cols = idx[-1].detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            nz = torch.nonzero(mask, as_tuple=False)
            rows = nz[:, -2].detach().cpu().numpy().astype(np.int64, copy=False)
            cols = nz[:, -1].detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
        rows = rows.astype(np.int64, copy=False)
        cols = cols.astype(np.int64, copy=False)
    rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
    key = rows * np.int64(n) + cols
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    keep = np.concatenate([[True], key[1:] != key[:-1]])
    order = order[keep]
    rows = rows[order]
    cols = cols[order]
    return rows, cols


def _ensure_sample_sparse_cache(sample: BenchmarkSample) -> None:
    if (
        sample.mask_rows is not None
        and sample.mask_cols is not None
        and sample.mask_key is not None
        and sample.diag_inv is not None
    ):
        return
    rows, cols = _mask_to_indices(sample.mask, sample.A.shape)
    n = int(sample.A.shape[0])
    key = rows * np.int64(n) + cols
    diag = sample.A.diagonal().astype(np.float64, copy=False)
    sample.mask_rows = rows
    sample.mask_cols = cols
    sample.mask_key = key
    sample.diag_inv = 1.0 / np.sqrt(diag)


def _remask_benchmark_sample(
    sample: BenchmarkSample,
    *,
    mask_percentile: float | None,
    use_mask_projection: bool,
) -> BenchmarkSample:
    if mask_percentile is None or not use_mask_projection:
        return sample
    from dpcg.data import mask_indices_from_lower_triangle

    cloned = _clone_benchmark_sample(sample)
    lower_tri = tril(cloned.A).tocoo()
    rows, cols = mask_indices_from_lower_triangle(lower_tri, float(mask_percentile))
    n = int(cloned.A.shape[0])
    key = rows * np.int64(n) + cols
    if torch is None:
        raise RuntimeError("torch is required to build per-model benchmark masks")
    indices = torch.as_tensor(np.vstack((rows, cols)), dtype=torch.int64)
    values = torch.ones(indices.shape[1], dtype=torch.bool)
    cloned.mask = torch.sparse_coo_tensor(
        indices,
        values,
        size=cloned.A.shape,
        dtype=torch.bool,
    ).coalesce()
    cloned.mask_rows = rows
    cloned.mask_cols = cols
    cloned.mask_key = key
    if cloned.diag_inv is None:
        diag = cloned.A.diagonal().astype(np.float64, copy=False)
        cloned.diag_inv = 1.0 / np.sqrt(diag)
    cloned.metadata = dict(cloned.metadata)
    cloned.metadata["mask_percentile"] = float(mask_percentile)
    return cloned


def _sparse_to_csr(tensor: Any) -> csr_matrix:
    from dpcg.utils import torch_sparse_to_scipy_csr

    return torch_sparse_to_scipy_csr(tensor).astype(np.float64)


def normalize_batch(batch: tuple[Any, ...], sample_id: str | None = None) -> BenchmarkSample:
    if isinstance(batch, BenchmarkSample):
        return batch
    m_tensor, _solutions, rhs, l_tensor, ref, mask = batch
    resolved_sample_id = sample_id or getattr(batch, "sample_id", "sample")
    metadata = dict(getattr(batch, "metadata", {}) or {})
    A = _sparse_to_csr(m_tensor)
    b = _to_numpy_vector(rhs)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{resolved_sample_id}: matrix must be square")
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"{resolved_sample_id}: rhs length does not match matrix shape")
    return BenchmarkSample(
        sample_id=resolved_sample_id,
        A=A,
        b=b,
        l_tensor=l_tensor,
        mask=mask,
        ref=float(ref) if ref is not None else None,
        metadata=metadata,
    )


def _prepare_learning_model(model, device: str):
    if model is None or torch is None or not hasattr(model, "to"):
        return model
    target_device = torch.device(device)
    current_device = None
    for tensor in list(model.parameters()) + list(model.buffers()):
        current_device = tensor.device
        break
    if current_device != target_device:
        model = model.to(target_device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _normalize_benchmark_samples(
    dataloader: Iterable[Any],
) -> tuple[list[BenchmarkSample], dict[str, int]]:
    samples = [normalize_batch(batch) for batch in dataloader]
    return samples, {
        "num_input_samples": len(samples),
        "num_ignition_samples": 1 if samples else 0,
        "num_measured_samples": len(samples),
    }


def _iter_streaming_grouped_samples(
    dataloader: Iterable[Any],
    *,
    model,
    backend: str,
    learning_device: str,
    methods: Sequence[str],
    learning_output_kind: str,
    learning_diag_strategy: str,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    ic0_diagcomp: float,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
    setup_warmup_runs_gpu_learning: int,
    setup_warmup_runs_other: int,
    apply_warmup_runs_cpu: int,
    solve_warmup_runs_cpu: int,
    steady_forward_repeats: int,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str,
    petsc_options: str | None,
    petsc_ksp_norm_type: str,
    petsc_factor_solve_mode: str,
    petsc_factor_operator_mode: str,
    petsc_cond_mode: str,
    petsc_apply_measure_repeats: int,
    petsc_learning_internal_rtol_ratio: float,
) -> tuple[list[tuple[BenchmarkSample, dict[str, Any] | None]], dict[str, int]]:
    counts = {
        "num_input_samples": 0,
        "num_ignition_samples": 0,
        "num_measured_samples": 0,
    }
    measured: list[tuple[BenchmarkSample, dict[str, Any] | None]] = []
    needs_learning_context = model is not None and getattr(model, "input_kind", None) == "spconv"
    learning_group_context = (
        _make_learning_group_context(device=learning_device, model=model)
        if needs_learning_context
        else None
    )
    first_sample = True

    for batch in dataloader:
        sample = normalize_batch(batch)
        counts["num_input_samples"] += 1
        if first_sample:
            _run_group_ignition(
                sample,
                methods,
                model=model,
                backend=backend,
                learning_output_kind=learning_output_kind,
                learning_diag_strategy=learning_diag_strategy,
                learning_device=learning_device,
                rtol=rtol,
                maxiter=maxiter,
                ssor_omega=ssor_omega,
                ic0_diagcomp=ic0_diagcomp,
                ilu_drop_tol=ilu_drop_tol,
                ilu_fill_factor=ilu_fill_factor,
                learning_group_context=learning_group_context,
                setup_warmup_runs_gpu_learning=setup_warmup_runs_gpu_learning,
                setup_warmup_runs_other=setup_warmup_runs_other,
                cpu_apply_warmup_runs=apply_warmup_runs_cpu,
                cpu_solve_warmup_runs=solve_warmup_runs_cpu,
                steady_forward_repeats=steady_forward_repeats,
                petsc_amg_backend=petsc_amg_backend,
                petsc_ic0_shift_type=petsc_ic0_shift_type,
                petsc_options=petsc_options,
                petsc_ksp_norm_type=petsc_ksp_norm_type,
                petsc_factor_solve_mode=petsc_factor_solve_mode,
                petsc_factor_operator_mode=petsc_factor_operator_mode,
                petsc_cond_mode=petsc_cond_mode,
                petsc_apply_warmup_runs=_FIXED_PETSC_APPLY_WARMUP_RUNS,
                petsc_apply_measure_repeats=petsc_apply_measure_repeats,
                petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
            )
            counts["num_ignition_samples"] += 1
            first_sample = False
            continue
        measured.append((sample, learning_group_context))
        counts["num_measured_samples"] += 1

    return measured, counts


def _check_spd(A: csr_matrix, sym_tol: float = 1e-10) -> None:
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square for CG benchmark")
    diff = (A - A.transpose()).tocsr()
    if diff.nnz > 0 and np.max(np.abs(diff.data)) > sym_tol:
        raise ValueError("matrix must be symmetric for CG benchmark")
    diag = A.diagonal()
    if np.any(diag <= 0.0):
        raise ValueError("matrix diagonal must be positive for CG benchmark")
    try:
        lambda_min = float(
            eigsh(
                A,
                k=1,
                sigma=0.0,
                which="LM",
                return_eigenvectors=False,
            )[0]
        )
    except Exception:
        try:
            lambda_min = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
        except Exception as sa_exc:
            raise ValueError(
                "failed to confirm SPD property via eigsh(sigma=0) or eigsh(which='SA')"
            ) from sa_exc
    if lambda_min <= 0.0:
        raise ValueError("matrix is not positive definite")


def _factor_stats(nnz: int | None, n: int) -> tuple[float | None, float | None]:
    if nnz is None:
        return None, None
    return float(nnz), float(nnz) / float(n * n)


def _make_jacobi_preconditioner(A: csr_matrix) -> PreparedPreconditioner:
    start = time.perf_counter()
    diag = A.diagonal()
    if np.any(np.abs(diag) <= 0.0):
        raise RuntimeError("jacobi requires a nonzero diagonal")
    inv_diag = 1.0 / diag
    setup = time.perf_counter() - start

    def matvec(x: np.ndarray) -> np.ndarray:
        return inv_diag * x

    operator = LinearOperator(A.shape, matvec=matvec, dtype=np.float64)
    nnz, density = _factor_stats(int(inv_diag.size), A.shape[0])
    return PreparedPreconditioner(
        method="jacobi",
        operator=operator,
        apply_kind="inverse_diagonal",
        setup_time_sec=setup,
        factor_matrix=diags(inv_diag, offsets=0, format="csr"),
        factor_nnz=nnz,
        factor_density=density,
    )


def _make_identity_preconditioner(A: csr_matrix) -> PreparedPreconditioner:
    start = time.perf_counter()
    setup = time.perf_counter() - start

    def matvec(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64).reshape(-1)

    operator = LinearOperator(A.shape, matvec=matvec, dtype=np.float64)
    return PreparedPreconditioner(
        method="none",
        operator=operator,
        apply_kind="identity",
        setup_time_sec=setup,
    )


def _make_split_preconditioner(
    A: csr_matrix,
    *,
    method: str,
    omega: float,
) -> PreparedPreconditioner:
    start = time.perf_counter()
    diag = A.diagonal()
    if np.any(np.abs(diag) <= 0.0):
        raise RuntimeError(f"{method} requires a nonzero diagonal")
    D = diags(diag, offsets=0, format="csr")
    D_inv = diags(1.0 / diag, offsets=0, format="csr")
    if method == "sgs":
        lower_factor = tril(A, format="csr")
        upper_factor = lower_factor.transpose().tocsc()
    else:
        strict_lower = tril(A, k=-1, format="csr")
        lower_factor = (D + omega * strict_lower).tocsr()
        upper_factor = (D + omega * strict_lower.transpose()).tocsc()
    lower_factor.sum_duplicates()
    lower_factor.sort_indices()
    upper_factor.sum_duplicates()
    upper_factor.sort_indices()
    M1 = (lower_factor @ D_inv).tocsr()
    M1.sum_duplicates()
    M1.sort_indices()
    M2 = upper_factor
    setup = time.perf_counter() - start

    def matvec(x: np.ndarray) -> np.ndarray:
        y = spsolve_triangular(M1, x, lower=True)
        return spsolve_triangular(M2, y, lower=False)

    operator = LinearOperator(A.shape, matvec=matvec, dtype=np.float64)
    nnz = M1.nnz + M2.nnz - A.shape[0]
    factor_nnz, factor_density = _factor_stats(int(nnz), A.shape[0])
    return PreparedPreconditioner(
        method=method,
        operator=operator,
        apply_kind="split_triangular_solve",
        setup_time_sec=setup,
        factor_matrix=M1 if method == "sgs" else lower_factor,
        factor_nnz=factor_nnz,
        factor_density=factor_density,
        metadata={"omega": omega},
    )


def _row_dot(
    left_cols: np.ndarray,
    left_vals: np.ndarray,
    left_stop: int,
    right_cols: np.ndarray,
    right_vals: np.ndarray,
    right_stop: int,
) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < left_stop and j < right_stop:
        ci = int(left_cols[i])
        cj = int(right_cols[j])
        if ci == cj:
            total += float(left_vals[i] * right_vals[j])
            i += 1
            j += 1
        elif ci < cj:
            i += 1
        else:
            j += 1
    return total


def incomplete_cholesky_zero_fill(A: csr_matrix, diagcomp: float = 0.0) -> csr_matrix:
    A = A.tocsr().astype(np.float64)
    n = A.shape[0]
    lower = tril(A, format="csr")
    indptr = lower.indptr.copy()
    indices = lower.indices.copy()
    data = lower.data.copy()
    shifted_diag = A.diagonal() * (1.0 + float(diagcomp))

    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row_cols = indices[row_start:row_end]
        row_data = data[row_start:row_end]
        diag_offset = int(np.searchsorted(row_cols, i))
        if diag_offset >= row_cols.size or int(row_cols[diag_offset]) != i:
            raise RuntimeError(f"IC(0) requires diagonal entry at row {i}")

        for offset in range(diag_offset):
            k = int(row_cols[offset])
            k_start = indptr[k]
            k_end = indptr[k + 1]
            k_cols = indices[k_start:k_end]
            k_data = data[k_start:k_end]
            k_diag_offset = int(np.searchsorted(k_cols, k))
            dot = _row_dot(row_cols, row_data, offset, k_cols, k_data, k_diag_offset)
            diag_k = float(k_data[k_diag_offset])
            if diag_k <= 0.0:
                raise RuntimeError(f"IC(0) encountered non-positive pivot at row {k}")
            row_data[offset] = (row_data[offset] - dot) / diag_k

        diag_value = float(shifted_diag[i] - np.dot(row_data[:diag_offset], row_data[:diag_offset]))
        if diag_value <= 0.0:
            raise RuntimeError(
                f"IC(0) failed at row {i}: non-positive pivot after diagcomp={diagcomp}"
            )
        row_data[diag_offset] = math.sqrt(diag_value)
        data[row_start:row_end] = row_data

    return csr_matrix((data, indices, indptr), shape=A.shape)


def _make_ic0_preconditioner(A: csr_matrix, diagcomp: float) -> PreparedPreconditioner:
    start = time.perf_counter()
    L = incomplete_cholesky_zero_fill(A, diagcomp=diagcomp)
    L.sum_duplicates()
    L.sort_indices()
    LT = L.transpose().tocsc()
    LT.sum_duplicates()
    LT.sort_indices()
    setup = time.perf_counter() - start

    def matvec(x: np.ndarray) -> np.ndarray:
        y = spsolve_triangular(L, x, lower=True)
        return spsolve_triangular(LT, y, lower=False)

    operator = LinearOperator(A.shape, matvec=matvec, dtype=np.float64)
    factor_nnz, factor_density = _factor_stats(int(L.nnz), A.shape[0])
    return PreparedPreconditioner(
        method="ic0",
        operator=operator,
        apply_kind="ichol0_triangular_solve",
        setup_time_sec=setup,
        factor_matrix=L,
        factor_nnz=factor_nnz,
        factor_density=factor_density,
        metadata={"diagcomp": diagcomp},
    )


def _make_ilu_preconditioner(
    A: csr_matrix,
    *,
    drop_tol: float,
    fill_factor: float,
) -> PreparedPreconditioner:
    start = time.perf_counter()
    ilu = spilu(A.tocsc(), drop_tol=float(drop_tol), fill_factor=float(fill_factor))
    setup = time.perf_counter() - start

    def matvec(x: np.ndarray) -> np.ndarray:
        return np.asarray(ilu.solve(x), dtype=np.float64).reshape(-1)

    L = ilu.L.tocsr()
    U = ilu.U.tocsr()
    factor_nnz, factor_density = _factor_stats(int(L.nnz + U.nnz - A.shape[0]), A.shape[0])
    return PreparedPreconditioner(
        method="ilu",
        operator=LinearOperator(A.shape, matvec=matvec, dtype=np.float64),
        apply_kind="spilu_solve",
        setup_time_sec=setup,
        factor_nnz=factor_nnz,
        factor_density=factor_density,
        metadata={"drop_tol": float(drop_tol), "fill_factor": float(fill_factor)},
    )


def _make_amg_preconditioner(A: csr_matrix) -> PreparedPreconditioner:
    if smoothed_aggregation_solver is None:
        raise RuntimeError("pyamg is not installed")
    start = time.perf_counter()
    ml = smoothed_aggregation_solver(A)
    precond = ml.aspreconditioner()
    setup = time.perf_counter() - start

    def matvec(x: np.ndarray) -> np.ndarray:
        return np.asarray(precond(x), dtype=np.float64)

    operator = LinearOperator(A.shape, matvec=matvec, dtype=np.float64)
    return PreparedPreconditioner(
        method="amg",
        operator=operator,
        apply_kind="pyamg_operator",
        setup_time_sec=setup,
        metadata={"levels": len(getattr(ml, "levels", []))},
    )


def build_learning_preconditioner_from_sparse_factor(
    prediction: "SparseFactorPrediction",
    *,
    sample: BenchmarkSample,
    diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
) -> PreparedPreconditioner:
    _ensure_sample_sparse_cache(sample)
    if (
        sample.mask_rows is None
        or sample.mask_cols is None
        or sample.mask_key is None
        or sample.diag_inv is None
    ):
        raise RuntimeError("sample sparse cache is incomplete")
    if prediction.coords.ndim != 2 or prediction.coords.shape[1] != 2:
        raise ValueError("SparseFactorPrediction.coords must have shape (nnz, 2)")

    if isinstance(prediction.coords, torch.Tensor):
        coords = prediction.coords.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        coords = np.asarray(prediction.coords, dtype=np.int64)
    if isinstance(prediction.values, torch.Tensor):
        values = prediction.values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        values = np.asarray(prediction.values, dtype=np.float64).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError("SparseFactorPrediction values/coords length mismatch")
    assembled = assemble_sparse_factor_from_prediction_numpy(
        coords=coords,
        values=values,
        mask_rows=sample.mask_rows,
        mask_cols=sample.mask_cols,
        mask_key=sample.mask_key,
        diag_inv=sample.diag_inv,
        shape=sample.A.shape,
        diag_strategy=normalize_diag_strategy(diag_strategy),
        force_unit_diag=True,
        keep_matched_zero=False,
        use_mask_projection=bool(use_mask_projection),
    )
    matrix = assembled["matrix"]
    matrix_t = matrix.transpose().tocsr()
    n = int(sample.A.shape[0])
    factor_nnz, factor_density = _factor_stats(int(matrix.nnz), n)

    def matvec(x: np.ndarray) -> np.ndarray:
        y = matrix_t @ x
        return matrix @ y

    return PreparedPreconditioner(
        method="learning",
        operator=LinearOperator(matrix.shape, matvec=matvec, dtype=np.float64),
        apply_kind="factor_LLt",
        setup_time_sec=0.0,
        factor_matrix=matrix,
        factor_nnz=factor_nnz,
        factor_density=factor_density,
        metadata={
            "output_kind": "sparse_factor_L",
            "diag_strategy": normalize_diag_strategy(diag_strategy),
            "mask_projection_enabled": bool(use_mask_projection),
        },
    )


def _measure_learning_forward_wall(model, model_input, *, use_cuda: bool):
    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = _run_learning_model_forward(model, model_input, profile=False)
    if use_cuda:
        torch.cuda.synchronize()
    return prediction, time.perf_counter() - start


def _measure_learning_forward_steady(
    model,
    model_input,
    *,
    use_cuda: bool,
    repeats: int = 10,
) -> float:
    if repeats <= 0:
        raise ValueError("steady forward repeats must be positive")
    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _run_learning_model_forward(model, model_input, profile=False)
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / float(repeats)


def _prepare_learning_preconditioner(
    sample: BenchmarkSample,
    *,
    model,
    device: str,
    output_kind: str,
    diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    group_context: dict[str, Any] | None = None,
    steady_forward_repeats: int = 50,
) -> PreparedPreconditioner:
    if torch is None:
        raise RuntimeError("torch is required for the learning benchmark path")
    from dpcg.data import sample_to_half_graph
    from dpcg.models import SparseFactorPrediction
    from dpcg.utils import torch_sparse_to_spconv

    del steady_forward_repeats
    if output_kind != "sparse_factor_L":
        raise ValueError("learning benchmark only supports learning_output_kind='sparse_factor_L'")
    _ensure_sample_sparse_cache(sample)
    if use_mask_projection and sample.mask is None:
        raise RuntimeError("learning benchmark requires mask in the sample")
    use_cuda = str(device).startswith("cuda")
    input_kind = getattr(model, "input_kind", "spconv")
    graph_build_time = 0.0
    rss_before_mb = _cpu_rss_mb()
    peak_gpu_memory_mb = None
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    if input_kind == "graph":
        graph_start = time.perf_counter()
        model_input = sample_to_half_graph(sample).to(device)
        if use_cuda:
            torch.cuda.synchronize()
        graph_build_time = time.perf_counter() - graph_start
    else:
        if sample.l_tensor is None:
            raise RuntimeError("learning benchmark requires l_tensor in the sample")
        if group_context is not None and group_context.get("base_spconv_input") is not None:
            feature_values = _coalesced_sample_values(sample, device)
            model_input = group_context["base_spconv_input"].replace_feature(feature_values)
        else:
            l_tensor = sample.l_tensor.to(device)
            model_input = torch_sparse_to_spconv(l_tensor)
            if group_context is not None:
                group_context["base_spconv_input"] = model_input
        if use_cuda:
            torch.cuda.synchronize()
    prediction, forward_time = _measure_learning_forward_wall(
        model,
        model_input,
        use_cuda=use_cuda,
    )
    transfer_start = time.perf_counter()
    if not isinstance(prediction, SparseFactorPrediction):
        raise TypeError("learning benchmark expects model output to be SparseFactorPrediction")
    sparse_prediction = SparseFactorPrediction(
        values=prediction.values.detach().cpu(),
        coords=prediction.coords.detach().cpu(),
        shape=prediction.shape,
        timings=dict(prediction.timings),
    )
    if use_cuda:
        torch.cuda.synchronize()
    transfer_time = time.perf_counter() - transfer_start
    postprocess_start = time.perf_counter()
    prepared = build_learning_preconditioner_from_sparse_factor(
        sparse_prediction,
        sample=sample,
        diag_strategy=diag_strategy,
        use_mask_projection=use_mask_projection,
    )
    postprocess_time = time.perf_counter() - postprocess_start
    setup_time_sec = graph_build_time + forward_time + transfer_time + postprocess_time
    prepared.setup_time_sec = setup_time_sec
    prepared.metadata["device"] = device
    prepared.metadata["graph_build_time_sec"] = graph_build_time
    prepared.metadata["wall_forward_time_sec"] = forward_time
    prepared.metadata["steady_forward_time_sec"] = forward_time
    prepared.metadata["forward_wall_time_sec"] = forward_time
    prepared.metadata["forward_steady_time_sec"] = forward_time
    prepared.metadata["transfer_time_sec"] = transfer_time
    prepared.metadata["postprocess_time_sec"] = postprocess_time
    prepared.metadata["factor_assembly_time_sec"] = postprocess_time
    prepared.metadata["factor_gpu_to_host_time_sec"] = None
    prepared.metadata["factor_petsc_build_time_sec"] = None
    prepared.metadata["a_petsc_build_time_sec"] = None
    prepared.metadata["solve_ready_build_time_sec"] = None
    prepared.metadata["excluded_materialization_time_sec"] = None
    prepared.metadata["diag_strategy"] = normalize_diag_strategy(diag_strategy)
    prepared.metadata["mask_projection_enabled"] = bool(use_mask_projection)
    if use_cuda:
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    prepared.metadata["inference_peak_gpu_memory_mb"] = peak_gpu_memory_mb
    prepared.metadata["cpu_rss_delta_mb"] = max(_cpu_rss_mb() - rss_before_mb, 0.0)
    return prepared


def _benchmark_method_once(
    sample: BenchmarkSample,
    method: str,
    *,
    model=None,
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str = "cuda",
    rtol: float = 1e-5,
    maxiter: int | None = None,
    ssor_omega: float = 1.2,
    ic0_diagcomp: float = 1e-1,
    ilu_drop_tol: float = 1e-4,
    ilu_fill_factor: float = 10.0,
    enable_spectral_metrics: bool = False,
    spectral_dense_limit: int = 256,
    residual_dir: Path | None = None,
    learning_group_context: dict[str, Any] | None = None,
    steady_forward_repeats: int = 50,
) -> MethodResult:
    operator_density = _matrix_density(sample.A)
    operator_nnz = float(sample.A.nnz)
    try:
        prepared = prepare_preconditioner(
            method,
            sample,
            model=model,
            learning_output_kind=learning_output_kind,
            learning_diag_strategy=learning_diag_strategy,
            use_mask_projection=use_mask_projection,
            learning_device=learning_device,
            ssor_omega=ssor_omega,
            ic0_diagcomp=ic0_diagcomp,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            learning_group_context=learning_group_context,
            steady_forward_repeats=steady_forward_repeats,
        )
    except Exception as exc:
        return MethodResult(
            sample_id=sample.sample_id,
            split_name=sample.metadata.get("split"),
            family=sample.metadata.get("family"),
            sampling_mode=sample.metadata.get("sampling_mode"),
            method=method,
            status="skipped" if method == "amg" else "failed",
            setup_time_sec=0.0,
            apply_time_sec=None,
            solve_time_sec=None,
            total_time_sec=None,
            iterations=None,
            info=None,
            converged=False,
            relative_residual=None,
            final_residual_norm=None,
            factor_nnz=None,
            factor_density=None,
            operator_nnz=operator_nnz,
            operator_density=operator_density,
            apply_kind="unavailable",
            message=str(exc),
        )

    try:
        timing_fields = _result_timing_fields(prepared)
        _, apply_time = _run_apply(prepared, sample.b)
        if method == "learning" and prepared.factor_matrix is not None:
            x, info, solve_time, residuals, iterations = _run_factor_system_cg(
                sample.A,
                sample.b,
                factor=prepared.factor_matrix,
                rtol=rtol,
                maxiter=maxiter,
            )
        else:
            x, info, solve_time, residuals, iterations = _run_cg(
                sample.A,
                sample.b,
                M=prepared.operator,
                rtol=rtol,
                maxiter=maxiter,
            )
        converged = info == 0
        final_residual = float(np.linalg.norm(sample.b - sample.A @ x))
        relative_residual = float(rtol) if converged else None
        residual_file = _save_residual_history(residual_dir, sample.sample_id, method, residuals)
        _cond_est, eig_cv, fro_error = _compute_spectral_metrics(
            sample.A,
            prepared,
            enabled=enable_spectral_metrics,
            dense_limit=spectral_dense_limit,
        )
        setup_time = float(prepared.metadata.get("setup_time_sec", prepared.setup_time_sec))
        return MethodResult(
            sample_id=sample.sample_id,
            split_name=sample.metadata.get("split"),
            family=sample.metadata.get("family"),
            sampling_mode=sample.metadata.get("sampling_mode"),
            method=method,
            status="ok",
            setup_time_sec=setup_time,
            apply_time_sec=float(apply_time),
            solve_time_sec=float(solve_time),
            total_time_sec=float(setup_time + solve_time),
            iterations=iterations,
            info=info,
            converged=converged,
            relative_residual=relative_residual,
            final_residual_norm=final_residual,
            factor_nnz=prepared.factor_nnz,
            factor_density=prepared.factor_density,
            operator_nnz=operator_nnz,
            operator_density=operator_density,
            apply_kind=prepared.apply_kind,
            inference_peak_gpu_memory_mb=prepared.metadata.get("inference_peak_gpu_memory_mb"),
            cpu_rss_delta_mb=prepared.metadata.get("cpu_rss_delta_mb"),
            residual_history_file=residual_file,
            eig_cv=eig_cv,
            fro_error=fro_error,
            model_name="learning" if method == "learning" else "baseline",
            model_label="learning" if method == "learning" else "baseline",
            is_learning_model=(method == "learning"),
            diag_strategy=learning_diag_strategy if method == "learning" else "",
            **timing_fields,
        )
    except Exception as exc:
        timing_fields = _result_timing_fields(prepared)
        setup_time = float(prepared.metadata.get("setup_time_sec", prepared.setup_time_sec))
        return MethodResult(
            sample_id=sample.sample_id,
            split_name=sample.metadata.get("split"),
            family=sample.metadata.get("family"),
            sampling_mode=sample.metadata.get("sampling_mode"),
            method=method,
            status="failed",
            setup_time_sec=setup_time,
            apply_time_sec=None,
            solve_time_sec=None,
            total_time_sec=None,
            iterations=None,
            info=None,
            converged=False,
            relative_residual=None,
            final_residual_norm=None,
            factor_nnz=prepared.factor_nnz,
            factor_density=prepared.factor_density,
            operator_nnz=operator_nnz,
            operator_density=operator_density,
            apply_kind=prepared.apply_kind,
            inference_peak_gpu_memory_mb=prepared.metadata.get("inference_peak_gpu_memory_mb"),
            cpu_rss_delta_mb=prepared.metadata.get("cpu_rss_delta_mb"),
            message=str(exc),
            model_name="learning" if method == "learning" else "baseline",
            model_label="learning" if method == "learning" else "baseline",
            is_learning_model=(method == "learning"),
            diag_strategy=learning_diag_strategy if method == "learning" else "",
            **timing_fields,
        )


def prepare_preconditioner(
    method: str,
    sample: BenchmarkSample,
    *,
    model=None,
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str = "cuda",
    ssor_omega: float = 1.2,
    ic0_diagcomp: float = 1e-1,
    ilu_drop_tol: float = 1e-4,
    ilu_fill_factor: float = 10.0,
    learning_group_context: dict[str, Any] | None = None,
    steady_forward_repeats: int = 50,
) -> PreparedPreconditioner:
    method = method.lower()
    if method == "learning":
        if model is None:
            raise RuntimeError("learning method requires a model")
        learning_diag_strategy = normalize_diag_strategy(learning_diag_strategy)
        model = _prepare_learning_model(model, learning_device)
        return _prepare_learning_preconditioner(
            sample,
            model=model,
            device=learning_device,
            output_kind=learning_output_kind,
            diag_strategy=learning_diag_strategy,
            use_mask_projection=use_mask_projection,
            group_context=learning_group_context,
            steady_forward_repeats=steady_forward_repeats,
        )
    if method == "none":
        return _make_identity_preconditioner(sample.A)
    if method == "jacobi":
        return _make_jacobi_preconditioner(sample.A)
    if method == "sgs":
        return _make_split_preconditioner(sample.A, method="sgs", omega=1.0)
    if method == "ssor":
        return _make_split_preconditioner(sample.A, method="ssor", omega=ssor_omega)
    if method == "ic0":
        return _make_ic0_preconditioner(sample.A, diagcomp=ic0_diagcomp)
    if method == "ilu":
        return _make_ilu_preconditioner(
            sample.A,
            drop_tol=ilu_drop_tol,
            fill_factor=ilu_fill_factor,
        )
    if method == "amg":
        return _make_amg_preconditioner(sample.A)
    raise ValueError(f"Unsupported benchmark method: {method}")


def _timing_breakdown(
    prepared: PreparedPreconditioner,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    return (
        prepared.metadata.get("wall_forward_time_sec"),
        prepared.metadata.get("steady_forward_time_sec"),
        prepared.metadata.get("forward_wall_time_sec"),
        prepared.metadata.get("forward_steady_time_sec"),
        prepared.metadata.get("transfer_time_sec"),
        prepared.metadata.get("postprocess_time_sec"),
        prepared.metadata.get("factor_assembly_time_sec"),
        prepared.metadata.get("factor_gpu_to_host_time_sec"),
        prepared.metadata.get("factor_petsc_build_time_sec"),
        prepared.metadata.get("a_petsc_build_time_sec"),
        prepared.metadata.get("solve_ready_build_time_sec"),
        prepared.metadata.get("excluded_materialization_time_sec"),
    )


def _result_timing_fields(prepared: PreparedPreconditioner) -> dict[str, float | None]:
    (
        wall_forward_time,
        steady_forward_time,
        forward_wall_time,
        forward_steady_time,
        transfer_time,
        postprocess_time,
        factor_assembly_time,
        factor_gpu_to_host_time,
        factor_petsc_build_time,
        a_petsc_build_time,
        solve_ready_build_time,
        excluded_materialization_time,
    ) = _timing_breakdown(prepared)
    return {
        "wall_forward_time_sec": wall_forward_time,
        "steady_forward_time_sec": steady_forward_time,
        "forward_wall_time_sec": forward_wall_time,
        "forward_steady_time_sec": forward_steady_time,
        "transfer_time_sec": transfer_time,
        "postprocess_time_sec": postprocess_time,
        "factor_assembly_time_sec": factor_assembly_time,
        "factor_gpu_to_host_time_sec": factor_gpu_to_host_time,
        "factor_petsc_build_time_sec": factor_petsc_build_time,
        "a_petsc_build_time_sec": a_petsc_build_time,
        "solve_ready_build_time_sec": solve_ready_build_time,
        "excluded_materialization_time_sec": excluded_materialization_time,
        "graph_build_time_sec": prepared.metadata.get("graph_build_time_sec"),
    }


def _run_cg(
    A: csr_matrix,
    b: np.ndarray,
    *,
    M: LinearOperator | None,
    rtol: float,
    maxiter: int | None,
) -> tuple[np.ndarray, int, float, list[float], int]:
    counter = _IterationCounter(A, b)
    t0 = time.perf_counter()
    kwargs = {"x0": None, "maxiter": maxiter, "M": M, "callback": counter}
    try:
        x, info = scipy_cg(A, b, rtol=rtol, atol=0.0, **kwargs)
    except TypeError:  # pragma: no cover - SciPy compatibility
        x, info = scipy_cg(A, b, tol=rtol, **kwargs)
    elapsed = time.perf_counter() - t0
    iterations = len(counter.residuals)
    residuals = counter.residuals
    if not residuals:
        residuals = [float(np.linalg.norm(b - A @ x))]
    return x, int(info), elapsed, residuals, iterations


def _run_apply(prepared: PreparedPreconditioner, x: np.ndarray) -> tuple[np.ndarray, float]:
    if prepared.operator is None:
        raise RuntimeError("prepared preconditioner does not expose an apply operator")
    t0 = time.perf_counter()
    y = np.asarray(prepared.operator @ x, dtype=np.float64).reshape(-1)
    elapsed = time.perf_counter() - t0
    return y, elapsed


def _run_factor_system_cg(
    A: csr_matrix,
    b: np.ndarray,
    *,
    factor: csr_matrix,
    rtol: float,
    maxiter: int | None,
) -> tuple[np.ndarray, int, float, list[float], int]:
    factor = factor.tocsr().astype(np.float64)
    factor_t = factor.transpose().tocsr()
    rhs = np.asarray(factor @ b, dtype=np.float64).reshape(-1)
    counter = _FactorSystemIterationCounter(A, b, factor_t)
    transformed = LinearOperator(
        A.shape,
        matvec=lambda y: np.asarray(factor @ (A @ (factor_t @ y)), dtype=np.float64).reshape(-1),
        dtype=np.float64,
    )
    t0 = time.perf_counter()
    kwargs = {"x0": None, "maxiter": maxiter, "callback": counter}
    try:
        y, info = scipy_cg(transformed, rhs, rtol=rtol, atol=0.0, **kwargs)
    except TypeError:  # pragma: no cover - SciPy compatibility
        y, info = scipy_cg(transformed, rhs, tol=rtol, **kwargs)
    elapsed = time.perf_counter() - t0
    x = np.asarray(factor_t @ y, dtype=np.float64).reshape(-1)
    residuals = counter.residuals
    if not residuals:
        residuals = [float(np.linalg.norm(b - A @ x))]
    return x, int(info), elapsed, residuals, len(residuals)


def _explicit_preconditioned_matrix(
    A: csr_matrix,
    prepared: PreparedPreconditioner,
    dense_limit: int,
) -> np.ndarray | None:
    n = A.shape[0]
    if n > dense_limit:
        return None
    P = prepared.explicit_operator
    if P is None:
        identity = np.eye(n, dtype=np.float64)
        P = np.column_stack([prepared.operator @ identity[:, i] for i in range(n)])
    return np.asarray(P @ A.toarray(), dtype=np.float64)


def _preconditioned_operator(
    A: csr_matrix,
    prepared: PreparedPreconditioner,
) -> LinearOperator | None:
    if prepared.operator is None:
        return None
    shape = A.shape

    def matvec(x: np.ndarray) -> np.ndarray:
        ax = np.asarray(A @ x, dtype=np.float64).reshape(-1)
        return np.asarray(prepared.operator @ ax, dtype=np.float64).reshape(-1)

    return LinearOperator(shape, matvec=matvec, dtype=np.float64)


def _estimate_sparse_condition_number(
    A: csr_matrix,
    prepared: PreparedPreconditioner,
) -> float | None:
    operator = _preconditioned_operator(A, prepared)
    if operator is None:
        return None
    n = int(A.shape[0])
    eig_kwargs = {
        "k": 1,
        "tol": 1.0e-2,
        "maxiter": min(max(n, 1000), 4000),
        "return_eigenvectors": False,
    }
    try:
        lambda_max = np.real_if_close(eigs(operator, which="LM", **eig_kwargs)[0])
        lambda_min = np.real_if_close(eigs(operator, which="SM", **eig_kwargs)[0])
    except Exception:
        return None
    lambda_max = float(np.abs(lambda_max))
    lambda_min = float(np.abs(lambda_min))
    if not np.isfinite(lambda_max) or not np.isfinite(lambda_min):
        return None
    if lambda_min <= 0.0:
        return None
    return lambda_max / lambda_min


def _compute_spectral_metrics(
    A: csr_matrix,
    prepared: PreparedPreconditioner,
    *,
    enabled: bool,
    dense_limit: int,
) -> tuple[float | None, float | None, float | None]:
    if not enabled:
        return None, None, None
    sparse_cond_est = _estimate_sparse_condition_number(A, prepared)
    preA = _explicit_preconditioned_matrix(A, prepared, dense_limit)
    if preA is None:
        return sparse_cond_est, None, None
    eigvals = np.linalg.eigvals(preA)
    eigvals = np.real_if_close(eigvals)
    eig_abs = np.abs(eigvals)
    eig_min = float(np.min(eig_abs))
    eig_max = float(np.max(eig_abs))
    dense_cond_est = None if eig_min <= 0.0 else eig_max / eig_min
    cond_est = sparse_cond_est if sparse_cond_est is not None else dense_cond_est
    mean_abs = float(np.mean(eig_abs))
    eig_cv = None if mean_abs == 0.0 else float(np.std(eig_abs) / mean_abs)
    fro_error = float(np.linalg.norm(np.eye(preA.shape[0]) - preA, ord="fro") ** 2)
    return cond_est, eig_cv, fro_error


def _run_learning_model_forward(model, model_input, *, profile: bool):
    if torch is None:
        raise RuntimeError("torch is required for the learning benchmark path")
    with torch.no_grad():
        if profile:
            try:
                return model(model_input, profile=True)
            except TypeError:
                return model(model_input)
        try:
            return model(model_input, profile=False)
        except TypeError:
            return model(model_input)


def _save_residual_history(
    residual_dir: Path | None,
    sample_id: str,
    method: str,
    residuals: Iterable[float],
) -> str | None:
    if residual_dir is None:
        return None
    residual_dir.mkdir(parents=True, exist_ok=True)
    path = residual_dir / f"{sample_id}__{method}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iter", "residual"])
        for it, value in enumerate(residuals):
            writer.writerow([it, value])
    return str(path)


def _coalesced_sample_values(sample: BenchmarkSample, device: str):
    if torch is None or sample.l_tensor is None:
        raise RuntimeError("learning benchmark requires torch sparse l_tensor")
    coalesced = sample.l_tensor.coalesce()
    return coalesced.values().unsqueeze(1).to(device=device)


def _make_learning_group_context(*, device: str, model) -> dict[str, Any]:
    return {
        "device": device,
        "input_kind": getattr(model, "input_kind", "spconv"),
        "base_spconv_input": None,
    }


def _run_group_ignition(
    sample: BenchmarkSample,
    methods: Sequence[str],
    *,
    model=None,
    backend: str = "scipy",
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str = "cuda",
    rtol: float = 1e-5,
    maxiter: int | None = None,
    ssor_omega: float = 1.2,
    ic0_diagcomp: float = 1e-1,
    ilu_drop_tol: float = 1e-4,
    ilu_fill_factor: float = 10.0,
    learning_group_context: dict[str, Any] | None = None,
    setup_warmup_runs_gpu_learning: int = _FIXED_SETUP_WARMUP_RUNS_GPU_LEARNING,
    setup_warmup_runs_other: int = _FIXED_SETUP_WARMUP_RUNS_OTHER,
    cpu_apply_warmup_runs: int = 1,
    cpu_solve_warmup_runs: int = 1,
    steady_forward_repeats: int = 50,
    petsc_amg_backend: str = "gamg",
    petsc_options: str | None = None,
    petsc_ksp_norm_type: str = "unpreconditioned",
    petsc_factor_solve_mode: str = "transformed_operator_native",
    petsc_factor_operator_mode: str = "explicit_aijcusparse",
    petsc_cond_mode: str = "accurate_ref",
    petsc_apply_warmup_runs: int = 0,
    petsc_apply_measure_repeats: int = 5,
    petsc_learning_internal_rtol_ratio: float = 1.0,
) -> None:
    backend = _parse_backend(backend)
    if backend == "petsc_gpu":
        from dpcg import petsc_benchmark

        petsc_benchmark.run_group_ignition(
            sample,
            methods,
            model=model,
            learning_output_kind=learning_output_kind,
            learning_diag_strategy=learning_diag_strategy,
            use_mask_projection=use_mask_projection,
            learning_device=learning_device,
            rtol=rtol,
            maxiter=maxiter,
            ssor_omega=ssor_omega,
            setup_warmup_runs_gpu_learning=setup_warmup_runs_gpu_learning,
            setup_warmup_runs_other=setup_warmup_runs_other,
            apply_warmup_runs=cpu_apply_warmup_runs,
            solve_warmup_runs=cpu_solve_warmup_runs,
            steady_forward_repeats=steady_forward_repeats,
            learning_group_context=learning_group_context,
            petsc_amg_backend=petsc_amg_backend,
            petsc_options=petsc_options,
            petsc_ksp_norm_type=petsc_ksp_norm_type,
            petsc_factor_solve_mode=petsc_factor_solve_mode,
            petsc_factor_operator_mode=petsc_factor_operator_mode,
            petsc_cond_mode=petsc_cond_mode,
            petsc_apply_warmup_runs=petsc_apply_warmup_runs,
            petsc_apply_measure_repeats=petsc_apply_measure_repeats,
            petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
        )
        return
    for method in methods:
        if method == "learning":
            if model is None:
                continue
            if str(learning_device).startswith("cuda"):
                setup_runs = max(int(setup_warmup_runs_gpu_learning), 0)
            else:
                setup_runs = max(int(setup_warmup_runs_other), 0)
        else:
            setup_runs = max(int(setup_warmup_runs_other), 0)

        prepared = None
        for _ in range(setup_runs):
            prepared = prepare_preconditioner(
                method,
                sample,
                model=model,
                learning_output_kind=learning_output_kind,
                learning_diag_strategy=learning_diag_strategy,
                use_mask_projection=use_mask_projection,
                learning_device=learning_device,
                ssor_omega=ssor_omega,
                ic0_diagcomp=ic0_diagcomp,
                ilu_drop_tol=ilu_drop_tol,
                ilu_fill_factor=ilu_fill_factor,
                learning_group_context=learning_group_context,
                steady_forward_repeats=steady_forward_repeats,
            )
        if prepared is None:
            try:
                prepared = prepare_preconditioner(
                    method,
                    sample,
                    model=model,
                    learning_output_kind=learning_output_kind,
                    learning_diag_strategy=learning_diag_strategy,
                    use_mask_projection=use_mask_projection,
                    learning_device=learning_device,
                    ssor_omega=ssor_omega,
                    ic0_diagcomp=ic0_diagcomp,
                    ilu_drop_tol=ilu_drop_tol,
                    ilu_fill_factor=ilu_fill_factor,
                    learning_group_context=learning_group_context,
                    steady_forward_repeats=steady_forward_repeats,
                )
            except Exception:
                continue
        for _ in range(max(int(cpu_apply_warmup_runs), 0)):
            _run_apply(prepared, sample.b)
        for _ in range(max(int(cpu_solve_warmup_runs), 0)):
            if method == "learning" and prepared.factor_matrix is not None:
                _run_factor_system_cg(
                    sample.A,
                    sample.b,
                    factor=prepared.factor_matrix,
                    rtol=rtol,
                    maxiter=maxiter,
                )
            else:
                _run_cg(
                    sample.A,
                    sample.b,
                    M=prepared.operator,
                    rtol=rtol,
                    maxiter=maxiter,
                )


def _resolve_method_specs(
    *,
    methods: Sequence[str],
    model,
    learning_models: Sequence[LearningBenchmarkModel] | None,
    learning_diag_strategy: str,
    learning_device: str,
    use_mask_projection: bool,
) -> list[_ResolvedMethodSpec]:
    resolved: list[_ResolvedMethodSpec] = []
    prepared_model = _prepare_learning_model(model, learning_device) if model is not None else None
    for method in methods:
        if method != "learning":
            resolved.append(
                _ResolvedMethodSpec(
                    method=method,
                    model=None,
                    model_name="baseline",
                    model_label="baseline",
                    is_learning_model=False,
                    diag_strategy="",
                    mask_percentile=None,
                    use_mask_projection=use_mask_projection,
                )
            )
            continue
        if learning_models:
            for learning_spec in learning_models:
                resolved.append(
                    _ResolvedMethodSpec(
                        method="learning",
                        model=learning_spec.model,
                        model_name=learning_spec.model_name,
                        model_label=learning_spec.label,
                        is_learning_model=True,
                        diag_strategy=learning_spec.diag_strategy,
                        mask_percentile=learning_spec.mask_percentile,
                        use_mask_projection=bool(
                            use_mask_projection
                            if learning_spec.use_mask_projection is None
                            else learning_spec.use_mask_projection
                        ),
                    )
                )
            continue
        if prepared_model is None:
            continue
        resolved.append(
            _ResolvedMethodSpec(
                method="learning",
                model=prepared_model,
                model_name="learning",
                model_label="learning",
                is_learning_model=True,
                diag_strategy=learning_diag_strategy,
                mask_percentile=None,
                use_mask_projection=use_mask_projection,
            )
        )
    return resolved


def _make_method_group_context(spec: _ResolvedMethodSpec, *, device: str) -> dict[str, Any] | None:
    if not spec.is_learning_model or spec.model is None:
        return None
    if getattr(spec.model, "input_kind", None) != "spconv":
        return None
    return _make_learning_group_context(device=device, model=spec.model)


def _make_residual_dir_for_rtol(
    residual_dir: Path | None,
    *,
    rtol_label: str,
    multi_rtol: bool,
) -> Path | None:
    if residual_dir is None:
        return None
    if not multi_rtol:
        residual_dir.mkdir(parents=True, exist_ok=True)
        return residual_dir
    path = residual_dir / f"rtol_{_safe_rtol_label(rtol_label)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cpu_prepare_once(builder) -> tuple[PreparedPreconditioner, float]:
    prepared = builder()
    return prepared, float(prepared.setup_time_sec)


def _median_optional(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.median(np.asarray(numeric, dtype=np.float64)))


def _median_timing_fields(
    observations: Sequence[dict[str, float | None]],
) -> dict[str, float | None]:
    if not observations:
        return {}
    keys = set().union(*(item.keys() for item in observations))
    return {key: _median_optional([item.get(key) for item in observations]) for key in sorted(keys)}


def _cpu_solve_once(
    sample: BenchmarkSample,
    prepared: PreparedPreconditioner,
    *,
    method: str,
    rtol: float,
    maxiter: int | None,
    residual_dir: Path | None,
    enable_spectral_metrics: bool,
    spectral_dense_limit: int,
    residual_method_name: str,
    save_residual_history: bool,
    collect_spectral_metrics: bool,
) -> dict[str, Any]:
    if method == "learning" and prepared.factor_matrix is not None:
        x, info, solve_time_sec, residuals, iterations = _run_factor_system_cg(
            sample.A,
            sample.b,
            factor=prepared.factor_matrix,
            rtol=rtol,
            maxiter=maxiter,
        )
    else:
        x, info, solve_time_sec, residuals, iterations = _run_cg(
            sample.A,
            sample.b,
            M=prepared.operator,
            rtol=rtol,
            maxiter=maxiter,
        )
    converged = int(info) == 0
    final_residual_norm = float(np.linalg.norm(sample.b - sample.A @ x))
    residual_history_file = None
    if save_residual_history:
        residual_history_file = _save_residual_history(
            residual_dir,
            sample.sample_id,
            residual_method_name,
            residuals,
        )
    cond_est = None
    eig_cv = None
    fro_error = None
    if collect_spectral_metrics:
        cond_est, eig_cv, fro_error = _compute_spectral_metrics(
            sample.A,
            prepared,
            enabled=enable_spectral_metrics,
            dense_limit=spectral_dense_limit,
        )
    return {
        "info": int(info),
        "solve_time_sec": float(solve_time_sec),
        "iterations": int(iterations),
        "converged": converged,
        "relative_residual": float(rtol) if converged else None,
        "final_residual_norm": final_residual_norm,
        "residual_history_file": residual_history_file,
        "cond_est": cond_est,
        "eig_cv": eig_cv,
        "fro_error": fro_error,
    }


def _cpu_failed_rows(
    sample: BenchmarkSample,
    spec: _ResolvedMethodSpec,
    *,
    rtol_values: Sequence[float],
    message: str,
    status: str,
    operator_nnz: float,
    operator_density: float,
    burnin_sample_id: str | None,
    setup_cold_time_sec: float | None = None,
    setup_time_sec: float = 0.0,
    apply_cold_time_sec: float | None = None,
    factor_gpu_to_host_cold_time_sec: float | None = None,
    factor_petsc_build_cold_time_sec: float | None = None,
    a_petsc_build_cold_time_sec: float | None = None,
    solve_ready_build_cold_time_sec: float | None = None,
    excluded_materialization_cold_time_sec: float | None = None,
    timing_fields: dict[str, float | None] | None = None,
    setup_warmup_runs: int | None = None,
    apply_warmup_runs: int | None = None,
    solve_warmup_runs: int | None = None,
    timing_stable: bool | None = None,
) -> list[MethodResult]:
    rows: list[MethodResult] = []
    timing_fields = dict(timing_fields or {})
    for rtol_value in rtol_values:
        rows.append(
            MethodResult(
                sample_id=sample.sample_id,
                split_name=sample.metadata.get("split"),
                family=sample.metadata.get("family"),
                sampling_mode=sample.metadata.get("sampling_mode"),
                method=spec.method,
                status=status,
                setup_time_sec=setup_time_sec,
                apply_time_sec=None,
                solve_time_sec=None,
                total_time_sec=None,
                iterations=None,
                info=None,
                converged=False,
                relative_residual=None,
                final_residual_norm=None,
                factor_nnz=None,
                factor_density=None,
                operator_nnz=operator_nnz,
                operator_density=operator_density,
                apply_kind="unavailable",
                message=message,
                setup_cold_time_sec=setup_cold_time_sec,
                apply_cold_time_sec=apply_cold_time_sec,
                factor_gpu_to_host_cold_time_sec=factor_gpu_to_host_cold_time_sec,
                factor_petsc_build_cold_time_sec=factor_petsc_build_cold_time_sec,
                a_petsc_build_cold_time_sec=a_petsc_build_cold_time_sec,
                solve_ready_build_cold_time_sec=solve_ready_build_cold_time_sec,
                excluded_materialization_cold_time_sec=excluded_materialization_cold_time_sec,
                setup_warmup_runs=setup_warmup_runs,
                apply_warmup_runs=apply_warmup_runs,
                solve_warmup_runs=solve_warmup_runs,
                setup_measure_repeats=_SETUP_MEASURE_REPEATS,
                apply_measure_repeats=_APPLY_MEASURE_REPEATS,
                solve_measure_repeats=_SOLVE_MEASURE_REPEATS,
                timing_stable=timing_stable,
                burnin_sample_id=burnin_sample_id,
                model_name=spec.model_name,
                model_label=spec.model_label,
                is_learning_model=spec.is_learning_model,
                diag_strategy=spec.diag_strategy,
                rtol=float(rtol_value),
                rtol_label=_normalize_rtol_label(float(rtol_value)),
                **timing_fields,
            )
        )
    return rows


def _run_cpu_method_burnin(
    sample: BenchmarkSample,
    spec: _ResolvedMethodSpec,
    *,
    learning_output_kind: str,
    use_mask_projection: bool,
    learning_device: str,
    maxiter: Any,
    ssor_omega: float,
    ic0_diagcomp: float,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
    learning_group_context: dict[str, Any] | None,
    steady_forward_repeats: int,
    burnin_rtol: float,
) -> None:
    _reset_learning_group_context(learning_group_context)
    prepared = prepare_preconditioner(
        spec.method,
        sample,
        model=spec.model,
        learning_output_kind=learning_output_kind,
        learning_diag_strategy=spec.diag_strategy or "learned_exp",
        use_mask_projection=use_mask_projection,
        learning_device=learning_device,
        ssor_omega=ssor_omega,
        ic0_diagcomp=ic0_diagcomp,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        learning_group_context=learning_group_context,
        steady_forward_repeats=steady_forward_repeats,
    )
    _adaptive_warmup(
        lambda: _run_apply(prepared, sample.b),
        policy=_apply_timing_policy(),
    )
    if spec.method == "learning" and prepared.factor_matrix is not None:

        def solve_runner():
            return _run_factor_system_cg(
                sample.A,
                sample.b,
                factor=prepared.factor_matrix,
                rtol=burnin_rtol,
                maxiter=_resolve_runtime_maxiter(maxiter, burnin_rtol),
            )
    else:

        def solve_runner():
            return _run_cg(
                sample.A,
                sample.b,
                M=prepared.operator,
                rtol=burnin_rtol,
                maxiter=_resolve_runtime_maxiter(maxiter, burnin_rtol),
            )

    _adaptive_warmup(
        solve_runner,
        policy=_solve_timing_policy(),
    )


def _benchmark_cpu_method_sample(
    sample: BenchmarkSample,
    spec: _ResolvedMethodSpec,
    *,
    rtol_values: Sequence[float],
    learning_output_kind: str,
    use_mask_projection: bool,
    learning_device: str,
    maxiter: Any,
    ssor_omega: float,
    ic0_diagcomp: float,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
    enable_spectral_metrics: bool,
    spectral_dense_limit: int,
    residual_dir: Path | None,
    learning_group_context: dict[str, Any] | None,
    steady_forward_repeats: int,
    burnin_sample_id: str | None,
) -> list[MethodResult]:
    operator_density = _matrix_density(sample.A)
    operator_nnz = float(sample.A.nnz)

    def build_prepared() -> PreparedPreconditioner:
        _reset_learning_group_context(learning_group_context)
        return prepare_preconditioner(
            spec.method,
            sample,
            model=spec.model,
            learning_output_kind=learning_output_kind,
            learning_diag_strategy=spec.diag_strategy or "learned_exp",
            use_mask_projection=use_mask_projection,
            learning_device=learning_device,
            ssor_omega=ssor_omega,
            ic0_diagcomp=ic0_diagcomp,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            learning_group_context=learning_group_context,
            steady_forward_repeats=steady_forward_repeats,
        )

    try:
        _check_spd(sample.A)
        cold_prepared, cold_setup_time_sec = _cpu_prepare_once(build_prepared)
        cold_timing_fields = _result_timing_fields(cold_prepared)
        del cold_prepared
    except Exception as exc:
        status = "skipped" if spec.method == "amg" else "failed"
        return _cpu_failed_rows(
            sample,
            spec,
            rtol_values=rtol_values,
            message=str(exc),
            status=status,
            operator_nnz=operator_nnz,
            operator_density=operator_density,
            burnin_sample_id=burnin_sample_id,
        )

    try:
        setup_warmup_runs, setup_stable = _adaptive_warmup(
            lambda: _cpu_prepare_once(build_prepared)[1],
            policy=_setup_timing_policy(),
            elapsed_resolver=lambda value: value,
        )
        setup_observations: list[float] = []
        timing_observations: list[dict[str, float | None]] = []
        for _ in range(_SETUP_MEASURE_REPEATS):
            measured_prepared, setup_time_sec = _cpu_prepare_once(build_prepared)
            setup_observations.append(setup_time_sec)
            timing_observations.append(_result_timing_fields(measured_prepared))
            del measured_prepared
        active_prepared = build_prepared()
    except Exception as exc:
        status = "skipped" if spec.method == "amg" else "failed"
        return _cpu_failed_rows(
            sample,
            spec,
            rtol_values=rtol_values,
            message=str(exc),
            status=status,
            operator_nnz=operator_nnz,
            operator_density=operator_density,
            burnin_sample_id=burnin_sample_id,
            setup_cold_time_sec=cold_setup_time_sec,
            factor_gpu_to_host_cold_time_sec=cold_timing_fields.get("factor_gpu_to_host_time_sec"),
            factor_petsc_build_cold_time_sec=cold_timing_fields.get("factor_petsc_build_time_sec"),
            a_petsc_build_cold_time_sec=cold_timing_fields.get("a_petsc_build_time_sec"),
            solve_ready_build_cold_time_sec=cold_timing_fields.get("solve_ready_build_time_sec"),
            excluded_materialization_cold_time_sec=cold_timing_fields.get(
                "excluded_materialization_time_sec"
            ),
            setup_warmup_runs=setup_warmup_runs,
            timing_stable=setup_stable,
        )

    setup_time_sec = float(np.median(np.asarray(setup_observations, dtype=np.float64)))
    setup_cold_time_sec = cold_setup_time_sec
    timing_fields = _median_timing_fields(timing_observations)

    try:
        _apply_y, apply_cold_time_sec = _run_apply(active_prepared, sample.b)
        del _apply_y
        apply_warmup_runs, apply_stable = _adaptive_warmup(
            lambda: _run_apply(active_prepared, sample.b),
            policy=_apply_timing_policy(),
        )
        apply_observations = _measure_repeated_calls(
            lambda: _run_apply(active_prepared, sample.b),
            repeats=_APPLY_MEASURE_REPEATS,
        )
        apply_time_sec = _median_elapsed(apply_observations)
    except Exception as exc:
        return _cpu_failed_rows(
            sample,
            spec,
            rtol_values=rtol_values,
            message=str(exc),
            status="failed",
            operator_nnz=operator_nnz,
            operator_density=operator_density,
            burnin_sample_id=burnin_sample_id,
            setup_cold_time_sec=setup_cold_time_sec,
            setup_time_sec=setup_time_sec,
            apply_cold_time_sec=None,
            factor_gpu_to_host_cold_time_sec=cold_timing_fields.get("factor_gpu_to_host_time_sec"),
            factor_petsc_build_cold_time_sec=cold_timing_fields.get("factor_petsc_build_time_sec"),
            a_petsc_build_cold_time_sec=cold_timing_fields.get("a_petsc_build_time_sec"),
            solve_ready_build_cold_time_sec=cold_timing_fields.get("solve_ready_build_time_sec"),
            excluded_materialization_cold_time_sec=cold_timing_fields.get(
                "excluded_materialization_time_sec"
            ),
            timing_fields=timing_fields,
            setup_warmup_runs=setup_warmup_runs,
            timing_stable=False,
        )

    total_time_basis = "setup_plus_solve"
    rows: list[MethodResult] = []
    multi_rtol = len(rtol_values) > 1
    for rtol_value in rtol_values:
        rtol_label = _normalize_rtol_label(float(rtol_value))
        solve_residual_dir = _make_residual_dir_for_rtol(
            residual_dir,
            rtol_label=rtol_label,
            multi_rtol=multi_rtol,
        )
        residual_method_name = spec.method

        def solve_runner():
            return _cpu_solve_once(
                sample,
                active_prepared,
                method=spec.method,
                rtol=float(rtol_value),
                maxiter=_resolve_runtime_maxiter(maxiter, float(rtol_value)),
                residual_dir=solve_residual_dir,
                enable_spectral_metrics=enable_spectral_metrics,
                spectral_dense_limit=spectral_dense_limit,
                residual_method_name=residual_method_name,
                save_residual_history=False,
                collect_spectral_metrics=False,
            )

        try:
            solve_warmup_runs, solve_stable = _adaptive_warmup(
                solve_runner,
                policy=_solve_timing_policy(),
                elapsed_resolver=lambda item: item["solve_time_sec"],
            )
            solve_observations = _measure_repeated_calls(
                solve_runner,
                repeats=_SOLVE_MEASURE_REPEATS,
                elapsed_resolver=lambda item: item["solve_time_sec"],
            )
            diagnostic_solve = _cpu_solve_once(
                sample,
                active_prepared,
                method=spec.method,
                rtol=float(rtol_value),
                maxiter=_resolve_runtime_maxiter(maxiter, float(rtol_value)),
                residual_dir=solve_residual_dir,
                enable_spectral_metrics=enable_spectral_metrics,
                spectral_dense_limit=spectral_dense_limit,
                residual_method_name=residual_method_name,
                save_residual_history=True,
                collect_spectral_metrics=True,
            )
            solve_time_sec = _median_elapsed(solve_observations)
            rows.append(
                MethodResult(
                    sample_id=sample.sample_id,
                    split_name=sample.metadata.get("split"),
                    family=sample.metadata.get("family"),
                    sampling_mode=sample.metadata.get("sampling_mode"),
                    method=spec.method,
                    status="ok" if diagnostic_solve["converged"] else "failed",
                    setup_time_sec=setup_time_sec,
                    apply_time_sec=apply_time_sec,
                    solve_time_sec=solve_time_sec,
                    total_time_sec=float(setup_time_sec + solve_time_sec),
                    iterations=diagnostic_solve["iterations"],
                    info=diagnostic_solve["info"],
                    converged=diagnostic_solve["converged"],
                    relative_residual=diagnostic_solve["relative_residual"],
                    final_residual_norm=diagnostic_solve["final_residual_norm"],
                    factor_nnz=active_prepared.factor_nnz,
                    factor_density=active_prepared.factor_density,
                    operator_nnz=operator_nnz,
                    operator_density=operator_density,
                    apply_kind=active_prepared.apply_kind,
                    inference_peak_gpu_memory_mb=_prepared_metric(
                        active_prepared,
                        "inference_peak_gpu_memory_mb",
                    ),
                    cpu_rss_delta_mb=_prepared_metric(active_prepared, "cpu_rss_delta_mb"),
                    residual_history_file=diagnostic_solve["residual_history_file"],
                    eig_cv=diagnostic_solve["eig_cv"],
                    fro_error=diagnostic_solve["fro_error"],
                    total_time_basis=total_time_basis,
                    setup_cold_time_sec=setup_cold_time_sec,
                    apply_cold_time_sec=float(apply_cold_time_sec),
                    factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                        "factor_gpu_to_host_time_sec"
                    ),
                    factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                        "factor_petsc_build_time_sec"
                    ),
                    a_petsc_build_cold_time_sec=cold_timing_fields.get("a_petsc_build_time_sec"),
                    solve_ready_build_cold_time_sec=cold_timing_fields.get(
                        "solve_ready_build_time_sec"
                    ),
                    excluded_materialization_cold_time_sec=cold_timing_fields.get(
                        "excluded_materialization_time_sec"
                    ),
                    setup_warmup_runs=setup_warmup_runs,
                    apply_warmup_runs=apply_warmup_runs,
                    solve_warmup_runs=solve_warmup_runs,
                    setup_measure_repeats=_SETUP_MEASURE_REPEATS,
                    apply_measure_repeats=_APPLY_MEASURE_REPEATS,
                    solve_measure_repeats=_SOLVE_MEASURE_REPEATS,
                    timing_stable=bool(setup_stable and apply_stable and solve_stable),
                    burnin_sample_id=burnin_sample_id,
                    model_name=spec.model_name,
                    model_label=spec.model_label,
                    is_learning_model=spec.is_learning_model,
                    diag_strategy=spec.diag_strategy,
                    rtol=float(rtol_value),
                    rtol_label=rtol_label,
                    **timing_fields,
                )
            )
        except Exception as exc:
            rows.extend(
                _cpu_failed_rows(
                    sample,
                    spec,
                    rtol_values=[float(rtol_value)],
                    message=str(exc),
                    status="failed",
                    operator_nnz=operator_nnz,
                    operator_density=operator_density,
                    burnin_sample_id=burnin_sample_id,
                    setup_cold_time_sec=setup_cold_time_sec,
                    setup_time_sec=setup_time_sec,
                    apply_cold_time_sec=float(apply_cold_time_sec),
                    factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                        "factor_gpu_to_host_time_sec"
                    ),
                    factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                        "factor_petsc_build_time_sec"
                    ),
                    a_petsc_build_cold_time_sec=cold_timing_fields.get("a_petsc_build_time_sec"),
                    solve_ready_build_cold_time_sec=cold_timing_fields.get(
                        "solve_ready_build_time_sec"
                    ),
                    excluded_materialization_cold_time_sec=cold_timing_fields.get(
                        "excluded_materialization_time_sec"
                    ),
                    timing_fields=timing_fields,
                    setup_warmup_runs=setup_warmup_runs,
                    apply_warmup_runs=apply_warmup_runs,
                    timing_stable=False,
                )
            )
    return rows


def _benchmark_samples(
    samples: Sequence[BenchmarkSample],
    *,
    methods: Sequence[str],
    model,
    learning_models: Sequence[LearningBenchmarkModel] | None,
    backend: str,
    learning_output_kind: str,
    learning_diag_strategy: str,
    use_mask_projection: bool,
    learning_device: str,
    rtol_values: Sequence[float],
    maxiter: int | None,
    ssor_omega: float,
    ic0_diagcomp: float,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
    enable_spectral_metrics: bool,
    spectral_dense_limit: int,
    residual_dir: Path | None,
    steady_forward_repeats: int,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str,
    petsc_options: str | None,
    petsc_ksp_norm_type: str,
    petsc_sgs_ssor_impl: str,
    petsc_factor_solve_mode: str,
    petsc_factor_operator_mode: str,
    petsc_cond_mode: str,
    petsc_apply_measure_repeats: int,
    petsc_learning_internal_rtol_ratio: float,
    petsc_learning_pc_impl: str,
) -> tuple[list[MethodResult], dict[str, int]]:
    backend = _parse_backend(backend)
    if backend != "petsc_gpu" and "parasails" in methods:
        raise ValueError(
            "method 'parasails' is only supported by BENCHMARK_BACKEND='petsc_gpu'"
        )
    resolved_specs = _resolve_method_specs(
        methods=methods,
        model=model,
        learning_models=learning_models,
        learning_diag_strategy=learning_diag_strategy,
        learning_device=learning_device,
        use_mask_projection=use_mask_projection,
    )
    counts = {
        "num_input_samples": len(samples),
        "num_ignition_samples": 1 if samples else 0,
        "num_measured_samples": len(samples),
    }
    all_rows: list[MethodResult] = []
    burnin_sample_id = samples[0].sample_id if samples else None
    for spec in resolved_specs:
        spec_samples = [
            _remask_benchmark_sample(
                sample,
                mask_percentile=spec.mask_percentile,
                use_mask_projection=spec.use_mask_projection,
            )
            for sample in samples
        ]
        method_group_context = _make_method_group_context(spec, device=learning_device)
        if spec_samples:
            burnin_sample = _clone_benchmark_sample(spec_samples[0])
            try:
                if backend == "petsc_gpu":
                    from dpcg import petsc_benchmark

                    petsc_benchmark.run_method_burnin(
                        burnin_sample,
                        method=spec.method,
                        model=spec.model,
                        learning_output_kind=learning_output_kind,
                        learning_diag_strategy=spec.diag_strategy or learning_diag_strategy,
                        use_mask_projection=spec.use_mask_projection,
                        learning_device=learning_device,
                        rtol=max(float(item) for item in rtol_values),
                        maxiter=maxiter,
                        ssor_omega=ssor_omega,
                        learning_group_context=method_group_context,
                        steady_forward_repeats=steady_forward_repeats,
                        petsc_amg_backend=petsc_amg_backend,
                        petsc_ic0_shift_type=petsc_ic0_shift_type,
                        petsc_options=petsc_options,
                        petsc_ksp_norm_type=petsc_ksp_norm_type,
                        petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                        petsc_factor_solve_mode=petsc_factor_solve_mode,
                        petsc_factor_operator_mode=petsc_factor_operator_mode,
                        petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
                        petsc_learning_pc_impl=petsc_learning_pc_impl,
                    )
                else:
                    _run_cpu_method_burnin(
                        burnin_sample,
                        spec,
                        learning_output_kind=learning_output_kind,
                        use_mask_projection=spec.use_mask_projection,
                        learning_device=learning_device,
                        maxiter=maxiter,
                        ssor_omega=ssor_omega,
                        ic0_diagcomp=ic0_diagcomp,
                        ilu_drop_tol=ilu_drop_tol,
                        ilu_fill_factor=ilu_fill_factor,
                        learning_group_context=method_group_context,
                        steady_forward_repeats=steady_forward_repeats,
                        burnin_rtol=max(float(item) for item in rtol_values),
                    )
            except Exception:
                pass
        if backend == "petsc_gpu":
            from dpcg import petsc_benchmark

            all_rows.extend(
                petsc_benchmark.benchmark_method_group(
                    spec_samples,
                    method=spec.method,
                    model=spec.model,
                    model_name=spec.model_name,
                    model_label=spec.model_label,
                    is_learning_model=spec.is_learning_model,
                    learning_output_kind=learning_output_kind,
                    learning_diag_strategy=spec.diag_strategy or learning_diag_strategy,
                    use_mask_projection=spec.use_mask_projection,
                    learning_device=learning_device,
                    rtol_values=rtol_values,
                    maxiter=maxiter,
                    ssor_omega=ssor_omega,
                    residual_dir=residual_dir,
                    learning_group_context=method_group_context,
                    steady_forward_repeats=steady_forward_repeats,
                    petsc_amg_backend=petsc_amg_backend,
                    petsc_ic0_shift_type=petsc_ic0_shift_type,
                    petsc_options=petsc_options,
                    petsc_ksp_norm_type=petsc_ksp_norm_type,
                    petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                    petsc_factor_solve_mode=petsc_factor_solve_mode,
                    petsc_factor_operator_mode=petsc_factor_operator_mode,
                    petsc_cond_mode=petsc_cond_mode,
                    petsc_enable_approx_spectral_diagnostics=enable_spectral_metrics,
                    petsc_apply_measure_repeats=petsc_apply_measure_repeats,
                    petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
                    petsc_learning_pc_impl=petsc_learning_pc_impl,
                    burnin_sample_id=burnin_sample_id,
                )
            )
            continue
        for sample in spec_samples:
            all_rows.extend(
                _benchmark_cpu_method_sample(
                    sample,
                    spec,
                    rtol_values=rtol_values,
                    learning_output_kind=learning_output_kind,
                    use_mask_projection=spec.use_mask_projection,
                    learning_device=learning_device,
                    maxiter=maxiter,
                    ssor_omega=ssor_omega,
                    ic0_diagcomp=ic0_diagcomp,
                    ilu_drop_tol=ilu_drop_tol,
                    ilu_fill_factor=ilu_fill_factor,
                    enable_spectral_metrics=enable_spectral_metrics,
                    spectral_dense_limit=spectral_dense_limit,
                    residual_dir=residual_dir,
                    learning_group_context=method_group_context,
                    steady_forward_repeats=steady_forward_repeats,
                    burnin_sample_id=burnin_sample_id,
                )
            )
    return all_rows, counts


def benchmark_sample(
    sample: BenchmarkSample,
    methods: Sequence[str],
    *,
    model=None,
    backend: str = "scipy",
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str = "cuda",
    rtol: float | Sequence[float] = 1e-5,
    maxiter: int | None = None,
    ssor_omega: float = 1.2,
    ic0_diagcomp: float = 1e-1,
    ilu_drop_tol: float = 1e-4,
    ilu_fill_factor: float = 10.0,
    enable_spectral_metrics: bool = False,
    spectral_dense_limit: int = 256,
    residual_dir: Path | None = None,
    learning_group_context: dict[str, Any] | None = None,
    steady_forward_repeats: int = 50,
    petsc_amg_backend: str = "gamg",
    petsc_options: str | None = None,
    petsc_ksp_norm_type: str = "unpreconditioned",
    petsc_sgs_ssor_impl: str = "petsc_sor_legacy",
    petsc_factor_solve_mode: str = "transformed_operator_native",
    petsc_factor_operator_mode: str = "explicit_aijcusparse",
    petsc_cond_mode: str = "accurate_ref",
    petsc_apply_warmup_runs: int = 0,
    petsc_apply_measure_repeats: int = 5,
    petsc_learning_internal_rtol_ratio: float = 1.0,
    petsc_learning_pc_impl: str = "shell_native",
) -> list[MethodResult]:
    del learning_group_context, petsc_apply_warmup_runs
    rows, _counts = _benchmark_samples(
        [normalize_batch(sample)],
        methods=methods,
        model=model,
        learning_models=None,
        backend=backend,
        learning_output_kind=learning_output_kind,
        learning_diag_strategy=learning_diag_strategy,
        use_mask_projection=use_mask_projection,
        learning_device=learning_device,
        rtol_values=_parse_rtol_values(rtol),
        maxiter=maxiter,
        ssor_omega=ssor_omega,
        ic0_diagcomp=ic0_diagcomp,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        enable_spectral_metrics=enable_spectral_metrics,
        spectral_dense_limit=spectral_dense_limit,
        residual_dir=residual_dir,
        steady_forward_repeats=steady_forward_repeats,
        petsc_amg_backend=petsc_amg_backend,
        petsc_ic0_shift_type="none",
        petsc_options=petsc_options,
        petsc_ksp_norm_type=petsc_ksp_norm_type,
        petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
        petsc_factor_solve_mode=petsc_factor_solve_mode,
        petsc_factor_operator_mode=petsc_factor_operator_mode,
        petsc_cond_mode=petsc_cond_mode,
        petsc_apply_measure_repeats=petsc_apply_measure_repeats,
        petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
        petsc_learning_pc_impl=petsc_learning_pc_impl,
    )
    return rows


def save_results(rows: Sequence[MethodResult], csv_file: str | Path) -> Path:
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [_raw_output_row(row) for row in rows]
    fieldnames = list(_RAW_OUTPUT_FIELD_ORDER)
    if output_rows:
        fieldnames = [
            field
            for field in _RAW_OUTPUT_FIELD_ORDER
            if any(_has_output_value(output_row.get(field)) for output_row in output_rows)
        ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for output_row in output_rows:
            writer.writerow({field: output_row.get(field) for field in fieldnames})
    return path


def summarize_method_results(
    rows: Sequence[MethodResult],
    *,
    group_fields: Sequence[str] = (
        "method",
        "model_name",
        "model_label",
        "is_learning_model",
        "diag_strategy",
    ),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[MethodResult]] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    numeric_fields = [
        "setup_time_sec",
        "apply_time_sec",
        "solve_time_sec",
        "total_time_sec",
        "iterations",
        "relative_residual",
        "true_relative_residual",
        "true_final_residual_norm",
        "ksp_monitor_initial_residual",
        "ksp_monitor_final_residual",
        "factor_nnz",
        "factor_density",
        "operator_nnz",
        "operator_density",
        "cond_est_ref",
        "cond_est_approx",
        "lambda_min_ref",
        "lambda_max_ref",
        "cond_ref_time_sec",
        "eig_cv",
        "fro_error",
        "wall_forward_time_sec",
        "steady_forward_time_sec",
        "forward_wall_time_sec",
        "forward_steady_time_sec",
        "transfer_time_sec",
        "postprocess_time_sec",
        "factor_assembly_time_sec",
        "factor_gpu_to_host_time_sec",
        "factor_petsc_build_time_sec",
        "a_petsc_build_time_sec",
        "solve_ready_build_time_sec",
        "excluded_materialization_time_sec",
        "operator_build_time_sec",
        "graph_build_time_sec",
        "inference_peak_gpu_memory_mb",
        "cpu_rss_delta_mb",
    ]

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "min": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    for group_key, items in grouped.items():
        row: dict[str, Any] = {
            "n_samples": len(items),
            "n_ok": sum(item.status == "ok" for item in items),
            "n_converged": sum(item.converged for item in items),
            "n_ksp_converged": sum(bool(item.ksp_converged) for item in items),
            "n_true_residual_converged": sum(bool(item.true_residual_converged) for item in items),
        }
        for group_field, value in zip(group_fields, group_key, strict=True):
            row[group_field] = value
        for metric_name in numeric_fields:
            values = [
                getattr(item, metric_name)
                for item in items
                if getattr(item, metric_name) is not None
            ]
            stats = _stats(values)
            row[metric_name] = stats["mean"]
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_median"] = stats["median"]
            row[f"{metric_name}_min"] = stats["min"]
            row[f"{metric_name}_max"] = stats["max"]
        summary_rows.append(row)
    return summary_rows


def build_method_summary_rows(
    rows: Sequence[MethodResult],
    *,
    include_split_name: bool = False,
    include_family: bool = False,
) -> list[dict[str, Any]]:
    learning_labels = {row.model_label for row in rows if row.is_learning_model}
    include_learning_label = len(learning_labels) > 1
    grouped: dict[tuple[Any, ...], list[MethodResult]] = {}
    for row in rows:
        method_name = _summary_method_name(
            row,
            include_learning_label=include_learning_label,
            include_split_name=include_split_name,
        )
        key_parts: list[Any] = [method_name]
        if include_family:
            key_parts.append(str(row.family or "unknown"))
        grouped.setdefault(tuple(key_parts), []).append(row)

    metric_extractors = {
        "setup_time_sec": lambda item: item.setup_time_sec,
        "apply_time_sec": lambda item: item.apply_time_sec,
        "solve_time_sec": lambda item: item.solve_time_sec,
        "total_time_sec": lambda item: item.total_time_sec,
        "iterations": lambda item: item.iterations,
        "relative_residual": lambda item: item.relative_residual,
        "factor_nnz": lambda item: item.factor_nnz,
        "factor_density": lambda item: item.factor_density,
        "operator_nnz": lambda item: item.operator_nnz,
        "operator_density": lambda item: item.operator_density,
        "cond_est_ref": lambda item: item.cond_est_ref,
        "cond_est_approx": lambda item: item.cond_est_approx,
    }

    summary_rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        method_name = str(key[0])
        family_name = str(key[1]) if include_family else None
        n_samples = len(items)
        n_converged = sum(item.converged for item in items)
        for stat in _SUMMARY_STATS:
            summary_row: dict[str, Any] = {
                "method": method_name,
                "stat": stat,
                "n_samples": n_samples,
                "n_converged": n_converged,
            }
            if include_family:
                summary_row["family"] = family_name
            for metric_name, extractor in metric_extractors.items():
                values = [value for item in items if (value := extractor(item)) is not None]
                summary_row[metric_name] = _aggregate_stat(values, stat)
            summary_rows.append(summary_row)
    return summary_rows


def _spectral_summary_by_method(rows: Sequence[MethodResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[MethodResult]] = {}
    for row in rows:
        key = row.method if not row.is_learning_model else f"{row.model_name}:{row.method}"
        grouped.setdefault(key, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for method, items in grouped.items():
        cond_ref_values = [item.cond_est_ref for item in items if item.cond_est_ref is not None]
        factor_nnz_values = [item.factor_nnz for item in items if item.factor_nnz is not None]
        factor_density_values = [
            item.factor_density for item in items if item.factor_density is not None
        ]
        operator_nnz_values = [item.operator_nnz for item in items if item.operator_nnz is not None]
        operator_density_values = [
            item.operator_density for item in items if item.operator_density is not None
        ]
        lambda_min_ref_values = [
            item.lambda_min_ref for item in items if item.lambda_min_ref is not None
        ]
        lambda_max_ref_values = [
            item.lambda_max_ref for item in items if item.lambda_max_ref is not None
        ]
        cond_ref_time_values = [
            item.cond_ref_time_sec for item in items if item.cond_ref_time_sec is not None
        ]
        cond_approx_values = [
            item.cond_est_approx for item in items if item.cond_est_approx is not None
        ]
        sigma_max_approx_values = [
            item.sigma_max_approx for item in items if item.sigma_max_approx is not None
        ]
        sigma_min_approx_values = [
            item.sigma_min_approx for item in items if item.sigma_min_approx is not None
        ]
        spectral_solve_time_values = [
            item.spectral_solve_time_sec
            for item in items
            if item.spectral_solve_time_sec is not None
        ]
        summary[method] = {
            "n_samples": len(items),
            "n_cond_ref_estimated": len(cond_ref_values),
            "n_cond_approx_estimated": len(cond_approx_values),
            "factor_nnz": None if not factor_nnz_values else float(np.mean(factor_nnz_values)),
            "factor_density": None
            if not factor_density_values
            else float(np.mean(factor_density_values)),
            "operator_nnz": None
            if not operator_nnz_values
            else float(np.mean(operator_nnz_values)),
            "operator_density": None
            if not operator_density_values
            else float(np.mean(operator_density_values)),
            "cond_est_ref": None if not cond_ref_values else float(np.mean(cond_ref_values)),
            "lambda_min_ref": None
            if not lambda_min_ref_values
            else float(np.mean(lambda_min_ref_values)),
            "lambda_max_ref": None
            if not lambda_max_ref_values
            else float(np.mean(lambda_max_ref_values)),
            "cond_ref_time_sec": None
            if not cond_ref_time_values
            else float(np.mean(cond_ref_time_values)),
            "cond_est_approx": None
            if not cond_approx_values
            else float(np.mean(cond_approx_values)),
            "sigma_max_approx": None
            if not sigma_max_approx_values
            else float(np.mean(sigma_max_approx_values)),
            "sigma_min_approx": None
            if not sigma_min_approx_values
            else float(np.mean(sigma_min_approx_values)),
            "spectral_solve_time_sec": None
            if not spectral_solve_time_values
            else float(np.mean(spectral_solve_time_values)),
        }
    return summary


def save_method_summary(
    rows: Sequence[MethodResult],
    csv_file: str | Path,
    *,
    shared_fields: dict[str, Any] | None = None,
    group_fields: Sequence[str] = (
        "method",
        "model_name",
        "model_label",
        "is_learning_model",
        "diag_strategy",
    ),
) -> Path:
    del shared_fields, group_fields
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = build_method_summary_rows(rows, include_split_name=False)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_SUMMARY_FIELD_ORDER))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in _SUMMARY_FIELD_ORDER})
    return path


def save_model_method_summary(rows: Sequence[MethodResult], csv_file: str | Path) -> Path:
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = build_method_summary_rows(rows, include_split_name=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_SUMMARY_FIELD_ORDER))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in _SUMMARY_FIELD_ORDER})
    return path


def save_family_method_summary(rows: Sequence[MethodResult], csv_file: str | Path) -> Path:
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = build_method_summary_rows(rows, include_family=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_FAMILY_SUMMARY_FIELD_ORDER))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in _FAMILY_SUMMARY_FIELD_ORDER})
    return path


def _collect_metadata(
    cfg: dict[str, Any],
    model_name: str | None,
    methods: Sequence[str],
    backend: str,
    learning_device: str,
    learning_diag_strategy: str | None,
    mask_projection_enabled: bool | None,
    *,
    split_name: str,
    manifest: dict[str, Any] | None,
    rows: Sequence[MethodResult] | None = None,
    benchmark_counts: dict[str, int] | None = None,
    petsc_amg_backend: str = "gamg",
    petsc_options: str | None = None,
    learning_models: Sequence[LearningBenchmarkModel] | None = None,
) -> dict[str, Any]:
    metadata = {
        "model_name": model_name,
        "methods": list(methods),
        "benchmark_backend": backend,
        "learning_device": learning_device,
        "learning_diag_strategy": learning_diag_strategy,
        "mask_projection_enabled": mask_projection_enabled,
        "learning_models": None
        if not learning_models
        else [
            {
                "model_name": item.model_name,
                "model_label": item.label,
                "diag_strategy": item.diag_strategy,
                "mask_percentile": item.mask_percentile,
                "mask_projection_enabled": item.use_mask_projection,
                "checkpoint_path": item.checkpoint_path,
                "model_kwargs": item.model_kwargs,
            }
            for item in learning_models
        ],
        "split_name": split_name,
        "dataset_root": None if manifest is None else manifest.get("dataset_root"),
        "split_source": None if manifest is None else manifest.get("split_source"),
        "family_counts": None
        if manifest is None
        else manifest.get("family_counts_by_split", {}).get(split_name),
        "iid_family_quotas": None if manifest is None else manifest.get("iid_family_quotas"),
        "ood_family_quotas": None if manifest is None else manifest.get("ood_family_quotas"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": getattr(torch, "__version__", None),
        "pyamg_available": True,
        "pyamg_version": sys.modules.get("pyamg").__version__
        if "pyamg" in sys.modules
        else "unknown",
        "benchmark_strategy": "method_major_burnin_then_steady_state_reuse_across_rtol",
        "timing_protocol": {
            "setup": {
                "cold_metric": "single fresh setup wall time",
                "steady_metric": "fresh setup after adaptive warmup, median of repeated builds",
                "learning_semantics": {
                    "steady_setup_boundary": "input materialize + warmed forward + GPU postprocess to factor_gpu ready",
                    "excluded_tail_columns": [
                        "factor_gpu_to_host_time_sec",
                        "factor_petsc_build_time_sec",
                        "a_petsc_build_time_sec",
                        "solve_ready_build_time_sec",
                        "excluded_materialization_time_sec",
                    ],
                },
                "policy": {
                    "min_runs": _setup_timing_policy().min_runs,
                    "max_runs": _setup_timing_policy().max_runs,
                    "window": _setup_timing_policy().window,
                    "rel_tol": _setup_timing_policy().rel_tol,
                },
                "measure_repeats": _SETUP_MEASURE_REPEATS,
            },
            "apply": {
                "cold_metric": "first apply on the active preconditioner",
                "steady_metric": "same active preconditioner after adaptive warmup",
                "policy": {
                    "min_runs": _apply_timing_policy().min_runs,
                    "max_runs": _apply_timing_policy().max_runs,
                    "window": _apply_timing_policy().window,
                    "rel_tol": _apply_timing_policy().rel_tol,
                },
                "measure_repeats": _APPLY_MEASURE_REPEATS,
            },
            "solve": {
                "steady_metric": "fresh solver state per rtol on a reused active preconditioner",
                "policy": {
                    "min_runs": _solve_timing_policy().min_runs,
                    "max_runs": _solve_timing_policy().max_runs,
                    "window": _solve_timing_policy().window,
                    "rel_tol": _solve_timing_policy().rel_tol,
                },
                "measure_repeats": _SOLVE_MEASURE_REPEATS,
                "reuse_policy": "reuse setup across rtols, rebuild solver/KSP for each rtol",
            },
            "burnin": {
                "strategy": "method-level burnin on a clone of the first sample",
                "recorded_in_results": False,
            },
        },
        "num_input_samples": None
        if benchmark_counts is None
        else benchmark_counts["num_input_samples"],
        "num_ignition_samples": None
        if benchmark_counts is None
        else benchmark_counts["num_ignition_samples"],
        "num_measured_samples": None
        if benchmark_counts is None
        else benchmark_counts["num_measured_samples"],
        "config": cfg,
    }
    if rows is not None:
        metadata["spectral_summary_by_method"] = _spectral_summary_by_method(rows)
    if backend == "petsc_gpu":
        from dpcg import petsc_benchmark

        metadata.update(
            petsc_benchmark.metadata_from_runtime(
                petsc_options=petsc_options,
                petsc_amg_backend=petsc_amg_backend,
            )
        )
    return metadata


def benchmark_model(
    model,
    dataloader,
    output_csv: str | Path | None = None,
    *,
    learning_models: Sequence[LearningBenchmarkModel] | None = None,
    residual_dir: str | Path | None = None,
    methods: Sequence[str] | None = None,
    backend: str = "scipy",
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str | None = None,
    rtol: float | Sequence[float] = 1e-5,
    maxiter: int | None = 20_000,
    ssor_omega: float = 1.2,
    ic0_diagcomp: float = 1e-1,
    ilu_drop_tol: float = 1e-4,
    ilu_fill_factor: float = 10.0,
    enable_spectral_metrics: bool = False,
    spectral_dense_limit: int = 256,
    setup_warmup_runs_gpu_learning: int = _FIXED_SETUP_WARMUP_RUNS_GPU_LEARNING,
    setup_warmup_runs_other: int = _FIXED_SETUP_WARMUP_RUNS_OTHER,
    apply_warmup_runs_cpu: int = _FIXED_APPLY_WARMUP_RUNS_CPU,
    solve_warmup_runs_cpu: int = _FIXED_SOLVE_WARMUP_RUNS_CPU,
    steady_forward_repeats: int = 50,
    benchmark_stats: dict[str, int] | None = None,
    petsc_amg_backend: str = "gamg",
    petsc_ic0_shift_type: str = "none",
    petsc_options: str | None = None,
    petsc_ksp_norm_type: str = "unpreconditioned",
    petsc_factor_solve_mode: str = "transformed_operator_native",
    petsc_factor_operator_mode: str = "explicit_aijcusparse",
    petsc_cond_mode: str = "accurate_ref",
    petsc_apply_warmup_runs: int = 0,
    petsc_apply_measure_repeats: int = 5,
    petsc_learning_internal_rtol_ratio: float = 1.0,
    petsc_learning_pc_impl: str = "shell_native",
) -> list[MethodResult]:
    backend = _parse_backend(backend)
    methods = _default_methods_for_backend(backend) if methods is None else _parse_methods(methods)
    residual_path = Path(residual_dir) if residual_dir is not None else None
    if residual_path is not None:
        residual_path.mkdir(parents=True, exist_ok=True)
    device = learning_device or "cuda"
    if learning_models is None and model is not None and "learning" in methods:
        prepared_model = _prepare_learning_model(model, device)
        learning_models = [
            LearningBenchmarkModel(
                model_name="learning",
                checkpoint_path=None,
                model_kwargs={},
                diag_strategy=learning_diag_strategy,
                label="learning",
                model=prepared_model,
                mask_percentile=None,
                use_mask_projection=use_mask_projection,
            )
        ]
    elif learning_models is not None:
        learning_models = [
            LearningBenchmarkModel(
                model_name=item.model_name,
                checkpoint_path=item.checkpoint_path,
                model_kwargs=dict(item.model_kwargs),
                diag_strategy=normalize_diag_strategy(item.diag_strategy),
                label=item.label,
                model=_prepare_learning_model(item.model, device),
                mask_percentile=item.mask_percentile,
                use_mask_projection=item.use_mask_projection,
            )
            for item in learning_models
        ]

    del setup_warmup_runs_gpu_learning
    del setup_warmup_runs_other
    del apply_warmup_runs_cpu
    del solve_warmup_runs_cpu
    del petsc_apply_warmup_runs
    samples, counts = _normalize_benchmark_samples(dataloader)
    all_rows, counts = _benchmark_samples(
        samples,
        methods=methods,
        model=model,
        learning_models=learning_models,
        backend=backend,
        learning_output_kind=learning_output_kind,
        learning_diag_strategy=learning_diag_strategy,
        use_mask_projection=use_mask_projection,
        learning_device=device,
        rtol_values=_parse_rtol_values(rtol),
        maxiter=maxiter,
        ssor_omega=ssor_omega,
        ic0_diagcomp=ic0_diagcomp,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        enable_spectral_metrics=enable_spectral_metrics,
        spectral_dense_limit=spectral_dense_limit,
        residual_dir=residual_path,
        steady_forward_repeats=steady_forward_repeats,
        petsc_amg_backend=petsc_amg_backend,
        petsc_ic0_shift_type=petsc_ic0_shift_type,
        petsc_options=petsc_options,
        petsc_ksp_norm_type=petsc_ksp_norm_type,
        petsc_sgs_ssor_impl="petsc_sor_legacy",
        petsc_factor_solve_mode=petsc_factor_solve_mode,
        petsc_factor_operator_mode=petsc_factor_operator_mode,
        petsc_cond_mode=petsc_cond_mode,
        petsc_apply_measure_repeats=petsc_apply_measure_repeats,
        petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
        petsc_learning_pc_impl=petsc_learning_pc_impl,
    )
    if benchmark_stats is not None:
        benchmark_stats.update(counts)
    if output_csv is not None:
        save_results(all_rows, output_csv)
    return all_rows


def run_benchmark(
    cfg: dict[str, Any],
    model_name: str = "sunet0",
    rtol: float | Sequence[float] | str | None = None,
):
    from dpcg.data import build_dataloaders

    if _cfg_get(cfg, "WARMUP_RUNS", None) is not None:
        raise ValueError("WARMUP_RUNS is no longer supported.")
    backend = _parse_backend(str(_cfg_get(cfg, "BENCHMARK_BACKEND", "scipy")))
    methods = _validate_methods_for_backend(
        _parse_methods(
        _cfg_get(cfg, "BENCHMARK_METHODS", _default_methods_for_backend(backend))
        ),
        backend,
    )
    learning_output_kind = str(_cfg_get(cfg, "LEARNING_OUTPUT_KIND", "sparse_factor_L"))
    learning_diag_strategy = normalize_diag_strategy(
        str(_cfg_get(cfg, "LEARNING_DIAG_STRATEGY", _cfg_get(cfg, "diag_strategy", "learned_exp")))
    )
    learning_device = str(_cfg_get(cfg, "BENCHMARK_DEVICE", "cuda"))
    petsc_options = _cfg_get(cfg, "PETSC_OPTIONS", "")
    petsc_amg_backend = str(_cfg_get(cfg, "PETSC_AMG_BACKEND", "gamg"))
    petsc_ksp_norm_type = str(_cfg_get(cfg, "PETSC_KSP_NORM_TYPE", "unpreconditioned"))
    petsc_factor_apply_mode = str(_cfg_get(cfg, "PETSC_FACTOR_APPLY_MODE", "llt"))
    petsc_factor_solve_mode = str(
        _cfg_get(cfg, "PETSC_FACTOR_SOLVE_MODE", "transformed_operator_native")
    )
    petsc_factor_operator_mode = str(
        _cfg_get(cfg, "PETSC_FACTOR_OPERATOR_MODE", "explicit_aijcusparse")
    )
    petsc_ic0_shift_type = str(_cfg_get(cfg, "PETSC_IC0_SHIFT_TYPE", "none"))
    petsc_cond_mode = str(_cfg_get(cfg, "PETSC_COND_MODE", "accurate_ref"))
    petsc_learning_internal_rtol_ratio = float(
        _cfg_get(cfg, "PETSC_LEARNING_INTERNAL_RTOL_RATIO", 1.0)
    )
    petsc_learning_pc_impl = str(_cfg_get(cfg, "PETSC_LEARNING_PC_IMPL", "shell_native"))
    petsc_apply_warmup_runs = _FIXED_PETSC_APPLY_WARMUP_RUNS
    petsc_apply_measure_repeats = int(_cfg_get(cfg, "PETSC_APPLY_MEASURE_REPEATS", 5))
    petsc_strict_gpu = bool(_cfg_get(cfg, "PETSC_STRICT_GPU", True))
    if petsc_factor_apply_mode != "llt":
        raise ValueError("PETSC_FACTOR_APPLY_MODE currently only supports 'llt'")
    if petsc_factor_solve_mode not in {
        "transformed_operator",
        "transformed_operator_native",
        "factor_pc",
    }:
        raise ValueError(
            "PETSC_FACTOR_SOLVE_MODE currently only supports "
            "'transformed_operator_native', 'transformed_operator', or 'factor_pc'"
        )
    if backend == "petsc_gpu" and not petsc_strict_gpu:
        raise ValueError("BENCHMARK_BACKEND='petsc_gpu' currently requires PETSC_STRICT_GPU=true")
    setup_warmup_runs_gpu_learning = _FIXED_SETUP_WARMUP_RUNS_GPU_LEARNING
    setup_warmup_runs_other = _FIXED_SETUP_WARMUP_RUNS_OTHER
    apply_warmup_runs_cpu = _FIXED_APPLY_WARMUP_RUNS_CPU
    solve_warmup_runs_cpu = _FIXED_SOLVE_WARMUP_RUNS_CPU
    steady_forward_repeats = int(_cfg_get(cfg, "STEADY_FORWARD_REPEATS", 50))
    num_workers = int(_cfg_get(cfg, "NUM_WORKERS", _cfg_get(cfg, "num_workers", 0)))
    pin_memory = bool(_cfg_get(cfg, "PIN_MEMORY", _cfg_get(cfg, "pin_memory", False)))
    persistent_workers = bool(
        _cfg_get(cfg, "PERSISTENT_WORKERS", _cfg_get(cfg, "persistent_workers", False))
    )
    prefetch_factor = _cfg_get(cfg, "PREFETCH_FACTOR", _cfg_get(cfg, "prefetch_factor", None))
    prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else None
    cache_samples = bool(_cfg_get(cfg, "CACHE_SAMPLES", _cfg_get(cfg, "cache_samples", True)))
    cache_graph = bool(_cfg_get(cfg, "CACHE_GRAPH", _cfg_get(cfg, "cache_graph", True)))
    mask_percentile = float(_cfg_get(cfg, "MASK_PERCENTILE", 50.0))
    use_mask_projection = _mask_projection_enabled_from_percentile(mask_percentile)
    results_prefix = Path(
        _cfg_get(
            cfg, "RESULTS_CSV", _cfg_get(cfg, "TEST_FILE_M", "./results/benchmark_results.csv")
        )
        or "./results/benchmark_results.csv"
    )
    if results_prefix.suffix.lower() == ".csv":
        results_prefix = results_prefix.with_suffix("")
    results_prefix.parent.mkdir(parents=True, exist_ok=True)
    residual_dir = _cfg_get(cfg, "RESIDUAL_DIR", None)
    requested_rtol = _resolve_requested_rtol(cfg, rtol)
    rtol_values = _parse_rtol_values(requested_rtol)
    multi_rtol = len(rtol_values) > 1

    needs_learning = "learning" in methods
    model = None
    learning_models: list[LearningBenchmarkModel] | None = None
    if needs_learning:
        if torch is None:
            raise RuntimeError("torch is required for the learning benchmark path")
        learning_models = _load_learning_model_specs(
            cfg,
            cli_model_name=model_name,
            learning_device=learning_device,
            global_diag_strategy=learning_diag_strategy,
            global_mask_percentile=mask_percentile,
        )
        if _cfg_get(cfg, "LEARNING_MODELS", None) is None:
            model = learning_models[0].model

    loaders = build_dataloaders(
        data_root=_cfg_get(cfg, "DATA_ROOT", "./data"),
        n_train=int(_cfg_get(cfg, "PC_TRAIN", 0)),
        n_val=int(_cfg_get(cfg, "PC_VAL", 0)),
        n_test=int(_cfg_get(cfg, "PC_TEST", 1)),
        seed=int(_cfg_get(cfg, "SEED", 42)),
        mask_percentile=mask_percentile,
        split_source=str(_cfg_get(cfg, "DATASET_SPLIT_SOURCE", "meta_json")),
        return_manifest=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        cache_samples=cache_samples,
        cache_graph=cache_graph,
        train_max_samples_per_split=_cfg_get(cfg, "TRAIN_MAX_SAMPLES_PER_SPLIT", None),
        val_max_samples_per_split=_cfg_get(cfg, "VAL_MAX_SAMPLES_PER_SPLIT", None),
        iid_max_samples_per_split=_cfg_get(cfg, "IID_MAX_SAMPLES_PER_SPLIT", None),
        ood_max_samples_per_split=_cfg_get(cfg, "OOD_MAX_SAMPLES_PER_SPLIT", None),
        iid_family_quotas=_cfg_get(cfg, "IID_FAMILY_QUOTAS", None),
        ood_family_quotas=_cfg_get(cfg, "OOD_FAMILY_QUOTAS", None),
    )
    _train_loader, _val_loader, iid_loader, ood_loader, manifest = loaders
    all_rows: list[MethodResult] = []
    all_maxiters = {
        _normalize_rtol_label(float(item)): _resolve_benchmark_maxiter(cfg, float(item))
        for item in rtol_values
    }
    benchmark_maxiter: Any
    if len(all_maxiters) == 1:
        benchmark_maxiter = int(next(iter(all_maxiters.values())))
    else:
        benchmark_maxiter = dict(all_maxiters)
    split_rows_by_name: dict[str, list[MethodResult]] = {}
    split_counts_by_name: dict[str, dict[str, int]] = {}
    for split_name, dataloader in (("iid", iid_loader), ("ood", ood_loader)):
        split_counts: dict[str, int] = {}
        split_residual_root = (
            Path(residual_dir) / split_name
            if residual_dir
            else results_prefix.parent / f"{results_prefix.name}_{split_name}_residuals"
        )
        split_residual_root.mkdir(parents=True, exist_ok=True)
        rows = benchmark_model(
            model,
            dataloader,
            learning_models=learning_models,
            output_csv=None,
            residual_dir=split_residual_root,
            methods=methods,
            backend=backend,
            learning_output_kind=learning_output_kind,
            learning_diag_strategy=learning_diag_strategy,
            use_mask_projection=use_mask_projection,
            learning_device=learning_device,
            rtol=rtol_values,
            maxiter=benchmark_maxiter,
            ssor_omega=float(_cfg_get(cfg, "SSOR_OMEGA", 1.2)),
            ic0_diagcomp=float(_cfg_get(cfg, "IC0_DIAGCOMP", 1e-1)),
            ilu_drop_tol=float(_cfg_get(cfg, "ILU_DROP_TOL", 1e-4)),
            ilu_fill_factor=float(_cfg_get(cfg, "ILU_FILL_FACTOR", 10.0)),
            enable_spectral_metrics=bool(_cfg_get(cfg, "ENABLE_SPECTRAL_METRICS", False)),
            spectral_dense_limit=int(_cfg_get(cfg, "SPECTRAL_DENSE_LIMIT", 256)),
            setup_warmup_runs_gpu_learning=setup_warmup_runs_gpu_learning,
            setup_warmup_runs_other=setup_warmup_runs_other,
            apply_warmup_runs_cpu=apply_warmup_runs_cpu,
            solve_warmup_runs_cpu=solve_warmup_runs_cpu,
            steady_forward_repeats=steady_forward_repeats,
            benchmark_stats=split_counts,
            petsc_amg_backend=petsc_amg_backend,
            petsc_ic0_shift_type=petsc_ic0_shift_type,
            petsc_options=petsc_options,
            petsc_ksp_norm_type=petsc_ksp_norm_type,
            petsc_factor_solve_mode=petsc_factor_solve_mode,
            petsc_factor_operator_mode=petsc_factor_operator_mode,
            petsc_cond_mode=petsc_cond_mode,
            petsc_apply_warmup_runs=petsc_apply_warmup_runs,
            petsc_apply_measure_repeats=petsc_apply_measure_repeats,
            petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
            petsc_learning_pc_impl=petsc_learning_pc_impl,
        )
        split_rows_by_name[split_name] = rows
        split_counts_by_name[split_name] = split_counts
        all_rows.extend(rows)

    for rtol_value in rtol_values:
        rtol_label = _normalize_rtol_label(rtol_value)
        prefix = results_prefix
        if multi_rtol:
            prefix = _with_stem_suffix(
                results_prefix,
                f"rtol_{_safe_rtol_label(rtol_label)}",
            )
        rtol_rows: list[MethodResult] = []
        for split_name, rows in split_rows_by_name.items():
            filtered_rows = [
                row for row in rows if row.rtol is not None and float(row.rtol) == float(rtol_value)
            ]
            split_csv = prefix.parent / f"{prefix.name}_{split_name}.csv"
            split_summary_csv = split_csv.parent / f"{split_csv.stem}_method_summary.csv"
            split_family_summary_csv = split_csv.parent / f"{split_csv.stem}_family_method_summary.csv"
            split_metadata_path = split_csv.parent / f"{split_csv.stem}_metadata.json"
            save_results(filtered_rows, split_csv)
            save_method_summary(
                filtered_rows,
                split_summary_csv,
                shared_fields={
                    "num_input_samples": split_counts_by_name[split_name].get("num_input_samples"),
                    "num_ignition_samples": split_counts_by_name[split_name].get(
                        "num_ignition_samples"
                    ),
                    "num_measured_samples": split_counts_by_name[split_name].get(
                        "num_measured_samples"
                    ),
                },
            )
            save_family_method_summary(filtered_rows, split_family_summary_csv)
            split_residual_dir = (
                Path(residual_dir) / split_name / f"rtol_{_safe_rtol_label(rtol_label)}"
                if residual_dir
                else (
                    results_prefix.parent
                    / f"{results_prefix.name}_{split_name}_residuals"
                    / f"rtol_{_safe_rtol_label(rtol_label)}"
                )
            )
            metadata_mask_projection_enabled: bool | None = use_mask_projection
            if learning_models and len(learning_models) > 1:
                projection_flags = {
                    bool(
                        use_mask_projection
                        if item.use_mask_projection is None
                        else item.use_mask_projection
                    )
                    for item in learning_models
                }
                metadata_mask_projection_enabled = (
                    next(iter(projection_flags)) if len(projection_flags) == 1 else None
                )
            metadata = _collect_metadata(
                cfg,
                None if learning_models and len(learning_models) > 1 else model_name,
                methods,
                backend,
                learning_device,
                None if learning_models and len(learning_models) > 1 else learning_diag_strategy,
                metadata_mask_projection_enabled,
                split_name=split_name,
                manifest=manifest,
                rows=filtered_rows,
                benchmark_counts=split_counts_by_name[split_name],
                petsc_amg_backend=petsc_amg_backend,
                petsc_options=petsc_options,
                learning_models=learning_models,
            )
            metadata["rtol"] = float(rtol_value)
            metadata["rtol_label"] = rtol_label
            metadata["rtols"] = [float(item) for item in rtol_values]
            metadata["benchmark_maxiter"] = int(
                all_maxiters[_normalize_rtol_label(float(rtol_value))]
            )
            metadata["benchmark_maxiter_by_rtol"] = {
                str(key): int(value) for key, value in all_maxiters.items()
            }
            metadata["num_results"] = len(filtered_rows)
            metadata["results_csv"] = str(split_csv)
            metadata["method_summary_csv"] = str(split_summary_csv)
            metadata["family_method_summary_csv"] = str(split_family_summary_csv)
            metadata["residual_dir"] = str(split_residual_dir)
            split_metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rtol_rows.extend(filtered_rows)
        model_method_summary_csv = prefix.parent / f"{prefix.name}_model_method_summary.csv"
        save_model_method_summary(rtol_rows, model_method_summary_csv)
    return all_rows


def main() -> None:
    from dpcg.train import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark/benchmark_petsc_gpu_sunet_vs_amg.yaml",
    )
    parser.add_argument("--model", type=str, default="sunet0")
    parser.add_argument("--rtol", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_benchmark(cfg, model_name=args.model, rtol=args.rtol)


run_gpu_benchmark = run_benchmark


if __name__ == "__main__":
    main()
