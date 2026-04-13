"""Abaqus parsers and dataset builders."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg, eigsh, spsolve

from dpcg.io.abaqus_dat import (
    detect_beam_elements,
    read_dat_elements,
    read_dat_info,
    read_dat_nodes,
)
from dpcg.io.npz import save_npz_sample
from dpcg.sample import BenchmarkSample


class _NodeMapLike(Protocol):
    node_id_to_index: np.ndarray


class _DofMapLike(Protocol):
    dof_key_base: int
    dof_keys: np.ndarray


@dataclass(frozen=True)
class NodeMap:
    """Map Abaqus node ids to compact node indices."""

    node_ids: np.ndarray
    node_id_to_index: np.ndarray


@dataclass(frozen=True)
class DofMap:
    """Map ``(node_index, dof_label)`` pairs to compact DOF indices."""

    dof_key_base: int
    dof_keys: np.ndarray


@dataclass
class AbaqusBuildResult:
    """Result bundle for Abaqus dataset conversion."""

    sample_paths: list[Path]
    manifest_path: Path | None
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class CgSolveStats:
    """Detailed SciPy CG solve result."""

    x: np.ndarray
    info: int
    iterations: int
    residual_history: np.ndarray
    true_residual_norm: float
    true_relative_residual: float
    rtol: float
    atol: float
    maxiter: int | None


class SpdValidationError(RuntimeError):
    """Raised when SPD validation fails due to resource or numerical issues."""


__all__ = [
    "AbaqusBuildResult",
    "CgSolveStats",
    "DofMap",
    "NodeMap",
    "SpdValidationError",
    "build_free_system_from_mtx",
    "build_load_vector_from_cload",
    "convert_abaqus_directory",
    "convert_abaqus_to_npz",
    "detect_free_dofs_by_penalty",
    "estimate_condition_number",
    "is_spd_matrix",
    "load_abaqus_system",
    "read_cload_mtx",
    "read_dat_elements",
    "read_dat_info",
    "read_dat_nodes",
    "read_mtx_5col",
    "solve_system_cg_with_stats",
]


def _sample_sidecar_payload(sample) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "coords_by_node_id",
        "node_ids",
        "element_ids",
        "element_types",
        "element_indptr",
        "element_nodes",
        "instance_names",
        "instance_node_ptr",
        "instance_node_ids",
        "free_dof_node_ids",
        "free_dof_labels",
        "free_dof_indices",
        "encastre_node_ids",
        "tie_slave_ids",
        "tie_master_ptr",
        "tie_master_idx",
    ):
        value = getattr(sample, key, None)
        if value is not None:
            payload[key] = value
    return payload


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return min(8, max(1, os.cpu_count() or 1))
    resolved = int(num_workers)
    if resolved < 1:
        raise ValueError("num_workers must be >= 1")
    return resolved


class _CgIterationCounter:
    def __init__(self, A: csr_matrix, b: np.ndarray):
        self.A = A
        self.b = np.asarray(b, dtype=np.float64).reshape(-1)
        self.residuals: list[float] = []

    def __call__(self, xk: np.ndarray) -> None:
        x = np.asarray(xk, dtype=np.float64).reshape(-1)
        residual = self.b - self.A @ x
        self.residuals.append(float(np.linalg.norm(residual)))


def solve_system_cg_with_stats(
    A: csr_matrix,
    b: np.ndarray,
    *,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> CgSolveStats:
    """Solve a system with SciPy CG and collect convergence diagnostics."""
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    counter = _CgIterationCounter(A, rhs)
    try:
        x, info = cg(A, rhs, rtol=float(rtol), atol=float(atol), maxiter=maxiter, callback=counter)
    except TypeError:  # pragma: no cover - SciPy compatibility
        x, info = cg(A, rhs, tol=float(rtol), maxiter=maxiter, callback=counter)
    solution = np.asarray(x, dtype=np.float64).reshape(-1)
    residual = rhs - A @ solution
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs))
    residual_history = np.asarray(counter.residuals, dtype=np.float64)
    if residual_history.size == 0:
        residual_history = np.asarray([residual_norm], dtype=np.float64)
    return CgSolveStats(
        x=solution,
        info=int(info),
        iterations=int(residual_history.shape[0]),
        residual_history=residual_history,
        true_residual_norm=residual_norm,
        true_relative_residual=residual_norm / (rhs_norm if rhs_norm > 0.0 else 1.0),
        rtol=float(rtol),
        atol=float(atol),
        maxiter=None if maxiter is None else int(maxiter),
    )


def _convert_abaqus_directory_worker(
    task: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    index, payload = task
    result = convert_abaqus_to_npz(
        payload["stiffness_path"],
        payload["output_root"],
        load_mtx=payload["load_mtx"],
        dat_path=payload["dat_path"],
        rhs_mode=payload["rhs_mode"],
        num_samples=payload["num_samples"],
        solve_mode=payload["solve_mode"],
        seed=payload["seed"],
        scaled_load_range=payload["scaled_load_range"],
        synthetic_kind=payload["synthetic_kind"],
        synthetic_scale=payload["synthetic_scale"],
        require_spd=payload["require_spd"],
        strict_load=payload["strict_load"],
        compress_npz=payload["compress_npz"],
        write_per_case_manifest=payload["write_per_case_manifest"],
    )
    return (
        index,
        {
            "sample_paths": [str(path) for path in result.sample_paths],
            "manifest_path": None if result.manifest_path is None else str(result.manifest_path),
            "records": result.records,
        },
    )


def read_mtx_5col(path: str, chunksize: int = 2_000_000) -> Tuple[np.ndarray, ...]:
    """Read Abaqus 5-column stiffness ``.mtx``."""
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None:
        rn_list, rd_list, cn_list, cd_list, val_list = [], [], [], [], []
        for df in pd.read_csv(
            path,
            header=None,
            comment="*",
            sep=",",
            engine="c",
            usecols=[0, 1, 2, 3, 4],
            dtype={0: np.int32, 1: np.int16, 2: np.int32, 3: np.int16, 4: np.float64},
            chunksize=chunksize,
        ):
            df = df.dropna()
            arr = df.to_numpy(copy=False)
            rn_list.append(arr[:, 0].astype(np.int32, copy=False))
            rd_list.append(arr[:, 1].astype(np.int16, copy=False))
            cn_list.append(arr[:, 2].astype(np.int32, copy=False))
            cd_list.append(arr[:, 3].astype(np.int16, copy=False))
            val_list.append(arr[:, 4].astype(np.float64, copy=False))
        return (
            np.concatenate(rn_list),
            np.concatenate(rd_list),
            np.concatenate(cn_list),
            np.concatenate(cd_list),
            np.concatenate(val_list),
        )

    rnL, rdL, cnL, cdL, valL = [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("*"):
                continue
            arr = np.fromstring(s, sep=",")
            if arr.size < 5:
                continue
            rni, rdi, cni, cdi, v = arr[:5]
            rnL.append(int(rni))
            rdL.append(int(rdi))
            cnL.append(int(cni))
            cdL.append(int(cdi))
            valL.append(float(v))
    return (
        np.asarray(rnL, np.int32),
        np.asarray(rdL, np.int16),
        np.asarray(cnL, np.int32),
        np.asarray(cdL, np.int16),
        np.asarray(valL, np.float64),
    )


def read_cload_mtx(
    path: str, chunksize: int = 1_000_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Abaqus ``*CLOAD``/``LOAD`` matrix with columns ``(node, dof, value)``."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        n_list, d_list, v_list = [], [], []
        for df in pd.read_csv(
            path,
            header=None,
            comment="*",
            sep=",",
            engine="c",
            usecols=[0, 1, 2],
            dtype={0: np.float64, 1: np.float64, 2: np.float64},
            chunksize=chunksize,
        ):
            df = df.dropna()
            arr = df.to_numpy(copy=False)
            n_list.append(arr[:, 0].astype(np.int32, copy=False))
            d_list.append(arr[:, 1].astype(np.int16, copy=False))
            v_list.append(arr[:, 2].astype(np.float64, copy=False))
        if not n_list:
            raise RuntimeError(f"No CLOAD entries found in {path}")
        return (
            np.concatenate(n_list),
            np.concatenate(d_list),
            np.concatenate(v_list),
        )

    nL, dL, vL = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("*"):
                continue
            arr = np.fromstring(s, sep=",")
            if arr.size < 3:
                continue
            nL.append(int(arr[0]))
            dL.append(int(arr[1]))
            vL.append(float(arr[2]))
    if not nL:
        raise RuntimeError(f"No CLOAD entries found in {path}")
    return (
        np.asarray(nL, np.int32),
        np.asarray(dL, np.int16),
        np.asarray(vL, np.float64),
    )


def build_load_vector_from_cload(
    load_path: str,
    node_map: _NodeMapLike,
    dof_map: _DofMapLike,
    free_idx: np.ndarray,
    strict: bool = True,
) -> np.ndarray:
    """Build the compact free-DOF load vector from Abaqus ``LOAD/CLOAD``."""
    node_ids, dof_ids, vals = read_cload_mtx(load_path)
    in_bounds = (node_ids >= 0) & (node_ids < node_map.node_id_to_index.size)
    if not np.all(in_bounds):
        invalid_node = int(node_ids[~in_bounds][0])
        raise RuntimeError(f"CLOAD node id {invalid_node} exceeds stiffness node map")
    node_index = node_map.node_id_to_index[node_ids.astype(np.int64)]
    if np.any(node_index < 0):
        bad_node = int(node_ids[node_index < 0][0])
        raise RuntimeError(f"CLOAD node id {bad_node} is missing from stiffness.mtx")
    base = int(dof_map.dof_key_base)
    dof_keys = dof_map.dof_keys
    keys = node_index.astype(np.int64) * base + dof_ids.astype(np.int64)
    idx = np.searchsorted(dof_keys, keys)
    valid = (idx >= 0) & (idx < dof_keys.size) & (dof_keys[idx] == keys)
    if not np.all(valid):
        bad_pos = int(np.flatnonzero(~valid)[0])
        bad_node = int(node_ids[bad_pos])
        bad_dof = int(dof_ids[bad_pos])
        if strict:
            raise RuntimeError(f"CLOAD (node={bad_node}, dof={bad_dof}) not found in stiffness.mtx")
        idx = idx[valid]
        vals = vals[valid]
    b_full = np.zeros(dof_keys.size, dtype=np.float64)
    np.add.at(b_full, idx.astype(np.int64), vals)
    return b_full[free_idx]


def detect_free_dofs_by_penalty(
    K: csr_matrix,
    diag_tol: float = 0.0,
    penalty_abs_threshold: float = 1e25,
    penalty_scale_factor: float = 1e8,
    penalty_diag_dom_ratio: float = 1e12,
    offdiag_floor: float = 1e-30,
) -> np.ndarray:
    """Detect unconstrained DOFs from penalty-like diagonal patterns."""
    diag = K.diagonal()
    absdiag = np.abs(diag)
    absdata = np.abs(K.data)
    row_sum = np.add.reduceat(absdata, K.indptr[:-1])
    empty = K.indptr[:-1] == K.indptr[1:]
    row_sum[empty] = 0.0
    offdiag = np.maximum(row_sum - absdiag, 0.0)
    nz = absdiag[absdiag > 0]
    typ = float(np.median(nz)) if nz.size else 1.0
    penalty_thr = max(penalty_abs_threshold, typ * penalty_scale_factor)
    diag_dom = absdiag / np.maximum(offdiag, offdiag_floor)
    constrained = (absdiag >= penalty_thr) & (diag_dom >= penalty_diag_dom_ratio)
    free_mask = (~constrained) & (absdiag > diag_tol)
    return np.where(free_mask)[0].astype(np.int32)


def _build_node_map_from_mtx(rn: np.ndarray, cn: np.ndarray) -> NodeMap:
    node_ids = np.unique(np.concatenate([rn, cn]).astype(np.int64))
    if node_ids.size == 0:
        raise RuntimeError("no node ids found in stiffness.mtx")
    max_id = int(node_ids.max())
    node_id_to_index = np.full(max_id + 1, -1, dtype=np.int64)
    node_id_to_index[node_ids] = np.arange(node_ids.size, dtype=np.int64)
    return NodeMap(node_ids=node_ids, node_id_to_index=node_id_to_index)


def _build_dof_map(
    node_r: np.ndarray, rd: np.ndarray, node_c: np.ndarray, cd: np.ndarray
) -> DofMap:
    max_dof = int(max(int(rd.max()), int(cd.max())))
    base = int(max_dof + 1)
    if base <= 1:
        raise RuntimeError("invalid DOF label range")
    row_key = node_r.astype(np.int64) * base + rd.astype(np.int64)
    col_key = node_c.astype(np.int64) * base + cd.astype(np.int64)
    dof_keys = np.unique(np.concatenate([row_key, col_key]).astype(np.int64))
    return DofMap(dof_key_base=base, dof_keys=dof_keys)


def _map_keys_to_indices(keys: np.ndarray, dof_keys: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(dof_keys, keys)
    if np.any((idx < 0) | (idx >= dof_keys.size)) or np.any(dof_keys[idx] != keys):
        raise RuntimeError("DOF key mapping failed")
    return idx.astype(np.int64, copy=False)


def _detect_sparse_symmetry_mode(
    row: np.ndarray,
    col: np.ndarray,
    n_dof_full: int,
    *,
    paired_fraction_full_threshold: float = 0.95,
    paired_fraction_triangular_threshold: float = 0.05,
) -> tuple[str, dict[str, float | int]]:
    """Detect whether sparse Abaqus entries encode one triangle or a full near-symmetric matrix."""
    offdiag = row != col
    if not np.any(offdiag):
        stats = {
            "offdiag_unique_nnz": 0,
            "paired_offdiag_unique_nnz": 0,
            "paired_fraction": 1.0,
            "upper_only_unique_nnz": 0,
            "lower_only_unique_nnz": 0,
        }
        return "full_near_symmetric", stats

    key = row[offdiag].astype(np.int64) * np.int64(n_dof_full) + col[offdiag].astype(np.int64)
    key = np.unique(key)
    row_u = key // np.int64(n_dof_full)
    col_u = key % np.int64(n_dof_full)
    transpose_key = col_u * np.int64(n_dof_full) + row_u
    idx = np.searchsorted(key, transpose_key)
    has_pair = (idx < key.size) & (key[idx] == transpose_key)

    upper_only = int(np.count_nonzero((row_u < col_u) & ~has_pair))
    lower_only = int(np.count_nonzero((row_u > col_u) & ~has_pair))
    paired_count = int(np.count_nonzero(has_pair))
    paired_fraction = float(paired_count) / float(key.size)
    stats = {
        "offdiag_unique_nnz": int(key.size),
        "paired_offdiag_unique_nnz": paired_count,
        "paired_fraction": paired_fraction,
        "upper_only_unique_nnz": upper_only,
        "lower_only_unique_nnz": lower_only,
    }

    if paired_fraction >= float(paired_fraction_full_threshold):
        return "full_near_symmetric", stats
    if paired_fraction <= float(paired_fraction_triangular_threshold) and (
        upper_only == 0 or lower_only == 0
    ):
        return "triangular", stats
    raise ValueError(
        "Ambiguous Abaqus stiffness sparsity pattern: "
        f"paired_fraction={paired_fraction:.6f}, "
        f"upper_only={upper_only}, lower_only={lower_only}"
    )


def _assemble_symmetric_stiffness(
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray,
    n_dof_full: int,
) -> tuple[csr_matrix, str, dict[str, float | int]]:
    """Assemble a symmetric stiffness matrix from Abaqus sparse entries."""
    assembly_mode, pattern_stats = _detect_sparse_symmetry_mode(row, col, n_dof_full)
    K_raw = coo_matrix((val, (row, col)), shape=(n_dof_full, n_dof_full)).tocsr()
    K_raw.sum_duplicates()
    if assembly_mode == "triangular":
        K = (K_raw + K_raw.transpose().tocsr() - diags(K_raw.diagonal())).tocsr()
    elif assembly_mode == "full_near_symmetric":
        K = ((K_raw + K_raw.transpose().tocsr()) * 0.5).tocsr()
    else:
        raise ValueError(f"Unsupported assembly mode: {assembly_mode}")
    K.sum_duplicates()
    return K, assembly_mode, pattern_stats


def build_free_system_from_mtx(
    rn,
    rd,
    cn,
    cd,
    val,
    free_dof_filter=detect_free_dofs_by_penalty,
    diag_tol: float = 0.0,
) -> tuple[
    csr_matrix,
    np.ndarray,
    np.ndarray,
    int,
    NodeMap,
    DofMap,
    np.ndarray,
    str,
    dict[str, float | int],
]:
    """Build the compact free-DOF stiffness matrix from Abaqus 5-column MTX."""
    node_map = _build_node_map_from_mtx(rn, cn)
    node_r = node_map.node_id_to_index[rn.astype(np.int64)]
    node_c = node_map.node_id_to_index[cn.astype(np.int64)]
    if (node_r < 0).any() or (node_c < 0).any():
        raise RuntimeError("mtx node id mapping failed")
    n_nodes = int(node_map.node_ids.size)

    dof_map = _build_dof_map(node_r, rd, node_c, cd)
    row_key = node_r.astype(np.int64) * dof_map.dof_key_base + rd.astype(np.int64)
    col_key = node_c.astype(np.int64) * dof_map.dof_key_base + cd.astype(np.int64)
    row = _map_keys_to_indices(row_key, dof_map.dof_keys)
    col = _map_keys_to_indices(col_key, dof_map.dof_keys)
    n_dof_full = int(dof_map.dof_keys.size)

    K, assembly_mode, pattern_stats = _assemble_symmetric_stiffness(row, col, val, n_dof_full)

    free_idx = free_dof_filter(K, diag_tol=diag_tol)
    Kf = K[free_idx, :][:, free_idx].tocsr()
    dof_key = dof_map.dof_keys[free_idx.astype(np.int64, copy=False)]
    dof2node = (dof_key // np.int64(dof_map.dof_key_base)).astype(np.int32)
    dof2abq = (dof_key % np.int64(dof_map.dof_key_base)).astype(np.int16)
    return Kf, dof2node, dof2abq, n_nodes, node_map, dof_map, free_idx, assembly_mode, pattern_stats


def is_spd_matrix(
    A: csr_matrix,
    sym_tol: float = 1e-8,
    diag_tol: float = 0.0,
    eig_tol: float = 1e-12,
    eigsh_tol: float = 1e-6,
    eigsh_maxiter: int = 10_000,
) -> bool:
    """Check if a sparse matrix is SPD without densifying it."""
    if A.shape[0] != A.shape[1]:
        return False
    if sym_tol > 0:
        diff = A - A.transpose()
        if diff.nnz > 0:
            maxv = np.max(np.abs(diff.data)) if diff.data.size else 0.0
            if maxv > float(sym_tol):
                return False
    d = A.diagonal()
    if np.any(d <= float(diag_tol)) or np.any(~np.isfinite(d)):
        return False
    try:
        lambda_min = float(
            eigsh(
                A,
                k=1,
                sigma=0.0,
                which="LM",
                tol=eigsh_tol,
                maxiter=eigsh_maxiter,
                return_eigenvectors=False,
            )[0]
        )
    except MemoryError as exc:
        raise SpdValidationError(f"resource failure during SPD validation: {exc}") from exc
    except Exception as exc:
        raise SpdValidationError(f"numerical failure during SPD validation: {exc}") from exc
    if not np.isfinite(lambda_min):
        return False
    return lambda_min > float(max(diag_tol, eig_tol))


def estimate_condition_number(
    A: csr_matrix,
    *,
    tol: float = 1e-8,
    maxiter: int = 20_000,
) -> dict[str, Any]:
    """Estimate ``lambda_max / lambda_min`` for a sparse SPD matrix."""
    started = perf_counter()
    result: dict[str, Any] = {
        "condition_number_est": None,
        "condition_number_status": "failed",
        "condition_number_backend": "scipy.eigsh_sigma0",
        "condition_number_tol": float(tol),
        "condition_number_maxiter": int(maxiter),
        "condition_number_error": None,
        "lambda_min_est": None,
        "lambda_max_est": None,
        "condition_number_time_sec": None,
    }
    eigsh_error: str | None = None
    try:
        lambda_max = float(
            eigsh(A, k=1, which="LA", tol=tol, maxiter=maxiter, return_eigenvectors=False)[0]
        )
        lambda_min = float(
            eigsh(
                A,
                k=1,
                sigma=0.0,
                which="LM",
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=False,
            )[0]
        )
    except Exception as exc:
        eigsh_error = str(exc)
        try:
            condition_number = float(np.linalg.cond(A.toarray()))
        except Exception as cond_exc:
            result["condition_number_error"] = (
                "condition-number estimation failed with "
                "scipy.eigsh(which='LA') + scipy.eigsh(sigma=0, which='LM'): "
                f"{eigsh_error}; fallback numpy.linalg.cond failed: {cond_exc}"
            )
            result["condition_number_time_sec"] = perf_counter() - started
            return result
        result["condition_number_backend"] = "numpy.linalg.cond"
        result["condition_number_est"] = condition_number
        result["condition_number_time_sec"] = perf_counter() - started
        if not np.isfinite(condition_number):
            result["condition_number_error"] = "non-finite condition number estimate"
            return result
        if condition_number <= 0.0:
            result["condition_number_error"] = "condition_number <= 0"
            return result
        result["condition_number_status"] = "ok"
        return result

    result["lambda_min_est"] = lambda_min
    result["lambda_max_est"] = lambda_max
    result["condition_number_time_sec"] = perf_counter() - started
    if not np.isfinite(lambda_min) or not np.isfinite(lambda_max):
        result["condition_number_error"] = "non-finite eigenvalue estimate"
        return result
    if lambda_min <= 0.0:
        result["condition_number_error"] = "lambda_min <= 0"
        return result

    result["condition_number_est"] = float(lambda_max / lambda_min)
    result["condition_number_status"] = "ok"
    return result


def _validate_matrix(A: csr_matrix) -> None:
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    if A.nnz == 0:
        raise ValueError("matrix must not be empty")
    if np.any(~np.isfinite(A.data)):
        raise ValueError("matrix contains NaN/Inf")
    diag = A.diagonal()
    if np.any(~np.isfinite(diag)):
        raise ValueError("matrix diagonal contains NaN/Inf")


def _default_sample_name(stiffness_mtx: str | Path) -> str:
    stem = Path(stiffness_mtx).stem
    return re.sub(r"[^0-9A-Za-z._-]+", "_", stem)


def _resolve_dat_path(stiffness_path: Path, dat_path: str | Path | None) -> Path | None:
    if dat_path is not None:
        path = Path(dat_path)
        return path if path.exists() else None
    stem = stiffness_path.stem
    base = stem.split("_STIF")[0]
    candidates = [stiffness_path.with_name(f"{base}.dat")]
    stripped = re.sub(r"[-_]\d+$", "", base)
    if stripped != base:
        candidates.append(stiffness_path.with_name(f"{stripped}.dat"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_load_path(stiffness_path: Path, load_path: str | Path | None) -> Path | None:
    if load_path is not None:
        path = Path(load_path)
        return path if path.exists() else None
    stem = stiffness_path.stem
    if "_STIF" not in stem:
        return None
    prefix, suffix = stem.split("_STIF", maxsplit=1)
    candidate = stiffness_path.with_name(f"{prefix}_LOAD{suffix}{stiffness_path.suffix}")
    return candidate if candidate.exists() else None


def _solve_system_direct(A: csr_matrix, b: np.ndarray, mode: str) -> np.ndarray:
    mode_key = str(mode).strip().lower()
    if mode_key == "dense":
        return np.asarray(np.linalg.solve(A.toarray(), b), dtype=np.float64)
    if mode_key == "spsolve":
        return np.asarray(spsolve(A, b), dtype=np.float64)
    raise ValueError(f"Unsupported direct solve mode: {mode}")


def _solve_system(A: csr_matrix, b: np.ndarray, solve_mode: str) -> tuple[np.ndarray, str]:
    mode = str(solve_mode).strip().lower()
    if mode == "zero_placeholder":
        return np.zeros(A.shape[0], dtype=np.float64), "zero_placeholder"
    if mode == "dense":
        return _solve_system_direct(A, b, "dense"), "dense"
    if mode == "spsolve":
        return _solve_system_direct(A, b, "spsolve"), "spsolve"
    if mode == "cg":
        result = solve_system_cg_with_stats(
            A,
            b,
            rtol=1.0e-10,
            atol=0.0,
            maxiter=max(10 * A.shape[0], 200),
        )
        if result.info != 0:
            raise RuntimeError(f"CG failed to solve system, info={result.info}")
        return result.x, "cg"
    if mode == "cg_fallback_spsolve":
        try:
            return _solve_system(A, b, "cg")
        except RuntimeError:
            return _solve_system(A, b, "spsolve")
    if mode == "cg_fallback_dense":
        try:
            return _solve_system(A, b, "cg")
        except RuntimeError:
            return _solve_system(A, b, "dense")
    raise ValueError(f"Unsupported solve_mode: {solve_mode}")


def _single_entry_rhs(size: int, entry_index: int, scale: float) -> np.ndarray:
    rhs = np.zeros(size, dtype=np.float64)
    rhs[int(np.clip(entry_index, 0, size - 1))] = float(scale)
    return rhs


def _sine_rhs(size: int, frequency: float, amplitude: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size)
    return amplitude * np.sin(2.0 * np.pi * frequency * x)


def _synthetic_rhs(
    size: int,
    *,
    rng: np.random.Generator,
    synthetic_kind: str,
    synthetic_scale: float,
) -> np.ndarray:
    kind = str(synthetic_kind).strip().lower()
    if kind == "random":
        return rng.standard_normal(size).astype(np.float64) * float(synthetic_scale)
    if kind == "sine":
        frequency = max(1.0, float(rng.integers(1, 4)))
        amplitude = float(rng.uniform(0.5, 1.5)) * float(synthetic_scale)
        return _sine_rhs(size, frequency=frequency, amplitude=amplitude)
    if kind == "single_entry":
        entry_index = int(rng.integers(0, size))
        amplitude = float(rng.uniform(0.5, 1.5)) * float(synthetic_scale)
        return _single_entry_rhs(size, entry_index=entry_index, scale=amplitude)
    raise ValueError(f"Unsupported synthetic RHS kind: {synthetic_kind}")


def _build_rhs_vectors(
    *,
    A: csr_matrix,
    load_vector: np.ndarray | None,
    rhs_mode: str,
    num_samples: int,
    seed: int,
    scaled_load_range: tuple[float, float],
    synthetic_kind: str,
    synthetic_scale: float,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    mode = str(rhs_mode).strip().lower()
    if mode == "load":
        if load_vector is None:
            raise ValueError("rhs_mode='load' requires a LOAD/CLOAD file")
        return [np.asarray(load_vector, dtype=np.float64)]
    if mode == "scaled_load":
        if load_vector is None:
            raise ValueError("rhs_mode='scaled_load' requires a LOAD/CLOAD file")
        low, high = float(scaled_load_range[0]), float(scaled_load_range[1])
        if low <= 0.0 or high < low:
            raise ValueError("scaled_load_range must satisfy 0 < low <= high")
        return [
            rng.uniform(low, high) * np.asarray(load_vector, dtype=np.float64)
            for _ in range(num_samples)
        ]
    if mode == "synthetic":
        return [
            _synthetic_rhs(
                A.shape[0],
                rng=rng,
                synthetic_kind=synthetic_kind,
                synthetic_scale=synthetic_scale,
            )
            for _ in range(num_samples)
        ]
    raise ValueError(f"Unsupported rhs_mode: {rhs_mode}")


def load_abaqus_system(
    stiffness_mtx: str | Path,
    load_mtx: str | Path | None = None,
    dat_path: str | Path | None = None,
    *,
    sample_id: str | None = None,
    strict_load: bool = True,
    diag_tol: float = 0.0,
):
    """Load an Abaqus stiffness/load/dat triplet into a benchmark sample."""
    stiffness_path = Path(stiffness_mtx)
    resolved_load = _resolve_load_path(stiffness_path, load_mtx)
    resolved_dat = _resolve_dat_path(stiffness_path, dat_path)

    rn, rd, cn, cd, val = read_mtx_5col(str(stiffness_path))
    (
        A,
        dof2node,
        dof2abq,
        n_nodes,
        node_map,
        dof_map,
        free_idx,
        assembly_mode,
        pattern_stats,
    ) = build_free_system_from_mtx(
        rn,
        rd,
        cn,
        cd,
        val,
        diag_tol=diag_tol,
    )
    _validate_matrix(A)

    b = None
    if resolved_load is not None:
        b = build_load_vector_from_cload(
            str(resolved_load), node_map, dof_map, free_idx, strict=strict_load
        )

    coords_by_node_id = None
    node_ids = None
    element_ids = None
    element_types = None
    element_indptr = None
    element_nodes = None
    instance_names = None
    instance_node_ptr = None
    instance_node_ids = None
    encastre_node_ids = None
    tie_slave_ids = None
    tie_master_ptr = None
    tie_master_idx = None
    if resolved_dat is not None:
        dat_info = read_dat_info(str(resolved_dat))
        node_ids = dat_info.node_ids.astype(np.int64)
        coords_by_node_id = np.full((int(node_ids.max()) + 1, 3), np.nan, dtype=np.float64)
        coords_by_node_id[node_ids.astype(np.int64)] = dat_info.node_xyz
        element_ids = dat_info.elem_ids.astype(np.int64)
        element_types = dat_info.elem_types.astype(np.str_)
        element_indptr = dat_info.elem_conn_ptr.astype(np.int64)
        element_nodes = dat_info.elem_conn_idx.astype(np.int64)
        instance_names = dat_info.instance_names.astype(np.str_)
        instance_node_ptr = dat_info.instance_node_ptr.astype(np.int64)
        instance_node_ids = dat_info.instance_node_ids.astype(np.int64)
        encastre_node_ids = dat_info.encastre_node_ids.astype(np.int64)
        tie_slave_ids = dat_info.tie_slave_ids.astype(np.int64)
        tie_master_ptr = dat_info.tie_master_ptr.astype(np.int64)
        tie_master_idx = dat_info.tie_master_idx.astype(np.int64)

    free_dof_node_ids = node_map.node_ids[dof2node.astype(np.int64, copy=False)].astype(np.int64)
    free_dof_labels = dof2abq.astype(np.int16, copy=False)
    free_dof_indices = free_idx.astype(np.int64, copy=False)

    metadata = {
        "source_stiffness_mtx": str(stiffness_path),
        "source_load_mtx": None if resolved_load is None else str(resolved_load),
        "source_dat": None if resolved_dat is None else str(resolved_dat),
        "n": int(A.shape[0]),
        "nnz": int(A.nnz),
        "density": float(A.nnz) / float(A.shape[0] * A.shape[1]),
        "n_nodes": int(n_nodes),
        "total_dof_count": int(dof_map.dof_keys.size),
        "has_load": resolved_load is not None,
        "has_dat": resolved_dat is not None,
        "dof_per_node_labels": sorted({int(v) for v in np.unique(dof2abq)}),
        "free_dof_count": int(free_dof_indices.size),
        "matrix_assembly_mode": assembly_mode,
        "stiffness_pattern_stats": pattern_stats,
    }
    if element_types is not None:
        unique_types, counts = np.unique(element_types.astype(str), return_counts=True)
        metadata["element_type_counts"] = {
            name: int(count) for name, count in zip(unique_types, counts)
        }
        metadata["beam_element_count"] = int(np.sum(detect_beam_elements(element_types)))
    if encastre_node_ids is not None:
        metadata["encastre_node_count"] = int(encastre_node_ids.size)
    if tie_slave_ids is not None:
        metadata["tie_slave_count"] = int(tie_slave_ids.size)
    if instance_names is not None:
        metadata["instance_names"] = [str(name) for name in instance_names.tolist()]

    return BenchmarkSample(
        sample_id=sample_id or _default_sample_name(stiffness_path),
        A=A,
        b=np.zeros(A.shape[0], dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64),
        metadata=metadata,
        coords_by_node_id=coords_by_node_id,
        node_ids=node_ids,
        element_ids=element_ids,
        element_types=element_types,
        element_indptr=element_indptr,
        element_nodes=element_nodes,
        instance_names=instance_names,
        instance_node_ptr=instance_node_ptr,
        instance_node_ids=instance_node_ids,
        free_dof_node_ids=free_dof_node_ids,
        free_dof_labels=free_dof_labels,
        free_dof_indices=free_dof_indices,
        encastre_node_ids=encastre_node_ids,
        tie_slave_ids=tie_slave_ids,
        tie_master_ptr=tie_master_ptr,
        tie_master_idx=tie_master_idx,
    )


def convert_abaqus_to_npz(
    stiffness_mtx: str | Path,
    output_dir: str | Path,
    *,
    load_mtx: str | Path | None = None,
    dat_path: str | Path | None = None,
    rhs_mode: str = "load",
    num_samples: int = 1,
    solve_mode: str = "zero_placeholder",
    seed: int = 42,
    scaled_load_range: tuple[float, float] = (0.5, 1.5),
    synthetic_kind: str = "single_entry",
    synthetic_scale: float = 1.0,
    require_spd: bool = True,
    strict_load: bool = True,
    compress_npz: bool = False,
    write_per_case_manifest: bool = False,
) -> AbaqusBuildResult:
    """Convert an Abaqus system into one or more standard NPZ samples."""
    sample = load_abaqus_system(
        stiffness_mtx,
        load_mtx=load_mtx,
        dat_path=dat_path,
        strict_load=strict_load,
    )
    try:
        is_spd = bool(is_spd_matrix(sample.A))
    except SpdValidationError as exc:
        raise RuntimeError(
            f"Abaqus SPD validation failed due to resource or numerical error: {exc}"
        ) from exc
    if require_spd and not is_spd:
        raise ValueError("Abaqus stiffness matrix is not SPD")

    resolved_load = sample.metadata.get("source_load_mtx")
    rhs_vectors = _build_rhs_vectors(
        A=sample.A,
        load_vector=None if resolved_load is None else sample.b,
        rhs_mode=rhs_mode,
        num_samples=max(1, int(num_samples)),
        seed=seed,
        scaled_load_range=scaled_load_range,
        synthetic_kind=synthetic_kind,
        synthetic_scale=synthetic_scale,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    sample_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    base_name = sample.sample_id
    for idx, rhs in enumerate(rhs_vectors):
        solution, solve_mode_used = _solve_system(sample.A, rhs, solve_mode=solve_mode)
        file_name = f"{base_name}.npz" if len(rhs_vectors) == 1 else f"{base_name}_{idx:04d}.npz"
        output_path = output_root / file_name
        sample_metadata = dict(sample.metadata)
        sample_metadata.update(
            {
                "rhs_mode": rhs_mode,
                "solve_mode": solve_mode_used,
                "solve_mode_requested": solve_mode,
                "sample_index": idx,
                "seed": int(seed),
                "is_spd": is_spd,
            }
        )
        sample_metadata.update(_sample_sidecar_payload(sample))
        save_npz_sample(
            output_path,
            sample.A,
            rhs,
            x=solution,
            metadata=sample_metadata,
            compress=compress_npz,
        )
        record = {
            "sample_file": str(output_path),
            "sample_id": output_path.stem,
            "n": int(sample.A.shape[0]),
            "nnz": int(sample.A.nnz),
            "density": float(sample.A.nnz) / float(sample.A.shape[0] * sample.A.shape[1]),
            "stiffness_mtx": sample.metadata["source_stiffness_mtx"],
            "load_mtx": sample.metadata["source_load_mtx"],
            "dat_path": sample.metadata["source_dat"],
            "rhs_mode": rhs_mode,
            "solve_mode": solve_mode_used,
            "solve_mode_requested": solve_mode,
            "is_spd": is_spd,
        }
        sample_paths.append(output_path)
        records.append(record)

    manifest_path: Path | None = None
    if write_per_case_manifest:
        manifest_path = output_root / f"{base_name}_manifest.json"
        manifest = {
            "source_stiffness_mtx": sample.metadata["source_stiffness_mtx"],
            "source_load_mtx": sample.metadata["source_load_mtx"],
            "source_dat": sample.metadata["source_dat"],
            "rhs_mode": rhs_mode,
            "solve_mode": solve_mode,
            "num_samples": len(sample_paths),
            "records": records,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return AbaqusBuildResult(
        sample_paths=sample_paths, manifest_path=manifest_path, records=records
    )


def convert_abaqus_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    rhs_mode: str = "load",
    num_samples: int = 1,
    solve_mode: str = "cg",
    seed: int = 42,
    scaled_load_range: tuple[float, float] = (0.5, 1.5),
    synthetic_kind: str = "single_entry",
    synthetic_scale: float = 1.0,
    require_spd: bool = True,
    strict_load: bool = True,
    num_workers: int | None = None,
    compress_npz: bool = False,
    write_per_case_manifest: bool = False,
) -> AbaqusBuildResult:
    """Batch-convert a directory of Abaqus systems into an NPZ dataset."""
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stiffness_files = sorted(input_root.glob("*_STIF*.mtx"))
    if not stiffness_files:
        raise FileNotFoundError(f"no stiffness files matching '*_STIF*.mtx' in {input_root}")

    all_paths: list[Path] = []
    all_records: list[dict[str, Any]] = []
    tasks: list[tuple[int, dict[str, Any]]] = []
    for offset, stiffness_path in enumerate(stiffness_files):
        tasks.append(
            (
                offset,
                {
                    "stiffness_path": stiffness_path,
                    "output_root": output_root,
                    "load_mtx": None,
                    "dat_path": None,
                    "rhs_mode": rhs_mode,
                    "num_samples": num_samples,
                    "solve_mode": solve_mode,
                    "seed": seed + offset,
                    "scaled_load_range": scaled_load_range,
                    "synthetic_kind": synthetic_kind,
                    "synthetic_scale": synthetic_scale,
                    "require_spd": require_spd,
                    "strict_load": strict_load,
                    "compress_npz": compress_npz,
                    "write_per_case_manifest": write_per_case_manifest,
                },
            )
        )

    resolved_workers = _resolve_num_workers(num_workers)
    ordered_results: list[tuple[int, dict[str, Any]]] = []
    if resolved_workers == 1:
        for task in tasks:
            ordered_results.append(_convert_abaqus_directory_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_index = {
                executor.submit(_convert_abaqus_directory_worker, task): task[0] for task in tasks
            }
            for future in as_completed(future_to_index):
                ordered_results.append(future.result())
    ordered_results.sort(key=lambda item: item[0])
    for _index, result_payload in ordered_results:
        all_paths.extend(Path(path) for path in result_payload["sample_paths"])
        all_records.extend(result_payload["records"])

    manifest_path = output_root / "manifest.json"
    dataset_manifest = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "rhs_mode": rhs_mode,
        "solve_mode": solve_mode,
        "num_workers": resolved_workers,
        "compress_npz": bool(compress_npz),
        "write_per_case_manifest": bool(write_per_case_manifest),
        "num_systems": len(stiffness_files),
        "num_samples": len(all_paths),
        "records": all_records,
    }
    manifest_path.write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return AbaqusBuildResult(
        sample_paths=all_paths, manifest_path=manifest_path, records=all_records
    )


def _parse_scaled_range(text: str) -> tuple[float, float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--scaled-load-range must be 'low,high'")
    return float(parts[0]), float(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NPZ datasets from Abaqus stiffness/load/dat files."
    )
    parser.add_argument(
        "--from-abaqus", action="store_true", help="Enable Abaqus dataset conversion mode."
    )
    parser.add_argument(
        "--stiffness-mtx", type=Path, default=None, help="Single Abaqus stiffness .mtx file."
    )
    parser.add_argument(
        "--load-mtx", type=Path, default=None, help="Optional Abaqus LOAD/CLOAD .mtx file."
    )
    parser.add_argument("--dat", type=Path, default=None, help="Optional Abaqus .dat file.")
    parser.add_argument(
        "--input-dir", type=Path, default=None, help="Directory containing Abaqus STIF files."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for NPZ samples."
    )
    parser.add_argument(
        "--rhs-mode", type=str, default="load", choices=["load", "scaled_load", "synthetic"]
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--solve-mode",
        type=str,
        default="cg",
        choices=["cg", "spsolve", "dense", "cg_fallback_spsolve", "cg_fallback_dense"],
    )
    parser.add_argument("--scaled-load-range", type=str, default="0.5,1.5")
    parser.add_argument(
        "--synthetic-kind",
        type=str,
        default="single_entry",
        choices=["single_entry", "random", "sine"],
    )
    parser.add_argument("--synthetic-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-spd-check", action="store_true")
    parser.add_argument("--non-strict-load", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--compress-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--write-per-case-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if not args.from_abaqus:
        parser.error("this CLI currently supports only --from-abaqus mode")

    scaled_range = _parse_scaled_range(args.scaled_load_range)
    if args.input_dir is not None:
        result = convert_abaqus_directory(
            args.input_dir,
            args.output_dir,
            rhs_mode=args.rhs_mode,
            num_samples=args.num_samples,
            solve_mode=args.solve_mode,
            seed=args.seed,
            scaled_load_range=scaled_range,
            synthetic_kind=args.synthetic_kind,
            synthetic_scale=args.synthetic_scale,
            require_spd=not args.no_spd_check,
            strict_load=not args.non_strict_load,
            num_workers=args.num_workers,
            compress_npz=bool(args.compress_npz),
            write_per_case_manifest=bool(args.write_per_case_manifest),
        )
    elif args.stiffness_mtx is not None:
        result = convert_abaqus_to_npz(
            args.stiffness_mtx,
            args.output_dir,
            load_mtx=args.load_mtx,
            dat_path=args.dat,
            rhs_mode=args.rhs_mode,
            num_samples=args.num_samples,
            solve_mode=args.solve_mode,
            seed=args.seed,
            scaled_load_range=scaled_range,
            synthetic_kind=args.synthetic_kind,
            synthetic_scale=args.synthetic_scale,
            require_spd=not args.no_spd_check,
            strict_load=not args.non_strict_load,
            compress_npz=bool(args.compress_npz),
            write_per_case_manifest=bool(args.write_per_case_manifest),
        )
    else:
        parser.error("either --stiffness-mtx or --input-dir is required")

    summary = {
        "num_samples": len(result.sample_paths),
        "manifest_path": None if result.manifest_path is None else str(result.manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
