"""NPZ helpers for sparse-system datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, load_npz
from scipy.sparse.linalg import cg, eigsh

from dpcg.sample import BenchmarkSample

_ARRAY_FIELDS = {
    "coords_by_node_id": np.float64,
    "node_ids": np.int64,
    "element_ids": np.int64,
    "element_types": np.str_,
    "element_indptr": np.int64,
    "element_nodes": np.int64,
    "instance_names": np.str_,
    "instance_node_ptr": np.int64,
    "instance_node_ids": np.int64,
    "free_dof_node_ids": np.int64,
    "free_dof_labels": np.int16,
    "free_dof_indices": np.int64,
    "encastre_node_ids": np.int64,
    "tie_slave_ids": np.int64,
    "tie_master_ptr": np.int64,
    "tie_master_idx": np.int64,
}


def _pick_key(data, keys: Iterable[str]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _jsonify(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _restore_array_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    restored = dict(metadata)
    for key, dtype in _ARRAY_FIELDS.items():
        if key not in restored or restored[key] is None:
            continue
        restored[key] = np.asarray(restored[key], dtype=dtype)
    return restored


def read_npz_arrays(file: str | Path):
    """Read standard or legacy NPZ arrays."""
    data = np.load(file)
    indices = _pick_key(data, ["A_indices", "indices", "arr_0"])
    values = _pick_key(data, ["A_values", "values", "arr_1"])
    sol = _pick_key(data, ["x", "solution", "solutions", "arr_2"])
    rhs = _pick_key(data, ["b", "rhs", "arr_3"])
    if indices is None or values is None or rhs is None:
        raise ValueError(f"Unrecognized npz format: {file}")

    if indices.ndim == 2 and indices.shape[0] == 2:
        row, col = indices[0], indices[1]
    else:
        row, col = indices[:, 0], indices[:, 1]
    return row, col, values, sol, rhs


def read_sst_file(file: str | Path):
    """Read npz and return ``(A, solution, rhs)``."""
    row, col, values, sol, rhs = read_npz_arrays(file)
    shape = (len(rhs), len(rhs))
    A = coo_matrix((values, (row, col)), shape)
    return A, sol, rhs


def load_npz_sample(path: str | Path, sample_id: str | None = None):
    """Load a benchmark sample from NPZ and optional sidecar metadata."""
    row, col, values, sol, rhs = read_npz_arrays(path)
    matrix = coo_matrix((values, (row, col)), shape=(len(rhs), len(rhs))).tocsr()
    npz_path = Path(path)
    metadata: dict[str, Any] = {"source": str(npz_path)}
    sidecar = npz_path.with_suffix(".meta.json")
    if sidecar.exists():
        metadata["sidecar"] = str(sidecar)
        metadata.update(_restore_array_fields(json.loads(sidecar.read_text(encoding="utf-8"))))
    sample_kwargs = {}
    for key in _ARRAY_FIELDS:
        if key in metadata:
            sample_kwargs[key] = metadata.pop(key)
    return BenchmarkSample(
        sample_id=sample_id or npz_path.stem,
        A=matrix,
        b=rhs,
        x=sol,
        metadata=metadata,
        **sample_kwargs,
    )


def save_npz_sample(
    path: str | Path,
    A: csr_matrix | coo_matrix,
    b: np.ndarray,
    *,
    x: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    compress: bool = False,
) -> Path:
    """Save a sparse linear-system sample to standard NPZ."""
    matrix = A.tocoo() if hasattr(A, "tocoo") else coo_matrix(A)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    savez = np.savez_compressed if compress else np.savez
    savez(
        output,
        A_indices=np.vstack([matrix.row, matrix.col]),
        A_values=matrix.data,
        x=np.zeros(matrix.shape[0], dtype=np.float64) if x is None else np.asarray(x),
        b=np.asarray(b),
    )
    if metadata:
        sidecar = output.with_suffix(".meta.json")
        sidecar.write_text(
            json.dumps(_jsonify(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output


def list_npz_files(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).glob("*.npz"))


def is_symmetric_positive_definite(A) -> bool:
    """Check if sparse matrix is symmetric positive definite."""
    if (A != A.T).nnz != 0:
        return False
    try:
        vals, _ = eigsh(A, k=1, which="SA")
        return bool(vals[0] > 0)
    except Exception:
        return False


def read_foam_file(file: str | Path):
    """Read npz (foam) as a scipy sparse matrix."""
    return load_npz(file)


def read_csv_file(file: str | Path, shape: tuple[int, int]):
    """Read triplet CSV with columns ``(row, col, value)`` using 1-based indices."""
    data = np.loadtxt(file, delimiter=" ")
    row = data[:, 0] - 1
    col = data[:, 1] - 1
    values = data[:, 2]
    return coo_matrix((values, (row, col)), shape)


def read_K_mtx(file_path: str | Path):
    """Read Abaqus stiffness ``.mtx`` with 3 columns ``(row, col, value)``."""
    data = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            data.append((int(parts[0]), int(parts[1]), float(parts[2])))
    array = np.array(data)
    row = array[:, 0].astype(int) - 1
    col = array[:, 1].astype(int) - 1
    val = array[:, 2]
    shape = (int(array[-1, 0]), int(array[-1, 1]))
    return coo_matrix((val, (row, col)), shape)


def read_F_mtx(file_path: str | Path, shape: int):
    """Read Abaqus load vector ``.mtx`` with columns ``(node, value)``."""
    f = np.zeros(shape)
    load_data = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("*"):
                continue
            line_data = line.strip().split()
            if len(line_data) >= 2:
                load_data.append((int(line_data[0]) - 1, float(line_data[1])))
    data = np.array(load_data)
    f[data[:, 0].astype(int)] = data[:, 1]
    return f


def val_sst_solution(file: str | Path):
    """Validate stored NPZ solution by CG and return ``(diff, norm)``."""
    M, solutions, rhs = read_sst_file(file)
    x_cg, _ = cg(M, rhs)
    diff = np.abs(solutions - x_cg)
    return diff, np.linalg.norm(solutions - x_cg)


def count_npz_dimensions(npz_folder: str | Path) -> dict[int, int]:
    """Count SPD NPZ matrix dimensions in a folder."""
    dimension_counts: dict[int, int] = {}
    for filename in list_npz_files(npz_folder):
        row, col, values, _sol, rhs = read_npz_arrays(filename)
        shape = (len(rhs), len(rhs))
        M = coo_matrix((values, (row, col)), shape)
        if not is_symmetric_positive_definite(M):
            continue
        dimension_counts[shape[0]] = dimension_counts.get(shape[0], 0) + 1
    return dimension_counts


def copy_npz_dim(npz_folder: str | Path, target_folder: str | Path, dimension_target: int):
    """Copy NPZ files of a given dimension into target folder."""
    target_folder_path = Path(target_folder)
    target_folder_path.mkdir(parents=True, exist_ok=True)
    for file_path in list_npz_files(npz_folder):
        _row, _col, _values, sol, _rhs = read_npz_arrays(file_path)
        if sol is not None and sol.shape[0] == dimension_target:
            target_path = target_folder_path / file_path.name
            target_path.write_bytes(file_path.read_bytes())


def compare_npz_val(folder1: str | Path, folder2: str | Path, num1: int, num2: int):
    """Compare nonzero values between two NPZ files by index."""
    file1 = list_npz_files(folder1)[num1]
    file2 = list_npz_files(folder2)[num2]
    with np.load(file1) as data1, np.load(file2) as data2:
        v1 = _pick_key(data1, ["A_values", "values", "arr_1"])
        v2 = _pick_key(data2, ["A_values", "values", "arr_1"])
    return v1, v2


compare_npz_values = compare_npz_val
copy_npz_by_size = copy_npz_dim
count_npz_sizes = count_npz_dimensions
load_npz_matrix = read_foam_file
load_npz_system = read_sst_file
load_coo_from_triplet_csv = read_csv_file
validate_npz_solution = val_sst_solution
