"""PETSc/CUDA benchmark path for learned and classical preconditioners."""

from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np
from scipy.linalg import eigh as scipy_dense_eigh
from scipy.sparse import csr_matrix, diags, tril
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

from dpcg.data import sample_to_half_graph
from dpcg.models import SparseFactorPrediction
from dpcg.utils import (
    assemble_sparse_factor_from_prediction_torch,
    normalize_diag_strategy,
    torch_sparse_to_spconv,
)

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    import cupyx.scipy.sparse.linalg as cupy_sparse_linalg
except Exception:  # pragma: no cover - optional dependency
    cp = None
    cupy_sparse = None
    cupy_sparse_linalg = None

def _default_petsc_init_args() -> list[str]:
    option_tokens = shlex.split(os.environ.get("PETSC_OPTIONS", ""))
    has_gpu_aware_toggle = any(
        token.lstrip("-").strip().lower() == "use_gpu_aware_mpi" for token in option_tokens
    )
    if has_gpu_aware_toggle:
        return []
    return ["-use_gpu_aware_mpi", "0"]


try:
    import petsc4py
except Exception:  # pragma: no cover - optional dependency
    petsc4py = None
    PETSc = None
else:
    init_args = _default_petsc_init_args()
    if init_args:
        petsc4py.init(init_args)
    try:
        from petsc4py import PETSc
    except Exception:  # pragma: no cover - optional dependency
        PETSc = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

if TYPE_CHECKING:
    from dpcg.sample import BenchmarkSample


_REF_COND_DENSE_LIMIT = 256
_REF_COND_EIGSH_TOL = 1.0e-3
_REF_COND_EIGSH_MAXITER = 4_000


class PetscBenchmarkSkip(RuntimeError):
    """Raised when a PETSc method is intentionally skipped."""


@dataclass
class PetscPreparedMethod:
    method: str
    mode: str
    setup_time_sec: float
    apply_kind: str
    matrix_type: str | None
    vector_type: str | None
    preconditioner_impl: str
    resolved_pc_type: str | None
    resolved_factor_solver_type: str | None
    ksp_norm_type: str
    petsc_matrix: Any | None = None
    operator_matrix: Any | None = None
    operator_matrix_scipy: csr_matrix | None = None
    factor_matrix_petsc: Any | None = None
    factor_matrix_scipy: csr_matrix | None = None
    factor_matrix_gpu: Any | None = None
    shell_context: Any | None = None
    ksp: Any | None = None
    pc: Any | None = None
    factor_nnz: float | None = None
    factor_density: float | None = None
    operator_nnz: float | None = None
    operator_density: float | None = None
    wall_forward_time_sec: float | None = None
    steady_forward_time_sec: float | None = None
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
    inference_peak_gpu_memory_mb: float | None = None
    cpu_rss_delta_mb: float | None = None
    factor_matrix_type: str | None = None
    operator_matrix_type: str | None = None
    ic0_levels: int | None = None
    ic0_ordering: str | None = None
    ic0_shift_type: str | None = None
    pc_mode: str | None = None
    solve_operator_mode: str | None = None
    learning_transformed_internal_rtol: float | None = None
    learning_transformed_internal_rtol_ratio: float | None = None
    learning_convergence_basis: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _ShellTransformedOperatorContext:
    def __init__(self, A_gpu, triangular_gpu, sqrt_diag_gpu, scale: float):
        self.A_gpu = A_gpu.tocsr()
        self.triangular_gpu = triangular_gpu.tocsr()
        self.triangular_t_gpu = self.triangular_gpu.transpose().tocsr()
        self.sqrt_diag_gpu = cp.asarray(sqrt_diag_gpu, dtype=cp.float64).reshape(-1)
        self.diag_gpu = self.sqrt_diag_gpu * self.sqrt_diag_gpu
        self.scale = float(scale)
        self.sqrt_scale = float(np.sqrt(scale))
        self.n = int(self.A_gpu.shape[0])

    def _solve_lower(self, rhs_gpu):
        rhs = cp.asarray(rhs_gpu, dtype=cp.float64).reshape(-1, 1)
        solved = cupy_sparse_linalg.spsolve_triangular(
            self.triangular_gpu,
            rhs,
            lower=True,
            overwrite_A=False,
            overwrite_b=False,
            unit_diagonal=False,
        )
        return cp.asarray(solved, dtype=cp.float64).reshape(-1)

    def _solve_upper(self, rhs_gpu):
        rhs = cp.asarray(rhs_gpu, dtype=cp.float64).reshape(-1, 1)
        solved = cupy_sparse_linalg.spsolve_triangular(
            self.triangular_t_gpu,
            rhs,
            lower=False,
            overwrite_A=False,
            overwrite_b=False,
            unit_diagonal=False,
        )
        return cp.asarray(solved, dtype=cp.float64).reshape(-1)

    def apply_left(self, x_gpu):
        weighted = self.sqrt_diag_gpu * cp.asarray(x_gpu, dtype=cp.float64).reshape(-1)
        return self.sqrt_scale * self._solve_upper(weighted)

    def apply_right(self, x_gpu):
        lower_solved = self._solve_lower(x_gpu)
        return self.sqrt_scale * self.sqrt_diag_gpu * lower_solved

    def apply_preconditioner(self, x_gpu):
        lower_solved = self._solve_lower(x_gpu)
        weighted = self.scale * self.diag_gpu * lower_solved
        return self._solve_upper(weighted)

    def transformed_rhs(self, b_gpu):
        return self.apply_left(b_gpu)

    def recover_solution(self, y_gpu):
        return self.apply_right(y_gpu)

    def mult(self, mat, x, y):
        with _vec_cuda_array(x, self.n, mode="r") as x_arr:
            with _vec_cuda_array(y, self.n, mode="w") as y_arr:
                transformed_x = self.apply_right(x_arr)
                transformed_y = self.apply_left(self.A_gpu @ transformed_x)
                y_arr[...] = transformed_y


class _FactorizedPCContext:
    def __init__(self, factor_gpu, factor_petsc, n: int):
        self.factor_gpu = factor_gpu
        self.factor_petsc = factor_petsc
        self.n = int(n)
        self.work_vec = None

    def setUp(self, pc) -> None:
        if self.work_vec is None:
            self.work_vec = self.factor_petsc.createVecRight()
            self.work_vec.setType(PETSc.Vec.Type.CUDA)

    def apply(self, pc, x, y) -> None:
        if self.work_vec is None:
            self.setUp(pc)
        self.factor_petsc.multTranspose(x, self.work_vec)
        self.factor_petsc.mult(self.work_vec, y)

    def applySymmetricLeft(self, pc, x, y) -> None:
        self.factor_petsc.mult(x, y)

    def applySymmetricRight(self, pc, x, y) -> None:
        self.factor_petsc.multTranspose(x, y)

    def applyTranspose(self, pc, x, y) -> None:
        self.apply(pc, x, y)


_LEARNING_NATIVE_LIB = None
_LEARNING_NATIVE_LIB_NAME = "libdpcg_petsc_learning.so"
_PETSC_GPU_RUNTIME_STATUS: tuple[bool, str | None] | None = None


def _default_learning_native_library_path() -> Path:
    return Path(__file__).resolve().parent / "_native" / _LEARNING_NATIVE_LIB_NAME


def _current_petsc_library_path() -> Path | None:
    petsc_dir = os.environ.get("PETSC_DIR")
    petsc_arch = os.environ.get("PETSC_ARCH")
    if (not petsc_dir or not petsc_arch) and petsc4py is not None:
        cfg = petsc4py.get_config()
        petsc_dir = petsc_dir or cfg.get("PETSC_DIR")
        petsc_arch = petsc_arch or cfg.get("PETSC_ARCH")
    if not petsc_dir or not petsc_arch:
        return None
    if ":" in petsc_arch:
        petsc_arch = petsc_arch.split(":")[-1]
    return Path(petsc_dir) / petsc_arch / "lib" / "libpetsc.so"


def _load_learning_native_library():
    global _LEARNING_NATIVE_LIB
    if _LEARNING_NATIVE_LIB is not None:
        return _LEARNING_NATIVE_LIB
    candidate = _default_learning_native_library_path()
    if not candidate.exists():
        raise RuntimeError(
            "learning transformed native operator requires the sidecar library "
            f"{candidate}. Build it with scripts/build_petsc_learning_native.sh"
        )
    petsc_library = _current_petsc_library_path()
    if petsc_library is not None and petsc_library.exists():
        try:
            ctypes.CDLL(str(petsc_library), mode=ctypes.RTLD_GLOBAL | os.RTLD_NOLOAD)
        except OSError:
            ctypes.CDLL(str(petsc_library), mode=ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    library.dpcg_learning_shell_attach.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.dpcg_learning_shell_attach.restype = ctypes.c_int
    library.dpcg_learning_pcshell_attach.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.dpcg_learning_pcshell_attach.restype = ctypes.c_int
    _LEARNING_NATIVE_LIB = library
    return library


def _require_petsc_gpu_stack() -> None:
    if PETSc is None:
        raise RuntimeError("petsc4py is required for BENCHMARK_BACKEND='petsc_gpu'")
    if cp is None or cupy_sparse is None or cupy_sparse_linalg is None:
        raise RuntimeError("cupy is required for BENCHMARK_BACKEND='petsc_gpu'")
    if getattr(PETSc.Mat.Type, "AIJCUSPARSE", None) is None:
        raise RuntimeError("PETSc was not built with AIJCUSPARSE support")
    if getattr(PETSc.Vec.Type, "CUDA", None) is None:
        raise RuntimeError("PETSc was not built with CUDA Vec support")
    runtime_ok, runtime_error = _petsc_gpu_runtime_status()
    if not runtime_ok:
        detail = "" if not runtime_error else f" Original error: {runtime_error}"
        raise RuntimeError(
            "PETSc advertises CUDA support, but initializing CUDA Vec/AIJCUSPARSE failed."
            " Run `source scripts/use_petsc_hypre.sh` and `bash scripts/check_petsc_cuda_runtime.sh`"
            f" in the same HPC shell.{detail}"
        )


def _probe_petsc_gpu_runtime() -> tuple[bool, str | None]:
    init_args = _default_petsc_init_args()
    init_args_source = repr(init_args) if init_args else "[]"
    probe_code = """
import numpy as np
import petsc4py
_INIT_ARGS = """ + init_args_source + """
if _INIT_ARGS:
    petsc4py.init(_INIT_ARGS)
from petsc4py import PETSc

vec = PETSc.Vec().createSeq(1)
vec.setType(PETSc.Vec.Type.CUDA)
vec.set(1.0)

indptr = np.asarray([0, 1], dtype=np.int32)
indices = np.asarray([0], dtype=np.int32)
values = np.asarray([1.0], dtype=np.float64)

mat = PETSc.Mat().create()
mat.setSizes([1, 1])
mat.setType(PETSc.Mat.Type.AIJCUSPARSE)
mat.setPreallocationCSR((indptr, indices))
mat.setValuesCSR(indptr, indices, values)
mat.assemblyBegin()
mat.assemblyEnd()

x = mat.createVecRight()
x.setType(PETSc.Vec.Type.CUDA)
x.set(1.0)
y = mat.createVecLeft()
y.setType(PETSc.Vec.Type.CUDA)
mat.mult(x, y)

print("vec_type", vec.getType())
print("mat_type", mat.getType())
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe_code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode == 0:
        return True, None
    message = (completed.stderr or completed.stdout).strip()
    if not message:
        message = f"PETSc CUDA runtime probe failed with exit code {completed.returncode}"
    return False, message


def _reset_petsc_gpu_runtime_status_cache() -> None:
    global _PETSC_GPU_RUNTIME_STATUS
    _PETSC_GPU_RUNTIME_STATUS = None


def _petsc_gpu_runtime_status() -> tuple[bool, str | None]:
    global _PETSC_GPU_RUNTIME_STATUS
    if _PETSC_GPU_RUNTIME_STATUS is not None:
        return _PETSC_GPU_RUNTIME_STATUS
    try:
        _PETSC_GPU_RUNTIME_STATUS = _probe_petsc_gpu_runtime()
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        _PETSC_GPU_RUNTIME_STATUS = (False, message)
    return _PETSC_GPU_RUNTIME_STATUS


def petsc_capabilities() -> dict[str, Any]:
    if PETSc is None:
        return {
            "petsc_available": False,
            "petsc_has_cuda": False,
            "petsc_has_hypre": False,
            "petsc_version": None,
            "petsc_cuda_runtime_ok": False,
            "petsc_cuda_runtime_error": "petsc4py is unavailable",
        }
    runtime_ok = False
    runtime_error = None
    if (
        cp is not None
        and cupy_sparse is not None
        and cupy_sparse_linalg is not None
        and getattr(PETSc.Mat.Type, "AIJCUSPARSE", None) is not None
        and getattr(PETSc.Vec.Type, "CUDA", None) is not None
    ):
        runtime_ok, runtime_error = _petsc_gpu_runtime_status()
    return {
        "petsc_available": True,
        "petsc_has_cuda": bool(PETSc.Sys.hasExternalPackage("cuda")),
        "petsc_has_hypre": bool(PETSc.Sys.hasExternalPackage("hypre")),
        "petsc_version": ".".join(str(part) for part in PETSc.Sys.getVersion()),
        "petsc4py_version": getattr(PETSc, "__version__", None),
        "cupy_version": None if cp is None else cp.__version__,
        "petsc_cuda_runtime_ok": runtime_ok,
        "petsc_cuda_runtime_error": runtime_error,
    }


def metadata_from_runtime(
    *,
    petsc_options: str | None,
    petsc_amg_backend: str,
) -> dict[str, Any]:
    caps = petsc_capabilities()
    option_map = _parse_petsc_options(petsc_options or "")
    return {
        **caps,
        "resolved_amg_backend": str(petsc_amg_backend),
        "petsc_use_gpu_aware_mpi": option_map.get("use_gpu_aware_mpi", None),
    }


def _parse_petsc_options(option_string: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    tokens = shlex.split(str(option_string).strip())
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            index += 1
            continue
        key = token.lstrip("-")
        value = "1"
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            value = tokens[index + 1]
            index += 1
        parsed[key] = value
        index += 1
    return parsed


@contextmanager
def _petsc_options_scope(option_string: str | None) -> Iterator[None]:
    if PETSc is None:
        yield
        return
    updates = _parse_petsc_options(option_string or "")
    if not updates:
        yield
        return
    options = PETSc.Options()
    existing = options.getAll()
    sentinel = object()
    previous: dict[str, object] = {key: existing.get(key, sentinel) for key in updates}
    for key, value in updates.items():
        options.setValue(key, value)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is sentinel:
                options.delValue(key)
            else:
                options.setValue(key, str(old_value))


def _cuda_sync() -> None:
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()


def _factor_stats(nnz: int | None, n: int) -> tuple[float | None, float | None]:
    if nnz is None:
        return None, None
    return float(nnz), float(nnz) / float(n * n)


def _matrix_stats(matrix: csr_matrix | None) -> tuple[float | None, float | None]:
    if matrix is None:
        return None, None
    return _factor_stats(int(matrix.nnz), int(matrix.shape[0]))


def _symmetrize_csr(matrix: csr_matrix) -> csr_matrix:
    sym = (0.5 * (matrix + matrix.transpose())).tocsr()
    sym.sum_duplicates()
    sym.sort_indices()
    return sym


def _build_factorized_transformed_operator_scipy(
    A: csr_matrix,
    factor: csr_matrix,
) -> csr_matrix:
    return _symmetrize_csr((factor @ (A @ factor.transpose())).tocsr())


def _torch_tensor_to_cupy(tensor: "torch.Tensor"):
    detached = tensor.detach()
    if detached.device.type == "cuda":
        return cp.from_dlpack(detached)
    return cp.asarray(detached.cpu().numpy())


def _torch_sparse_to_cupy_csr(sparse_tensor: "torch.Tensor"):
    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices()
    values = coalesced.values().to(dtype=torch.float64)
    rows = _torch_tensor_to_cupy(indices[0].to(dtype=torch.int32))
    cols = _torch_tensor_to_cupy(indices[1].to(dtype=torch.int32))
    data = _torch_tensor_to_cupy(values)
    return cupy_sparse.coo_matrix(
        (data, (rows, cols)),
        shape=tuple(int(v) for v in coalesced.shape),
    ).tocsr()


def _scipy_to_cupy_csr(matrix: csr_matrix):
    csr = matrix.tocsr().astype(np.float64)
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    data = np.asarray(csr.data, dtype=np.float64)
    return cupy_sparse.csr_matrix(
        (cp.asarray(data), cp.asarray(indices), cp.asarray(indptr)),
        shape=csr.shape,
    )


def _cupy_csr_to_scipy_csr(matrix) -> csr_matrix:
    csr = matrix.tocsr()
    return csr_matrix(
        (
            cp.asnumpy(csr.data),
            cp.asnumpy(csr.indices),
            cp.asnumpy(csr.indptr),
        ),
        shape=csr.shape,
    )


def _petsc_matrix_to_scipy_csr(matrix) -> csr_matrix:
    indptr, indices, data = matrix.getValuesCSR()
    return csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=matrix.getSize(),
    )


@contextmanager
def _vec_cuda_array(vec, n: int, *, mode: str) -> Iterator[Any]:
    handle = vec.getCUDAHandle(mode)
    try:
        memory = cp.cuda.UnownedMemory(int(handle), n * np.dtype(np.float64).itemsize, vec)
        pointer = cp.cuda.MemoryPointer(memory, 0)
        yield cp.ndarray((n,), dtype=cp.float64, memptr=pointer)
    finally:
        vec.restoreCUDAHandle(handle, mode)


def _new_cuda_vec(n: int, values=None):
    vec = PETSc.Vec().createSeq(int(n))
    vec.setType(PETSc.Vec.Type.CUDA)
    vec.set(0.0)
    if values is not None:
        values_gpu = cp.asarray(values, dtype=cp.float64).reshape(-1)
        with _vec_cuda_array(vec, int(n), mode="w") as vec_arr:
            vec_arr[...] = values_gpu
    return vec


def _copy_solution_from_vec(vec, n: int):
    with _vec_cuda_array(vec, n, mode="r") as arr:
        return cp.asarray(arr).copy()


def _build_petsc_matrix(matrix: csr_matrix, *, matrix_type: str | None = None):
    csr = matrix.tocsr().astype(np.float64)
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    data = np.asarray(csr.data, dtype=np.float64)
    petsc_matrix = PETSc.Mat().create()
    petsc_matrix.setSizes(list(csr.shape))
    resolved_type = PETSc.Mat.Type.AIJCUSPARSE if matrix_type is None else str(matrix_type)
    petsc_matrix.setType(resolved_type)
    petsc_matrix.setPreallocationCSR((indptr, indices))
    petsc_matrix.setValuesCSR(indptr, indices, data)
    petsc_matrix.assemblyBegin()
    petsc_matrix.assemblyEnd()
    try:
        petsc_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    except Exception:
        pass
    try:
        petsc_matrix.setOption(PETSc.Mat.Option.SPD, True)
    except Exception:
        pass
    return petsc_matrix


def _make_native_learning_shell_operator(A_petsc, factor_petsc):
    operator = PETSc.Mat().create(comm=A_petsc.getComm())
    library = _load_learning_native_library()
    ierr = int(
        library.dpcg_learning_shell_attach(
            ctypes.c_void_p(int(operator.handle)),
            ctypes.c_void_p(int(A_petsc.handle)),
            ctypes.c_void_p(int(factor_petsc.handle)),
        )
    )
    if ierr != 0:
        raise RuntimeError(
            f"failed to attach native learning transformed shell operator (PETSc error code {ierr})"
        )
    operator.setVecType(PETSc.Vec.Type.CUDA)
    return operator


def _attach_native_learning_pcshell(pc, factor_petsc) -> None:
    library = _load_learning_native_library()
    ierr = int(
        library.dpcg_learning_pcshell_attach(
            ctypes.c_void_p(int(pc.handle)),
            ctypes.c_void_p(int(factor_petsc.handle)),
        )
    )
    if ierr != 0:
        raise RuntimeError(f"failed to attach native learning PC shell (PETSc error code {ierr})")


def _make_shell_operator(context: _ShellTransformedOperatorContext):
    operator = PETSc.Mat().createPython(
        [list(context.A_gpu.shape), list(context.A_gpu.shape)],
        context=context,
    )
    operator.setVecType(PETSc.Vec.Type.CUDA)
    operator.setUp()
    return operator


def _build_factorized_operator_matrix(A_petsc, factor_petsc):
    right = A_petsc.matTransposeMult(factor_petsc)
    operator = factor_petsc.matMult(right)
    try:
        operator.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    except Exception:
        pass
    try:
        operator.setOption(PETSc.Mat.Option.SPD, True)
    except Exception:
        pass
    return operator


def _normalize_ksp_norm_type(norm_type: str) -> tuple[int, str]:
    value = str(norm_type).strip().lower()
    if value in {"unpreconditioned", "norm_unpreconditioned"}:
        return PETSc.KSP.NormType.UNPRECONDITIONED, "unpreconditioned"
    if value in {"preconditioned", "norm_preconditioned"}:
        return PETSc.KSP.NormType.PRECONDITIONED, "preconditioned"
    if value in {"natural", "norm_natural"}:
        return PETSc.KSP.NormType.NATURAL, "natural"
    if value in {"none", "no", "norm_none"}:
        return PETSc.KSP.NormType.NONE, "none"
    raise ValueError(f"Unsupported PETSC_KSP_NORM_TYPE={norm_type!r}")


def _enum_ksp_norm_type_name(norm_type_value: int) -> str | None:
    mapping = {
        int(PETSc.KSP.NormType.NONE): "none",
        int(PETSc.KSP.NormType.PRECONDITIONED): "preconditioned",
        int(PETSc.KSP.NormType.UNPRECONDITIONED): "unpreconditioned",
        int(PETSc.KSP.NormType.NATURAL): "natural",
    }
    return mapping.get(int(norm_type_value))


def _make_prefixed_options(prefix: str, values: dict[str, str]) -> dict[str, str]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _set_ksp_tolerances(ksp, *, rtol: float, maxiter: int | None) -> None:
    kwargs: dict[str, Any] = {"rtol": float(rtol), "atol": 0.0}
    if maxiter is not None:
        kwargs["max_it"] = int(maxiter)
    ksp.setTolerances(**kwargs)


def _set_ksp_spectral_estimation(ksp, enabled: bool) -> None:
    if hasattr(ksp, "setComputeSingularValues"):
        ksp.setComputeSingularValues(bool(enabled))


def _set_ksp_norm_type(ksp, norm_type: str) -> str:
    enum_value, normalized = _normalize_ksp_norm_type(norm_type)
    ksp.setNormType(enum_value)
    return normalized


def _extract_ksp_spectral_metrics(ksp) -> tuple[float | None, float | None, float | None]:
    if not hasattr(ksp, "computeExtremeSingularValues"):
        return None, None, None
    try:
        sigma_max, sigma_min = ksp.computeExtremeSingularValues()
    except Exception:
        return None, None, None
    sigma_max_value = float(sigma_max) if np.isfinite(sigma_max) else None
    sigma_min_value = float(sigma_min) if np.isfinite(sigma_min) else None
    cond_est = None
    if sigma_max_value is not None and sigma_min_value is not None and sigma_min_value > 0.0:
        cond_est = float(sigma_max_value / sigma_min_value)
    return sigma_max_value, sigma_min_value, cond_est


def _resolve_run_status(
    *,
    ksp_converged: bool,
    true_residual_converged: bool,
) -> tuple[str, str | None]:
    if true_residual_converged:
        if ksp_converged:
            return "ok", None
        return (
            "ok",
            "True residual satisfies the requested rtol even though KSP did not report convergence",
        )
    if ksp_converged:
        return (
            "failed",
            "KSP reported convergence, but the true residual does not satisfy the requested rtol",
        )
    return (
        "failed",
        "KSP did not converge and the true residual does not satisfy the requested rtol",
    )


def _timed_median_seconds(*, runner, warmup_runs: int, repeats: int) -> float:
    for _ in range(max(int(warmup_runs), 0)):
        _cuda_sync()
        runner()
        _cuda_sync()
    elapsed: list[float] = []
    measure_repeats = max(int(repeats), 1)
    for _ in range(measure_repeats):
        _cuda_sync()
        start = time.perf_counter()
        runner()
        _cuda_sync()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(np.asarray(elapsed, dtype=np.float64)))


def _destroy_petsc_object(obj) -> None:
    if obj is None:
        return
    destroy = getattr(obj, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass


def _destroy_prepared_method(prepared: PetscPreparedMethod | None) -> None:
    if prepared is None:
        return
    seen: set[int] = set()
    for obj in (
        prepared.ksp,
        prepared.pc,
        prepared.operator_matrix,
        prepared.factor_matrix_petsc,
        prepared.petsc_matrix,
    ):
        if obj is None:
            continue
        object_id = id(obj)
        if object_id in seen:
            continue
        seen.add(object_id)
        _destroy_petsc_object(obj)


def _true_residual_stats(sample: "BenchmarkSample", x_gpu) -> tuple[float, float]:
    x = cp.asnumpy(cp.asarray(x_gpu, dtype=cp.float64).reshape(-1))
    residual = np.asarray(sample.b, dtype=np.float64).reshape(-1) - sample.A @ x
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(np.asarray(sample.b, dtype=np.float64).reshape(-1)))
    denom = rhs_norm if rhs_norm > 0.0 else 1.0
    return residual_norm, residual_norm / denom


def _reference_spectrum_explicit(
    matrix: csr_matrix,
) -> tuple[float | None, float | None, float | None, str | None]:
    sym = (0.5 * (matrix + matrix.transpose())).tocsr()
    n = int(sym.shape[0])
    if n <= 1:
        if n == 0:
            return None, None, None, None
        value = float(sym.diagonal()[0])
        cond = None if value <= 0.0 else 1.0
        return value, value, cond, "trivial"
    try:
        if n <= _REF_COND_DENSE_LIMIT:
            dense = np.asarray(sym.toarray(), dtype=np.float64)
            eigenvalues = np.linalg.eigvalsh(dense)
            lambda_min = float(eigenvalues[0])
            lambda_max = float(eigenvalues[-1])
            method = "dense_eigh"
        else:
            try:
                lambda_min = float(
                    np.real(
                        eigsh(
                            sym,
                            k=1,
                            sigma=0.0,
                            which="LM",
                            tol=_REF_COND_EIGSH_TOL,
                            maxiter=_REF_COND_EIGSH_MAXITER,
                            return_eigenvectors=False,
                        )[0]
                    )
                )
                method = "eigsh_sigma0"
            except Exception:
                lambda_min = float(
                    np.real(
                        eigsh(
                            sym,
                            k=1,
                            which="SA",
                            tol=_REF_COND_EIGSH_TOL,
                            maxiter=_REF_COND_EIGSH_MAXITER,
                            return_eigenvectors=False,
                        )[0]
                    )
                )
                method = "eigsh_sa"
            lambda_max = float(
                np.real(
                    eigsh(
                        sym,
                        k=1,
                        which="LA",
                        tol=_REF_COND_EIGSH_TOL,
                        maxiter=_REF_COND_EIGSH_MAXITER,
                        return_eigenvectors=False,
                    )[0]
                )
            )
            method = f"{method}+eigsh_la"
    except Exception:
        return None, None, None, None
    cond_est = None
    if np.isfinite(lambda_min) and np.isfinite(lambda_max) and lambda_min > 0.0:
        cond_est = float(lambda_max / lambda_min)
    return lambda_min, lambda_max, cond_est, method


def _reference_spectrum_generalized(
    A: csr_matrix,
    M: csr_matrix,
) -> tuple[float | None, float | None, float | None, str | None]:
    A_sym = _symmetrize_csr(A)
    M_sym = _symmetrize_csr(M)
    n = int(A_sym.shape[0])
    if n <= 1:
        if n == 0:
            return None, None, None, None
        a_value = float(A_sym.diagonal()[0])
        m_value = float(M_sym.diagonal()[0])
        if m_value <= 0.0:
            return None, None, None, None
        eigenvalue = a_value / m_value
        cond = None if eigenvalue <= 0.0 else 1.0
        return eigenvalue, eigenvalue, cond, "trivial_generalized"
    try:
        if n <= _REF_COND_DENSE_LIMIT:
            dense_A = np.asarray(A_sym.toarray(), dtype=np.float64)
            dense_M = np.asarray(M_sym.toarray(), dtype=np.float64)
            eigenvalues = scipy_dense_eigh(dense_A, dense_M, eigvals_only=True)
            lambda_min = float(eigenvalues[0])
            lambda_max = float(eigenvalues[-1])
            method = "dense_generalized_eigh"
        else:
            try:
                lambda_min = float(
                    np.real(
                        eigsh(
                            A_sym,
                            k=1,
                            M=M_sym,
                            sigma=0.0,
                            which="LM",
                            tol=_REF_COND_EIGSH_TOL,
                            maxiter=_REF_COND_EIGSH_MAXITER,
                            return_eigenvectors=False,
                        )[0]
                    )
                )
                method = "eigsh_generalized_sigma0"
            except Exception:
                lambda_min = float(
                    np.real(
                        eigsh(
                            A_sym,
                            k=1,
                            M=M_sym,
                            which="SA",
                            tol=_REF_COND_EIGSH_TOL,
                            maxiter=_REF_COND_EIGSH_MAXITER,
                            return_eigenvectors=False,
                        )[0]
                    )
                )
                method = "eigsh_generalized_sa"
            lambda_max = float(
                np.real(
                    eigsh(
                        A_sym,
                        k=1,
                        M=M_sym,
                        which="LA",
                        tol=_REF_COND_EIGSH_TOL,
                        maxiter=_REF_COND_EIGSH_MAXITER,
                        return_eigenvectors=False,
                    )[0]
                )
            )
            method = f"{method}+eigsh_generalized_la"
    except Exception:
        return None, None, None, None
    cond_est = None
    if np.isfinite(lambda_min) and np.isfinite(lambda_max) and lambda_min > 0.0:
        cond_est = float(lambda_max / lambda_min)
    return lambda_min, lambda_max, cond_est, method


def _reference_spectrum_pc_symmetric_operator(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
) -> tuple[float | None, float | None, float | None, str | None]:
    n = int(sample.A.shape[0])
    input_vec = _new_cuda_vec(n)
    right_vec = prepared.petsc_matrix.createVecRight()
    middle_vec = prepared.petsc_matrix.createVecRight()
    output_vec = prepared.petsc_matrix.createVecRight()

    def _matvec(x: np.ndarray) -> np.ndarray:
        with _vec_cuda_array(input_vec, n, mode="w") as input_arr:
            input_arr[...] = cp.asarray(np.asarray(x, dtype=np.float64).reshape(-1))
        prepared.pc.applySymmetricRight(input_vec, right_vec)
        prepared.petsc_matrix.mult(right_vec, middle_vec)
        prepared.pc.applySymmetricLeft(middle_vec, output_vec)
        return cp.asnumpy(_copy_solution_from_vec(output_vec, n))

    try:
        if n <= _REF_COND_DENSE_LIMIT:
            basis = np.eye(n, dtype=np.float64)
            dense = np.column_stack([_matvec(basis[:, i]) for i in range(n)])
            dense = 0.5 * (dense + dense.T)
            eigenvalues = np.linalg.eigvalsh(dense)
            lambda_min = float(eigenvalues[0])
            lambda_max = float(eigenvalues[-1])
            method = "dense_pc_symmetric"
        else:
            operator = LinearOperator((n, n), matvec=_matvec, dtype=np.float64)
            try:
                lambda_min = float(
                    np.real(
                        eigsh(
                            operator,
                            k=1,
                            sigma=0.0,
                            which="LM",
                            tol=_REF_COND_EIGSH_TOL,
                            maxiter=_REF_COND_EIGSH_MAXITER,
                            return_eigenvectors=False,
                        )[0]
                    )
                )
                method = "eigsh_pc_symmetric_sigma0"
            except Exception:
                lambda_min = float(
                    np.real(
                        eigsh(
                            operator,
                            k=1,
                            which="SA",
                            tol=_REF_COND_EIGSH_TOL,
                            maxiter=_REF_COND_EIGSH_MAXITER,
                            return_eigenvectors=False,
                        )[0]
                    )
                )
                method = "eigsh_pc_symmetric_sa"
            lambda_max = float(
                np.real(
                    eigsh(
                        operator,
                        k=1,
                        which="LA",
                        tol=_REF_COND_EIGSH_TOL,
                        maxiter=_REF_COND_EIGSH_MAXITER,
                        return_eigenvectors=False,
                    )[0]
                )
            )
            method = f"{method}+eigsh_pc_symmetric_la"
    except Exception:
        return None, None, None, None
    cond_est = None
    if np.isfinite(lambda_min) and np.isfinite(lambda_max) and lambda_min > 0.0:
        cond_est = float(lambda_max / lambda_min)
    return lambda_min, lambda_max, cond_est, method


def _reference_spectrum_left_preconditioned_operator(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
) -> tuple[float | None, float | None, float | None, str | None]:
    n = int(sample.A.shape[0])
    input_vec = _new_cuda_vec(n)
    output_vec = prepared.petsc_matrix.createVecRight()

    def _matvec(x: np.ndarray) -> np.ndarray:
        rhs = sample.A @ np.asarray(x, dtype=np.float64).reshape(-1)
        with _vec_cuda_array(input_vec, n, mode="w") as input_arr:
            input_arr[...] = cp.asarray(rhs, dtype=cp.float64)
        prepared.pc.apply(input_vec, output_vec)
        return cp.asnumpy(_copy_solution_from_vec(output_vec, n))

    try:
        if n <= _REF_COND_DENSE_LIMIT:
            basis = np.eye(n, dtype=np.float64)
            dense = np.column_stack([_matvec(basis[:, i]) for i in range(n)])
            eigenvalues = np.real_if_close(np.linalg.eigvals(dense))
            eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
            lambda_min = float(np.min(eigenvalues))
            lambda_max = float(np.max(eigenvalues))
            method = "dense_left_preconditioned"
        else:
            operator = LinearOperator((n, n), matvec=_matvec, dtype=np.float64)
            lambda_min = float(
                np.real(
                    eigs(
                        operator,
                        k=1,
                        which="SR",
                        tol=_REF_COND_EIGSH_TOL,
                        maxiter=_REF_COND_EIGSH_MAXITER,
                        return_eigenvectors=False,
                    )[0]
                )
            )
            lambda_max = float(
                np.real(
                    eigs(
                        operator,
                        k=1,
                        which="LR",
                        tol=_REF_COND_EIGSH_TOL,
                        maxiter=_REF_COND_EIGSH_MAXITER,
                        return_eigenvectors=False,
                    )[0]
                )
            )
            method = "eigs_left_preconditioned"
    except Exception:
        return None, None, None, None
    cond_est = None
    if np.isfinite(lambda_min) and np.isfinite(lambda_max) and lambda_min > 0.0:
        cond_est = float(lambda_max / lambda_min)
    return lambda_min, lambda_max, cond_est, method


def _compute_reference_condition_metrics(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    cond_mode: str,
) -> dict[str, Any]:
    mode = str(cond_mode).strip().lower()
    if mode == "off":
        return {
            "cond_est_ref": None,
            "lambda_min_ref": None,
            "lambda_max_ref": None,
            "cond_est_method": None,
        }
    if mode != "accurate_ref":
        raise ValueError(f"Unsupported PETSC_COND_MODE={cond_mode!r}")

    if prepared.method == "learning":
        if prepared.factor_matrix_scipy is None:
            return {
                "cond_est_ref": None,
                "lambda_min_ref": None,
                "lambda_max_ref": None,
                "cond_est_method": None,
            }
        operator_matrix = _build_factorized_transformed_operator_scipy(
            sample.A.tocsr().astype(np.float64),
            prepared.factor_matrix_scipy,
        )
        lambda_min, lambda_max, cond_est, method = _reference_spectrum_explicit(operator_matrix)
        return {
            "cond_est_ref": cond_est,
            "lambda_min_ref": lambda_min,
            "lambda_max_ref": lambda_max,
            "cond_est_method": method,
        }

    if prepared.method == "none":
        lambda_min, lambda_max, cond_est, method = _reference_spectrum_explicit(
            sample.A.tocsr().astype(np.float64)
        )
        return {
            "cond_est_ref": cond_est,
            "lambda_min_ref": lambda_min,
            "lambda_max_ref": lambda_max,
            "cond_est_method": method,
        }

    if prepared.method == "jacobi":
        diagonal = sample.A.diagonal().astype(np.float64, copy=False)
        mass_matrix = diags(diagonal, format="csr")
        lambda_min, lambda_max, cond_est, method = _reference_spectrum_generalized(
            sample.A, mass_matrix
        )
        return {
            "cond_est_ref": cond_est,
            "lambda_min_ref": lambda_min,
            "lambda_max_ref": lambda_max,
            "cond_est_method": method,
        }

    if prepared.method in {"sgs", "ssor"}:
        mass_matrix = prepared.metadata.get("reference_mass_matrix")
        if mass_matrix is None:
            return {
                "cond_est_ref": None,
                "lambda_min_ref": None,
                "lambda_max_ref": None,
                "cond_est_method": None,
            }
        lambda_min, lambda_max, cond_est, method = _reference_spectrum_generalized(
            sample.A, mass_matrix
        )
        return {
            "cond_est_ref": cond_est,
            "lambda_min_ref": lambda_min,
            "lambda_max_ref": lambda_max,
            "cond_est_method": method,
        }

    if prepared.method == "ic0":
        return {
            "cond_est_ref": None,
            "lambda_min_ref": None,
            "lambda_max_ref": None,
            "cond_est_method": None,
        }

    if prepared.method == "amg":
        return {
            "cond_est_ref": None,
            "lambda_min_ref": None,
            "lambda_max_ref": None,
            "cond_est_method": None,
        }

    return {
        "cond_est_ref": None,
        "lambda_min_ref": None,
        "lambda_max_ref": None,
        "cond_est_method": None,
    }


def _compute_reference_condition_metrics_timed(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    cond_mode: str,
) -> tuple[dict[str, Any], float | None, str | None]:
    mode = str(cond_mode).strip().lower()
    if mode == "off":
        return (
            _compute_reference_condition_metrics(sample, prepared, cond_mode=cond_mode),
            None,
            None,
        )
    start = time.perf_counter()
    metrics = _compute_reference_condition_metrics(sample, prepared, cond_mode=cond_mode)
    elapsed = time.perf_counter() - start
    if metrics.get("cond_est_ref") is None:
        return metrics, None, None
    return metrics, float(elapsed), "scipy_reference"


def _run_ksp(
    ksp,
    b_vec,
    x_vec,
    *,
    compute_singular_values: bool,
) -> tuple[float, list[float], int, int, float | None, float | None, float | None]:
    residuals: list[float] = []

    def _monitor(_ksp, _its: int, rnorm: float) -> None:
        residuals.append(float(rnorm))

    _set_ksp_spectral_estimation(ksp, compute_singular_values)
    x_vec.set(0.0)
    ksp.setMonitor(_monitor)
    _cuda_sync()
    start = time.perf_counter()
    ksp.solve(b_vec, x_vec)
    _cuda_sync()
    elapsed = time.perf_counter() - start
    reason = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())
    sigma_max, sigma_min, cond_est = (
        _extract_ksp_spectral_metrics(ksp) if compute_singular_values else (None, None, None)
    )
    return elapsed, residuals, iterations, reason, sigma_max, sigma_min, cond_est


def _make_transformed_ksp(
    operator_matrix,
    *,
    rtol: float,
    maxiter: int | None,
    ksp_norm_type: str,
    compute_singular_values: bool,
):
    ksp = PETSc.KSP().create()
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.NONE)
    ksp.setOperators(operator_matrix)
    _set_ksp_norm_type(ksp, ksp_norm_type)
    _set_ksp_spectral_estimation(ksp, compute_singular_values)
    _set_ksp_tolerances(ksp, rtol=rtol, maxiter=maxiter)
    ksp.setUp()
    return ksp


def _resolve_petsc_ic0_shift_type(petsc_ic0_shift_type: str) -> tuple[Any, str]:
    normalized = str(petsc_ic0_shift_type).strip().lower()
    if normalized in {"positive_definite", "positive-definite", "positive definite"}:
        return PETSc.Mat.FactorShiftType.POSITIVE_DEFINITE, "positive_definite"
    if normalized in {"none", "off"}:
        return PETSc.Mat.FactorShiftType.NONE, "none"
    raise ValueError(
        "PETSC_IC0_SHIFT_TYPE currently only supports 'positive_definite' or 'none'"
    )


def _configure_classical_ksp(
    A_petsc,
    *,
    method: str,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str,
    ksp_norm_type: str,
    petsc_sgs_ssor_impl: str,
    compute_singular_values: bool = False,
) -> tuple[Any, Any, str, str | None, str | None, str | None]:
    del petsc_sgs_ssor_impl
    prefix = f"dpcg_{method}_{id(A_petsc)}_"
    options: dict[str, str] = {}
    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix(prefix)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.setOperators(A_petsc)
    _set_ksp_norm_type(ksp, ksp_norm_type)
    _set_ksp_spectral_estimation(ksp, compute_singular_values)
    _set_ksp_tolerances(ksp, rtol=rtol, maxiter=maxiter)
    pc = ksp.getPC()
    preconditioner_impl = ""
    resolved_ordering = None
    resolved_shift_type = None
    if method == "none":
        pc.setType(PETSc.PC.Type.NONE)
        preconditioner_impl = "PCNONE"
    elif method == "jacobi":
        pc.setType(PETSc.PC.Type.JACOBI)
        preconditioner_impl = "PCJACOBI"
    elif method in {"sgs", "ssor"}:
        pc.setType(PETSc.PC.Type.SOR)
        options = {
            "pc_sor_symmetric": "",
            "pc_sor_its": "1",
            "pc_sor_omega": "1.0" if method == "sgs" else str(float(ssor_omega)),
        }
        preconditioner_impl = "PCSOR"
    elif method == "ic0":
        pc.setType(PETSc.PC.Type.ICC)
        pc.setFactorLevels(0)
        pc.setFactorSolverType("petsc")
        shift_type, resolved_shift_type = _resolve_petsc_ic0_shift_type(petsc_ic0_shift_type)
        pc.setFactorShift(shift_type)
        preconditioner_impl = "PCICC"
    elif method == "amg":
        backend = str(petsc_amg_backend).strip().lower()
        if backend == "hypre_boomeramg":
            if not PETSc.Sys.hasExternalPackage("hypre"):
                raise PetscBenchmarkSkip("PETSc hypre support is not available")
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType("boomeramg")
            preconditioner_impl = "PCHYPRE(boomeramg)"
        elif backend == "gamg":
            pc.setType(PETSc.PC.Type.GAMG)
            preconditioner_impl = "PCGAMG"
        else:
            raise ValueError(f"Unsupported PETSC_AMG_BACKEND={petsc_amg_backend!r}")
    elif method == "parasails":
        if not PETSc.Sys.hasExternalPackage("hypre"):
            raise PetscBenchmarkSkip("PETSc hypre support is not available")
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("parasails")
        preconditioner_impl = "PCHYPRE(parasails)"
    else:
        raise ValueError(f"Unsupported PETSc classical method: {method}")

    with _petsc_options_scope(
        " ".join(f"-{k} {v}".strip() for k, v in _make_prefixed_options(prefix, options).items())
    ):
        ksp.setFromOptions()
        ksp.setUp()
    resolved_factor_solver_type = None
    if hasattr(pc, "getFactorSolverType"):
        try:
            resolved_factor_solver_type = pc.getFactorSolverType()
        except Exception:
            resolved_factor_solver_type = None
    return (
        ksp,
        pc,
        preconditioner_impl,
        resolved_factor_solver_type,
        resolved_ordering,
        resolved_shift_type,
    )


def _configure_factorized_pc_ksp(
    A_petsc,
    *,
    factor_gpu,
    factor_petsc,
    rtol: float,
    maxiter: int | None,
    ksp_norm_type: str,
    pc_impl: str,
    compute_singular_values: bool = False,
):
    impl = str(pc_impl).strip().lower()
    if impl in {"pcpython", "python"}:
        resolved_impl = "pcpython"
    elif impl in {"shell_native", "pcshell", "pcshell_native"}:
        resolved_impl = "shell_native"
    else:
        raise ValueError(f"Unsupported factor PC implementation: {pc_impl!r}")
    ksp = PETSc.KSP().create()
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.setOperators(A_petsc)
    effective_norm_type = _set_ksp_norm_type(ksp, ksp_norm_type)
    _set_ksp_spectral_estimation(ksp, compute_singular_values)
    _set_ksp_tolerances(ksp, rtol=rtol, maxiter=maxiter)
    pc = ksp.getPC()
    if resolved_impl == "pcpython":
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(
            _FactorizedPCContext(
                factor_gpu=factor_gpu,
                factor_petsc=factor_petsc,
                n=int(A_petsc.getSize()[0]),
            )
        )
        preconditioner_impl = "PCPYTHON(factorized_llt_gpu)"
    else:
        pc.setType(PETSc.PC.Type.SHELL)
        _attach_native_learning_pcshell(pc, factor_petsc)
        preconditioner_impl = "PCSHELL(factorized_llt_gpu_native)"
    ksp.setUp()
    return ksp, pc, preconditioner_impl, effective_norm_type


def _resolve_learning_internal_rtol(
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
) -> float:
    if prepared.learning_transformed_internal_rtol_ratio is not None:
        internal_rtol = float(rtol) * float(prepared.learning_transformed_internal_rtol_ratio)
    elif prepared.learning_transformed_internal_rtol is not None:
        internal_rtol = float(prepared.learning_transformed_internal_rtol)
    else:
        internal_rtol = float(rtol)
    if internal_rtol <= 0.0:
        raise ValueError("Resolved learning internal rtol must be positive")
    return internal_rtol


def _measure_pc_apply(pc, input_vec, output_vec, *, warmup_runs: int, repeats: int) -> float:
    return _timed_median_seconds(
        runner=lambda: pc.apply(input_vec, output_vec),
        warmup_runs=warmup_runs,
        repeats=repeats,
    )


def _measure_factor_apply(
    factor_matrix,
    input_vec,
    work_vec,
    output_vec,
    *,
    warmup_runs: int,
    repeats: int,
) -> float:
    return _timed_median_seconds(
        runner=lambda: (
            factor_matrix.multTranspose(input_vec, work_vec),
            factor_matrix.mult(work_vec, output_vec),
        ),
        warmup_runs=warmup_runs,
        repeats=repeats,
    )


def _measure_shell_apply(
    context: _ShellTransformedOperatorContext,
    input_gpu,
    *,
    warmup_runs: int,
    repeats: int,
) -> float:
    return _timed_median_seconds(
        runner=lambda: context.apply_preconditioner(input_gpu),
        warmup_runs=warmup_runs,
        repeats=repeats,
    )


def _run_ksp_spectral_diagnostics(
    ksp,
    rhs_vec,
    work_vec,
) -> tuple[float | None, float | None, float | None, float | None]:
    try:
        spectral_solve_time_sec, _, _, _, sigma_max, sigma_min, cond_est = _run_ksp(
            ksp,
            rhs_vec,
            work_vec,
            compute_singular_values=True,
        )
    except Exception:
        return None, None, None, None
    return spectral_solve_time_sec, sigma_max, sigma_min, cond_est


def _should_enable_approx_spectral_diagnostics(
    method: str,
    requested: bool,
) -> bool:
    if bool(requested):
        return True
    return str(method).strip().lower() == "ic0"


def _measure_learning_forward_wall(model, model_input) -> tuple[Any, float]:
    _cuda_sync()
    start = time.perf_counter()
    with torch.no_grad():
        try:
            prediction = model(model_input, profile=False)
        except TypeError:
            prediction = model(model_input)
    _cuda_sync()
    return prediction, time.perf_counter() - start


def _measure_learning_forward_steady(model, model_input, *, repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("steady forward repeats must be positive")
    _cuda_sync()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(int(repeats)):
            try:
                model(model_input, profile=False)
            except TypeError:
                model(model_input)
    _cuda_sync()
    return (time.perf_counter() - start) / float(repeats)


def _coalesced_sample_values(sample: "BenchmarkSample", device: str):
    if torch is None or sample.l_tensor is None:
        raise RuntimeError("learning benchmark requires torch sparse l_tensor")
    coalesced = sample.l_tensor.coalesce()
    return coalesced.values().unsqueeze(1).to(device=device)


def _prepare_learning_factor(
    sample: "BenchmarkSample",
    *,
    model,
    output_kind: str,
    diag_strategy: str,
    use_mask_projection: bool,
    learning_device: str,
    learning_group_context: dict[str, Any] | None,
    steady_forward_repeats: int,
    factor_solve_mode: str,
    ksp_norm_type: str,
    rtol: float,
    maxiter: int | None,
    learning_internal_rtol_ratio: float,
    petsc_learning_pc_impl: str = "shell_native",
):
    from dpcg import benchmark as benchmark_mod

    del steady_forward_repeats
    if torch is None:
        raise RuntimeError("torch is required for the learning benchmark path")
    if output_kind != "sparse_factor_L":
        raise ValueError("PETSc learning path only supports sparse_factor_L")
    solve_mode = str(factor_solve_mode).strip().lower()
    if solve_mode == "transformed_operator":
        solve_mode = "transformed_operator_native"
    if solve_mode not in {"transformed_operator_native", "factor_pc"}:
        raise ValueError(
            "PETSc learning path only supports "
            "PETSC_FACTOR_SOLVE_MODE in "
            "{'transformed_operator_native', 'transformed_operator', 'factor_pc'}"
        )
    benchmark_mod._ensure_sample_sparse_cache(sample)
    if use_mask_projection and sample.mask is None:
        raise RuntimeError("learning benchmark requires mask in the sample")
    input_kind = getattr(model, "input_kind", "spconv")
    use_cuda = str(learning_device).startswith("cuda")
    graph_build_time_sec = 0.0
    peak_gpu_memory_mb = None
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    if input_kind == "graph":
        start = time.perf_counter()
        model_input = sample_to_half_graph(sample).to(learning_device)
        _cuda_sync()
        graph_build_time_sec = time.perf_counter() - start
    else:
        if sample.l_tensor is None:
            raise RuntimeError("learning benchmark requires l_tensor in the sample")
        if (
            learning_group_context is not None
            and learning_group_context.get("base_spconv_input") is not None
        ):
            features = _coalesced_sample_values(sample, learning_device)
            model_input = learning_group_context["base_spconv_input"].replace_feature(features)
        else:
            model_input = torch_sparse_to_spconv(sample.l_tensor.to(learning_device))
            if learning_group_context is not None:
                learning_group_context["base_spconv_input"] = model_input
        _cuda_sync()
    prediction, wall_forward_time_sec = _measure_learning_forward_wall(model, model_input)
    steady_forward_time_sec = wall_forward_time_sec
    if not isinstance(prediction, SparseFactorPrediction):
        raise TypeError("learning benchmark expects SparseFactorPrediction output")
    postprocess_start = time.perf_counter()
    assembled = assemble_sparse_factor_from_prediction_torch(
        coords=prediction.coords,
        values=prediction.values.to(dtype=torch.float64),
        mask_rows=torch.as_tensor(
            sample.mask_rows, dtype=torch.int64, device=prediction.values.device
        ),
        mask_cols=torch.as_tensor(
            sample.mask_cols, dtype=torch.int64, device=prediction.values.device
        ),
        mask_key=torch.as_tensor(
            sample.mask_key, dtype=torch.int64, device=prediction.values.device
        ),
        diag_inv=torch.as_tensor(
            sample.diag_inv, dtype=torch.float64, device=prediction.values.device
        ),
        shape=prediction.shape,
        diag_strategy=normalize_diag_strategy(diag_strategy),
        force_unit_diag=True,
        use_mask_projection=bool(use_mask_projection),
    )
    factor_gpu = _torch_sparse_to_cupy_csr(assembled["sparse_tensor"])
    _cuda_sync()
    postprocess_time_sec = time.perf_counter() - postprocess_start

    factor_gpu_to_host_start = time.perf_counter()
    factor_scipy = _cupy_csr_to_scipy_csr(factor_gpu)
    factor_gpu_to_host_time_sec = time.perf_counter() - factor_gpu_to_host_start

    factor_petsc_build_start = time.perf_counter()
    factor_petsc = _build_petsc_matrix(factor_scipy)
    _cuda_sync()
    factor_petsc_build_time_sec = time.perf_counter() - factor_petsc_build_start

    a_petsc_build_start = time.perf_counter()
    A_petsc = _build_petsc_matrix(sample.A)
    _cuda_sync()
    a_petsc_build_time_sec = time.perf_counter() - a_petsc_build_start
    learning_internal_rtol = None
    learning_internal_rtol_ratio_value = None
    ksp = None
    pc = None
    operator_build_time_sec = None
    solve_ready_build_time_sec = None
    operator_nnz, operator_density = _matrix_stats(sample.A.tocsr().astype(np.float64))
    if solve_mode == "transformed_operator_native":
        solve_ready_build_start = time.perf_counter()
        operator_matrix = _make_native_learning_shell_operator(A_petsc, factor_petsc)
        _cuda_sync()
        solve_ready_build_time_sec = time.perf_counter() - solve_ready_build_start
        operator_build_time_sec = solve_ready_build_time_sec
        learning_internal_rtol_ratio_value = float(learning_internal_rtol_ratio)
        if learning_internal_rtol_ratio_value <= 0.0:
            raise ValueError("PETSC_LEARNING_INTERNAL_RTOL_RATIO must be positive")
        learning_internal_rtol = float(rtol) * learning_internal_rtol_ratio_value
        preconditioner_impl = "native_matshell(factorized_transformed_gpu)"
        resolved_ksp_norm_type = _normalize_ksp_norm_type(ksp_norm_type)[1]
        operator_matrix_type = operator_matrix.getType()
        vector_type = operator_matrix.createVecRight().getType()
        mode = "transformed_learning_native"
        pc_mode = None
        resolved_pc_type = "none"
        solve_operator_mode = "transformed_operator_native"
        learning_convergence_basis = "transformed_internal"
        operator_nnz, operator_density = (None, None)
    else:
        solve_ready_build_start = time.perf_counter()
        ksp, pc, preconditioner_impl, resolved_ksp_norm_type = _configure_factorized_pc_ksp(
            A_petsc,
            factor_gpu=factor_gpu,
            factor_petsc=factor_petsc,
            rtol=rtol,
            maxiter=maxiter,
            ksp_norm_type=ksp_norm_type,
            pc_impl=petsc_learning_pc_impl,
            compute_singular_values=False,
        )
        _cuda_sync()
        solve_ready_build_time_sec = time.perf_counter() - solve_ready_build_start
        operator_matrix = A_petsc
        operator_matrix_type = A_petsc.getType()
        vector_type = A_petsc.createVecRight().getType()
        mode = "factor_pc"
        pc_mode = (
            "factorized_llt_gpu_shell_native"
            if "shell" in str(preconditioner_impl).lower()
            else "factorized_llt_gpu_python"
        )
        resolved_pc_type = pc.getType()
        solve_operator_mode = "original_operator_pc"
        learning_convergence_basis = "original_true_residual"
    setup_time_sec = graph_build_time_sec + steady_forward_time_sec + postprocess_time_sec
    excluded_materialization_time_sec = (
        factor_gpu_to_host_time_sec
        + factor_petsc_build_time_sec
        + a_petsc_build_time_sec
        + (0.0 if solve_ready_build_time_sec is None else solve_ready_build_time_sec)
    )
    if use_cuda:
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    factor_nnz, factor_density = _matrix_stats(factor_scipy)
    return PetscPreparedMethod(
        method="learning",
        mode=mode,
        setup_time_sec=setup_time_sec,
        apply_kind="factorized_llt" if mode == "transformed_learning_native" else "petsc_pc_apply",
        matrix_type=operator_matrix.getType(),
        vector_type=vector_type,
        preconditioner_impl=preconditioner_impl,
        resolved_pc_type=resolved_pc_type,
        resolved_factor_solver_type=None,
        ksp_norm_type=resolved_ksp_norm_type,
        petsc_matrix=A_petsc,
        operator_matrix=operator_matrix,
        operator_matrix_scipy=None,
        factor_matrix_petsc=factor_petsc,
        factor_matrix_scipy=factor_scipy,
        factor_matrix_gpu=factor_gpu,
        ksp=ksp,
        pc=pc,
        factor_nnz=factor_nnz,
        factor_density=factor_density,
        operator_nnz=operator_nnz,
        operator_density=operator_density,
        wall_forward_time_sec=wall_forward_time_sec,
        steady_forward_time_sec=steady_forward_time_sec,
        transfer_time_sec=factor_gpu_to_host_time_sec,
        postprocess_time_sec=postprocess_time_sec,
        factor_assembly_time_sec=postprocess_time_sec,
        factor_gpu_to_host_time_sec=factor_gpu_to_host_time_sec,
        factor_petsc_build_time_sec=factor_petsc_build_time_sec,
        a_petsc_build_time_sec=a_petsc_build_time_sec,
        solve_ready_build_time_sec=solve_ready_build_time_sec,
        excluded_materialization_time_sec=excluded_materialization_time_sec,
        operator_build_time_sec=operator_build_time_sec,
        graph_build_time_sec=graph_build_time_sec,
        inference_peak_gpu_memory_mb=peak_gpu_memory_mb,
        cpu_rss_delta_mb=0.0,
        factor_matrix_type=factor_petsc.getType(),
        operator_matrix_type=operator_matrix_type,
        pc_mode=pc_mode,
        solve_operator_mode=solve_operator_mode,
        learning_transformed_internal_rtol=learning_internal_rtol,
        learning_transformed_internal_rtol_ratio=learning_internal_rtol_ratio_value,
        learning_convergence_basis=learning_convergence_basis,
    )
def _prepare_classical_method(
    sample: "BenchmarkSample",
    *,
    method: str,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str,
    ksp_norm_type: str,
):
    setup_start = time.perf_counter()
    matrix_type = PETSc.Mat.Type.AIJ if method == "parasails" else PETSc.Mat.Type.AIJCUSPARSE
    petsc_matrix = _build_petsc_matrix(sample.A, matrix_type=matrix_type)
    (
        ksp,
        pc,
        preconditioner_impl,
        resolved_factor_solver_type,
        resolved_ordering,
        resolved_shift_type,
    ) = _configure_classical_ksp(
        petsc_matrix,
        method=method,
        rtol=rtol,
        maxiter=maxiter,
        ssor_omega=ssor_omega,
        petsc_amg_backend=petsc_amg_backend,
        petsc_ic0_shift_type=petsc_ic0_shift_type,
        ksp_norm_type=ksp_norm_type,
        petsc_sgs_ssor_impl="petsc_sor_legacy",
    )
    setup_time_sec = time.perf_counter() - setup_start
    factor_matrix_petsc = None
    factor_matrix_scipy = None
    factor_matrix_type = None
    factor_nnz = None
    factor_density = None
    metadata: dict[str, Any] = {}
    if method == "ic0" and hasattr(pc, "getFactorMatrix"):
        try:
            factor_matrix_petsc = pc.getFactorMatrix()
        except Exception:
            factor_matrix_petsc = None
        if factor_matrix_petsc is not None:
            factor_matrix_type = factor_matrix_petsc.getType()
            info = factor_matrix_petsc.getInfo()
            factor_nnz = float(info.get("nz_used", 0.0))
            factor_density = factor_nnz / float(sample.A.shape[0] ** 2)
    if method in {"sgs", "ssor"}:
        diagonal = sample.A.diagonal().astype(np.float64, copy=False)
        lower = tril(sample.A, k=-1, format="csr")
        omega = 1.0 if method == "sgs" else float(ssor_omega)
        scale = 1.0 if method == "sgs" else omega * (2.0 - omega)
        triangular = (diags(diagonal, format="csr") + omega * lower).tocsr()
        inverse_diag = diags(1.0 / diagonal, format="csr")
        metadata["reference_mass_matrix"] = (
            (1.0 / scale) * (triangular @ inverse_diag @ triangular.transpose())
        ).tocsr()
    vector_type = _new_cuda_vec(int(sample.A.shape[0])).getType()
    operator_nnz, operator_density = _matrix_stats(sample.A)
    return PetscPreparedMethod(
        method=method,
        mode="classical_pc",
        setup_time_sec=setup_time_sec,
        apply_kind="petsc_pc_apply",
        matrix_type=petsc_matrix.getType(),
        vector_type=vector_type,
        preconditioner_impl=preconditioner_impl,
        resolved_pc_type=pc.getType(),
        resolved_factor_solver_type=resolved_factor_solver_type,
        ksp_norm_type=ksp_norm_type,
        petsc_matrix=petsc_matrix,
        operator_matrix=petsc_matrix,
        operator_matrix_scipy=sample.A.tocsr().astype(np.float64),
        factor_matrix_petsc=factor_matrix_petsc,
        factor_matrix_scipy=factor_matrix_scipy,
        ksp=ksp,
        pc=pc,
        factor_nnz=factor_nnz,
        factor_density=factor_density,
        operator_nnz=operator_nnz,
        operator_density=operator_density,
        factor_matrix_type=factor_matrix_type,
        operator_matrix_type=petsc_matrix.getType(),
        ic0_levels=0 if method == "ic0" else None,
        ic0_ordering=resolved_ordering,
        ic0_shift_type=resolved_shift_type,
        metadata=metadata,
    )


def prepare_method(
    sample: "BenchmarkSample",
    *,
    method: str,
    model,
    learning_output_kind: str,
    learning_diag_strategy: str,
    use_mask_projection: bool = True,
    learning_device: str,
    ssor_omega: float,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str = "none",
    rtol: float,
    maxiter: int | None,
    learning_group_context: dict[str, Any] | None,
    steady_forward_repeats: int,
    petsc_ksp_norm_type: str,
    petsc_sgs_ssor_impl: str,
    petsc_factor_solve_mode: str,
    petsc_factor_operator_mode: str,
    petsc_learning_internal_rtol_ratio: float,
    petsc_learning_pc_impl: str,
):
    _require_petsc_gpu_stack()
    del petsc_sgs_ssor_impl
    method = str(method).strip().lower()
    if method == "learning":
        if model is None:
            raise RuntimeError("learning method requires a model")
        return _prepare_learning_factor(
            sample,
            model=model,
            output_kind=learning_output_kind,
            diag_strategy=learning_diag_strategy,
            use_mask_projection=use_mask_projection,
            learning_device=learning_device,
            learning_group_context=learning_group_context,
            steady_forward_repeats=steady_forward_repeats,
            factor_solve_mode=petsc_factor_solve_mode,
            ksp_norm_type=petsc_ksp_norm_type,
            rtol=rtol,
            maxiter=maxiter,
            learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
            petsc_learning_pc_impl=petsc_learning_pc_impl,
        )
    return _prepare_classical_method(
        sample,
        method=method,
        rtol=rtol,
        maxiter=maxiter,
        ssor_omega=ssor_omega,
        petsc_amg_backend=petsc_amg_backend,
        petsc_ic0_shift_type=petsc_ic0_shift_type,
        ksp_norm_type=petsc_ksp_norm_type,
    )


def _prepare_method_once(builder):
    prepared = builder()
    return prepared, float(prepared.setup_time_sec)


def _petsc_result_timing_fields(prepared: PetscPreparedMethod) -> dict[str, float | None]:
    return {
        "wall_forward_time_sec": prepared.wall_forward_time_sec,
        "steady_forward_time_sec": prepared.steady_forward_time_sec,
        "forward_wall_time_sec": prepared.wall_forward_time_sec,
        "forward_steady_time_sec": prepared.steady_forward_time_sec,
        "transfer_time_sec": prepared.transfer_time_sec,
        "postprocess_time_sec": prepared.postprocess_time_sec,
        "factor_assembly_time_sec": prepared.factor_assembly_time_sec,
        "factor_gpu_to_host_time_sec": prepared.factor_gpu_to_host_time_sec,
        "factor_petsc_build_time_sec": prepared.factor_petsc_build_time_sec,
        "a_petsc_build_time_sec": prepared.a_petsc_build_time_sec,
        "solve_ready_build_time_sec": prepared.solve_ready_build_time_sec,
        "excluded_materialization_time_sec": prepared.excluded_materialization_time_sec,
        "operator_build_time_sec": prepared.operator_build_time_sec,
        "graph_build_time_sec": prepared.graph_build_time_sec,
    }


def _make_petsc_apply_runner(sample: "BenchmarkSample", prepared: PetscPreparedMethod):
    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    if prepared.mode in {"transformed_explicit", "transformed_learning_native"}:
        input_vec = _new_cuda_vec(n, b_gpu)
        work_vec = prepared.factor_matrix_petsc.createVecRight()
        output_vec = prepared.factor_matrix_petsc.createVecLeft()
        return lambda: (
            prepared.factor_matrix_petsc.multTranspose(input_vec, work_vec),
            prepared.factor_matrix_petsc.mult(work_vec, output_vec),
        )
    if prepared.mode == "transformed_shell":
        return lambda: prepared.shell_context.apply_preconditioner(b_gpu)
    rhs_vec = _new_cuda_vec(n, b_gpu)
    apply_vec = _new_cuda_vec(n)
    return lambda: prepared.pc.apply(rhs_vec, apply_vec)


def _factor_pc_impl_for_method(
    method: str,
    *,
    petsc_learning_pc_impl: str,
) -> str:
    del method
    return petsc_learning_pc_impl


def _run_petsc_solve_core(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    petsc_amg_backend: str,
    petsc_sgs_ssor_impl: str,
    factor_pc_impl: str,
) -> dict[str, Any]:
    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    cleanup_objects: list[Any] = []
    try:
        if prepared.mode == "transformed_learning_native":
            input_vec = _new_cuda_vec(n, b_gpu)
            rhs_vec = prepared.operator_matrix.createVecRight()
            prepared.factor_matrix_petsc.mult(input_vec, rhs_vec)
            y_vec = prepared.operator_matrix.createVecLeft()
            internal_rtol = _resolve_learning_internal_rtol(prepared, rtol=rtol)
            ksp = _make_transformed_ksp(
                prepared.operator_matrix,
                rtol=internal_rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                compute_singular_values=False,
            )
            cleanup_objects.extend([input_vec, rhs_vec, y_vec, ksp])
            solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
                ksp,
                rhs_vec,
                y_vec,
                compute_singular_values=False,
            )
            x_vec = prepared.factor_matrix_petsc.createVecRight()
            cleanup_objects.append(x_vec)
            prepared.factor_matrix_petsc.multTranspose(y_vec, x_vec)
            x_gpu = _copy_solution_from_vec(x_vec, n)
            true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
            ksp_converged = reason > 0
            return {
                "status": "ok" if ksp_converged else "failed",
                "message": None
                if ksp_converged
                else "Transformed KSP did not converge to the requested internal tolerance",
                "solve_time_sec": solve_time_sec,
                "residuals": residuals,
                "iterations": iterations,
                "info": reason,
                "ksp_reason": reason,
                "ksp_converged": ksp_converged,
                "true_residual_converged": bool(true_relative_residual <= float(rtol)),
                "converged": ksp_converged,
                "relative_residual": true_relative_residual,
                "final_residual_norm": true_final_residual_norm,
                "true_relative_residual": true_relative_residual,
                "true_final_residual_norm": true_final_residual_norm,
                "ksp_monitor_initial_residual": None if not residuals else residuals[0],
                "ksp_monitor_final_residual": None if not residuals else residuals[-1],
            }
        if prepared.mode == "transformed_explicit":
            input_vec = _new_cuda_vec(n, b_gpu)
            rhs_vec = prepared.operator_matrix.createVecRight()
            prepared.factor_matrix_petsc.mult(input_vec, rhs_vec)
            y_vec = prepared.operator_matrix.createVecLeft()
            ksp = _make_transformed_ksp(
                prepared.operator_matrix,
                rtol=rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                compute_singular_values=False,
            )
            cleanup_objects.extend([input_vec, rhs_vec, y_vec, ksp])
            solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
                ksp,
                rhs_vec,
                y_vec,
                compute_singular_values=False,
            )
            x_vec = prepared.factor_matrix_petsc.createVecRight()
            cleanup_objects.append(x_vec)
            prepared.factor_matrix_petsc.multTranspose(y_vec, x_vec)
            x_gpu = _copy_solution_from_vec(x_vec, n)
        elif prepared.mode == "transformed_shell":
            rhs_gpu = prepared.shell_context.transformed_rhs(b_gpu)
            rhs_vec = prepared.operator_matrix.createVecRight()
            with _vec_cuda_array(rhs_vec, n, mode="w") as rhs_arr:
                rhs_arr[...] = rhs_gpu
            y_vec = prepared.operator_matrix.createVecLeft()
            ksp = _make_transformed_ksp(
                prepared.operator_matrix,
                rtol=rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                compute_singular_values=False,
            )
            cleanup_objects.extend([rhs_vec, y_vec, ksp])
            solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
                ksp,
                rhs_vec,
                y_vec,
                compute_singular_values=False,
            )
            y_gpu = _copy_solution_from_vec(y_vec, n)
            x_gpu = prepared.shell_context.recover_solution(y_gpu)
        elif prepared.mode == "factor_pc":
            rhs_vec = _new_cuda_vec(n, b_gpu)
            x_vec = _new_cuda_vec(n)
            ksp, _pc, _impl, _norm_type = _configure_factorized_pc_ksp(
                prepared.petsc_matrix,
                factor_gpu=prepared.factor_matrix_gpu,
                factor_petsc=prepared.factor_matrix_petsc,
                rtol=rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                pc_impl=factor_pc_impl,
                compute_singular_values=False,
            )
            cleanup_objects.extend([rhs_vec, x_vec, ksp])
            solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
                ksp,
                rhs_vec,
                x_vec,
                compute_singular_values=False,
            )
            x_gpu = _copy_solution_from_vec(x_vec, n)
        else:
            rhs_vec = _new_cuda_vec(n, b_gpu)
            x_vec = _new_cuda_vec(n)
            (
                ksp,
                _pc,
                _impl,
                _solver_type,
                _ordering,
                _shift_type,
            ) = _configure_classical_ksp(
                prepared.petsc_matrix,
                method=prepared.method,
                rtol=rtol,
                maxiter=maxiter,
                ssor_omega=ssor_omega,
                petsc_amg_backend=petsc_amg_backend,
                petsc_ic0_shift_type=prepared.ic0_shift_type or "none",
                ksp_norm_type=prepared.ksp_norm_type,
                petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                compute_singular_values=False,
            )
            cleanup_objects.extend([rhs_vec, x_vec, ksp])
            solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
                ksp,
                rhs_vec,
                x_vec,
                compute_singular_values=False,
            )
            x_gpu = _copy_solution_from_vec(x_vec, n)
        true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
        ksp_converged = reason > 0
        true_residual_converged = bool(true_relative_residual <= float(rtol))
        status, message = _resolve_run_status(
            ksp_converged=ksp_converged,
            true_residual_converged=true_residual_converged,
        )
        return {
            "status": status,
            "message": message,
            "solve_time_sec": solve_time_sec,
            "residuals": residuals,
            "iterations": iterations,
            "info": reason,
            "ksp_reason": reason,
            "ksp_converged": ksp_converged,
            "true_residual_converged": true_residual_converged,
            "converged": true_residual_converged,
            "relative_residual": true_relative_residual,
            "final_residual_norm": true_final_residual_norm,
            "true_relative_residual": true_relative_residual,
            "true_final_residual_norm": true_final_residual_norm,
            "ksp_monitor_initial_residual": None if not residuals else residuals[0],
            "ksp_monitor_final_residual": None if not residuals else residuals[-1],
        }
    finally:
        for obj in cleanup_objects:
            _destroy_petsc_object(obj)


def _run_petsc_spectral_diagnostics(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    petsc_amg_backend: str,
    petsc_sgs_ssor_impl: str,
    factor_pc_impl: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    cleanup_objects: list[Any] = []
    try:
        if prepared.mode == "transformed_learning_native":
            input_vec = _new_cuda_vec(n, b_gpu)
            rhs_vec = prepared.operator_matrix.createVecRight()
            prepared.factor_matrix_petsc.mult(input_vec, rhs_vec)
            y_vec = prepared.operator_matrix.createVecLeft()
            internal_rtol = _resolve_learning_internal_rtol(prepared, rtol=rtol)
            spectral_ksp = _make_transformed_ksp(
                prepared.operator_matrix,
                rtol=internal_rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                compute_singular_values=True,
            )
            cleanup_objects.extend([input_vec, rhs_vec, y_vec, spectral_ksp])
            return _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, y_vec)
        if prepared.mode == "transformed_explicit":
            input_vec = _new_cuda_vec(n, b_gpu)
            rhs_vec = prepared.operator_matrix.createVecRight()
            prepared.factor_matrix_petsc.mult(input_vec, rhs_vec)
            y_vec = prepared.operator_matrix.createVecLeft()
            spectral_ksp = _make_transformed_ksp(
                prepared.operator_matrix,
                rtol=rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                compute_singular_values=True,
            )
            cleanup_objects.extend([input_vec, rhs_vec, y_vec, spectral_ksp])
            return _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, y_vec)
        if prepared.mode == "transformed_shell":
            rhs_gpu = prepared.shell_context.transformed_rhs(b_gpu)
            rhs_vec = prepared.operator_matrix.createVecRight()
            with _vec_cuda_array(rhs_vec, n, mode="w") as rhs_arr:
                rhs_arr[...] = rhs_gpu
            y_vec = prepared.operator_matrix.createVecLeft()
            spectral_ksp = _make_transformed_ksp(
                prepared.operator_matrix,
                rtol=rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                compute_singular_values=True,
            )
            cleanup_objects.extend([rhs_vec, y_vec, spectral_ksp])
            return _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, y_vec)
        if prepared.mode == "factor_pc":
            rhs_vec = _new_cuda_vec(n, b_gpu)
            x_vec = _new_cuda_vec(n)
            spectral_ksp, _pc, _impl, _norm_type = _configure_factorized_pc_ksp(
                prepared.petsc_matrix,
                factor_gpu=prepared.factor_matrix_gpu,
                factor_petsc=prepared.factor_matrix_petsc,
                rtol=rtol,
                maxiter=maxiter,
                ksp_norm_type=prepared.ksp_norm_type,
                pc_impl=factor_pc_impl,
                compute_singular_values=True,
            )
            cleanup_objects.extend([rhs_vec, x_vec, spectral_ksp])
            return _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, x_vec)
        rhs_vec = _new_cuda_vec(n, b_gpu)
        x_vec = _new_cuda_vec(n)
        (
            spectral_ksp,
            _pc,
            _impl,
            _solver_type,
            _ordering,
            _shift_type,
        ) = _configure_classical_ksp(
            prepared.petsc_matrix,
            method=prepared.method,
            rtol=rtol,
            maxiter=maxiter,
            ssor_omega=ssor_omega,
            petsc_amg_backend=petsc_amg_backend,
            petsc_ic0_shift_type=prepared.ic0_shift_type or "none",
            ksp_norm_type=prepared.ksp_norm_type,
            petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
            compute_singular_values=True,
        )
        cleanup_objects.extend([rhs_vec, x_vec, spectral_ksp])
        return _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, x_vec)
    finally:
        for obj in cleanup_objects:
            _destroy_petsc_object(obj)


def _petsc_failed_rows(
    sample: "BenchmarkSample",
    *,
    method: str,
    model_name: str,
    model_label: str,
    is_learning_model: bool,
    learning_diag_strategy: str,
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
) -> list[Any]:
    from dpcg import benchmark as benchmark_mod

    rows: list[Any] = []
    timing_fields = dict(timing_fields or {})
    for rtol_value in rtol_values:
        rows.append(
            benchmark_mod.MethodResult(
                sample_id=sample.sample_id,
                split_name=sample.metadata.get("split"),
                family=sample.metadata.get("family"),
                sampling_mode=sample.metadata.get("sampling_mode"),
                method=method,
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
                backend="petsc_gpu",
                total_time_basis="setup_plus_solve",
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
                setup_measure_repeats=benchmark_mod._SETUP_MEASURE_REPEATS,
                apply_measure_repeats=benchmark_mod._APPLY_MEASURE_REPEATS,
                solve_measure_repeats=benchmark_mod._SOLVE_MEASURE_REPEATS,
                timing_stable=timing_stable,
                burnin_sample_id=burnin_sample_id,
                model_name=model_name,
                model_label=model_label,
                is_learning_model=is_learning_model,
                diag_strategy=learning_diag_strategy if method == "learning" else "",
                rtol=float(rtol_value),
                rtol_label=benchmark_mod._normalize_rtol_label(float(rtol_value)),
                **timing_fields,
            )
        )
    return rows


def run_method_burnin(
    sample: "BenchmarkSample",
    *,
    method: str,
    model,
    learning_output_kind: str,
    learning_diag_strategy: str,
    use_mask_projection: bool,
    learning_device: str,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    learning_group_context: dict[str, Any] | None,
    steady_forward_repeats: int,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str = "none",
    petsc_options: str | None,
    petsc_ksp_norm_type: str,
    petsc_sgs_ssor_impl: str,
    petsc_factor_solve_mode: str,
    petsc_factor_operator_mode: str,
    petsc_learning_internal_rtol_ratio: float,
    petsc_learning_pc_impl: str,
) -> None:
    from dpcg import benchmark as benchmark_mod

    with _petsc_options_scope(petsc_options):
        benchmark_mod._reset_learning_group_context(learning_group_context)
        prepared = prepare_method(
            sample,
            method=method,
            model=model,
            learning_output_kind=learning_output_kind,
            learning_diag_strategy=learning_diag_strategy,
            use_mask_projection=use_mask_projection,
            learning_device=learning_device,
            ssor_omega=ssor_omega,
            petsc_amg_backend=petsc_amg_backend,
            petsc_ic0_shift_type=petsc_ic0_shift_type,
            rtol=rtol,
            maxiter=maxiter,
            learning_group_context=learning_group_context,
            steady_forward_repeats=steady_forward_repeats,
            petsc_ksp_norm_type=petsc_ksp_norm_type,
            petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
            petsc_factor_solve_mode=petsc_factor_solve_mode,
            petsc_factor_operator_mode=petsc_factor_operator_mode,
            petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
            petsc_learning_pc_impl=petsc_learning_pc_impl,
        )
        try:
            apply_runner = _make_petsc_apply_runner(sample, prepared)
            benchmark_mod._adaptive_warmup(
                apply_runner,
                policy=benchmark_mod._apply_timing_policy(),
                sync=_cuda_sync,
            )

            def solve_runner():
                return _run_petsc_solve_core(
                    sample,
                    prepared,
                    rtol=rtol,
                    maxiter=maxiter,
                    ssor_omega=ssor_omega,
                    petsc_amg_backend=petsc_amg_backend,
                    petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                    factor_pc_impl=_factor_pc_impl_for_method(
                        method,
                        petsc_learning_pc_impl=petsc_learning_pc_impl,
                    ),
                )

            benchmark_mod._adaptive_warmup(
                solve_runner,
                policy=benchmark_mod._solve_timing_policy(),
                elapsed_resolver=lambda item: item["solve_time_sec"],
            )
        finally:
            _destroy_prepared_method(prepared)


def benchmark_method_group(
    samples: Sequence["BenchmarkSample"],
    *,
    method: str,
    model,
    model_name: str,
    model_label: str,
    is_learning_model: bool,
    learning_output_kind: str,
    learning_diag_strategy: str,
    use_mask_projection: bool,
    learning_device: str,
    rtol_values: Sequence[float],
    maxiter: Any,
    ssor_omega: float,
    residual_dir: Path | None,
    learning_group_context: dict[str, Any] | None,
    steady_forward_repeats: int,
    petsc_amg_backend: str,
    petsc_ic0_shift_type: str = "none",
    petsc_options: str | None,
    petsc_ksp_norm_type: str,
    petsc_sgs_ssor_impl: str,
    petsc_factor_solve_mode: str,
    petsc_factor_operator_mode: str,
    petsc_cond_mode: str,
    petsc_enable_approx_spectral_diagnostics: bool,
    petsc_apply_measure_repeats: int,
    petsc_learning_internal_rtol_ratio: float,
    petsc_learning_pc_impl: str,
    burnin_sample_id: str | None,
) -> list[Any]:
    from dpcg import benchmark as benchmark_mod

    results: list[Any] = []
    with _petsc_options_scope(petsc_options):
        for sample in samples:
            operator_nnz, operator_density = _matrix_stats(sample.A.tocsr().astype(np.float64))

            def build_prepared():
                benchmark_mod._reset_learning_group_context(learning_group_context)
                return prepare_method(
                    sample,
                    method=method,
                    model=model,
                    learning_output_kind=learning_output_kind,
                    learning_diag_strategy=learning_diag_strategy,
                    use_mask_projection=use_mask_projection,
                    learning_device=learning_device,
                    ssor_omega=ssor_omega,
                    petsc_amg_backend=petsc_amg_backend,
                    petsc_ic0_shift_type=petsc_ic0_shift_type,
                    rtol=max(float(item) for item in rtol_values),
                    maxiter=benchmark_mod._resolve_runtime_maxiter(
                        maxiter,
                        max(float(item) for item in rtol_values),
                    ),
                    learning_group_context=learning_group_context,
                    steady_forward_repeats=steady_forward_repeats,
                    petsc_ksp_norm_type=petsc_ksp_norm_type,
                    petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                    petsc_factor_solve_mode=petsc_factor_solve_mode,
                    petsc_factor_operator_mode=petsc_factor_operator_mode,
                    petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
                    petsc_learning_pc_impl=petsc_learning_pc_impl,
                )

            try:
                cold_prepared, cold_setup_time_sec = _prepare_method_once(build_prepared)
                cold_timing_fields = _petsc_result_timing_fields(cold_prepared)
                _destroy_prepared_method(cold_prepared)
            except PetscBenchmarkSkip as exc:
                results.extend(
                    _petsc_failed_rows(
                        sample,
                        method=method,
                        model_name=model_name,
                        model_label=model_label,
                        is_learning_model=is_learning_model,
                        learning_diag_strategy=learning_diag_strategy,
                        rtol_values=rtol_values,
                        message=str(exc),
                        status="skipped",
                        operator_nnz=operator_nnz or 0.0,
                        operator_density=operator_density or 0.0,
                        burnin_sample_id=burnin_sample_id,
                    )
                )
                continue
            except Exception as exc:
                results.extend(
                    _petsc_failed_rows(
                        sample,
                        method=method,
                        model_name=model_name,
                        model_label=model_label,
                        is_learning_model=is_learning_model,
                        learning_diag_strategy=learning_diag_strategy,
                        rtol_values=rtol_values,
                        message=str(exc),
                        status="failed",
                        operator_nnz=operator_nnz or 0.0,
                        operator_density=operator_density or 0.0,
                        burnin_sample_id=burnin_sample_id,
                    )
                )
                continue

            def build_and_destroy() -> float:
                prepared, setup_time_sec = _prepare_method_once(build_prepared)
                _destroy_prepared_method(prepared)
                return setup_time_sec

            try:
                setup_warmup_runs, setup_stable = benchmark_mod._adaptive_warmup(
                    build_and_destroy,
                    policy=benchmark_mod._setup_timing_policy(),
                    sync=_cuda_sync,
                    elapsed_resolver=lambda value: value,
                )
                setup_observations: list[float] = []
                timing_observations: list[dict[str, float | None]] = []
                for _ in range(benchmark_mod._SETUP_MEASURE_REPEATS):
                    measured_prepared, setup_time_sec = _prepare_method_once(build_prepared)
                    setup_observations.append(setup_time_sec)
                    timing_observations.append(_petsc_result_timing_fields(measured_prepared))
                    _destroy_prepared_method(measured_prepared)
                active_prepared = build_prepared()
            except Exception as exc:
                results.extend(
                    _petsc_failed_rows(
                        sample,
                        method=method,
                        model_name=model_name,
                        model_label=model_label,
                        is_learning_model=is_learning_model,
                        learning_diag_strategy=learning_diag_strategy,
                        rtol_values=rtol_values,
                        message=str(exc),
                        status="failed",
                        operator_nnz=operator_nnz or 0.0,
                        operator_density=operator_density or 0.0,
                        burnin_sample_id=burnin_sample_id,
                        setup_cold_time_sec=cold_setup_time_sec,
                        factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                            "factor_gpu_to_host_time_sec"
                        ),
                        factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                            "factor_petsc_build_time_sec"
                        ),
                        a_petsc_build_cold_time_sec=cold_timing_fields.get(
                            "a_petsc_build_time_sec"
                        ),
                        solve_ready_build_cold_time_sec=cold_timing_fields.get(
                            "solve_ready_build_time_sec"
                        ),
                        excluded_materialization_cold_time_sec=cold_timing_fields.get(
                            "excluded_materialization_time_sec"
                        ),
                        setup_warmup_runs=setup_warmup_runs,
                        timing_stable=setup_stable,
                    )
                )
                continue

            setup_time_sec = float(np.median(np.asarray(setup_observations, dtype=np.float64)))
            setup_cold_time_sec = cold_setup_time_sec
            timing_fields = benchmark_mod._median_timing_fields(timing_observations)
            apply_runner = _make_petsc_apply_runner(sample, active_prepared)
            try:
                apply_cold_time_sec = benchmark_mod._timed_call(
                    apply_runner,
                    sync=_cuda_sync,
                ).elapsed_sec
                apply_warmup_runs, apply_stable = benchmark_mod._adaptive_warmup(
                    apply_runner,
                    policy=benchmark_mod._apply_timing_policy(),
                    sync=_cuda_sync,
                )
                apply_observations = benchmark_mod._measure_repeated_calls(
                    apply_runner,
                    repeats=benchmark_mod._APPLY_MEASURE_REPEATS,
                    sync=_cuda_sync,
                )
                apply_time_sec = benchmark_mod._median_elapsed(apply_observations)
            except Exception as exc:
                _destroy_prepared_method(active_prepared)
                results.extend(
                    _petsc_failed_rows(
                        sample,
                        method=method,
                        model_name=model_name,
                        model_label=model_label,
                        is_learning_model=is_learning_model,
                        learning_diag_strategy=learning_diag_strategy,
                        rtol_values=rtol_values,
                        message=str(exc),
                        status="failed",
                        operator_nnz=operator_nnz or 0.0,
                        operator_density=operator_density or 0.0,
                        burnin_sample_id=burnin_sample_id,
                        setup_cold_time_sec=setup_cold_time_sec,
                        setup_time_sec=setup_time_sec,
                        factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                            "factor_gpu_to_host_time_sec"
                        ),
                        factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                            "factor_petsc_build_time_sec"
                        ),
                        a_petsc_build_cold_time_sec=cold_timing_fields.get(
                            "a_petsc_build_time_sec"
                        ),
                        solve_ready_build_cold_time_sec=cold_timing_fields.get(
                            "solve_ready_build_time_sec"
                        ),
                        excluded_materialization_cold_time_sec=cold_timing_fields.get(
                            "excluded_materialization_time_sec"
                        ),
                        timing_fields=timing_fields,
                        setup_warmup_runs=setup_warmup_runs,
                        timing_stable=False,
                    )
                )
                continue

            total_time_basis = "setup_plus_solve"
            try:
                ref_metrics, cond_ref_time_sec, cond_ref_backend = (
                    _compute_reference_condition_metrics_timed(
                        sample,
                        active_prepared,
                        cond_mode=petsc_cond_mode,
                    )
                )
            except Exception as exc:
                _destroy_prepared_method(active_prepared)
                results.extend(
                    _petsc_failed_rows(
                        sample,
                        method=method,
                        model_name=model_name,
                        model_label=model_label,
                        is_learning_model=is_learning_model,
                        learning_diag_strategy=learning_diag_strategy,
                        rtol_values=rtol_values,
                        message=str(exc),
                        status="failed",
                        operator_nnz=operator_nnz or 0.0,
                        operator_density=operator_density or 0.0,
                        burnin_sample_id=burnin_sample_id,
                        setup_cold_time_sec=setup_cold_time_sec,
                        setup_time_sec=setup_time_sec,
                        apply_cold_time_sec=apply_cold_time_sec,
                        factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                            "factor_gpu_to_host_time_sec"
                        ),
                        factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                            "factor_petsc_build_time_sec"
                        ),
                        a_petsc_build_cold_time_sec=cold_timing_fields.get(
                            "a_petsc_build_time_sec"
                        ),
                        solve_ready_build_cold_time_sec=cold_timing_fields.get(
                            "solve_ready_build_time_sec"
                        ),
                        excluded_materialization_cold_time_sec=cold_timing_fields.get(
                            "excluded_materialization_time_sec"
                        ),
                        timing_fields=timing_fields,
                        setup_warmup_runs=setup_warmup_runs,
                        apply_warmup_runs=apply_warmup_runs,
                        solve_warmup_runs=0,
                        timing_stable=False,
                    )
                )
                continue
            for rtol_value in rtol_values:
                runtime_maxiter = benchmark_mod._resolve_runtime_maxiter(
                    maxiter,
                    float(rtol_value),
                )
                learning_internal_rtol_for_row = (
                    _resolve_learning_internal_rtol(
                        active_prepared,
                        rtol=float(rtol_value),
                    )
                    if active_prepared.mode == "transformed_learning_native"
                    else active_prepared.learning_transformed_internal_rtol
                )
                solve_residual_dir = benchmark_mod._make_residual_dir_for_rtol(
                    residual_dir,
                    rtol_label=benchmark_mod._normalize_rtol_label(float(rtol_value)),
                    multi_rtol=len(rtol_values) > 1,
                )
                factor_pc_impl = _factor_pc_impl_for_method(
                    method,
                    petsc_learning_pc_impl=petsc_learning_pc_impl,
                )

                def solve_runner():
                    return _run_petsc_solve_core(
                        sample,
                        active_prepared,
                        rtol=float(rtol_value),
                        maxiter=runtime_maxiter,
                        ssor_omega=ssor_omega,
                        petsc_amg_backend=petsc_amg_backend,
                        petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                        factor_pc_impl=factor_pc_impl,
                    )

                try:
                    solve_warmup_runs, solve_stable = benchmark_mod._adaptive_warmup(
                        solve_runner,
                        policy=benchmark_mod._solve_timing_policy(),
                        elapsed_resolver=lambda item: item["solve_time_sec"],
                    )
                    solve_observations = benchmark_mod._measure_repeated_calls(
                        solve_runner,
                        repeats=benchmark_mod._SOLVE_MEASURE_REPEATS,
                        elapsed_resolver=lambda item: item["solve_time_sec"],
                    )
                    diagnostic_solve = _run_petsc_solve_core(
                        sample,
                        active_prepared,
                        rtol=float(rtol_value),
                        maxiter=runtime_maxiter,
                        ssor_omega=ssor_omega,
                        petsc_amg_backend=petsc_amg_backend,
                        petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                        factor_pc_impl=factor_pc_impl,
                    )
                    spectral_solve_time_sec = None
                    sigma_max_approx = None
                    sigma_min_approx = None
                    cond_est_approx = None
                    if _should_enable_approx_spectral_diagnostics(
                        method,
                        petsc_enable_approx_spectral_diagnostics,
                    ):
                        (
                            spectral_solve_time_sec,
                            sigma_max_approx,
                            sigma_min_approx,
                            cond_est_approx,
                        ) = _run_petsc_spectral_diagnostics(
                            sample,
                            active_prepared,
                            rtol=float(rtol_value),
                            maxiter=runtime_maxiter,
                            ssor_omega=ssor_omega,
                            petsc_amg_backend=petsc_amg_backend,
                            petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                            factor_pc_impl=factor_pc_impl,
                        )
                    residual_history_file = None
                    if solve_residual_dir is not None:
                        residual_history_file = benchmark_mod._save_residual_history(
                            solve_residual_dir,
                            sample.sample_id,
                            method,
                            diagnostic_solve["residuals"],
                        )
                    solve_time_sec = benchmark_mod._median_elapsed(solve_observations)
                    results.append(
                        benchmark_mod.MethodResult(
                            sample_id=sample.sample_id,
                            split_name=sample.metadata.get("split"),
                            family=sample.metadata.get("family"),
                            sampling_mode=sample.metadata.get("sampling_mode"),
                            method=method,
                            status=diagnostic_solve["status"],
                            setup_time_sec=setup_time_sec,
                            apply_time_sec=apply_time_sec,
                            solve_time_sec=solve_time_sec,
                            total_time_sec=float(setup_time_sec + solve_time_sec),
                            iterations=diagnostic_solve["iterations"],
                            info=diagnostic_solve["info"],
                            converged=diagnostic_solve["converged"],
                            relative_residual=diagnostic_solve["relative_residual"],
                            final_residual_norm=diagnostic_solve["final_residual_norm"],
                            ksp_reason=diagnostic_solve["ksp_reason"],
                            ksp_converged=diagnostic_solve["ksp_converged"],
                            true_residual_converged=diagnostic_solve["true_residual_converged"],
                            true_relative_residual=diagnostic_solve["true_relative_residual"],
                            true_final_residual_norm=diagnostic_solve["true_final_residual_norm"],
                            ksp_norm_type=active_prepared.ksp_norm_type,
                            ksp_monitor_initial_residual=diagnostic_solve[
                                "ksp_monitor_initial_residual"
                            ],
                            ksp_monitor_final_residual=diagnostic_solve[
                                "ksp_monitor_final_residual"
                            ],
                            factor_nnz=active_prepared.factor_nnz,
                            factor_density=active_prepared.factor_density,
                            operator_nnz=active_prepared.operator_nnz,
                            operator_density=active_prepared.operator_density,
                            apply_kind=active_prepared.apply_kind,
                            inference_peak_gpu_memory_mb=active_prepared.inference_peak_gpu_memory_mb,
                            cpu_rss_delta_mb=active_prepared.cpu_rss_delta_mb,
                            residual_history_file=residual_history_file,
                            cond_est_ref=ref_metrics["cond_est_ref"],
                            lambda_min_ref=ref_metrics["lambda_min_ref"],
                            lambda_max_ref=ref_metrics["lambda_max_ref"],
                            cond_est_method=ref_metrics["cond_est_method"],
                            cond_est_approx=cond_est_approx,
                            sigma_max_approx=sigma_max_approx,
                            sigma_min_approx=sigma_min_approx,
                            spectral_solve_time_sec=spectral_solve_time_sec,
                            cond_ref_time_sec=cond_ref_time_sec,
                            cond_ref_backend=cond_ref_backend,
                            message=diagnostic_solve["message"],
                            backend="petsc_gpu",
                            matrix_type=active_prepared.matrix_type,
                            vector_type=active_prepared.vector_type,
                            operator_matrix_type=active_prepared.operator_matrix_type,
                            factor_matrix_type=active_prepared.factor_matrix_type,
                            preconditioner_impl=active_prepared.preconditioner_impl,
                            resolved_pc_type=active_prepared.resolved_pc_type,
                            resolved_factor_solver_type=active_prepared.resolved_factor_solver_type,
                            ic0_shift_type=active_prepared.ic0_shift_type,
                            solve_operator_mode=active_prepared.solve_operator_mode,
                            learning_transformed_internal_rtol=learning_internal_rtol_for_row,
                            learning_convergence_basis=active_prepared.learning_convergence_basis,
                            total_time_basis=total_time_basis,
                            setup_cold_time_sec=setup_cold_time_sec,
                            apply_cold_time_sec=apply_cold_time_sec,
                            factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                                "factor_gpu_to_host_time_sec"
                            ),
                            factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                                "factor_petsc_build_time_sec"
                            ),
                            a_petsc_build_cold_time_sec=cold_timing_fields.get(
                                "a_petsc_build_time_sec"
                            ),
                            solve_ready_build_cold_time_sec=cold_timing_fields.get(
                                "solve_ready_build_time_sec"
                            ),
                            excluded_materialization_cold_time_sec=cold_timing_fields.get(
                                "excluded_materialization_time_sec"
                            ),
                            setup_warmup_runs=setup_warmup_runs,
                            apply_warmup_runs=apply_warmup_runs,
                            solve_warmup_runs=solve_warmup_runs,
                            setup_measure_repeats=benchmark_mod._SETUP_MEASURE_REPEATS,
                            apply_measure_repeats=benchmark_mod._APPLY_MEASURE_REPEATS,
                            solve_measure_repeats=benchmark_mod._SOLVE_MEASURE_REPEATS,
                            timing_stable=bool(setup_stable and apply_stable and solve_stable),
                            burnin_sample_id=burnin_sample_id,
                            model_name=model_name,
                            model_label=model_label,
                            is_learning_model=is_learning_model,
                            diag_strategy=learning_diag_strategy if method == "learning" else "",
                            rtol=float(rtol_value),
                            rtol_label=benchmark_mod._normalize_rtol_label(float(rtol_value)),
                            **timing_fields,
                        )
                    )
                except Exception as exc:
                    results.extend(
                        _petsc_failed_rows(
                            sample,
                            method=method,
                            model_name=model_name,
                            model_label=model_label,
                            is_learning_model=is_learning_model,
                            learning_diag_strategy=learning_diag_strategy,
                            rtol_values=[float(rtol_value)],
                            message=str(exc),
                            status="failed",
                            operator_nnz=operator_nnz or 0.0,
                            operator_density=operator_density or 0.0,
                            burnin_sample_id=burnin_sample_id,
                            setup_cold_time_sec=setup_cold_time_sec,
                            setup_time_sec=setup_time_sec,
                            apply_cold_time_sec=apply_cold_time_sec,
                            factor_gpu_to_host_cold_time_sec=cold_timing_fields.get(
                                "factor_gpu_to_host_time_sec"
                            ),
                            factor_petsc_build_cold_time_sec=cold_timing_fields.get(
                                "factor_petsc_build_time_sec"
                            ),
                            a_petsc_build_cold_time_sec=cold_timing_fields.get(
                                "a_petsc_build_time_sec"
                            ),
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
            _destroy_prepared_method(active_prepared)
    return results


def _run_explicit_transformed_method(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    residual_dir: Path | None,
    petsc_cond_mode: str,
    petsc_enable_approx_spectral_diagnostics: bool,
    petsc_apply_warmup_runs: int,
    petsc_apply_measure_repeats: int,
):
    from dpcg import benchmark as benchmark_mod

    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    input_vec = _new_cuda_vec(n, b_gpu)
    apply_work_vec = prepared.factor_matrix_petsc.createVecRight()
    apply_output_vec = prepared.factor_matrix_petsc.createVecLeft()
    apply_time_sec = _measure_factor_apply(
        prepared.factor_matrix_petsc,
        input_vec,
        apply_work_vec,
        apply_output_vec,
        warmup_runs=petsc_apply_warmup_runs,
        repeats=petsc_apply_measure_repeats,
    )
    rhs_vec = prepared.operator_matrix.createVecRight()
    prepared.factor_matrix_petsc.mult(input_vec, rhs_vec)
    y_vec = prepared.operator_matrix.createVecLeft()
    ksp = _make_transformed_ksp(
        prepared.operator_matrix,
        rtol=rtol,
        maxiter=maxiter,
        ksp_norm_type=prepared.ksp_norm_type,
        compute_singular_values=False,
    )
    solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
        ksp,
        rhs_vec,
        y_vec,
        compute_singular_values=False,
    )
    x_vec = prepared.factor_matrix_petsc.createVecRight()
    prepared.factor_matrix_petsc.multTranspose(y_vec, x_vec)
    x_gpu = _copy_solution_from_vec(x_vec, n)
    true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
    ksp_converged = reason > 0
    true_residual_converged = bool(true_relative_residual <= float(rtol))
    spectral_solve_time_sec = None
    sigma_max_approx = None
    sigma_min_approx = None
    cond_est_approx = None
    if petsc_enable_approx_spectral_diagnostics:
        spectral_ksp = _make_transformed_ksp(
            prepared.operator_matrix,
            rtol=rtol,
            maxiter=maxiter,
            ksp_norm_type=prepared.ksp_norm_type,
            compute_singular_values=True,
        )
        spectral_solve_time_sec, sigma_max_approx, sigma_min_approx, cond_est_approx = (
            _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, y_vec)
        )
    residual_history_file = None
    if residual_dir is not None:
        residual_history_file = benchmark_mod._save_residual_history(
            residual_dir,
            sample.sample_id,
            prepared.method,
            residuals,
        )
    ref_metrics, cond_ref_time_sec, cond_ref_backend = _compute_reference_condition_metrics_timed(
        sample,
        prepared,
        cond_mode=petsc_cond_mode,
    )
    status, message = _resolve_run_status(
        ksp_converged=ksp_converged,
        true_residual_converged=true_residual_converged,
    )
    return {
        "status": status,
        "message": message,
        "apply_time_sec": apply_time_sec,
        "solve_time_sec": solve_time_sec,
        "spectral_solve_time_sec": spectral_solve_time_sec,
        "cond_ref_time_sec": cond_ref_time_sec,
        "cond_ref_backend": cond_ref_backend,
        "iterations": iterations,
        "info": reason,
        "ksp_reason": reason,
        "ksp_converged": ksp_converged,
        "true_residual_converged": true_residual_converged,
        "converged": true_residual_converged,
        "relative_residual": true_relative_residual,
        "final_residual_norm": true_final_residual_norm,
        "true_relative_residual": true_relative_residual,
        "true_final_residual_norm": true_final_residual_norm,
        "ksp_monitor_initial_residual": None if not residuals else residuals[0],
        "ksp_monitor_final_residual": None if not residuals else residuals[-1],
        "residual_history_file": residual_history_file,
        "cond_est": cond_est_approx,
        "sigma_max": sigma_max_approx,
        "sigma_min": sigma_min_approx,
        "cond_est_approx": cond_est_approx,
        "sigma_max_approx": sigma_max_approx,
        "sigma_min_approx": sigma_min_approx,
        **ref_metrics,
    }


def _run_learning_transformed_native_method(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    residual_dir: Path | None,
    petsc_cond_mode: str,
    petsc_enable_approx_spectral_diagnostics: bool,
    petsc_apply_warmup_runs: int,
    petsc_apply_measure_repeats: int,
):
    from dpcg import benchmark as benchmark_mod

    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    input_vec = _new_cuda_vec(n, b_gpu)
    apply_work_vec = prepared.factor_matrix_petsc.createVecRight()
    apply_output_vec = prepared.factor_matrix_petsc.createVecLeft()
    apply_time_sec = _measure_factor_apply(
        prepared.factor_matrix_petsc,
        input_vec,
        apply_work_vec,
        apply_output_vec,
        warmup_runs=petsc_apply_warmup_runs,
        repeats=petsc_apply_measure_repeats,
    )
    rhs_vec = prepared.operator_matrix.createVecRight()
    prepared.factor_matrix_petsc.mult(input_vec, rhs_vec)
    y_vec = prepared.operator_matrix.createVecLeft()
    internal_rtol = _resolve_learning_internal_rtol(prepared, rtol=rtol)
    ksp = _make_transformed_ksp(
        prepared.operator_matrix,
        rtol=internal_rtol,
        maxiter=maxiter,
        ksp_norm_type=prepared.ksp_norm_type,
        compute_singular_values=False,
    )
    solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
        ksp,
        rhs_vec,
        y_vec,
        compute_singular_values=False,
    )
    x_vec = prepared.factor_matrix_petsc.createVecRight()
    prepared.factor_matrix_petsc.multTranspose(y_vec, x_vec)
    x_gpu = _copy_solution_from_vec(x_vec, n)
    true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
    ksp_converged = reason > 0
    true_residual_converged = bool(true_relative_residual <= float(rtol))
    spectral_solve_time_sec = None
    sigma_max_approx = None
    sigma_min_approx = None
    cond_est_approx = None
    if petsc_enable_approx_spectral_diagnostics:
        spectral_ksp = _make_transformed_ksp(
            prepared.operator_matrix,
            rtol=internal_rtol,
            maxiter=maxiter,
            ksp_norm_type=prepared.ksp_norm_type,
            compute_singular_values=True,
        )
        spectral_solve_time_sec, sigma_max_approx, sigma_min_approx, cond_est_approx = (
            _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, y_vec)
        )
    residual_history_file = None
    if residual_dir is not None:
        residual_history_file = benchmark_mod._save_residual_history(
            residual_dir,
            sample.sample_id,
            prepared.method,
            residuals,
        )
    ref_metrics, cond_ref_time_sec, cond_ref_backend = _compute_reference_condition_metrics_timed(
        sample,
        prepared,
        cond_mode=petsc_cond_mode,
    )
    status = "ok" if ksp_converged else "failed"
    message = (
        None
        if ksp_converged
        else "Transformed KSP did not converge to the requested internal tolerance"
    )
    return {
        "status": status,
        "message": message,
        "apply_time_sec": apply_time_sec,
        "solve_time_sec": solve_time_sec,
        "spectral_solve_time_sec": spectral_solve_time_sec,
        "cond_ref_time_sec": cond_ref_time_sec,
        "cond_ref_backend": cond_ref_backend,
        "iterations": iterations,
        "info": reason,
        "ksp_reason": reason,
        "ksp_converged": ksp_converged,
        "true_residual_converged": true_residual_converged,
        "converged": ksp_converged,
        "relative_residual": true_relative_residual,
        "final_residual_norm": true_final_residual_norm,
        "true_relative_residual": true_relative_residual,
        "true_final_residual_norm": true_final_residual_norm,
        "ksp_monitor_initial_residual": None if not residuals else residuals[0],
        "ksp_monitor_final_residual": None if not residuals else residuals[-1],
        "residual_history_file": residual_history_file,
        "cond_est": cond_est_approx,
        "sigma_max": sigma_max_approx,
        "sigma_min": sigma_min_approx,
        "cond_est_approx": cond_est_approx,
        "sigma_max_approx": sigma_max_approx,
        "sigma_min_approx": sigma_min_approx,
        **ref_metrics,
    }


def _run_factor_pc_method(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    residual_dir: Path | None,
    petsc_cond_mode: str,
    petsc_enable_approx_spectral_diagnostics: bool,
    petsc_apply_warmup_runs: int,
    petsc_apply_measure_repeats: int,
    factor_pc_impl: str,
):
    from dpcg import benchmark as benchmark_mod

    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    rhs_vec = prepared.petsc_matrix.createVecRight()
    with _vec_cuda_array(rhs_vec, n, mode="w") as rhs_arr:
        rhs_arr[...] = b_gpu
    apply_vec = prepared.petsc_matrix.createVecRight()
    apply_vec.set(0.0)
    apply_time_sec = _measure_pc_apply(
        prepared.pc,
        rhs_vec,
        apply_vec,
        warmup_runs=petsc_apply_warmup_runs,
        repeats=petsc_apply_measure_repeats,
    )
    x_vec = prepared.petsc_matrix.createVecRight()
    solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
        prepared.ksp,
        rhs_vec,
        x_vec,
        compute_singular_values=False,
    )
    x_gpu = _copy_solution_from_vec(x_vec, n)
    true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
    ksp_converged = reason > 0
    true_residual_converged = bool(true_relative_residual <= float(rtol))
    spectral_solve_time_sec = None
    sigma_max_approx = None
    sigma_min_approx = None
    cond_est_approx = None
    if petsc_enable_approx_spectral_diagnostics:
        spectral_ksp, _spectral_pc, _impl, _norm_type = _configure_factorized_pc_ksp(
            prepared.petsc_matrix,
            factor_gpu=prepared.factor_matrix_gpu,
            factor_petsc=prepared.factor_matrix_petsc,
            rtol=rtol,
            maxiter=maxiter,
            ksp_norm_type=prepared.ksp_norm_type,
            pc_impl=factor_pc_impl,
            compute_singular_values=True,
        )
        spectral_solve_time_sec, sigma_max_approx, sigma_min_approx, cond_est_approx = (
            _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, x_vec)
        )
    residual_history_file = None
    if residual_dir is not None:
        residual_history_file = benchmark_mod._save_residual_history(
            residual_dir,
            sample.sample_id,
            prepared.method,
            residuals,
        )
    ref_metrics, cond_ref_time_sec, cond_ref_backend = _compute_reference_condition_metrics_timed(
        sample,
        prepared,
        cond_mode=petsc_cond_mode,
    )
    status, message = _resolve_run_status(
        ksp_converged=ksp_converged,
        true_residual_converged=true_residual_converged,
    )
    return {
        "status": status,
        "message": message,
        "apply_time_sec": apply_time_sec,
        "solve_time_sec": solve_time_sec,
        "spectral_solve_time_sec": spectral_solve_time_sec,
        "cond_ref_time_sec": cond_ref_time_sec,
        "cond_ref_backend": cond_ref_backend,
        "iterations": iterations,
        "info": reason,
        "ksp_reason": reason,
        "ksp_converged": ksp_converged,
        "true_residual_converged": true_residual_converged,
        "converged": true_residual_converged,
        "relative_residual": true_relative_residual,
        "final_residual_norm": true_final_residual_norm,
        "true_relative_residual": true_relative_residual,
        "true_final_residual_norm": true_final_residual_norm,
        "ksp_monitor_initial_residual": None if not residuals else residuals[0],
        "ksp_monitor_final_residual": None if not residuals else residuals[-1],
        "residual_history_file": residual_history_file,
        "cond_est": cond_est_approx,
        "sigma_max": sigma_max_approx,
        "sigma_min": sigma_min_approx,
        "cond_est_approx": cond_est_approx,
        "sigma_max_approx": sigma_max_approx,
        "sigma_min_approx": sigma_min_approx,
        **ref_metrics,
    }


def _run_shell_transformed_method(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    residual_dir: Path | None,
    petsc_cond_mode: str,
    petsc_enable_approx_spectral_diagnostics: bool,
    petsc_apply_warmup_runs: int,
    petsc_apply_measure_repeats: int,
):
    from dpcg import benchmark as benchmark_mod

    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    apply_time_sec = _measure_shell_apply(
        prepared.shell_context,
        b_gpu,
        warmup_runs=petsc_apply_warmup_runs,
        repeats=petsc_apply_measure_repeats,
    )
    rhs_gpu = prepared.shell_context.transformed_rhs(b_gpu)
    rhs_vec = prepared.operator_matrix.createVecRight()
    with _vec_cuda_array(rhs_vec, n, mode="w") as rhs_arr:
        rhs_arr[...] = rhs_gpu
    y_vec = prepared.operator_matrix.createVecLeft()
    ksp = _make_transformed_ksp(
        prepared.operator_matrix,
        rtol=rtol,
        maxiter=maxiter,
        ksp_norm_type=prepared.ksp_norm_type,
        compute_singular_values=False,
    )
    solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
        ksp,
        rhs_vec,
        y_vec,
        compute_singular_values=False,
    )
    y_gpu = _copy_solution_from_vec(y_vec, n)
    x_gpu = prepared.shell_context.recover_solution(y_gpu)
    true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
    ksp_converged = reason > 0
    true_residual_converged = bool(true_relative_residual <= float(rtol))
    spectral_solve_time_sec = None
    sigma_max_approx = None
    sigma_min_approx = None
    cond_est_approx = None
    if petsc_enable_approx_spectral_diagnostics:
        spectral_ksp = _make_transformed_ksp(
            prepared.operator_matrix,
            rtol=rtol,
            maxiter=maxiter,
            ksp_norm_type=prepared.ksp_norm_type,
            compute_singular_values=True,
        )
        spectral_solve_time_sec, sigma_max_approx, sigma_min_approx, cond_est_approx = (
            _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, y_vec)
        )
    residual_history_file = None
    if residual_dir is not None:
        residual_history_file = benchmark_mod._save_residual_history(
            residual_dir,
            sample.sample_id,
            prepared.method,
            residuals,
        )
    ref_metrics, cond_ref_time_sec, cond_ref_backend = _compute_reference_condition_metrics_timed(
        sample,
        prepared,
        cond_mode=petsc_cond_mode,
    )
    status, message = _resolve_run_status(
        ksp_converged=ksp_converged,
        true_residual_converged=true_residual_converged,
    )
    return {
        "status": status,
        "message": message,
        "apply_time_sec": apply_time_sec,
        "solve_time_sec": solve_time_sec,
        "spectral_solve_time_sec": spectral_solve_time_sec,
        "cond_ref_time_sec": cond_ref_time_sec,
        "cond_ref_backend": cond_ref_backend,
        "iterations": iterations,
        "info": reason,
        "ksp_reason": reason,
        "ksp_converged": ksp_converged,
        "true_residual_converged": true_residual_converged,
        "converged": true_residual_converged,
        "relative_residual": true_relative_residual,
        "final_residual_norm": true_final_residual_norm,
        "true_relative_residual": true_relative_residual,
        "true_final_residual_norm": true_final_residual_norm,
        "ksp_monitor_initial_residual": None if not residuals else residuals[0],
        "ksp_monitor_final_residual": None if not residuals else residuals[-1],
        "residual_history_file": residual_history_file,
        "cond_est": cond_est_approx,
        "sigma_max": sigma_max_approx,
        "sigma_min": sigma_min_approx,
        "cond_est_approx": cond_est_approx,
        "sigma_max_approx": sigma_max_approx,
        "sigma_min_approx": sigma_min_approx,
        **ref_metrics,
    }


def _run_classical_method(
    sample: "BenchmarkSample",
    prepared: PetscPreparedMethod,
    *,
    rtol: float,
    maxiter: int | None,
    ssor_omega: float,
    petsc_amg_backend: str,
    residual_dir: Path | None,
    petsc_enable_approx_spectral_diagnostics: bool,
    petsc_apply_warmup_runs: int,
    petsc_apply_measure_repeats: int,
    petsc_sgs_ssor_impl: str,
    petsc_cond_mode: str,
):
    from dpcg import benchmark as benchmark_mod

    n = int(sample.A.shape[0])
    b_gpu = cp.asarray(np.asarray(sample.b, dtype=np.float64).reshape(-1))
    rhs_vec = prepared.petsc_matrix.createVecRight()
    with _vec_cuda_array(rhs_vec, n, mode="w") as rhs_arr:
        rhs_arr[...] = b_gpu
    apply_vec = prepared.petsc_matrix.createVecRight()
    apply_vec.set(0.0)
    apply_time_sec = _measure_pc_apply(
        prepared.pc,
        rhs_vec,
        apply_vec,
        warmup_runs=petsc_apply_warmup_runs,
        repeats=petsc_apply_measure_repeats,
    )
    x_vec = prepared.petsc_matrix.createVecRight()
    solve_time_sec, residuals, iterations, reason, _, _, _ = _run_ksp(
        prepared.ksp,
        rhs_vec,
        x_vec,
        compute_singular_values=False,
    )
    x_gpu = _copy_solution_from_vec(x_vec, n)
    true_final_residual_norm, true_relative_residual = _true_residual_stats(sample, x_gpu)
    ksp_converged = reason > 0
    true_residual_converged = bool(true_relative_residual <= float(rtol))
    spectral_solve_time_sec = None
    sigma_max_approx = None
    sigma_min_approx = None
    cond_est_approx = None
    if petsc_enable_approx_spectral_diagnostics:
        (
            spectral_ksp,
            _spectral_pc,
            _impl,
            _solver_type,
            _ordering,
            _shift_type,
        ) = _configure_classical_ksp(
            prepared.petsc_matrix,
            method=prepared.method,
            rtol=rtol,
            maxiter=maxiter,
            ssor_omega=ssor_omega,
            petsc_amg_backend=petsc_amg_backend,
            petsc_ic0_shift_type=prepared.ic0_shift_type or "none",
            ksp_norm_type=prepared.ksp_norm_type,
            petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
            compute_singular_values=True,
        )
        spectral_solve_time_sec, sigma_max_approx, sigma_min_approx, cond_est_approx = (
            _run_ksp_spectral_diagnostics(spectral_ksp, rhs_vec, x_vec)
        )
    residual_history_file = None
    if residual_dir is not None:
        residual_history_file = benchmark_mod._save_residual_history(
            residual_dir,
            sample.sample_id,
            prepared.method,
            residuals,
        )
    ref_metrics, cond_ref_time_sec, cond_ref_backend = _compute_reference_condition_metrics_timed(
        sample,
        prepared,
        cond_mode=petsc_cond_mode,
    )
    status, message = _resolve_run_status(
        ksp_converged=ksp_converged,
        true_residual_converged=true_residual_converged,
    )
    return {
        "status": status,
        "message": message,
        "apply_time_sec": apply_time_sec,
        "solve_time_sec": solve_time_sec,
        "spectral_solve_time_sec": spectral_solve_time_sec,
        "cond_ref_time_sec": cond_ref_time_sec,
        "cond_ref_backend": cond_ref_backend,
        "iterations": iterations,
        "info": reason,
        "ksp_reason": reason,
        "ksp_converged": ksp_converged,
        "true_residual_converged": true_residual_converged,
        "converged": true_residual_converged,
        "relative_residual": true_relative_residual,
        "final_residual_norm": true_final_residual_norm,
        "true_relative_residual": true_relative_residual,
        "true_final_residual_norm": true_final_residual_norm,
        "ksp_monitor_initial_residual": None if not residuals else residuals[0],
        "ksp_monitor_final_residual": None if not residuals else residuals[-1],
        "residual_history_file": residual_history_file,
        "cond_est": cond_est_approx,
        "sigma_max": sigma_max_approx,
        "sigma_min": sigma_min_approx,
        "cond_est_approx": cond_est_approx,
        "sigma_max_approx": sigma_max_approx,
        "sigma_min_approx": sigma_min_approx,
        **ref_metrics,
    }


def benchmark_sample(
    sample: "BenchmarkSample",
    methods: Sequence[str],
    *,
    model=None,
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str = "cuda",
    rtol: float = 1.0e-5,
    maxiter: int | None = None,
    ssor_omega: float = 1.2,
    residual_dir: Path | None = None,
    learning_group_context: dict[str, Any] | None = None,
    steady_forward_repeats: int = 50,
    petsc_amg_backend: str = "gamg",
    petsc_ic0_shift_type: str = "none",
    petsc_options: str | None = None,
    petsc_ksp_norm_type: str = "unpreconditioned",
    petsc_sgs_ssor_impl: str = "petsc_sor_legacy",
    petsc_factor_solve_mode: str = "transformed_operator_native",
    petsc_factor_operator_mode: str = "explicit_aijcusparse",
    petsc_cond_mode: str = "accurate_ref",
    petsc_enable_approx_spectral_diagnostics: bool = False,
    petsc_apply_warmup_runs: int = 0,
    petsc_apply_measure_repeats: int = 5,
    petsc_learning_internal_rtol_ratio: float = 1.0,
    petsc_learning_pc_impl: str = "shell_native",
) -> list[Any]:
    del petsc_apply_warmup_runs
    from dpcg import benchmark as benchmark_mod

    _require_petsc_gpu_stack()
    benchmark_mod._check_spd(sample.A)
    prepared_model = (
        benchmark_mod._prepare_learning_model(model, learning_device)
        if "learning" in methods
        else model
    )
    results: list[Any] = []
    for method in methods:
        results.extend(
            benchmark_method_group(
                [sample],
                method=method,
                model=prepared_model if method == "learning" else None,
                model_name="learning" if method == "learning" else "baseline",
                model_label="learning" if method == "learning" else "baseline",
                is_learning_model=(method == "learning"),
                learning_output_kind=learning_output_kind,
                learning_diag_strategy=learning_diag_strategy,
                use_mask_projection=use_mask_projection,
                learning_device=learning_device,
                rtol_values=[float(rtol)],
                maxiter=maxiter,
                ssor_omega=ssor_omega,
                residual_dir=residual_dir,
                learning_group_context=learning_group_context,
                steady_forward_repeats=steady_forward_repeats,
                petsc_amg_backend=petsc_amg_backend,
                petsc_ic0_shift_type=petsc_ic0_shift_type,
                petsc_options=petsc_options,
                petsc_ksp_norm_type=petsc_ksp_norm_type,
                petsc_sgs_ssor_impl=petsc_sgs_ssor_impl,
                petsc_factor_solve_mode=petsc_factor_solve_mode,
                petsc_factor_operator_mode=petsc_factor_operator_mode,
                petsc_cond_mode=petsc_cond_mode,
                petsc_enable_approx_spectral_diagnostics=petsc_enable_approx_spectral_diagnostics,
                petsc_apply_measure_repeats=petsc_apply_measure_repeats,
                petsc_learning_internal_rtol_ratio=petsc_learning_internal_rtol_ratio,
                petsc_learning_pc_impl=petsc_learning_pc_impl,
                burnin_sample_id=sample.sample_id,
            )
        )
    return results


def run_group_ignition(
    sample: "BenchmarkSample",
    methods: Sequence[str],
    *,
    model=None,
    learning_output_kind: str = "sparse_factor_L",
    learning_diag_strategy: str = "learned_exp",
    use_mask_projection: bool = True,
    learning_device: str = "cuda",
    rtol: float = 1.0e-5,
    maxiter: int | None = None,
    ssor_omega: float = 1.2,
    setup_warmup_runs_gpu_learning: int = 1,
    setup_warmup_runs_other: int = 0,
    apply_warmup_runs: int = 0,
    solve_warmup_runs: int = 0,
    steady_forward_repeats: int = 50,
    learning_group_context: dict[str, Any] | None = None,
    petsc_amg_backend: str = "gamg",
    petsc_ic0_shift_type: str = "none",
    petsc_options: str | None = None,
    petsc_ksp_norm_type: str = "unpreconditioned",
    petsc_sgs_ssor_impl: str = "petsc_sor_legacy",
    petsc_factor_solve_mode: str = "transformed_operator_native",
    petsc_factor_operator_mode: str = "explicit_aijcusparse",
    petsc_cond_mode: str = "accurate_ref",
    petsc_enable_approx_spectral_diagnostics: bool = False,
    petsc_apply_warmup_runs: int = 0,
    petsc_apply_measure_repeats: int = 5,
    petsc_learning_internal_rtol_ratio: float = 1.0,
    petsc_learning_pc_impl: str = "shell_native",
) -> None:
    del setup_warmup_runs_gpu_learning
    del setup_warmup_runs_other
    del apply_warmup_runs
    del solve_warmup_runs
    del petsc_cond_mode
    del petsc_enable_approx_spectral_diagnostics
    del petsc_apply_warmup_runs
    del petsc_apply_measure_repeats
    for method in methods:
        try:
            run_method_burnin(
                sample,
                method=method,
                model=model if method == "learning" else None,
                learning_output_kind=learning_output_kind,
                learning_diag_strategy=learning_diag_strategy,
                use_mask_projection=use_mask_projection,
                learning_device=learning_device,
                rtol=rtol,
                maxiter=maxiter,
                ssor_omega=ssor_omega,
                learning_group_context=learning_group_context,
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
        except Exception:
            continue
