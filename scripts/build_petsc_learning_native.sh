#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_FILE="${ROOT_DIR}/src/dpcg/_native/petsc_learning_transformed_mat.c"
OUT_FILE="${DPCG_PETSC_NATIVE_OUT:-${ROOT_DIR}/src/dpcg/_native/libdpcg_petsc_learning.so}"
PYTHON_BIN="${DPCG_PYTHON_BIN:-/home/zhike/miniconda3/envs/HPC/bin/python}"

normalize_petsc_arch() {
  local raw_arch="${1-}"
  local selected_arch=""
  local item

  IFS=':' read -r -a PETSC_ARCH_ITEMS <<< "${raw_arch}"
  for item in "${PETSC_ARCH_ITEMS[@]}"; do
    if [[ -n "${item}" ]]; then
      selected_arch="${item}"
    fi
  done

  printf '%s' "${selected_arch}"
}

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "missing source: ${SRC_FILE}" >&2
  exit 1
fi

if [[ -z "${PETSC_DIR:-}" || -z "${PETSC_ARCH:-}" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing python interpreter: ${PYTHON_BIN}" >&2
    echo "set PETSC_DIR/PETSC_ARCH explicitly, or override DPCG_PYTHON_BIN" >&2
    exit 1
  fi

  PETSC_CFG_RAW="$("${PYTHON_BIN}" - <<'PY'
import petsc4py

cfg = petsc4py.get_config()
print(cfg.get("PETSC_DIR", ""))
print(cfg.get("PETSC_ARCH", ""))
PY
)" || {
    echo "failed to query PETSc config from petsc4py using ${PYTHON_BIN}" >&2
    echo "set PETSC_DIR and PETSC_ARCH explicitly, or install petsc4py in the HPC env" >&2
    exit 1
  }
  mapfile -t PETSC_CFG <<< "${PETSC_CFG_RAW}"
  export PETSC_DIR="${PETSC_DIR:-${PETSC_CFG[0]}}"
  export PETSC_ARCH="${PETSC_ARCH:-${PETSC_CFG[1]}}"
fi

export PETSC_ARCH="$(normalize_petsc_arch "${PETSC_ARCH:-}")"

if [[ -z "${PETSC_DIR:-}" || -z "${PETSC_ARCH:-}" ]]; then
  echo "PETSC_DIR and PETSC_ARCH must be configured" >&2
  exit 1
fi

PETSC_LIB_DIR="${PETSC_DIR}/${PETSC_ARCH}/lib"
PETSC_INCLUDE_DIR="${PETSC_DIR}/${PETSC_ARCH}/include"
PETSC_INCLUDE_FLAGS=(
  "-I${PETSC_DIR}/include"
  "-I${PETSC_INCLUDE_DIR}"
)
PETSC_LINK_FLAGS=(
  "-Wl,--disable-new-dtags"
  "-Wl,-rpath,${PETSC_LIB_DIR}"
  "${PETSC_LIB_DIR}/libpetsc.so"
)

if [[ ! -d "${PETSC_LIB_DIR}" ]]; then
  echo "missing PETSc library directory: ${PETSC_LIB_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PETSC_INCLUDE_DIR}" ]]; then
  echo "missing PETSc arch include directory: ${PETSC_INCLUDE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PETSC_LIB_DIR}/libpetsc.so" ]]; then
  echo "missing PETSc shared library: ${PETSC_LIB_DIR}/libpetsc.so" >&2
  exit 1
fi

if [[ -n "${MPICC:-}" ]]; then
  CC_BIN="${MPICC}"
elif [[ -x "${PETSC_DIR}/${PETSC_ARCH}/bin/mpicc" ]]; then
  CC_BIN="${PETSC_DIR}/${PETSC_ARCH}/bin/mpicc"
else
  CC_BIN="$(command -v mpicc || true)"
  if [[ -z "${CC_BIN}" && -x "/usr/bin/mpicc.openmpi" ]]; then
    CC_BIN="/usr/bin/mpicc.openmpi"
  fi
fi
if [[ -z "${CC_BIN}" ]]; then
  echo "unable to find mpicc; set MPICC explicitly" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_FILE}")"

"${CC_BIN}" \
  -O3 \
  -fPIC \
  -shared \
  "${SRC_FILE}" \
  -o "${OUT_FILE}" \
  "${PETSC_INCLUDE_FLAGS[@]}" \
  "${PETSC_LINK_FLAGS[@]}"

echo "built ${OUT_FILE}"
