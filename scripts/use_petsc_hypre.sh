#!/usr/bin/env bash

# Configure a clean PETSc+hypre runtime environment for DPCG benchmarks.
# This script is optional: source it only when you want to run the PETSc GPU path.

set -euo pipefail

_dpcg_filter_path_list() {
  local input="${1-}"
  local mode="${2-path}"
  local petsc_dir="${3-/home/zhike/petsc}"
  local IFS=':'
  local -a items=()
  local -a kept=()
  local item
  read -r -a items <<< "${input}"
  for item in "${items[@]}"; do
    if [[ -z "${item}" ]]; then
      continue
    fi
    case "${mode}" in
      path)
        if [[ "${item}" == "${petsc_dir}"/arch-*/bin ]]; then
          continue
        fi
        ;;
      lib)
        if [[ "${item}" == "${petsc_dir}"/arch-*/lib ]]; then
          continue
        fi
        ;;
      *)
        ;;
    esac
    kept+=("${item}")
  done
  local result=""
  for item in "${kept[@]}"; do
    if [[ -z "${result}" ]]; then
      result="${item}"
    else
      result="${result}:${item}"
    fi
  done
  printf '%s' "${result}"
}

_dpcg_join_nonempty() {
  local result=""
  local item
  for item in "$@"; do
    if [[ -z "${item}" ]]; then
      continue
    fi
    case ":${result}:" in
      *":${item}:"*)
        continue
        ;;
      *)
        ;;
    esac
    if [[ -z "${result}" ]]; then
      result="${item}"
    else
      result="${result}:${item}"
    fi
  done
  printf '%s' "${result}"
}

_dpcg_detect_default_petsc_arch() {
  local petsc_dir="${1-}"
  local preferred_arch="arch-dpcg-hypre-opt"
  if [[ -d "${petsc_dir}/${preferred_arch}/lib" ]]; then
    printf '%s' "${preferred_arch}"
    return 0
  fi
  return 0
}

_dpcg_configure_pythonpath() {
  local petsc_lib="${1-}"
  local filtered_pythonpath=""

  if [[ -z "${petsc_lib}" || ! -d "${petsc_lib}" ]]; then
    return 0
  fi

  filtered_pythonpath="$(_dpcg_filter_path_list "${PYTHONPATH-}" lib "${PETSC_DIR}")"
  export PYTHONPATH="$(_dpcg_join_nonempty "${petsc_lib}" "${filtered_pythonpath}")"
}

_dpcg_read_make_var() {
  local file="${1-}"
  local key="${2-}"
  if [[ -z "${file}" || -z "${key}" || ! -f "${file}" ]]; then
    return 0
  fi
  sed -n "s/^${key}[[:space:]]*=[[:space:]]*//p" "${file}" | sed -n '1p'
}

_dpcg_unquote() {
  local value="${1-}"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s' "${value}"
}

_dpcg_first_command_token() {
  local value="${1-}"
  local first=""
  read -r first _ <<< "${value}"
  _dpcg_unquote "${first}"
}

_dpcg_first_library_dir_from_flags() {
  local value="${1-}"
  local token
  local -a tokens=()
  read -r -a tokens <<< "${value}"
  for token in "${tokens[@]}"; do
    token="$(_dpcg_unquote "${token}")"
    case "${token}" in
      -L*)
        token="${token#-L}"
        if [[ -d "${token}" ]]; then
          printf '%s' "${token}"
          return 0
        fi
        ;;
      *)
        ;;
    esac
  done
  return 0
}

_dpcg_resolve_petsc_mpi_paths() {
  local conf_file="${1-}"
  local mpi_exec=""
  local mpi_cc=""
  local mpi_lib_flags=""
  local mpi_bin=""
  local mpi_lib=""
  local candidate=""

  mpi_exec="$(_dpcg_read_make_var "${conf_file}" "MPIEXEC")"
  mpi_cc="$(_dpcg_read_make_var "${conf_file}" "PCC")"
  mpi_lib_flags="$(_dpcg_read_make_var "${conf_file}" "MPICXX_LIBS")"

  candidate="$(_dpcg_first_command_token "${mpi_exec}")"
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    mpi_bin="$(dirname "${candidate}")"
  fi
  if [[ -z "${mpi_bin}" ]]; then
    candidate="$(_dpcg_first_command_token "${mpi_cc}")"
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      mpi_bin="$(dirname "${candidate}")"
    fi
  fi

  mpi_lib="$(_dpcg_first_library_dir_from_flags "${mpi_lib_flags}")"
  if [[ -z "${mpi_lib}" && -n "${mpi_bin}" ]]; then
    candidate="$(cd "${mpi_bin}/.." && pwd)/lib"
    if [[ -d "${candidate}" ]]; then
      mpi_lib="${candidate}"
    fi
  fi

  printf '%s\n%s\n' "${mpi_bin}" "${mpi_lib}"
}

_dpcg_apply_petsc_hypre_env() {
  local petsc_dir="${DPCG_PETSC_DIR:-/home/zhike/petsc}"
  local requested_arch="${DPCG_PETSC_ARCH:-}"
  local petsc_arch=""
  local petsc_options="${DPCG_PETSC_OPTIONS:--use_gpu_aware_mpi 0}"
  local cuda_lib="${DPCG_CUDA_LIB_DIR:-/usr/local/cuda-12.4/lib64}"
  local wsl_lib="${DPCG_WSL_LIB_DIR:-/usr/lib/wsl/lib}"
  local petsc_lib=""
  local petsc_bin=""
  local petsc_conf=""
  local mpi_paths=()
  local petsc_mpi_bin=""
  local petsc_mpi_lib=""

  if [[ ! -d "${petsc_dir}" ]]; then
    echo "PETSC_DIR does not exist: ${petsc_dir}" >&2
    return 1
  fi

  if [[ -n "${requested_arch}" ]]; then
    petsc_arch="${requested_arch}"
  else
    petsc_arch="$(_dpcg_detect_default_petsc_arch "${petsc_dir}")"
  fi
  if [[ -z "${petsc_arch}" ]]; then
    echo "unable to detect a PETSc arch under ${petsc_dir}" >&2
    echo "expected default runtime arch: ${petsc_dir}/arch-dpcg-hypre-opt/lib" >&2
    echo "set DPCG_PETSC_ARCH explicitly before running this script only if you really want a different build" >&2
    return 1
  fi

  petsc_lib="${petsc_dir}/${petsc_arch}/lib"
  petsc_bin="${petsc_dir}/${petsc_arch}/bin"
  petsc_conf="${petsc_lib}/petsc/conf/petscvariables"
  if [[ ! -d "${petsc_lib}" ]]; then
    echo "PETSc arch library directory does not exist: ${petsc_lib}" >&2
    echo "default runtime on this machine is: ${petsc_dir}/arch-dpcg-hypre-opt/lib" >&2
    echo "Build the target arch first or fix DPCG_PETSC_ARCH if you overrode it." >&2
    return 1
  fi

  export PETSC_DIR="${petsc_dir}"
  export PETSC_ARCH="${petsc_arch}"
  export PETSC_OPTIONS="${petsc_options}"

  if [[ -f "${petsc_conf}" ]]; then
    mapfile -t mpi_paths < <(_dpcg_resolve_petsc_mpi_paths "${petsc_conf}")
    petsc_mpi_bin="${mpi_paths[0]:-}"
    petsc_mpi_lib="${mpi_paths[1]:-}"
  fi

  local filtered_ld
  local filtered_path
  filtered_ld="$(_dpcg_filter_path_list "${LD_LIBRARY_PATH-}" lib "${petsc_dir}")"
  filtered_path="$(_dpcg_filter_path_list "${PATH-}" path "${petsc_dir}")"

  export LD_LIBRARY_PATH="$(_dpcg_join_nonempty "${cuda_lib}" "${wsl_lib}" "${petsc_lib}" "${petsc_mpi_lib}" "${filtered_ld}")"
  export PATH="$(_dpcg_join_nonempty "${petsc_mpi_bin}" "${petsc_bin}" "${filtered_path}")"
  _dpcg_configure_pythonpath "${petsc_lib}"
}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _dpcg_apply_petsc_hypre_env
  echo "Configured PETSc+hypre environment: PETSC_ARCH=${PETSC_ARCH}" >&2
  return 0
fi

_dpcg_apply_petsc_hypre_env

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

cat <<EOF
Configured PETSc+hypre environment for this shell process:
  PETSC_DIR=${PETSC_DIR}
  PETSC_ARCH=${PETSC_ARCH}
  PETSC_OPTIONS=${PETSC_OPTIONS}
  mpicc=$(command -v mpicc || echo "<missing>")
  mpiexec=$(command -v mpiexec || echo "<missing>")
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
  PYTHONPATH=${PYTHONPATH-}

Usage:
  source scripts/use_petsc_hypre.sh
  scripts/use_petsc_hypre.sh /home/zhike/miniconda3/envs/HPC/bin/python -m dpcg.benchmark --config ...

Optional overrides:
  DPCG_PETSC_DIR
  DPCG_PETSC_ARCH
  DPCG_PETSC_OPTIONS
  DPCG_CUDA_LIB_DIR
  DPCG_WSL_LIB_DIR
EOF
