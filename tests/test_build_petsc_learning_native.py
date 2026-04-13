from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def _write_fake_mpicc(script_path: Path) -> None:
    script_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

printf '%s\\n' "$0" > "${FAKE_MPICC_LOG}"
printf '%s\\n' "$@" >> "${FAKE_MPICC_LOG}"

out_file=""
prev=""
for arg in "$@"; do
  if [[ "${prev}" == "-o" ]]; then
    out_file="${arg}"
    break
  fi
  prev="${arg}"
done

if [[ -z "${out_file}" ]]; then
  echo "missing -o argument" >&2
  exit 1
fi

mkdir -p "$(dirname "${out_file}")"
: > "${out_file}"
""",
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)


def test_build_script_uses_selected_petsc_arch_for_mpicc(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_petsc_learning_native.sh"

    petsc_dir = tmp_path / "petsc"
    selected_arch = "arch-dpcg-hypre-opt"
    skipped_arch = "arch-linux-c-debug"

    (petsc_dir / "include").mkdir(parents=True)
    (petsc_dir / selected_arch / "include").mkdir(parents=True)
    (petsc_dir / selected_arch / "lib").mkdir(parents=True)
    (petsc_dir / selected_arch / "bin").mkdir(parents=True)
    (petsc_dir / selected_arch / "lib" / "libpetsc.so").write_text("", encoding="utf-8")

    fake_mpicc = petsc_dir / selected_arch / "bin" / "mpicc"
    fake_mpicc_log = tmp_path / "fake_mpicc.log"
    _write_fake_mpicc(fake_mpicc)

    out_file = tmp_path / "libdpcg_petsc_learning.so"
    env = os.environ.copy()
    env.update(
        {
            "PETSC_DIR": str(petsc_dir),
            "PETSC_ARCH": f"{skipped_arch}:{selected_arch}",
            "DPCG_PETSC_NATIVE_OUT": str(out_file),
            "FAKE_MPICC_LOG": str(fake_mpicc_log),
        }
    )
    env.pop("MPICC", None)

    subprocess.run(
        ["bash", str(script_path)],
        check=True,
        cwd=repo_root,
        env=env,
    )

    assert out_file.exists()
    log_lines = fake_mpicc_log.read_text(encoding="utf-8").splitlines()
    assert log_lines[0] == str(fake_mpicc)
    assert f"-I{petsc_dir / selected_arch / 'include'}" in log_lines
    assert str(petsc_dir / selected_arch / "lib" / "libpetsc.so") in log_lines
