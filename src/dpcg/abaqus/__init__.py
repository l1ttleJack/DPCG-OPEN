"""Public Abaqus-facing helpers for dataset construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dpcg.io.abaqus import load_abaqus_system
from dpcg.io.abaqus_case_library import (
    build_case_library,
    convert_case_library,
    validate_case_library,
)


def build_dataset_from_inp(
    *,
    dataset_id: str,
    master_inp: str | Path,
    output_root: str | Path,
    baseline_type: str | None = None,
    seed: int = 42,
    num_cases: int = 500,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 4,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the formal Abaqus case library from a marked master input."""
    return build_case_library(
        dataset_id=dataset_id,
        master_inp=Path(master_inp),
        output_root=Path(output_root),
        baseline_type=baseline_type,
        seed=seed,
        num_cases=num_cases,
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


__all__ = [
    "build_case_library",
    "build_dataset_from_inp",
    "convert_case_library",
    "load_abaqus_system",
    "validate_case_library",
]
