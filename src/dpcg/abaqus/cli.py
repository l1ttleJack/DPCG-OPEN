"""Unified dataset-building CLI for the lightweight DPCG release."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpcg.io.abaqus_case_library import (
    build_case_library,
    convert_case_library,
    validate_case_library,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, validate, and convert an Abaqus dataset release."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build formal Abaqus cases from a master .inp.")
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--master-inp", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--baseline-type", type=str, default=None)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--num-cases", type=int, default=500)
    build.add_argument("--abaqus-command", type=str, default=r"D:\Abaqus2021\Command\abaqus.bat")
    build.add_argument("--cpus", type=int, default=4)
    build.add_argument("--dataset-rules-config", type=Path, default=None)

    validate = subparsers.add_parser("validate", help="Validate a generated Abaqus case library.")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--skip-condition-number", action="store_true")
    validate.add_argument("--condition-number-tol", type=float, default=1.0e-8)
    validate.add_argument("--condition-number-maxiter", type=int, default=20_000)

    convert = subparsers.add_parser("convert", help="Convert a validated case library to NPZ.")
    convert.add_argument("--dataset-root", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, default=None)
    convert.add_argument("--rhs-mode", type=str, default="load")
    convert.add_argument("--solve-mode", type=str, default="zero_placeholder")
    convert.add_argument("--bh-solve-mode", type=str, default="zero_placeholder")
    convert.add_argument("--num-workers", type=int, default=None)
    convert.add_argument("--compress-npz", action=argparse.BooleanOptionalAction, default=False)
    convert.add_argument(
        "--write-per-case-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        build_case_library(
            dataset_id=args.dataset_id,
            master_inp=args.master_inp,
            output_root=args.output_root,
            baseline_type=args.baseline_type,
            seed=args.seed,
            num_cases=args.num_cases,
            abaqus_command=args.abaqus_command,
            cpus=args.cpus,
            dataset_rules_path=args.dataset_rules_config,
        )
        return
    if args.command == "validate":
        validate_case_library(
            args.dataset_root,
            estimate_condition_number_enabled=not bool(args.skip_condition_number),
            condition_number_tol=args.condition_number_tol,
            condition_number_maxiter=args.condition_number_maxiter,
        )
        return
    if args.command == "convert":
        convert_case_library(
            args.dataset_root,
            output_dir=args.output_dir,
            rhs_mode=args.rhs_mode,
            solve_mode=args.solve_mode,
            bh_solve_mode=args.bh_solve_mode,
            num_workers=args.num_workers,
            compress_npz=bool(args.compress_npz),
            write_per_case_manifest=bool(args.write_per_case_manifest),
        )
        return
    raise RuntimeError(f"unsupported command: {args.command}")
