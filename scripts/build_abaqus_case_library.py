"""Build an Abaqus case library from one marked master input."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpcg.cli_config import add_config_argument, bootstrap_config, ensure_required
from dpcg.io.abaqus_case_library import (
    add_build_arguments,
    build_case_library,
    prepare_dataset_bh_cases,
    prepare_dataset_pilot,
    prepare_frame33_bh_pilot,
    resolve_pathological_dataset_id,
)

DEFAULTS = {
    "mode": "cases",
    "seed": 42,
    "num_cases": 500,
    "abaqus_command": r"D:\Abaqus2021\Command\abaqus.bat",
    "cpus": 4,
    "baseline_type": None,
    "dataset_rules_config": None,
}

CONFIG_KEYS = {
    "dataset_id",
    "baseline_type",
    "master_inp",
    "output_root",
    "seed",
    "num_cases",
    "abaqus_command",
    "cpus",
    "mode",
    "source_dataset_root",
    "dataset_rules_config",
}


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.master_inp = Path(args.master_inp)
    args.output_root = Path(args.output_root)
    if args.source_dataset_root is not None:
        args.source_dataset_root = str(args.source_dataset_root)
    if args.dataset_rules_config is not None:
        args.dataset_rules_config = Path(args.dataset_rules_config)
    return args


def _apply_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-A preparation of an Abaqus case library from a marked master input."
    )
    add_config_argument(parser)
    add_build_arguments(parser, required=False, include_defaults=False)
    parser.add_argument(
        "--mode",
        choices=("cases", "pilot", "pathological-cases", "bh-pilot"),
        default=None,
        help=(
            "cases: prepare the formal dataset; "
            "pilot/bh-pilot: prepare deterministic BH pilot cases; "
            "pathological-cases: prepare BH-only cases from accepted pilot templates."
        ),
    )
    parser.add_argument(
        "--source-dataset-root",
        type=str,
        default=None,
        help="Required for --mode pathological-cases. Points to the base dataset root with bh_pilot_results.json.",
    )
    parser.add_argument(
        "--dataset-rules-config",
        type=Path,
        default=None,
        help="Optional YAML file that defines dataset-specific case-library rules.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    _config_path, remaining, config_values = bootstrap_config(
        argv,
        section_name="build_abaqus_case_library",
        allowed_keys=CONFIG_KEYS,
    )
    if config_values:
        parser.set_defaults(**config_values)
    args = parser.parse_args(remaining)
    args = _apply_runtime_defaults(args)
    ensure_required(parser, args, required=("dataset_id", "master_inp", "output_root"))
    return _normalize_args(args)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode in {"pilot", "bh-pilot"}:
        if args.mode == "bh-pilot" and args.dataset_id == "frame33-b":
            prepare_frame33_bh_pilot(
                master_inp=args.master_inp,
                dataset_root=args.output_root / args.dataset_id,
                abaqus_command=args.abaqus_command,
                cpus=args.cpus,
                dataset_rules_path=args.dataset_rules_config,
            )
            return
        prepare_dataset_pilot(
            master_inp=args.master_inp,
            dataset_root=args.output_root / args.dataset_id,
            dataset_id=args.dataset_id,
            baseline_type=args.baseline_type,
            abaqus_command=args.abaqus_command,
            cpus=args.cpus,
            dataset_rules_path=args.dataset_rules_config,
        )
        return
    if args.mode == "pathological-cases":
        if args.source_dataset_root is None:
            raise SystemExit("--source-dataset-root is required for --mode pathological-cases")
        prepare_dataset_bh_cases(
            master_inp=args.master_inp,
            dataset_root=args.output_root
            / resolve_pathological_dataset_id(
                args.dataset_id,
                baseline_type=args.baseline_type,
                dataset_rules_path=args.dataset_rules_config,
            ),
            dataset_id=args.dataset_id,
            baseline_type=args.baseline_type,
            source_dataset_root=args.source_dataset_root,
            seed=args.seed,
            num_cases=args.num_cases,
            abaqus_command=args.abaqus_command,
            cpus=args.cpus,
            dataset_rules_path=args.dataset_rules_config,
        )
        return
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


if __name__ == "__main__":
    main()
