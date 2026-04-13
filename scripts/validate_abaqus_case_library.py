"""Validate an Abaqus case library and compute structural statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpcg.cli_config import add_config_argument, bootstrap_config, ensure_required
from dpcg.io.abaqus_case_library import (
    collect_dataset_pilot_results,
    collect_frame33_bh_pilot_results,
    validate_case_library,
)

DEFAULTS = {
    "estimate_condition_number": False,
    "skip_condition_number": False,
    "condition_number_tol": 1e-8,
    "condition_number_maxiter": 20_000,
    "collect_frame33_bh_pilot_results": False,
    "collect_pilot_results": False,
    "dataset_id": None,
    "baseline_type": None,
    "dataset_rules_config": None,
}

CONFIG_KEYS = {
    "dataset_root",
    "estimate_condition_number",
    "skip_condition_number",
    "condition_number_tol",
    "condition_number_maxiter",
    "collect_frame33_bh_pilot_results",
    "collect_pilot_results",
    "dataset_id",
    "baseline_type",
    "dataset_rules_config",
}


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.dataset_root = Path(args.dataset_root)
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
        description="Validate an Abaqus case library and compute structural statistics."
    )
    add_config_argument(parser)
    parser.add_argument("--dataset-root", type=Path, required=False, default=None)
    parser.add_argument(
        "--estimate-condition-number",
        action="store_true",
        default=None,
        help="Deprecated compatibility flag. Condition numbers are estimated by default.",
    )
    parser.add_argument(
        "--skip-condition-number",
        action="store_true",
        default=None,
        help="Skip condition-number estimation and run structure-only validation.",
    )
    parser.add_argument("--condition-number-tol", type=float, default=None)
    parser.add_argument("--condition-number-maxiter", type=int, default=None)
    parser.add_argument(
        "--collect-frame33-bh-pilot-results",
        action="store_true",
        default=None,
        help="After validation, collect BH pilot results into meta/bh_pilot_results.json.",
    )
    parser.add_argument(
        "--collect-pilot-results",
        action="store_true",
        default=None,
        help="After validation, collect generic BH pilot results into meta/bh_pilot_results.json.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional dataset id used with --collect-pilot-results.",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default=None,
        help="Optional baseline type used with --collect-pilot-results.",
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
        section_name="validate_abaqus_case_library",
        allowed_keys=CONFIG_KEYS,
    )
    if config_values:
        parser.set_defaults(**config_values)
    args = parser.parse_args(remaining)
    args = _apply_runtime_defaults(args)
    ensure_required(parser, args, required=("dataset_root",))
    return _normalize_args(args)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    estimate_condition_number_enabled = not bool(args.skip_condition_number)
    validate_case_library(
        args.dataset_root,
        estimate_condition_number_enabled=estimate_condition_number_enabled,
        condition_number_tol=args.condition_number_tol,
        condition_number_maxiter=args.condition_number_maxiter,
    )
    if args.collect_frame33_bh_pilot_results:
        collect_frame33_bh_pilot_results(
            args.dataset_root,
            condition_number_tol=args.condition_number_tol,
            condition_number_maxiter=args.condition_number_maxiter,
            dataset_rules_path=args.dataset_rules_config,
        )
    if args.collect_pilot_results:
        dataset_id = args.dataset_id or args.dataset_root.name
        collect_dataset_pilot_results(
            args.dataset_root,
            dataset_id=dataset_id,
            baseline_type=args.baseline_type,
            condition_number_tol=args.condition_number_tol,
            condition_number_maxiter=args.condition_number_maxiter,
            dataset_rules_path=args.dataset_rules_config,
        )


if __name__ == "__main__":
    main()
