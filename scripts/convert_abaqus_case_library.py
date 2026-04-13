"""Convert a validated Abaqus case library into NPZ samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpcg.cli_config import add_config_argument, bootstrap_config, ensure_required
from dpcg.io.abaqus_case_library import convert_case_library

DEFAULTS = {
    "rhs_mode": "load",
    "solve_mode": "zero_placeholder",
    "bh_solve_mode": "zero_placeholder",
    "num_workers": None,
    "compress_npz": False,
    "write_per_case_manifest": False,
}

CONFIG_KEYS = {
    "dataset_root",
    "output_dir",
    "rhs_mode",
    "solve_mode",
    "bh_solve_mode",
    "num_workers",
    "compress_npz",
    "write_per_case_manifest",
}


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.dataset_root = Path(args.dataset_root)
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
    return args


def _apply_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a validated Abaqus case library into NPZ samples."
    )
    add_config_argument(parser)
    parser.add_argument("--dataset-root", type=Path, required=False, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rhs-mode", type=str, default=None)
    parser.add_argument("--solve-mode", type=str, default=None)
    parser.add_argument("--bh-solve-mode", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--compress-npz", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--write-per-case-manifest",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    _config_path, remaining, config_values = bootstrap_config(
        argv,
        section_name="convert_abaqus_case_library",
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


if __name__ == "__main__":
    main()
