"""Shared helpers for YAML/JSON-backed script configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--config`` option to a script parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config file. CLI flags override config values.",
    )


def bootstrap_config(
    argv: list[str] | None,
    *,
    section_name: str,
    allowed_keys: Iterable[str],
) -> tuple[Path | None, list[str], dict[str, Any]]:
    """Read config values before the main parser handles CLI overrides."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    config_args, remaining = bootstrap.parse_known_args(argv)
    config_values = load_config_mapping(
        config_args.config,
        section_name=section_name,
        allowed_keys=allowed_keys,
    )
    return config_args.config, remaining, config_values


def load_config_mapping(
    config_path: str | Path | None,
    *,
    section_name: str,
    allowed_keys: Iterable[str],
) -> dict[str, Any]:
    """Load config values from YAML/JSON and validate supported keys."""
    if config_path is None:
        return {}
    path = Path(config_path)
    payload = _read_config(path)
    config_values = payload
    section_payload = payload.get(section_name)
    if section_payload is not None:
        if not isinstance(section_payload, dict):
            raise SystemExit(f"config section '{section_name}' must be a mapping: {path}")
        config_values = section_payload
    if not isinstance(config_values, dict):
        raise SystemExit(f"config root must be a mapping: {path}")

    allowed = set(allowed_keys)
    unknown = sorted(str(key) for key in config_values.keys() if key not in allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise SystemExit(f"unsupported config keys in {path}: {joined}")
    return dict(config_values)


def ensure_required(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    required: Iterable[str],
) -> None:
    """Emulate argparse required-option checks after config defaults are applied."""
    missing = [
        f"--{name.replace('_', '-')}" for name in required if getattr(args, name, None) is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")


def _read_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    else:
        raise SystemExit(f"unsupported config format for {path}; expected .json/.yaml/.yml")
    if not isinstance(payload, dict):
        raise SystemExit(f"config root must be a mapping: {path}")
    return payload
