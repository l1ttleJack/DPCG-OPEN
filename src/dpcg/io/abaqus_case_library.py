"""Utilities for building and validating Abaqus case libraries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dpcg.io.abaqus import (
    _resolve_num_workers,
    build_free_system_from_mtx,
    convert_abaqus_to_npz,
    estimate_condition_number,
    load_abaqus_system,
    read_mtx_5col,
)

MATERIALS_BEGIN = "**<MATERIALS_BEGIN>"
MATERIALS_END = "**<MATERIALS_END>"
DEFAULT_DATASET_RULES_PATH = Path(__file__).with_name("abaqus_case_library_rules.yaml")
CSV_FIELDS: tuple[str, ...] = (
    "dataset_id",
    "case_id",
    "split",
    "family",
    "sampling_mode",
    "template_id",
    "run_status",
    "validation_status",
    "node_count",
    "total_dof_count",
    "free_dof_count",
    "nnz",
    "sparsity_density",
    "condition_number_est",
    "condition_number_status",
    "condition_ratio_to_ref",
    "condition_target_min",
    "condition_target_max",
    "bh_attempt_index",
    "generated_inp_sha256",
    "alpha_global",
    "sigma_f",
    "sigma_s",
    "rho_f",
    "rho_s",
    "target_group",
    "soft_layer_indices",
    "soft_layer_material_names",
    "soft_alpha_values",
    "variable_material_names",
    "frame_material_names",
    "soil_material_names",
    "E0_values",
    "E_values",
    "inp_path",
    "dat_path",
    "stiffness_mtx_path",
    "load_mtx_path",
    "env_path",
    "run_case_cmd_path",
)


@dataclass(frozen=True)
class MaterialLayer:
    """Single mutable material definition inside the marked block."""

    name: str
    group: str
    order: int
    elastic_modulus: float
    poisson_ratio: float
    material_line_idx: int
    elastic_value_line_idx: int


@dataclass(frozen=True)
class ParsedMasterInput:
    """Parsed master input with mutable material layers."""

    lines: tuple[str, ...]
    marker_begin_idx: int
    marker_end_idx: int
    materials: tuple[MaterialLayer, ...]


@dataclass(frozen=True)
class CaseAssignment:
    """Split/family assignment for a generated case."""

    case_id: str
    split: str
    family: str


@dataclass(frozen=True)
class BhPilotTemplate:
    """Deterministic pilot template for BH/pathological calibration."""

    template_id: str
    sampling_mode: str
    target_group: str
    soft_layer_indices: tuple[int, ...]
    soft_alpha_values: tuple[float, ...]


@dataclass(frozen=True)
class MaterialGroupMatcher:
    """Name-based matcher for one material group."""

    prefixes: tuple[str, ...]
    exact_names: tuple[str, ...]
    regex: tuple[str, ...]


@dataclass(frozen=True)
class FamilyASpec:
    """Sampling rules for family A."""

    alpha_ranges: dict[str, tuple[tuple[float, float], ...]]


@dataclass(frozen=True)
class FamilyBSpec:
    """Sampling rules for family B."""

    sigma_ranges: dict[str, dict[str, tuple[tuple[float, float], ...]]]
    rho: dict[str, float]
    alpha_clip: tuple[float, float]


@dataclass(frozen=True)
class FamilyBHSpec:
    """Sampling rules for family BH and pilot selection."""

    family_name: str
    pathological_dataset_suffix: str
    pilot_dir_name: str
    target_groups: tuple[str, ...]
    pilot_story_alphas: dict[str, tuple[float, ...]]
    pilot_band_alphas: dict[str, tuple[float, ...]]
    condition_ratio_min: float
    condition_ratio_max: float
    soft_alpha_log10_jitter: float
    other_alpha_range: tuple[float, float]
    pilot_required_material_counts: dict[str, int]


@dataclass(frozen=True)
class BaselineSpec:
    """Sampling rules for one reusable baseline type."""

    baseline_type: str
    default_group: str
    material_group_matchers: dict[str, MaterialGroupMatcher]
    base_quotas: tuple[tuple[str, str, int], ...]
    bh_only_base_quotas: tuple[tuple[str, str, int], ...]
    reference_condition_source: str
    reference_condition_tol: float
    reference_condition_maxiter: int
    family_a: FamilyASpec
    family_b: FamilyBSpec
    family_bh: FamilyBHSpec


@dataclass(frozen=True)
class CaseLibraryRules:
    """Loaded case-library sampling rules."""

    baselines: dict[str, BaselineSpec]


LEGACY_DATASET_BASELINE_TYPES: dict[str, str] = {
    "frame33-b": "pure_frame",
    "frame99-b": "pure_frame",
    "station3622-b": "frame_soil_mixed",
    "frame33-b-bh-v1": "pure_frame",
    "frame99-b-bh-v1": "pure_frame",
    "station3622-b-bh-v1": "frame_soil_mixed",
}


def sha256_file(path: str | Path) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_rules_path(dataset_rules_path: str | Path | None) -> Path:
    if dataset_rules_path is None:
        return DEFAULT_DATASET_RULES_PATH
    return Path(dataset_rules_path)


def _parse_quota_rows(rows: Any, *, field_name: str) -> tuple[tuple[str, str, int], ...]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{field_name} must be a non-empty list")
    parsed: list[tuple[str, str, int]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{field_name}[{idx}] must be a mapping")
        split = str(row.get("split", "")).strip()
        family = str(row.get("family", "")).strip()
        count = int(row.get("count", 0))
        if not split or not family or count <= 0:
            raise RuntimeError(
                f"{field_name}[{idx}] requires non-empty split/family and positive count"
            )
        parsed.append((split, family, count))
    return tuple(parsed)


def _parse_alpha_map(raw: Any, *, field_name: str) -> dict[str, tuple[float, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"{field_name} must be a non-empty mapping")
    parsed: dict[str, tuple[float, ...]] = {}
    for group, values in raw.items():
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"{field_name}.{group} must be a non-empty list")
        parsed[str(group)] = tuple(float(value) for value in values)
    return parsed


def _parse_required_material_counts(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError("pilot_required_material_counts must be a mapping when provided")
    return {str(group): int(count) for group, count in raw.items()}


def _parse_ranges(raw: Any, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{field_name} must be a non-empty list")
    parsed: list[tuple[float, float]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != 2:
            raise RuntimeError(f"{field_name}[{idx}] must contain exactly two values")
        low = float(row[0])
        high = float(row[1])
        if low <= 0.0 or high <= 0.0 or low >= high:
            raise RuntimeError(f"{field_name}[{idx}] must satisfy 0 < low < high")
        parsed.append((low, high))
    return tuple(parsed)


def _parse_group_matcher(group_name: str, raw: Any, *, field_name: str) -> MaterialGroupMatcher:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"{field_name}.{group_name} must be a mapping")
    prefixes = tuple(str(item) for item in raw.get("prefixes", []))
    exact_names = tuple(str(item) for item in raw.get("exact_names", []))
    regex = tuple(str(item) for item in raw.get("regex", []))
    return MaterialGroupMatcher(prefixes=prefixes, exact_names=exact_names, regex=regex)


def _parse_material_groups(
    raw: Any,
    *,
    field_name: str,
) -> tuple[str, dict[str, MaterialGroupMatcher]]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"{field_name} must be a mapping")
    default_group = str(raw.get("default_group", "")).strip()
    groups_raw = raw.get("groups")
    if not default_group:
        raise RuntimeError(f"{field_name}.default_group must be a non-empty string")
    if not isinstance(groups_raw, dict) or not groups_raw:
        raise RuntimeError(f"{field_name}.groups must be a non-empty mapping")
    parsed = {
        str(group_name): _parse_group_matcher(
            str(group_name),
            matcher_raw,
            field_name=f"{field_name}.groups",
        )
        for group_name, matcher_raw in groups_raw.items()
    }
    if default_group not in parsed:
        raise RuntimeError(f"{field_name}.default_group must exist in {field_name}.groups")
    return default_group, parsed


def _parse_split_ranges(
    raw: Any,
    *,
    field_name: str,
) -> dict[str, tuple[tuple[float, float], ...]]:
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"{field_name} must be a non-empty mapping")
    return {
        str(split): _parse_ranges(values, field_name=f"{field_name}.{split}")
        for split, values in raw.items()
    }


def _parse_group_split_ranges(
    raw: Any,
    *,
    field_name: str,
) -> dict[str, dict[str, tuple[tuple[float, float], ...]]]:
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"{field_name} must be a non-empty mapping")
    parsed: dict[str, dict[str, tuple[tuple[float, float], ...]]] = {}
    for split, split_payload in raw.items():
        if not isinstance(split_payload, dict) or not split_payload:
            raise RuntimeError(f"{field_name}.{split} must be a non-empty mapping")
        parsed[str(split)] = {
            str(group): _parse_ranges(
                values,
                field_name=f"{field_name}.{split}.{group}",
            )
            for group, values in split_payload.items()
        }
    return parsed


def _parse_float_map(raw: Any, *, field_name: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"{field_name} must be a non-empty mapping")
    return {str(key): float(value) for key, value in raw.items()}


def _parse_baseline_spec(baseline_type: str, payload: Any) -> BaselineSpec:
    if not isinstance(payload, dict):
        raise RuntimeError(f"baselines.{baseline_type} must be a mapping")
    default_group, matchers = _parse_material_groups(
        payload.get("material_groups"),
        field_name=f"baselines.{baseline_type}.material_groups",
    )
    reference_condition = payload.get("reference_condition")
    if not isinstance(reference_condition, dict):
        raise RuntimeError(f"baselines.{baseline_type}.reference_condition must be a mapping")
    family_b_raw = payload.get("family_b")
    if not isinstance(family_b_raw, dict):
        raise RuntimeError(f"baselines.{baseline_type}.family_b must be a mapping")
    family_bh_raw = payload.get("family_bh")
    if not isinstance(family_bh_raw, dict):
        raise RuntimeError(f"baselines.{baseline_type}.family_bh must be a mapping")
    other_alpha_range = family_bh_raw.get("other_alpha_range")
    if not isinstance(other_alpha_range, list) or len(other_alpha_range) != 2:
        raise RuntimeError(
            f"baselines.{baseline_type}.family_bh.other_alpha_range must contain two values"
        )
    target_groups = family_bh_raw.get("target_groups")
    if not isinstance(target_groups, list) or not target_groups:
        raise RuntimeError(
            f"baselines.{baseline_type}.family_bh.target_groups must be a non-empty list"
        )
    alpha_clip = family_b_raw.get("alpha_clip")
    if not isinstance(alpha_clip, list) or len(alpha_clip) != 2:
        raise RuntimeError(f"baselines.{baseline_type}.family_b.alpha_clip must contain two values")
    return BaselineSpec(
        baseline_type=baseline_type,
        default_group=default_group,
        material_group_matchers=matchers,
        base_quotas=_parse_quota_rows(
            payload.get("quotas"),
            field_name=f"baselines.{baseline_type}.quotas",
        ),
        bh_only_base_quotas=_parse_quota_rows(
            payload.get("bh_only_quotas"),
            field_name=f"baselines.{baseline_type}.bh_only_quotas",
        ),
        reference_condition_source=str(reference_condition.get("source", "")).strip(),
        reference_condition_tol=float(reference_condition.get("tol", 1.0e-8)),
        reference_condition_maxiter=int(reference_condition.get("maxiter", 20_000)),
        family_a=FamilyASpec(
            alpha_ranges=_parse_split_ranges(
                payload.get("family_a", {}).get("alpha_ranges"),
                field_name=f"baselines.{baseline_type}.family_a.alpha_ranges",
            )
        ),
        family_b=FamilyBSpec(
            sigma_ranges=_parse_group_split_ranges(
                family_b_raw.get("sigma_ranges"),
                field_name=f"baselines.{baseline_type}.family_b.sigma_ranges",
            ),
            rho=_parse_float_map(
                family_b_raw.get("rho"),
                field_name=f"baselines.{baseline_type}.family_b.rho",
            ),
            alpha_clip=(float(alpha_clip[0]), float(alpha_clip[1])),
        ),
        family_bh=FamilyBHSpec(
            family_name=str(family_bh_raw.get("family_name", "BH")).strip(),
            pathological_dataset_suffix=str(
                family_bh_raw.get("pathological_dataset_suffix", "-bh-v1")
            ).strip(),
            pilot_dir_name=str(family_bh_raw.get("pilot_dir_name", "bh_pilot")).strip(),
            target_groups=tuple(str(group) for group in target_groups),
            pilot_story_alphas=_parse_alpha_map(
                family_bh_raw.get("pilot_story_alphas"),
                field_name=f"baselines.{baseline_type}.family_bh.pilot_story_alphas",
            ),
            pilot_band_alphas=_parse_alpha_map(
                family_bh_raw.get("pilot_band_alphas"),
                field_name=f"baselines.{baseline_type}.family_bh.pilot_band_alphas",
            ),
            condition_ratio_min=float(family_bh_raw.get("condition_ratio_min")),
            condition_ratio_max=float(family_bh_raw.get("condition_ratio_max")),
            soft_alpha_log10_jitter=float(family_bh_raw.get("soft_alpha_log10_jitter")),
            other_alpha_range=(float(other_alpha_range[0]), float(other_alpha_range[1])),
            pilot_required_material_counts=_parse_required_material_counts(
                family_bh_raw.get("pilot_required_material_counts")
            ),
        ),
    )


@lru_cache(maxsize=None)
def _load_case_library_rules_cached(resolved_path: str) -> CaseLibraryRules:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset rules yaml not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"dataset rules yaml root must be a mapping: {path}")
    baselines_raw = payload.get("baselines")
    if not isinstance(baselines_raw, dict) or not baselines_raw:
        raise RuntimeError(
            f"dataset rules yaml must define a non-empty 'baselines' mapping: {path}"
        )
    rules = CaseLibraryRules(
        baselines={
            str(baseline_type): _parse_baseline_spec(str(baseline_type), raw_spec)
            for baseline_type, raw_spec in baselines_raw.items()
        }
    )
    return rules


def load_case_library_rules(dataset_rules_path: str | Path | None = None) -> CaseLibraryRules:
    """Load case-library rules from the default or user-provided YAML file."""
    path = _normalize_rules_path(dataset_rules_path)
    return _load_case_library_rules_cached(str(path.resolve()))


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _find_marker_line(lines: list[str], marker: str) -> int:
    matches = [idx for idx, line in enumerate(lines) if line.strip().upper() == marker.upper()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one '{marker}' marker, found {len(matches)}")
    return matches[0]


def _material_name_from_line(line: str) -> str:
    match = re.search(r"name\s*=\s*([^,\s]+)", line, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"material line is missing NAME=...: {line.strip()}")
    return match.group(1).strip()


def _parse_elastic_values(line: str) -> tuple[float, float]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise RuntimeError(f"invalid *ELASTIC value line: {line.rstrip()}")
    return float(parts[0]), float(parts[1])


def _find_elastic_value_line(lines: list[str], block_start: int, block_end: int) -> int:
    for idx in range(block_start, block_end):
        if lines[idx].strip().upper().startswith("*ELASTIC"):
            for value_idx in range(idx + 1, block_end):
                stripped = lines[value_idx].strip()
                if not stripped or stripped.startswith("**"):
                    continue
                if stripped.startswith("*"):
                    raise RuntimeError(
                        f"material block at line {block_start + 1} is missing an *ELASTIC value row"
                    )
                return value_idx
    raise RuntimeError(f"material block at line {block_start + 1} is missing *ELASTIC")


def parse_master_input(master_text: str) -> ParsedMasterInput:
    """Parse the mutable materials inside the marked master input region."""
    lines = master_text.splitlines(keepends=True)
    begin_idx = _find_marker_line(lines, MATERIALS_BEGIN)
    end_idx = _find_marker_line(lines, MATERIALS_END)
    if end_idx <= begin_idx:
        raise RuntimeError("materials end marker must appear after the begin marker")

    material_starts = [
        idx
        for idx in range(begin_idx + 1, end_idx)
        if lines[idx].strip().upper().startswith("*MATERIAL")
    ]
    if not material_starts:
        raise RuntimeError("no *MATERIAL blocks found inside the marked material region")

    materials: list[MaterialLayer] = []
    for order, block_start in enumerate(material_starts):
        next_candidates = [idx for idx in material_starts if idx > block_start]
        block_end = next_candidates[0] if next_candidates else end_idx
        name = _material_name_from_line(lines[block_start])
        elastic_value_line_idx = _find_elastic_value_line(lines, block_start, block_end)
        elastic_modulus, poisson_ratio = _parse_elastic_values(lines[elastic_value_line_idx])
        materials.append(
            MaterialLayer(
                name=name,
                group="unassigned",
                order=order,
                elastic_modulus=elastic_modulus,
                poisson_ratio=poisson_ratio,
                material_line_idx=block_start,
                elastic_value_line_idx=elastic_value_line_idx,
            )
        )

    names = [item.name.lower() for item in materials]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate material names inside the marked material region")
    return ParsedMasterInput(
        lines=tuple(lines),
        marker_begin_idx=begin_idx,
        marker_end_idx=end_idx,
        materials=tuple(materials),
    )


def _replace_first_value(line: str, new_value: float) -> str:
    line_ending_match = re.search(r"(\r?\n)$", line)
    line_ending = "" if line_ending_match is None else line_ending_match.group(1)
    body = line[: -len(line_ending)] if line_ending else line
    match = re.match(r"^(\s*)([^,]+)(.*)$", body)
    if match is None:
        raise RuntimeError(f"cannot replace elastic modulus in line: {line.rstrip()}")
    prefix, _old_value, suffix = match.groups()
    return f"{prefix}{new_value:.6e}{suffix}{line_ending}"


def render_case_input(parsed: ParsedMasterInput, e_values_by_name: dict[str, float]) -> str:
    """Render a concrete case input by replacing only marked elastic moduli."""
    lines = list(parsed.lines)
    expected = {material.name for material in parsed.materials}
    if set(e_values_by_name) != expected:
        missing = sorted(expected.difference(e_values_by_name))
        extra = sorted(set(e_values_by_name).difference(expected))
        raise RuntimeError(f"material update mismatch: missing={missing}, extra={extra}")
    for material in parsed.materials:
        lines[material.elastic_value_line_idx] = _replace_first_value(
            lines[material.elastic_value_line_idx], float(e_values_by_name[material.name])
        )
    return "".join(lines)


def _resolve_baseline_type(dataset_id: str, baseline_type: str | None) -> str:
    if baseline_type is not None and str(baseline_type).strip():
        return str(baseline_type).strip()
    dataset_key = str(dataset_id).strip().lower()
    if dataset_key in LEGACY_DATASET_BASELINE_TYPES:
        return LEGACY_DATASET_BASELINE_TYPES[dataset_key]
    raise ValueError(
        "baseline_type is required when dataset_id is not one of the historical built-in IDs"
    )


def _get_baseline_spec(
    dataset_id: str,
    baseline_type: str | None,
    dataset_rules_path: str | Path | None = None,
) -> BaselineSpec:
    resolved_type = _resolve_baseline_type(dataset_id, baseline_type)
    rules = load_case_library_rules(dataset_rules_path)
    try:
        return rules.baselines[resolved_type]
    except KeyError as exc:
        raise ValueError(f"dataset rules do not define baseline_type: {resolved_type}") from exc


def _match_material_group(name: str, matcher: MaterialGroupMatcher) -> bool:
    lowered = name.lower()
    if lowered in {item.lower() for item in matcher.exact_names}:
        return True
    if any(lowered.startswith(prefix.lower()) for prefix in matcher.prefixes):
        return True
    return any(
        re.search(pattern, name, flags=re.IGNORECASE) is not None for pattern in matcher.regex
    )


def _material_groups_for_baseline(
    parsed: ParsedMasterInput,
    baseline: BaselineSpec,
) -> dict[str, list[MaterialLayer]]:
    grouped = {group: [] for group in baseline.material_group_matchers}
    for material in parsed.materials:
        matched_groups = [
            group
            for group, matcher in baseline.material_group_matchers.items()
            if _match_material_group(material.name, matcher)
        ]
        if len(matched_groups) > 1:
            raise RuntimeError(
                f"material {material.name} matches multiple material groups: {matched_groups}"
            )
        group = matched_groups[0] if matched_groups else baseline.default_group
        grouped.setdefault(group, []).append(material)
    return grouped


def _base_quotas_for_baseline(
    dataset_id: str,
    baseline_type: str | None,
    dataset_rules_path: str | Path | None = None,
) -> tuple[tuple[str, str, int], ...]:
    return _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path).base_quotas


def resolve_pathological_dataset_id(
    dataset_id: str,
    *,
    baseline_type: str | None = None,
    dataset_rules_path: str | Path | None = None,
) -> str:
    baseline = _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path)
    return f"{dataset_id}{baseline.family_bh.pathological_dataset_suffix}"


def _scaled_counts_from_base(
    base_quotas: tuple[tuple[str, str, int], ...], num_cases: int
) -> list[tuple[str, str, int]]:
    base_total = sum(count for _split, _family, count in base_quotas)
    if base_total <= 0:
        raise ValueError("base quotas must contain a positive total count")
    exact = [
        (split, family, (count / float(base_total)) * float(num_cases))
        for split, family, count in base_quotas
    ]
    floors = [(split, family, int(math.floor(value))) for split, family, value in exact]
    assigned = sum(count for _split, _family, count in floors)
    remainders = [
        (value - math.floor(value), idx) for idx, (_split, _family, value) in enumerate(exact)
    ]
    remaining = num_cases - assigned
    for _fraction, idx in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        split, family, count = floors[idx]
        floors[idx] = (split, family, count + 1)
    return floors


def _scaled_counts(
    dataset_id: str,
    num_cases: int,
    baseline_type: str | None = None,
    dataset_rules_path: str | Path | None = None,
) -> list[tuple[str, str, int]]:
    return _scaled_counts_from_base(
        _base_quotas_for_baseline(dataset_id, baseline_type, dataset_rules_path),
        num_cases,
    )


def _build_case_assignments(
    base_quotas: tuple[tuple[str, str, int], ...],
    *,
    num_cases: int,
    seed: int,
    case_prefix: str,
) -> list[CaseAssignment]:
    if num_cases <= 0:
        raise ValueError("num_cases must be positive")
    rows: list[CaseAssignment] = []
    for split, family, count in _scaled_counts_from_base(base_quotas, num_cases):
        for _ in range(count):
            rows.append(CaseAssignment(case_id="", split=split, family=family))
    if len(rows) != num_cases:
        raise RuntimeError("failed to allocate the requested number of cases")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [
        CaseAssignment(case_id=f"{case_prefix}{idx + 1:04d}", split=item.split, family=item.family)
        for idx, item in enumerate(rows)
    ]


def build_case_plan(
    dataset_id: str,
    num_cases: int,
    seed: int,
    *,
    baseline_type: str | None = None,
    dataset_rules_path: str | Path | None = None,
) -> list[CaseAssignment]:
    """Create a deterministic shuffled list of split/family assignments."""
    return _build_case_assignments(
        _base_quotas_for_baseline(dataset_id, baseline_type, dataset_rules_path),
        num_cases=num_cases,
        seed=seed,
        case_prefix="case_",
    )


def _pathological_template_prefix(target_group: str, *, include_group: bool) -> str:
    if include_group:
        return f"bh_soft_{target_group}"
    return "bh_soft"


def _pathological_sampling_mode(target_group: str, mode: str, *, include_group: bool) -> str:
    if include_group:
        return f"BH-soft-{target_group}-{mode}"
    return f"BH-soft-{mode}"


def _build_pathological_pilot_templates(
    parsed: ParsedMasterInput,
    baseline: BaselineSpec,
) -> list[BhPilotTemplate]:
    materials_by_group = _material_groups_for_baseline(parsed, baseline)
    target_groups = baseline.family_bh.target_groups
    for group in target_groups:
        if not materials_by_group[group]:
            raise RuntimeError(
                f"{baseline.baseline_type} pilot expects at least one mutable {group} material"
            )
    for group, expected in baseline.family_bh.pilot_required_material_counts.items():
        actual = len(materials_by_group.get(group, []))
        if actual != int(expected):
            raise RuntimeError(
                f"{baseline.baseline_type} BH pilot expects exactly {expected} mutable {group} "
                f"materials inside the marked block, found {actual}"
            )
    templates: list[BhPilotTemplate] = []
    include_group = len(target_groups) > 1
    for target_group in target_groups:
        group_materials = materials_by_group[target_group]
        prefix = _pathological_template_prefix(target_group, include_group=include_group)
        story_mode = _pathological_sampling_mode(target_group, "story", include_group=include_group)
        band_mode = _pathological_sampling_mode(target_group, "band", include_group=include_group)
        for soft_idx in range(len(group_materials)):
            for alpha in baseline.family_bh.pilot_story_alphas[target_group]:
                templates.append(
                    BhPilotTemplate(
                        template_id=f"{prefix}_story_{soft_idx + 1}_{alpha:.0e}",
                        sampling_mode=story_mode,
                        target_group=target_group,
                        soft_layer_indices=(soft_idx,),
                        soft_alpha_values=(float(alpha),),
                    )
                )
        for start_idx in range(max(0, len(group_materials) - 1)):
            for alpha in baseline.family_bh.pilot_band_alphas[target_group]:
                templates.append(
                    BhPilotTemplate(
                        template_id=f"{prefix}_band_{start_idx + 1}_{start_idx + 2}_{alpha:.0e}",
                        sampling_mode=band_mode,
                        target_group=target_group,
                        soft_layer_indices=(start_idx, start_idx + 1),
                        soft_alpha_values=(float(alpha), float(alpha)),
                    )
                )
    return templates


def _reference_condition_for_dataset(
    dataset_root: Path,
    dataset_id: str,
    baseline: BaselineSpec,
) -> float:
    if baseline.reference_condition_source != "master_stiffness":
        raise RuntimeError(
            f"unsupported reference_condition.source for {baseline.baseline_type}: "
            f"{baseline.reference_condition_source}"
        )
    master_dir = dataset_root / "master"
    stiffness_path = _resolve_reference_stiffness_path(master_dir)
    if stiffness_path is None:
        raise FileNotFoundError(
            f"reference stiffness matrix not found for {dataset_id} under: {master_dir}"
        )
    return _estimate_reference_condition_number(
        stiffness_path,
        dataset_id=dataset_id,
        tol=baseline.reference_condition_tol,
        maxiter=baseline.reference_condition_maxiter,
    )


def _resolve_reference_stiffness_path(master_dir: Path) -> Path | None:
    if not master_dir.exists():
        return None
    inp_candidates = sorted(master_dir.glob("*.inp"))
    for inp_path in inp_candidates:
        exact = sorted(master_dir.glob(f"{inp_path.stem}_STIF*.mtx"))
        if exact:
            return exact[0]
    generic = sorted(master_dir.glob("*_STIF*.mtx"))
    if generic:
        return generic[0]
    return None


def _estimate_reference_condition_number(
    stiffness_path: Path,
    *,
    dataset_id: str,
    tol: float = 1.0e-8,
    maxiter: int = 20_000,
) -> float:
    rn, rd, cn, cd, val = read_mtx_5col(str(stiffness_path))
    A, *_rest = build_free_system_from_mtx(rn, rd, cn, cd, val)
    cond_meta = estimate_condition_number(A, tol=tol, maxiter=maxiter)
    if cond_meta["condition_number_status"] != "ok" or cond_meta["condition_number_est"] is None:
        raise RuntimeError(
            f"failed to estimate reference condition number for {dataset_id} from {stiffness_path}: "
            f"{cond_meta.get('condition_number_error')}"
        )
    return float(cond_meta["condition_number_est"])


def _copy_master_reference_artifacts(master_inp: Path, target_master_dir: Path) -> dict[str, Any]:
    target_master_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    shutil.copy2(master_inp, target_master_dir / "master.inp")
    copied["master_inp_copy"] = str(target_master_dir / "master.inp")
    if master_inp.name != "master.inp":
        shutil.copy2(master_inp, target_master_dir / master_inp.name)
        copied["master_inp_original_copy"] = str(target_master_dir / master_inp.name)
    for suffix in ("_STIF*.mtx", "_LOAD*.mtx"):
        for source_path in sorted(master_inp.parent.glob(f"{master_inp.stem}{suffix}")):
            target_path = target_master_dir / source_path.name
            shutil.copy2(source_path, target_path)
            copied[source_path.name] = str(target_path)
    dat_path = master_inp.with_suffix(".dat")
    if dat_path.exists():
        target_path = target_master_dir / dat_path.name
        shutil.copy2(dat_path, target_path)
        copied[dat_path.name] = str(target_path)
    return copied


def _soft_alpha_value(base_alpha: float, rng: np.random.Generator, *, jitter: float) -> float:
    log_alpha = math.log10(float(base_alpha))
    perturbed = log_alpha + float(rng.uniform(-jitter, jitter))
    return float(10.0**perturbed)


def _material_names_by_indices(
    materials: list[MaterialLayer], indices: tuple[int, ...]
) -> list[str]:
    return [materials[idx].name for idx in indices]


def _load_pathological_templates(
    dataset_root: Path,
    *,
    dataset_id: str,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    results_path = dataset_root / "meta" / "bh_pilot_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{dataset_id} BH pilot results not found; prepare and solve BH pilot cases first: "
            f"{results_path}"
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    accepted = list(payload.get("accepted_templates", []))
    if not accepted:
        raise RuntimeError(f"{dataset_id} BH pilot produced no accepted templates")
    c_ref = float(payload["reference_condition_number"])
    target = {
        "condition_target_min": float(payload["condition_target_min"]),
        "condition_target_max": float(payload["condition_target_max"]),
    }
    return c_ref, target, accepted


def _ranges_for_split(
    split: str,
    ranges_by_split: dict[str, tuple[tuple[float, float], ...]],
    *,
    field_name: str,
) -> tuple[tuple[float, float], ...]:
    if split in ranges_by_split:
        return ranges_by_split[split]
    if "default" in ranges_by_split:
        return ranges_by_split["default"]
    raise RuntimeError(f"{field_name} does not define sampling ranges for split={split}")


def _sample_from_ranges(
    ranges: tuple[tuple[float, float], ...],
    rng: np.random.Generator,
) -> float:
    low, high = ranges[int(rng.integers(0, len(ranges)))]
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _sample_sigma(
    split: str,
    group: str,
    rng: np.random.Generator,
    *,
    family_b: FamilyBSpec,
) -> float:
    split_ranges = family_b.sigma_ranges.get(split) or family_b.sigma_ranges.get("default")
    if split_ranges is None:
        raise RuntimeError(f"family_b.sigma_ranges does not define split={split}")
    ranges = split_ranges.get(group) or split_ranges.get("default")
    if ranges is None:
        raise RuntimeError(f"family_b.sigma_ranges.{split} does not define group={group}")
    low, high = ranges[int(rng.integers(0, len(ranges)))]
    return float(rng.uniform(low, high))


def _correlated_alphas(
    size: int, sigma: float, rho: float, rng: np.random.Generator, clip: tuple[float, float]
) -> np.ndarray:
    if size <= 0:
        return np.empty((0,), dtype=np.float64)
    field = np.zeros(size, dtype=np.float64)
    field[0] = float(rng.standard_normal())
    for idx in range(1, size):
        noise = float(rng.standard_normal())
        field[idx] = rho * field[idx - 1] + math.sqrt(max(0.0, 1.0 - rho * rho)) * noise
    alphas = np.exp(sigma * field)
    return np.clip(alphas, clip[0], clip[1]).astype(np.float64, copy=False)


def sample_material_parameters(
    parsed: ParsedMasterInput,
    assignment: CaseAssignment,
    rng: np.random.Generator,
    *,
    dataset_id: str,
    baseline_type: str | None = None,
    dataset_rules_path: str | Path | None = None,
    bh_reference_condition: float | None = None,
    bh_target: dict[str, float] | None = None,
    bh_templates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sample elastic moduli for one case."""
    variable_material_names = [material.name for material in parsed.materials]
    baseline = _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path)
    grouped_materials = _material_groups_for_baseline(parsed, baseline)
    frame_materials = list(grouped_materials.get("frame", []))
    soil_materials = list(grouped_materials.get("soil", []))
    e0_by_name = {material.name: float(material.elastic_modulus) for material in parsed.materials}
    e_values_by_name = dict(e0_by_name)
    metadata: dict[str, Any] = {
        "variable_material_names": variable_material_names,
        "frame_material_names": [material.name for material in frame_materials],
        "soil_material_names": [material.name for material in soil_materials],
        "E0_values": [float(e0_by_name[name]) for name in variable_material_names],
        "E_values": [],
        "alpha_global": None,
        "alpha_frame": None,
        "alpha_soil": None,
        "sigma_f": None,
        "sigma_s": None,
        "rho_f": None,
        "rho_s": None,
        "target_group": None,
        "sampling_mode": None,
        "template_id": None,
        "soft_layer_indices": [],
        "soft_layer_material_names": [],
        "soft_alpha_values": [],
        "condition_ratio_to_ref": None,
        "condition_target_min": None,
        "condition_target_max": None,
        "bh_attempt_index": None,
    }

    if assignment.family == "A":
        alpha = _sample_from_ranges(
            _ranges_for_split(
                assignment.split,
                baseline.family_a.alpha_ranges,
                field_name=f"{baseline.baseline_type}.family_a.alpha_ranges",
            ),
            rng,
        )
        for material in parsed.materials:
            e_values_by_name[material.name] = alpha * float(material.elastic_modulus)
        metadata["alpha_global"] = alpha
        metadata["sampling_mode"] = "A-global-scale"
    elif assignment.family == "B":
        if frame_materials:
            sigma_f = _sample_sigma(
                assignment.split,
                "frame",
                rng,
                family_b=baseline.family_b,
            )
            alpha_frame = _correlated_alphas(
                len(frame_materials),
                sigma_f,
                float(baseline.family_b.rho["frame"]),
                rng,
                clip=baseline.family_b.alpha_clip,
            )
            for material, alpha in zip(frame_materials, alpha_frame.tolist(), strict=True):
                e_values_by_name[material.name] = alpha * float(material.elastic_modulus)
            metadata["alpha_frame"] = [float(item) for item in alpha_frame.tolist()]
            metadata["sigma_f"] = float(sigma_f)
            metadata["rho_f"] = float(baseline.family_b.rho["frame"])
        if soil_materials:
            sigma_s = _sample_sigma(
                assignment.split,
                "soil",
                rng,
                family_b=baseline.family_b,
            )
            alpha_soil = _correlated_alphas(
                len(soil_materials),
                sigma_s,
                float(baseline.family_b.rho["soil"]),
                rng,
                clip=baseline.family_b.alpha_clip,
            )
            for material, alpha in zip(soil_materials, alpha_soil.tolist(), strict=True):
                e_values_by_name[material.name] = alpha * float(material.elastic_modulus)
            metadata["alpha_soil"] = [float(item) for item in alpha_soil.tolist()]
            metadata["sigma_s"] = float(sigma_s)
            metadata["rho_s"] = float(baseline.family_b.rho["soil"])
        metadata["sampling_mode"] = "B-normal"
    elif assignment.family == "BH":
        if bh_templates is None or not bh_templates:
            raise RuntimeError("BH sampling requires accepted pilot templates")
        if bh_reference_condition is None or bh_target is None:
            raise RuntimeError("BH sampling requires reference condition number and target window")
        template = bh_templates[int(rng.integers(0, len(bh_templates)))]
        target_group = str(template.get("target_group", "frame"))
        if target_group not in baseline.family_bh.target_groups:
            raise RuntimeError(f"unsupported BH target_group: {target_group}")
        soft_indices = tuple(int(idx) for idx in template["soft_layer_indices"])
        target_materials = frame_materials if target_group == "frame" else soil_materials
        if not target_materials:
            raise RuntimeError(
                f"{baseline.baseline_type} BH sampling expects mutable {target_group} materials"
            )
        soft_alpha_values: list[float] = []
        alpha_frame = [1.0] * len(frame_materials)
        alpha_soil = [1.0] * len(soil_materials)
        for material_idx, material in enumerate(target_materials):
            if material_idx in soft_indices:
                soft_pos = soft_indices.index(material_idx)
                base_alpha = float(template["soft_alpha_values"][soft_pos])
                alpha = _soft_alpha_value(
                    base_alpha,
                    rng,
                    jitter=baseline.family_bh.soft_alpha_log10_jitter,
                )
                soft_alpha_values.append(alpha)
            else:
                alpha = _sample_from_ranges((baseline.family_bh.other_alpha_range,), rng)
            e_values_by_name[material.name] = alpha * float(material.elastic_modulus)
            if target_group == "frame":
                alpha_frame[material_idx] = alpha
            else:
                alpha_soil[material_idx] = alpha
        metadata["sampling_mode"] = str(template["sampling_mode"])
        metadata["template_id"] = str(template["template_id"])
        metadata["target_group"] = target_group
        metadata["alpha_frame"] = [float(value) for value in alpha_frame] if frame_materials else []
        metadata["alpha_soil"] = [float(value) for value in alpha_soil] if soil_materials else []
        metadata["soft_layer_indices"] = [int(idx) for idx in soft_indices]
        metadata["soft_layer_material_names"] = _material_names_by_indices(
            target_materials, soft_indices
        )
        metadata["soft_alpha_values"] = [float(value) for value in soft_alpha_values]
        metadata["condition_target_min"] = float(bh_target["condition_target_min"])
        metadata["condition_target_max"] = float(bh_target["condition_target_max"])
    else:
        raise ValueError(f"unsupported family: {assignment.family}")

    metadata["E_values"] = [float(e_values_by_name[name]) for name in variable_material_names]
    metadata["E0_by_material"] = {name: float(value) for name, value in e0_by_name.items()}
    metadata["E_by_material"] = {name: float(value) for name, value in e_values_by_name.items()}
    metadata["e_values_by_name"] = e_values_by_name
    return metadata


def run_abaqus_job(
    case_dir: Path,
    case_name: str,
    *,
    abaqus_command: str = "abaqus",
    cpus: int = 4,
    extra_args: list[str] | None = None,
) -> int:
    """Run Abaqus in the case directory and capture stdout/stderr to ``run.log``."""
    log_path = case_dir / "run.log"
    env = dict(os.environ)
    executable = str(abaqus_command).strip() or "abaqus"
    if executable.lower().endswith(".bat"):
        executable_text = f'"{executable}"'
    else:
        executable_text = executable
    command_text = f"{executable_text} job={case_name} input={case_name}.inp cpus={int(cpus)}"
    if extra_args:
        command_text += " " + " ".join(extra_args)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_SHLVL",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    completed = subprocess.run(
        command_text,
        cwd=case_dir,
        text=True,
        check=False,
        shell=True,
        env=env,
    )
    log_path.write_text(
        f"command: {command_text}\nreturncode: {completed.returncode}\n",
        encoding="utf-8",
    )
    return int(completed.returncode)


def _resolve_case_artifacts(case_dir: Path, case_name: str) -> dict[str, Any]:
    stiffness_files = sorted(case_dir.glob(f"{case_name}_STIF*.mtx"))
    load_files = sorted(case_dir.glob(f"{case_name}_LOAD*.mtx"))
    dat_path = case_dir / f"{case_name}.dat"
    return {
        "stiffness_mtx_files": [str(path) for path in stiffness_files],
        "load_mtx_files": [str(path) for path in load_files],
        "stiffness_mtx_path": None if not stiffness_files else str(stiffness_files[0]),
        "load_mtx_path": None if not load_files else str(load_files[0]),
        "dat_path": str(dat_path) if dat_path.exists() else None,
    }


def _prune_case_outputs(case_dir: Path, case_name: str) -> None:
    keep_patterns = [
        re.compile(rf"^{re.escape(case_name)}\.inp$", re.IGNORECASE),
        re.compile(rf"^{re.escape(case_name)}\.dat$", re.IGNORECASE),
        re.compile(rf"^{re.escape(case_name)}_STIF.*\.mtx$", re.IGNORECASE),
        re.compile(rf"^{re.escape(case_name)}_LOAD.*\.mtx$", re.IGNORECASE),
        re.compile(r"^sample_meta\.json$", re.IGNORECASE),
        re.compile(r"^run\.log$", re.IGNORECASE),
    ]
    for entry in case_dir.iterdir():
        if any(pattern.match(entry.name) for pattern in keep_patterns):
            continue
        if entry.is_file():
            try:
                entry.unlink()
            except PermissionError:
                continue
        elif entry.is_dir():
            shutil.rmtree(entry)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_jsonify(value), ensure_ascii=False)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_samples_csv(meta_dir: Path, records: list[dict[str, Any]]) -> Path:
    csv_path = meta_dir / "samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record.get(field)) for field in CSV_FIELDS})
    return csv_path


def _load_case_meta(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / "sample_meta.json").read_text(encoding="utf-8"))


def _write_case_meta(case_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(case_dir / "sample_meta.json", payload)


def _scan_case_dirs(dataset_root: Path) -> list[Path]:
    cases_dir = dataset_root / "cases"
    results: list[Path] = []
    for pattern in ("case_*", "pilot_*"):
        results.extend(
            path
            for path in cases_dir.glob(pattern)
            if path.is_dir() and (path / "sample_meta.json").exists()
        )
    return sorted(results)


def _write_abaqus_env(case_dir: Path) -> Path:
    env_path = case_dir / "abaqus_v6.env"
    env_path.write_text(
        "\n".join(
            [
                "# Generated by DPCG phase-A case preparation.",
                "ask_delete = OFF",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return env_path


def _write_run_case_cmd(
    case_dir: Path,
    case_name: str,
    *,
    abaqus_command: str,
    cpus: int,
    extra_args: list[str] | None = None,
) -> Path:
    cmd_path = case_dir / "run_case.cmd"
    command_parts = [
        f'"{abaqus_command}"',
        f"job={case_name}",
        f"input={case_name}.inp",
        "interactive",
        f"cpus={int(cpus)}",
    ]
    if extra_args:
        command_parts.extend(extra_args)
    command_line = " ".join(command_parts)
    cmd_path.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                "cd /d %~dp0",
                "set LOG=%~dp0run.log",
                '> "%LOG%" echo command: ' + command_line,
                '>> "%LOG%" echo started: %DATE% %TIME%',
                "call " + command_line + ' >> "%LOG%" 2>&1',
                "set EXITCODE=%ERRORLEVEL%",
                '>> "%LOG%" echo exit_code: %EXITCODE%',
                '>> "%LOG%" echo finished: %DATE% %TIME%',
                "exit /b %EXITCODE%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return cmd_path


def _write_cases_to_run_csv(dataset_root: Path, records: list[dict[str, Any]]) -> Path:
    csv_path = dataset_root / "meta" / "cases_to_run.csv"
    fieldnames = [
        "dataset_id",
        "case_id",
        "split",
        "family",
        "case_dir",
        "inp_path",
        "env_path",
        "run_case_cmd_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            case_dir = Path(str(record["inp_path"])).parent.resolve()
            writer.writerow(
                {
                    "dataset_id": record["dataset_id"],
                    "case_id": record["case_id"],
                    "split": record["split"],
                    "family": record["family"],
                    "case_dir": str(case_dir),
                    "inp_path": str(Path(str(record["inp_path"])).resolve()),
                    "env_path": str(Path(str(record["env_path"])).resolve()),
                    "run_case_cmd_path": str(Path(str(record["run_case_cmd_path"])).resolve()),
                }
            )
    return csv_path


def _write_dispatch_scripts(dataset_root: Path, cases_csv: Path) -> tuple[Path, Path]:
    ps1_path = dataset_root / "run_all_cases.ps1"
    cmd_path = dataset_root / "run_all_cases.cmd"
    ps1_path.write_text(
        "\n".join(
            [
                "param(",
                "    [int]$MaxParallelJobs = 1,",
                "    [int]$PollIntervalMs = 500",
                ")",
                '$ErrorActionPreference = "Stop"',
                'if ($MaxParallelJobs -lt 1) { throw "MaxParallelJobs must be >= 1" }',
                '$casesCsv = Join-Path $PSScriptRoot "meta\\cases_to_run.csv"',
                '$summaryCsv = Join-Path $PSScriptRoot "meta\\run_summary.csv"',
                "$cases = Import-Csv -Path $casesCsv",
                "function New-CaseResult {",
                "    param(",
                "        [pscustomobject]$Case,",
                "        [int]$ExitCode,",
                "        [datetime]$StartedAt,",
                "        [datetime]$FinishedAt,",
                '        [string]$LaunchError = ""',
                "    )",
                "    $caseDir = [string]$Case.case_dir",
                "    $caseId = [string]$Case.case_id",
                '    $datPath = Join-Path $caseDir ($caseId + ".dat")',
                '    $stifFiles = Get-ChildItem -Path $caseDir -Filter ($caseId + "_STIF*.mtx") -ErrorAction SilentlyContinue',
                '    $loadFiles = Get-ChildItem -Path $caseDir -Filter ($caseId + "_LOAD*.mtx") -ErrorAction SilentlyContinue',
                '    $status = if ($ExitCode -eq 0 -and [string]::IsNullOrEmpty($LaunchError) -and (Test-Path $datPath) -and $stifFiles.Count -gt 0 -and $loadFiles.Count -gt 0) { "success" } else { "failed" }',
                "    return [pscustomobject]@{",
                "        dataset_id = $Case.dataset_id",
                "        case_id = $Case.case_id",
                "        split = $Case.split",
                "        family = $Case.family",
                "        case_dir = $Case.case_dir",
                "        run_case_cmd_path = $Case.run_case_cmd_path",
                "        exit_code = $ExitCode",
                '        dat_path = if (Test-Path $datPath) { $datPath } else { "" }',
                "        stiffness_mtx_count = $stifFiles.Count",
                "        load_mtx_count = $loadFiles.Count",
                "        launch_error = $LaunchError",
                "        started_at = $StartedAt.ToString('o')",
                "        finished_at = $FinishedAt.ToString('o')",
                "        duration_sec = ($FinishedAt - $StartedAt).TotalSeconds",
                "        run_status = $status",
                "    }",
                "}",
                '$results = New-Object "System.Collections.Generic.List[object]"',
                '$running = New-Object "System.Collections.Generic.List[object]"',
                "$nextIndex = 0",
                "while ($nextIndex -lt $cases.Count -or $running.Count -gt 0) {",
                "    while ($nextIndex -lt $cases.Count -and $running.Count -lt $MaxParallelJobs) {",
                "        $case = $cases[$nextIndex]",
                "        $startedAt = Get-Date",
                "        try {",
                "            $process = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', ([string]$case.run_case_cmd_path) -WorkingDirectory ([string]$case.case_dir) -PassThru -WindowStyle Hidden",
                "            $running.Add([pscustomobject]@{ case = $case; process = $process; started_at = $startedAt })",
                "        } catch {",
                "            $finishedAt = Get-Date",
                "            $results.Add((New-CaseResult -Case $case -ExitCode -1 -StartedAt $startedAt -FinishedAt $finishedAt -LaunchError $_.Exception.Message))",
                "        }",
                "        $nextIndex += 1",
                "    }",
                "    if ($running.Count -eq 0) {",
                "        continue",
                "    }",
                "    Start-Sleep -Milliseconds $PollIntervalMs",
                "    for ($idx = $running.Count - 1; $idx -ge 0; $idx -= 1) {",
                "        $item = $running[$idx]",
                "        if ($item.process.HasExited) {",
                "            $item.process.Refresh()",
                "            $finishedAt = Get-Date",
                "            $results.Add((New-CaseResult -Case $item.case -ExitCode $item.process.ExitCode -StartedAt $item.started_at -FinishedAt $finishedAt))",
                "            $running.RemoveAt($idx)",
                "        }",
                "    }",
                "}",
                "$results | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8",
                'Write-Host "Run summary written to $summaryCsv (MaxParallelJobs=$MaxParallelJobs)"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cmd_path.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                'powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run_all_cases.ps1" %*',
                "exit /b %ERRORLEVEL%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return ps1_path, cmd_path


def prepare_dataset_pilot(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    dataset_id: str,
    baseline_type: str | None = None,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 4,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare deterministic BH pilot cases for one configured dataset."""
    baseline = _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path)
    dataset_path = Path(dataset_root)
    master_path = Path(master_inp)
    pilot_root = dataset_path / "meta" / baseline.family_bh.pilot_dir_name
    pilot_master_dir = pilot_root / "master"
    pilot_cases_dir = pilot_root / "cases"
    pilot_meta_dir = pilot_root / "meta"
    pilot_master_dir.mkdir(parents=True, exist_ok=True)
    pilot_cases_dir.mkdir(parents=True, exist_ok=True)
    pilot_meta_dir.mkdir(parents=True, exist_ok=True)

    master_text = master_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_master_input(master_text)
    base_master_dir = dataset_path / "master"
    copied_master_files = _copy_master_reference_artifacts(master_path, base_master_dir)
    templates = _build_pathological_pilot_templates(parsed, baseline)
    grouped_materials = _material_groups_for_baseline(parsed, baseline)
    frame_materials = list(grouped_materials.get("frame", []))
    soil_materials = list(grouped_materials.get("soil", []))
    c_ref = _reference_condition_for_dataset(dataset_path, dataset_id, baseline)
    target_min = float(c_ref * baseline.family_bh.condition_ratio_min)
    target_max = float(c_ref * baseline.family_bh.condition_ratio_max)
    _copy_master_reference_artifacts(master_path, pilot_master_dir)

    records: list[dict[str, Any]] = []
    pilot_assignments = [
        CaseAssignment(
            case_id=f"pilot_{idx + 1:03d}",
            split="pilot",
            family=baseline.family_bh.family_name,
        )
        for idx in range(len(templates))
    ]
    for assignment, template in zip(pilot_assignments, templates, strict=True):
        case_dir = pilot_cases_dir / assignment.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        e_values_by_name = {
            material.name: float(material.elastic_modulus) for material in parsed.materials
        }
        alpha_frame = [1.0] * len(frame_materials)
        alpha_soil = [1.0] * len(soil_materials)
        soft_indices = tuple(int(idx) for idx in template.soft_layer_indices)
        target_materials = frame_materials if template.target_group == "frame" else soil_materials
        for material_idx, material in enumerate(target_materials):
            if material_idx in soft_indices:
                soft_pos = soft_indices.index(material_idx)
                alpha = float(template.soft_alpha_values[soft_pos])
            else:
                alpha = 1.0
            e_values_by_name[material.name] = alpha * float(material.elastic_modulus)
            if template.target_group == "frame":
                alpha_frame[material_idx] = alpha
            else:
                alpha_soil[material_idx] = alpha
        inp_text = render_case_input(parsed, e_values_by_name)
        inp_path = case_dir / f"{assignment.case_id}.inp"
        inp_path.write_text(inp_text, encoding="utf-8")
        env_path = _write_abaqus_env(case_dir)
        run_case_cmd_path = _write_run_case_cmd(
            case_dir,
            assignment.case_id,
            abaqus_command=abaqus_command,
            cpus=cpus,
        )
        (case_dir / "run.log").write_text("", encoding="utf-8")
        record = {
            "dataset_id": dataset_id,
            "baseline_type": baseline.baseline_type,
            "case_id": assignment.case_id,
            "split": "pilot",
            "family": baseline.family_bh.family_name,
            "sampling_mode": template.sampling_mode,
            "template_id": template.template_id,
            "target_group": template.target_group,
            "soft_layer_indices": [int(idx) for idx in soft_indices],
            "soft_layer_material_names": _material_names_by_indices(target_materials, soft_indices),
            "soft_alpha_values": [float(value) for value in template.soft_alpha_values],
            "condition_ratio_to_ref": None,
            "condition_target_min": target_min,
            "condition_target_max": target_max,
            "bh_attempt_index": 1,
            "run_status": "pending",
            "validation_status": "pending",
            "master_inp_sha256": sha256_file(master_path),
            "generated_inp_sha256": sha256_file(inp_path),
            "inp_path": str(inp_path),
            "env_path": str(env_path),
            "run_case_cmd_path": str(run_case_cmd_path),
            "run_log_path": str(case_dir / "run.log"),
            "abaqus_returncode": None,
            "run_error": None,
            "stiffness_mtx_files": [],
            "load_mtx_files": [],
            "stiffness_mtx_path": None,
            "load_mtx_path": None,
            "dat_path": None,
            "variable_material_names": [material.name for material in parsed.materials],
            "frame_material_names": [material.name for material in frame_materials],
            "soil_material_names": [material.name for material in soil_materials],
            "E0_values": [float(material.elastic_modulus) for material in parsed.materials],
            "E_values": [float(e_values_by_name[material.name]) for material in parsed.materials],
            "alpha_global": None,
            "alpha_frame": [float(value) for value in alpha_frame] if frame_materials else [],
            "alpha_soil": [float(value) for value in alpha_soil] if soil_materials else [],
            "sigma_f": None,
            "sigma_s": None,
            "rho_f": None,
            "rho_s": None,
        }
        _write_case_meta(case_dir, record)
        records.append(record)

    csv_path = _write_samples_csv(pilot_meta_dir, records)
    cases_to_run_csv = _write_cases_to_run_csv(pilot_root, records)
    run_all_ps1, run_all_cmd = _write_dispatch_scripts(pilot_root, cases_to_run_csv)
    summary = {
        "dataset_id": dataset_id,
        "baseline_type": baseline.baseline_type,
        "pilot_root": str(pilot_root),
        "master_reference_dir": str(base_master_dir),
        "master_reference_files": copied_master_files,
        "reference_condition_number": c_ref,
        "condition_target_min": target_min,
        "condition_target_max": target_max,
        "num_cases": len(records),
        "samples_csv": str(csv_path),
        "cases_to_run_csv": str(cases_to_run_csv),
        "run_all_cases_ps1": str(run_all_ps1),
        "run_all_cases_cmd": str(run_all_cmd),
    }
    _write_json(pilot_meta_dir / "dataset_summary.json", summary)
    return summary


def prepare_frame33_bh_pilot(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 4,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare deterministic BH pilot cases for frame33-b under ``meta/bh_pilot``."""
    return prepare_dataset_pilot(
        master_inp,
        dataset_root,
        dataset_id="frame33-b",
        baseline_type="pure_frame",
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


def prepare_frame99_bh_pilot(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 4,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    return prepare_dataset_pilot(
        master_inp,
        dataset_root,
        dataset_id="frame99-b",
        baseline_type="pure_frame",
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


def prepare_station_bh_pilot(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 4,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    return prepare_dataset_pilot(
        master_inp,
        dataset_root,
        dataset_id="station3622-b",
        baseline_type="frame_soil_mixed",
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


def collect_dataset_pilot_results(
    dataset_root: str | Path,
    *,
    dataset_id: str,
    baseline_type: str | None = None,
    condition_number_tol: float = 1e-8,
    condition_number_maxiter: int = 20_000,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect validated BH pilot cases and write accepted templates for one dataset."""
    baseline = _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path)
    dataset_path = Path(dataset_root)
    pilot_root = dataset_path / "meta" / baseline.family_bh.pilot_dir_name
    if not pilot_root.exists():
        raise FileNotFoundError(f"BH pilot root not found: {pilot_root}")
    c_ref = _reference_condition_for_dataset(dataset_path, dataset_id, baseline)
    target_min = float(c_ref * baseline.family_bh.condition_ratio_min)
    target_max = float(c_ref * baseline.family_bh.condition_ratio_max)
    validate_case_library(
        pilot_root,
        estimate_condition_number_enabled=True,
        condition_number_tol=condition_number_tol,
        condition_number_maxiter=condition_number_maxiter,
    )

    all_results: list[dict[str, Any]] = []
    accepted_templates: list[dict[str, Any]] = []
    for case_dir in _scan_case_dirs(pilot_root):
        record = _load_case_meta(case_dir)
        if record.get("validation_status") != "success":
            all_results.append(record)
            continue
        cond = float(record["condition_number_est"])
        ratio = float(cond / c_ref)
        record["condition_ratio_to_ref"] = ratio
        record["condition_target_min"] = target_min
        record["condition_target_max"] = target_max
        record["accepted_for_bh_release"] = bool(target_min <= cond <= target_max)
        _write_case_meta(case_dir, record)
        all_results.append(record)
        if record["accepted_for_bh_release"]:
            accepted_templates.append(
                {
                    "template_id": record["template_id"],
                    "sampling_mode": record["sampling_mode"],
                    "target_group": record.get("target_group"),
                    "soft_layer_indices": [int(idx) for idx in record["soft_layer_indices"]],
                    "soft_layer_material_names": list(record["soft_layer_material_names"]),
                    "soft_alpha_values": [float(value) for value in record["soft_alpha_values"]],
                    "condition_number_est": cond,
                    "condition_ratio_to_ref": ratio,
                }
            )

    payload = {
        "dataset_id": dataset_id,
        "baseline_type": baseline.baseline_type,
        "pilot_root": str(pilot_root),
        "reference_condition_number": c_ref,
        "condition_target_min": target_min,
        "condition_target_max": target_max,
        "accepted_templates": accepted_templates,
        "all_results": all_results,
    }
    _write_json(dataset_path / "meta" / "bh_pilot_results.json", payload)
    return payload


def collect_frame33_bh_pilot_results(
    dataset_root: str | Path,
    *,
    condition_number_tol: float = 1e-8,
    condition_number_maxiter: int = 20_000,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect validated BH pilot cases and write accepted templates for frame33-b."""
    return collect_dataset_pilot_results(
        dataset_root,
        dataset_id="frame33-b",
        baseline_type="pure_frame",
        condition_number_tol=condition_number_tol,
        condition_number_maxiter=condition_number_maxiter,
        dataset_rules_path=dataset_rules_path,
    )


def prepare_dataset_bh_cases(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    dataset_id: str,
    baseline_type: str | None = None,
    source_dataset_root: str | Path,
    seed: int = 42,
    num_cases: int = 120,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 1,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare a BH-only case library using accepted pilot templates."""
    baseline = _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path)
    pathological_dataset_id = f"{dataset_id}{baseline.family_bh.pathological_dataset_suffix}"
    dataset_path = Path(dataset_root)
    source_root = Path(source_dataset_root)
    master_path = Path(master_inp)
    master_dir = dataset_path / "master"
    cases_dir = dataset_path / "cases"
    meta_dir = dataset_path / "meta"
    master_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    bh_results_path = source_root / "meta" / "bh_pilot_results.json"
    if not bh_results_path.exists():
        raise FileNotFoundError(f"{dataset_id} BH pilot results not found: {bh_results_path}")
    bh_reference_condition, bh_target, bh_templates = _load_pathological_templates(
        source_root,
        dataset_id=dataset_id,
    )

    master_text = master_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_master_input(master_text)
    copied_master_files = _copy_master_reference_artifacts(master_path, master_dir)
    shutil.copy2(bh_results_path, meta_dir / "bh_pilot_results.json")

    assignments = _build_case_assignments(
        baseline.bh_only_base_quotas,
        num_cases=num_cases,
        seed=seed,
        case_prefix="case_bh_",
    )
    _write_json(
        meta_dir / "split.json",
        {
            "dataset_id": pathological_dataset_id,
            "baseline_type": baseline.baseline_type,
            "cases": [asdict(item) for item in assignments],
        },
    )
    _write_json(
        meta_dir / "materials_manifest.json",
        {
            "dataset_id": pathological_dataset_id,
            "baseline_type": baseline.baseline_type,
            "source_dataset_root": str(source_root),
            "master_inp": str(master_path),
            "mutable_materials": [asdict(item) for item in parsed.materials],
            "bh_pilot_results_path": str(bh_results_path),
        },
    )

    np_rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for assignment in assignments:
        case_name = assignment.case_id
        case_dir = cases_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        sample_info = sample_material_parameters(
            parsed,
            assignment,
            np_rng,
            dataset_id=dataset_id,
            baseline_type=baseline.baseline_type,
            dataset_rules_path=dataset_rules_path,
            bh_reference_condition=bh_reference_condition,
            bh_target=bh_target,
            bh_templates=bh_templates,
        )
        sample_info["bh_attempt_index"] = 1
        new_inp_text = render_case_input(parsed, sample_info["e_values_by_name"])
        inp_path = case_dir / f"{case_name}.inp"
        inp_path.write_text(new_inp_text, encoding="utf-8")
        env_path = _write_abaqus_env(case_dir)
        run_case_cmd_path = _write_run_case_cmd(
            case_dir,
            case_name,
            abaqus_command=abaqus_command,
            cpus=cpus,
        )
        (case_dir / "run.log").write_text("", encoding="utf-8")
        record = {
            "dataset_id": pathological_dataset_id,
            "source_dataset_id": dataset_id,
            "baseline_type": baseline.baseline_type,
            "case_id": case_name,
            "split": assignment.split,
            "family": baseline.family_bh.family_name,
            "run_status": "pending",
            "validation_status": "pending",
            "master_inp_sha256": sha256_file(master_path),
            "generated_inp_sha256": sha256_file(inp_path),
            "inp_path": str(inp_path),
            "env_path": str(env_path),
            "run_case_cmd_path": str(run_case_cmd_path),
            "run_log_path": str(case_dir / "run.log"),
            "abaqus_returncode": None,
            "run_error": None,
            "stiffness_mtx_files": [],
            "load_mtx_files": [],
            "stiffness_mtx_path": None,
            "load_mtx_path": None,
            "dat_path": None,
        }
        record.update(
            {key: value for key, value in sample_info.items() if key != "e_values_by_name"}
        )
        _write_case_meta(case_dir, record)
        records.append(record)

    csv_path = _write_samples_csv(meta_dir, records)
    cases_to_run_csv = _write_cases_to_run_csv(dataset_path, records)
    run_all_ps1, run_all_cmd = _write_dispatch_scripts(dataset_path, cases_to_run_csv)
    summary = {
        "dataset_id": pathological_dataset_id,
        "baseline_type": baseline.baseline_type,
        "source_dataset_root": str(source_root),
        "master_inp": str(master_path),
        "master_inp_copy": str(master_dir / "master.inp"),
        "master_reference_files": copied_master_files,
        "num_cases": int(num_cases),
        "n_prepared": int(len(records)),
        "n_pending": int(len(records)),
        "n_success": 0,
        "n_failed": 0,
        "samples_csv": str(csv_path),
        "cases_to_run_csv": str(cases_to_run_csv),
        "run_all_cases_ps1": str(run_all_ps1),
        "run_all_cases_cmd": str(run_all_cmd),
        "bh_pilot_results_path": str(meta_dir / "bh_pilot_results.json"),
        "consistency_ok": None,
        "consistency_error": None,
    }
    _write_json(meta_dir / "dataset_summary.json", summary)
    return summary


def prepare_frame33_bh_cases(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    source_dataset_root: str | Path,
    seed: int = 42,
    num_cases: int = 120,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 1,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare a BH-only case library using accepted frame33-b pilot templates."""
    return prepare_dataset_bh_cases(
        master_inp,
        dataset_root,
        dataset_id="frame33-b",
        baseline_type="pure_frame",
        source_dataset_root=source_dataset_root,
        seed=seed,
        num_cases=num_cases,
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


def prepare_frame99_bh_cases(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    source_dataset_root: str | Path,
    seed: int = 42,
    num_cases: int = 120,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 1,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    return prepare_dataset_bh_cases(
        master_inp,
        dataset_root,
        dataset_id="frame99-b",
        baseline_type="pure_frame",
        source_dataset_root=source_dataset_root,
        seed=seed,
        num_cases=num_cases,
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


def prepare_station_bh_cases(
    master_inp: str | Path,
    dataset_root: str | Path,
    *,
    source_dataset_root: str | Path,
    seed: int = 42,
    num_cases: int = 120,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 1,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    return prepare_dataset_bh_cases(
        master_inp,
        dataset_root,
        dataset_id="station3622-b",
        baseline_type="frame_soil_mixed",
        source_dataset_root=source_dataset_root,
        seed=seed,
        num_cases=num_cases,
        abaqus_command=abaqus_command,
        cpus=cpus,
        dataset_rules_path=dataset_rules_path,
    )


def build_case_library(
    dataset_id: str,
    master_inp: str | Path,
    output_root: str | Path,
    *,
    baseline_type: str | None = None,
    seed: int = 42,
    num_cases: int = 500,
    abaqus_command: str = r"D:\Abaqus2021\Command\abaqus.bat",
    cpus: int = 4,
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Phase-A generation of a case library under ``output_root / dataset_id``."""
    master_path = Path(master_inp)
    dataset_root = Path(output_root) / dataset_id
    master_dir = dataset_root / "master"
    cases_dir = dataset_root / "cases"
    meta_dir = dataset_root / "meta"
    master_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    master_text = master_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_master_input(master_text)
    copied_master_files = _copy_master_reference_artifacts(master_path, master_dir)

    bh_reference_condition = None
    bh_target = None
    bh_templates = None
    baseline = _get_baseline_spec(dataset_id, baseline_type, dataset_rules_path)
    families = {
        family
        for _split, family, _count in _scaled_counts(
            dataset_id,
            num_cases,
            baseline.baseline_type,
            dataset_rules_path,
        )
    }
    if baseline.family_bh.family_name in families:
        bh_reference_condition, bh_target, bh_templates = _load_pathological_templates(
            dataset_root,
            dataset_id=dataset_id,
        )

    assignments = build_case_plan(
        dataset_id=dataset_id,
        num_cases=num_cases,
        seed=seed,
        baseline_type=baseline.baseline_type,
        dataset_rules_path=dataset_rules_path,
    )
    _write_json(
        meta_dir / "split.json",
        {
            "dataset_id": dataset_id,
            "baseline_type": baseline.baseline_type,
            "cases": [asdict(item) for item in assignments],
        },
    )
    _write_json(
        meta_dir / "materials_manifest.json",
        {
            "dataset_id": dataset_id,
            "baseline_type": baseline.baseline_type,
            "master_inp": str(master_path),
            "mutable_materials": [asdict(item) for item in parsed.materials],
        },
    )

    np_rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for assignment in assignments:
        case_name = assignment.case_id
        case_dir = cases_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        sample_info = sample_material_parameters(
            parsed,
            assignment,
            np_rng,
            dataset_id=dataset_id,
            baseline_type=baseline.baseline_type,
            dataset_rules_path=dataset_rules_path,
            bh_reference_condition=bh_reference_condition,
            bh_target=bh_target,
            bh_templates=bh_templates,
        )
        new_inp_text = render_case_input(parsed, sample_info["e_values_by_name"])
        inp_path = case_dir / f"{case_name}.inp"
        inp_path.write_text(new_inp_text, encoding="utf-8")
        env_path = _write_abaqus_env(case_dir)
        run_case_cmd_path = _write_run_case_cmd(
            case_dir,
            case_name,
            abaqus_command=abaqus_command,
            cpus=cpus,
        )
        (case_dir / "run.log").write_text("", encoding="utf-8")
        record = {
            "dataset_id": dataset_id,
            "baseline_type": baseline.baseline_type,
            "case_id": case_name,
            "split": assignment.split,
            "family": assignment.family,
            "run_status": "pending",
            "validation_status": "pending",
            "master_inp_sha256": sha256_file(master_path),
            "generated_inp_sha256": sha256_file(inp_path),
            "inp_path": str(inp_path),
            "env_path": str(env_path),
            "run_case_cmd_path": str(run_case_cmd_path),
            "run_log_path": str(case_dir / "run.log"),
            "abaqus_returncode": None,
            "run_error": None,
            "stiffness_mtx_files": [],
            "load_mtx_files": [],
            "stiffness_mtx_path": None,
            "load_mtx_path": None,
            "dat_path": None,
        }
        record.update(
            {key: value for key, value in sample_info.items() if key != "e_values_by_name"}
        )
        _write_case_meta(case_dir, record)
        records.append(record)

    csv_path = _write_samples_csv(meta_dir, records)
    cases_to_run_csv = _write_cases_to_run_csv(dataset_root, records)
    run_all_ps1, run_all_cmd = _write_dispatch_scripts(dataset_root, cases_to_run_csv)
    summary = {
        "dataset_id": dataset_id,
        "baseline_type": baseline.baseline_type,
        "master_inp": str(master_path),
        "master_inp_copy": str(master_dir / "master.inp"),
        "master_reference_files": copied_master_files,
        "num_cases": int(num_cases),
        "n_prepared": int(len(records)),
        "n_pending": int(len(records)),
        "n_success": 0,
        "n_failed": 0,
        "samples_csv": str(csv_path),
        "cases_to_run_csv": str(cases_to_run_csv),
        "run_all_cases_ps1": str(run_all_ps1),
        "run_all_cases_cmd": str(run_all_cmd),
        "bh_pilot_results_path": None
        if bh_templates is None
        else str(dataset_root / "meta" / "bh_pilot_results.json"),
        "consistency_ok": None,
        "consistency_error": None,
    }
    _write_json(meta_dir / "dataset_summary.json", summary)
    return summary


def validate_case_library(
    dataset_root: str | Path,
    *,
    estimate_condition_number_enabled: bool = True,
    condition_number_tol: float = 1e-8,
    condition_number_maxiter: int = 20_000,
) -> dict[str, Any]:
    """Validate a built case library and enrich metadata with structural statistics."""
    dataset_path = Path(dataset_root)
    meta_dir = dataset_path / "meta"
    bh_reference_condition = None
    bh_target_min = None
    bh_target_max = None
    bh_results_path = dataset_path / "meta" / "bh_pilot_results.json"
    if bh_results_path.exists():
        bh_results = json.loads(bh_results_path.read_text(encoding="utf-8"))
        bh_reference_condition = float(bh_results["reference_condition_number"])
        bh_target_min = float(bh_results["condition_target_min"])
        bh_target_max = float(bh_results["condition_target_max"])
    records: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for case_dir in _scan_case_dirs(dataset_path):
        record = _load_case_meta(case_dir)
        artifacts = _resolve_case_artifacts(case_dir, str(record.get("case_id", case_dir.name)))
        if artifacts["stiffness_mtx_path"] is not None:
            record.update(artifacts)
            record["run_status"] = "success"
        elif record.get("run_status") != "success":
            record["validation_status"] = "skipped"
            _write_case_meta(case_dir, record)
            records.append(record)
            continue

        stiffness_path = record.get("stiffness_mtx_path")
        load_path = record.get("load_mtx_path")
        dat_path = record.get("dat_path")
        if not stiffness_path or not load_path or not dat_path:
            record["validation_status"] = "failed"
            record["validation_error"] = "missing one or more Abaqus artifacts"
            _write_case_meta(case_dir, record)
            records.append(record)
            continue

        try:
            sample = load_abaqus_system(stiffness_path, load_mtx=load_path, dat_path=dat_path)
            n_free_dof = int(sample.A.shape[0])
            density = float(sample.A.nnz) / float(n_free_dof * n_free_dof)
            record.update(
                {
                    "validation_status": "success",
                    "validation_error": None,
                    "node_count": int(sample.metadata["n_nodes"]),
                    "total_dof_count": int(sample.metadata["total_dof_count"]),
                    "free_dof_count": int(sample.metadata["free_dof_count"]),
                    "nnz": int(sample.A.nnz),
                    "sparsity_density": density,
                }
            )
            if estimate_condition_number_enabled:
                cond_meta = estimate_condition_number(
                    sample.A,
                    tol=condition_number_tol,
                    maxiter=condition_number_maxiter,
                )
                record.update(cond_meta)
                record["validation_status"] = (
                    "success" if cond_meta["condition_number_status"] == "ok" else "failed"
                )
                record["validation_error"] = cond_meta.get("condition_number_error")
                if (
                    record.get("family") == "BH"
                    and cond_meta["condition_number_status"] == "ok"
                    and bh_reference_condition is not None
                ):
                    ratio = float(cond_meta["condition_number_est"]) / float(bh_reference_condition)
                    record["condition_ratio_to_ref"] = ratio
                    record["condition_target_min"] = bh_target_min
                    record["condition_target_max"] = bh_target_max
                    if not (
                        float(bh_target_min)
                        <= float(cond_meta["condition_number_est"])
                        <= float(bh_target_max)
                    ):
                        record["validation_status"] = "failed"
                        record["validation_error"] = (
                            "BH condition number outside target window: "
                            f"{cond_meta['condition_number_est']:.6g} not in "
                            f"[{bh_target_min:.6g}, {bh_target_max:.6g}]"
                        )
            else:
                if record.get("condition_number_est") is None:
                    record.update(
                        {
                            "condition_number_est": None,
                            "condition_number_status": "skipped",
                            "condition_number_backend": None,
                            "condition_number_tol": None,
                            "condition_number_maxiter": None,
                            "condition_number_error": None,
                            "lambda_min_est": None,
                            "lambda_max_est": None,
                            "condition_number_time_sec": None,
                        }
                    )
            if record["validation_status"] == "success":
                successes.append(record)
        except Exception as exc:
            record["validation_status"] = "failed"
            record["validation_error"] = str(exc)
        _write_case_meta(case_dir, record)
        records.append(record)

    consistency_ok = True
    consistency_error = None
    if successes:
        node_counts = sorted({int(record["node_count"]) for record in successes})
        total_dof_counts = sorted({int(record["total_dof_count"]) for record in successes})
        if len(node_counts) != 1 or len(total_dof_counts) != 1:
            consistency_ok = False
            consistency_error = (
                f"inconsistent node_count={node_counts} or total_dof_count={total_dof_counts}"
            )
    else:
        consistency_ok = False
        consistency_error = "no successfully validated cases"

    csv_path = _write_samples_csv(meta_dir, records)
    free_dofs = [float(record["free_dof_count"]) for record in successes]
    conds = [
        float(record["condition_number_est"])
        for record in successes
        if record.get("condition_number_est") is not None
    ]
    densities = [float(record["sparsity_density"]) for record in successes]
    summary = {
        "dataset_id": dataset_path.name,
        "n_success": int(len(successes)),
        "n_failed": int(len(records) - len(successes)),
        "node_count": None if not successes else int(successes[0]["node_count"]),
        "total_dof_count": None if not successes else int(successes[0]["total_dof_count"]),
        "free_dof_count_min": None if not free_dofs else float(min(free_dofs)),
        "free_dof_count_max": None if not free_dofs else float(max(free_dofs)),
        "free_dof_count_mean": None if not free_dofs else float(np.mean(free_dofs)),
        "condition_number_est_min": None if not conds else float(min(conds)),
        "condition_number_est_max": None if not conds else float(max(conds)),
        "condition_number_est_median": None if not conds else float(np.median(conds)),
        "density_min": None if not densities else float(min(densities)),
        "density_max": None if not densities else float(max(densities)),
        "density_mean": None if not densities else float(np.mean(densities)),
        "consistency_ok": bool(consistency_ok),
        "consistency_error": consistency_error,
        "samples_csv": str(csv_path),
        "estimate_condition_number": bool(estimate_condition_number_enabled),
        "condition_number_backend": "scipy.eigsh" if estimate_condition_number_enabled else None,
        "condition_number_tol": (
            float(condition_number_tol) if estimate_condition_number_enabled else None
        ),
        "condition_number_maxiter": (
            int(condition_number_maxiter) if estimate_condition_number_enabled else None
        ),
    }
    _write_json(meta_dir / "dataset_summary.json", summary)
    return summary


def _merge_json_file(path: Path, extra_payload: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(_jsonify(extra_payload))
    _write_json(path, payload)


def _npz_sidecar_payload(
    record: dict[str, Any], *, dataset_id: str | None = None
) -> dict[str, Any]:
    case_dir = str(Path(str(record["inp_path"])).parent) if record.get("inp_path") else None
    return {
        "dataset_id": record["dataset_id"] if dataset_id is None else dataset_id,
        "source_dataset_id": record.get("source_dataset_id", record["dataset_id"]),
        "baseline_type": record.get("baseline_type"),
        "case_id": record["case_id"],
        "source_case_dir": case_dir,
        "source_sample_meta": None
        if case_dir is None
        else str(Path(case_dir) / "sample_meta.json"),
        "split": record["split"],
        "family": record["family"],
        "sampling_mode": record.get("sampling_mode"),
        "template_id": record.get("template_id"),
        "variable_material_names": record["variable_material_names"],
        "frame_material_names": record["frame_material_names"],
        "soil_material_names": record["soil_material_names"],
        "E0_values": record["E0_values"],
        "E_values": record["E_values"],
        "alpha_global": record.get("alpha_global"),
        "alpha_frame": record.get("alpha_frame"),
        "alpha_soil": record.get("alpha_soil"),
        "sigma_f": record.get("sigma_f"),
        "sigma_s": record.get("sigma_s"),
        "rho_f": record.get("rho_f"),
        "rho_s": record.get("rho_s"),
        "target_group": record.get("target_group"),
        "node_count": record.get("node_count"),
        "total_dof_count": record.get("total_dof_count"),
        "free_dof_count": record.get("free_dof_count"),
        "nnz": record.get("nnz"),
        "sparsity_density": record.get("sparsity_density"),
        "condition_number_est": record.get("condition_number_est"),
        "condition_number_status": record.get("condition_number_status"),
        "condition_ratio_to_ref": record.get("condition_ratio_to_ref"),
        "condition_target_min": record.get("condition_target_min"),
        "condition_target_max": record.get("condition_target_max"),
        "bh_attempt_index": record.get("bh_attempt_index"),
        "condition_number_backend": record.get("condition_number_backend"),
        "condition_number_tol": record.get("condition_number_tol"),
        "condition_number_maxiter": record.get("condition_number_maxiter"),
        "generated_inp_sha256": record.get("generated_inp_sha256"),
        "soft_layer_indices": record.get("soft_layer_indices"),
        "soft_layer_material_names": record.get("soft_layer_material_names"),
        "soft_alpha_values": record.get("soft_alpha_values"),
    }


def _convert_case_record_worker(
    task: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    index, payload = task
    result = convert_abaqus_to_npz(
        payload["stiffness_path"],
        payload["npz_root"],
        load_mtx=payload["load_path"],
        dat_path=payload["dat_path"],
        rhs_mode=payload["rhs_mode"],
        num_samples=1,
        solve_mode=payload["solve_mode"],
        compress_npz=payload["compress_npz"],
        write_per_case_manifest=payload["write_per_case_manifest"],
    )
    return (
        index,
        {
            "sample_paths": [str(path) for path in result.sample_paths],
            "records": result.records,
        },
    )


def _convert_case_records_to_npz(
    records: list[dict[str, Any]],
    *,
    npz_root: Path,
    dataset_id: str,
    rhs_mode: str,
    solve_mode: str,
    bh_solve_mode: str | None = None,
    num_workers: int | None = None,
    compress_npz: bool = False,
    write_per_case_manifest: bool = False,
) -> dict[str, Any]:
    manifest_records: list[dict[str, Any]] = []
    tasks: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        effective_solve_mode = (
            str(bh_solve_mode)
            if bh_solve_mode is not None and str(record.get("family")) == "BH"
            else solve_mode
        )
        tasks.append(
            (
                index,
                {
                    "stiffness_path": Path(record["stiffness_mtx_path"]),
                    "load_path": Path(record["load_mtx_path"]),
                    "dat_path": Path(record["dat_path"]),
                    "npz_root": npz_root,
                    "rhs_mode": rhs_mode,
                    "solve_mode": effective_solve_mode,
                    "compress_npz": compress_npz,
                    "write_per_case_manifest": write_per_case_manifest,
                },
            )
        )

    resolved_workers = _resolve_num_workers(num_workers)
    ordered_results: list[tuple[int, dict[str, Any]]] = []
    if resolved_workers == 1:
        for task in tasks:
            ordered_results.append(_convert_case_record_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_index = {
                executor.submit(_convert_case_record_worker, task): task[0] for task in tasks
            }
            for future in as_completed(future_to_index):
                ordered_results.append(future.result())
    ordered_results.sort(key=lambda item: item[0])

    for index, result_payload in ordered_results:
        record = records[index]
        for sample_path_text in result_payload["sample_paths"]:
            sample_path = Path(sample_path_text)
            sidecar = sample_path.with_suffix(".meta.json")
            _merge_json_file(sidecar, _npz_sidecar_payload(record, dataset_id=dataset_id))
            manifest_records.append(
                {
                    "sample_file": str(sample_path),
                    "sample_sidecar": str(sidecar),
                    "case_id": record["case_id"],
                    "source_case_dir": (
                        str(Path(str(record["inp_path"])).parent)
                        if record.get("inp_path") is not None
                        else None
                    ),
                    "source_sample_meta": (
                        str(Path(str(record["inp_path"])).parent / "sample_meta.json")
                        if record.get("inp_path") is not None
                        else None
                    ),
                    "source_stiffness_mtx": record.get("stiffness_mtx_path"),
                    "source_load_mtx": record.get("load_mtx_path"),
                    "source_dat": record.get("dat_path"),
                    "split": record["split"],
                    "family": record["family"],
                    "source_dataset_id": record.get("source_dataset_id", record["dataset_id"]),
                }
            )
    return {
        "dataset_id": dataset_id,
        "output_dir": str(npz_root),
        "num_samples": int(len(manifest_records)),
        "rhs_mode": rhs_mode,
        "solve_mode": solve_mode,
        "num_workers": resolved_workers,
        "compress_npz": bool(compress_npz),
        "write_per_case_manifest": bool(write_per_case_manifest),
        "records": manifest_records,
    }


def convert_case_library(
    dataset_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    rhs_mode: str = "load",
    solve_mode: str = "zero_placeholder",
    bh_solve_mode: str | None = "zero_placeholder",
    num_workers: int | None = None,
    compress_npz: bool = False,
    write_per_case_manifest: bool = False,
) -> dict[str, Any]:
    """Convert a validated case library into NPZ samples."""
    dataset_path = Path(dataset_root)
    summary_path = dataset_path / "meta" / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"dataset summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("consistency_ok")):
        raise RuntimeError(f"dataset consistency check failed: {summary.get('consistency_error')}")

    npz_root = Path(output_dir) if output_dir is not None else dataset_path / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)
    records_to_convert: list[dict[str, Any]] = []
    for case_dir in _scan_case_dirs(dataset_path):
        record = _load_case_meta(case_dir)
        if record.get("run_status") != "success" or record.get("validation_status") != "success":
            continue
        records_to_convert.append(record)
    manifest = _convert_case_records_to_npz(
        records_to_convert,
        npz_root=npz_root,
        dataset_id=dataset_path.name,
        rhs_mode=rhs_mode,
        solve_mode=solve_mode,
        bh_solve_mode=bh_solve_mode,
        num_workers=num_workers,
        compress_npz=compress_npz,
        write_per_case_manifest=write_per_case_manifest,
    )
    _write_json(npz_root / "manifest.json", manifest)
    return manifest


def _select_release_records(
    base_dataset_root: Path,
    bh_dataset_root: Path,
    *,
    base_quotas: tuple[tuple[str, str, int], ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for root, allowed_families in ((base_dataset_root, {"A", "B"}), (bh_dataset_root, {"BH"})):
        for case_dir in _scan_case_dirs(root):
            record = _load_case_meta(case_dir)
            if (
                record.get("run_status") != "success"
                or record.get("validation_status") != "success"
            ):
                continue
            family = str(record.get("family"))
            if family not in allowed_families:
                continue
            grouped.setdefault((str(record["split"]), family), []).append(record)
    selected: list[dict[str, Any]] = []
    for split, family, count in base_quotas:
        bucket = sorted(grouped.get((split, family), []), key=lambda item: str(item["case_id"]))
        if len(bucket) < count:
            raise RuntimeError(
                f"insufficient validated {family} cases for split={split}: required {count}, found {len(bucket)}"
            )
        selected.extend(bucket[:count])
    return selected


def build_frame33_bh_release_npz(
    base_dataset_root: str | Path,
    bh_dataset_root: str | Path,
    *,
    output_dir: str | Path,
    rhs_mode: str = "load",
    solve_mode: str = "zero_placeholder",
    bh_solve_mode: str = "zero_placeholder",
    dataset_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a new frame33-b release NPZ package using base A/B and BH-only cases."""
    base_root = Path(base_dataset_root)
    bh_root = Path(bh_dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    baseline = _get_baseline_spec("frame33-b", "pure_frame", dataset_rules_path)
    selected_records = _select_release_records(
        base_root,
        bh_root,
        base_quotas=baseline.base_quotas,
    )
    manifest = _convert_case_records_to_npz(
        selected_records,
        npz_root=output_path,
        dataset_id=f"frame33-b{baseline.family_bh.pathological_dataset_suffix}",
        rhs_mode=rhs_mode,
        solve_mode=solve_mode,
        bh_solve_mode=bh_solve_mode,
    )
    family_counts: dict[str, int] = {}
    split_family_counts: dict[str, int] = {}
    for record in selected_records:
        family = str(record["family"])
        split = str(record["split"])
        family_counts[family] = family_counts.get(family, 0) + 1
        key = f"{split}/{family}"
        split_family_counts[key] = split_family_counts.get(key, 0) + 1
    release_summary = {
        **manifest,
        "base_dataset_root": str(base_root),
        "bh_dataset_root": str(bh_root),
        "family_counts": family_counts,
        "split_family_counts": split_family_counts,
        "selected_cases": [
            {
                "case_id": record["case_id"],
                "split": record["split"],
                "family": record["family"],
                "source_dataset_id": record.get("source_dataset_id", record["dataset_id"]),
                "stiffness_mtx_path": record["stiffness_mtx_path"],
            }
            for record in selected_records
        ],
    }
    _write_json(output_path / "manifest.json", release_summary)
    return release_summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_build_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
    include_defaults: bool = True,
) -> None:
    """Add shared CLI arguments for the build script."""
    parser.add_argument("--dataset-id", type=str, required=required, default=None)
    parser.add_argument("--baseline-type", type=str, required=False, default=None)
    parser.add_argument("--master-inp", type=Path, required=required, default=None)
    parser.add_argument("--output-root", type=Path, required=required, default=None)
    parser.add_argument("--seed", type=int, default=42 if include_defaults else None)
    parser.add_argument(
        "--num-cases", type=_positive_int, default=500 if include_defaults else None
    )
    parser.add_argument(
        "--abaqus-command",
        type=str,
        default=r"D:\Abaqus2021\Command\abaqus.bat" if include_defaults else None,
    )
    parser.add_argument("--cpus", type=_positive_int, default=4 if include_defaults else None)
