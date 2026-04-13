"""Abaqus ``.dat`` parser with nodes, elements, ties, and instance node blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AbaqusDatInfo:
    """Structured Abaqus ``.dat`` contents used by dataset and ODB export flows."""

    node_ids: np.ndarray
    node_xyz: np.ndarray
    encastre_node_ids: np.ndarray
    elem_ids: np.ndarray
    elem_types: np.ndarray
    elem_props: np.ndarray
    elem_conn_ptr: np.ndarray
    elem_conn_idx: np.ndarray
    tie_slave_ids: np.ndarray
    tie_master_ptr: np.ndarray
    tie_master_idx: np.ndarray
    instance_names: np.ndarray
    instance_node_ptr: np.ndarray
    instance_node_ids: np.ndarray


def _header_key(line: str) -> str:
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return ""
    return "".join(letters).upper()


def _strip_line_prefix(line: str) -> str:
    s = line.strip()
    if s.upper().startswith("LINE"):
        parts = s.split()
        if len(parts) >= 3 and parts[0].upper() == "LINE":
            return " ".join(parts[2:])
    return s


def _keyword(line: str) -> str:
    s = line.strip()
    if not s.startswith("*"):
        return ""
    return s[1:].split(",", 1)[0].strip().upper().replace(" ", "")


def _parse_param(line: str, key: str) -> str:
    s = line.strip()
    if s.startswith("*"):
        s = s[1:]
    parts = s.split(",")
    key_u = key.strip().upper()
    for token in parts[1:]:
        if "=" not in token:
            continue
        lhs, rhs = token.split("=", 1)
        if lhs.strip().upper() == key_u:
            return rhs.strip()
    return ""


def _parse_part_instance_name(line: str) -> str | None:
    upper = line.upper()
    marker = "PART INSTANCE:"
    if marker not in upper:
        return None
    idx = upper.find(marker)
    return line[idx + len(marker) :].strip() or None


def _build_coords_by_node_id(node_ids: np.ndarray, node_xyz: np.ndarray) -> np.ndarray:
    max_id = int(node_ids.max())
    coords_by_id = np.full((max_id + 1, 3), np.nan, dtype=np.float64)
    coords_by_id[node_ids.astype(np.int64)] = node_xyz
    return coords_by_id


def _parse_node_row(line: str) -> tuple[int, float, float, float] | None:
    if "," in line:
        parts = [part.strip() for part in line.split(",") if part.strip()]
    else:
        parts = line.split()
    if len(parts) < 3:
        return None
    try:
        nid = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3]) if len(parts) > 3 else 0.0
    except ValueError:
        return None
    return nid, x, y, z


def read_dat_info(path_dat: str) -> AbaqusDatInfo:
    """Parse Abaqus ``.dat`` node, element, tie, and instance node-block information."""

    nodes: dict[int, tuple[float, float, float]] = {}
    encastre_list: list[int] = []
    node_sets: dict[str, list[int]] = {}
    node_set_seen: dict[str, set[int]] = {}
    elem_ids_list: list[int] = []
    elem_types_list: list[str] = []
    elem_props_list: list[int] = []
    elem_conn_ptr_list: list[int] = [0]
    elem_conn_idx_list: list[int] = []
    tie_slave_order: list[int] = []
    tie_map: dict[int, list[int]] = {}
    instance_nodes: dict[str, list[int]] = {}
    instance_seen: dict[str, set[int]] = {}
    elem_id_seen: set[int] = set()

    state = "none"
    current_elem_type = ""
    current_elem_prop = 0
    pending_instance: str | None = None
    current_instance: str | None = None
    current_node_set: str | None = None
    current_node_set_generate = False

    def append_node(nid: int, xyz: tuple[float, float, float]) -> None:
        prev = nodes.get(nid)
        if prev is None:
            nodes[nid] = xyz
            return
        if not np.allclose(np.asarray(prev), np.asarray(xyz), atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"duplicate node id with conflicting coords in .dat: {nid}")

    def append_instance_node(instance_name: str, nid: int) -> None:
        if instance_name not in instance_nodes:
            instance_nodes[instance_name] = []
            instance_seen[instance_name] = set()
        if nid in instance_seen[instance_name]:
            return
        instance_nodes[instance_name].append(nid)
        instance_seen[instance_name].add(nid)

    def append_element(eid: int, etype: str, prop: int, conn: list[int]) -> None:
        if not conn:
            return
        if eid in elem_id_seen:
            raise RuntimeError(f"duplicate element id in .dat: {eid}")
        elem_id_seen.add(eid)
        elem_ids_list.append(eid)
        elem_types_list.append(etype if etype else "UNKNOWN")
        elem_props_list.append(int(prop))
        elem_conn_idx_list.extend(conn)
        elem_conn_ptr_list.append(len(elem_conn_idx_list))

    def append_node_set_member(node_set_name: str, nid: int) -> None:
        key = node_set_name.upper()
        if key not in node_sets:
            node_sets[key] = []
            node_set_seen[key] = set()
        if nid in node_set_seen[key]:
            return
        node_sets[key].append(nid)
        node_set_seen[key].add(nid)

    def append_encastre_target(target: str) -> None:
        token = target.strip()
        if not token:
            return
        if token.lstrip("-").isdigit():
            encastre_list.append(int(token))
            return
        encastre_list.extend(node_sets.get(token.upper(), []))

    with open(path_dat, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if "OPTIONS BEING PROCESSED" in raw.upper():
                break
            s = _strip_line_prefix(raw)
            if not s:
                continue

            if s.startswith("**"):
                maybe_instance = _parse_part_instance_name(s)
                if maybe_instance is not None:
                    pending_instance = maybe_instance
                continue

            header = _header_key(s)
            if header == "ELEMENTDEFINITIONS":
                state = "elem_table"
                current_instance = None
                continue
            if header == "NODEDEFINITIONS":
                state = "node_table"
                current_instance = None
                continue
            if header == "TIECONSTRAINTS":
                state = "tie_table"
                current_instance = None
                continue

            if s.startswith("*"):
                kw = _keyword(s)
                current_instance = None
                if kw == "NODE":
                    current_node_set = None
                    current_node_set_generate = False
                    if pending_instance is not None:
                        state = "instance_node"
                        current_instance = pending_instance
                        pending_instance = None
                    else:
                        state = "node_table"
                    continue
                if kw == "ELEMENT":
                    current_node_set = None
                    current_node_set_generate = False
                    state = "elem_keyword"
                    current_elem_type = _parse_param(s, "TYPE")
                    current_elem_prop = 0
                    continue
                if kw == "NSET":
                    pending_instance = None
                    state = "nset_keyword"
                    current_node_set = _parse_param(s, "NSET").strip().upper() or None
                    current_node_set_generate = "GENERATE" in s.upper()
                    continue
                if kw == "BOUNDARY":
                    pending_instance = None
                    current_node_set = None
                    current_node_set_generate = False
                    state = "boundary_keyword"
                    continue
                pending_instance = None
                current_node_set = None
                current_node_set_generate = False
                state = "none"
                continue

            if state == "elem_table":
                parts = s.replace(",", " ").split()
                if len(parts) < 4 or not parts[0].lstrip("-").isdigit():
                    continue
                if not any(ch.isalpha() for ch in parts[1]):
                    continue
                try:
                    eid = int(parts[0])
                    etype = parts[1]
                    prop = int(parts[2])
                except ValueError:
                    continue
                conn = [int(tok) for tok in parts[3:] if tok.lstrip("-").isdigit()]
                append_element(eid, etype, prop, conn)
                continue

            if state == "elem_keyword":
                parts = s.replace(",", " ").split()
                if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
                    continue
                try:
                    eid = int(parts[0])
                except ValueError:
                    continue
                conn = [int(tok) for tok in parts[1:] if tok.lstrip("-").isdigit()]
                append_element(eid, current_elem_type, current_elem_prop, conn)
                continue

            if state in {"node_table", "instance_node"}:
                parsed = _parse_node_row(s)
                if parsed is None:
                    continue
                nid, x, y, z = parsed
                append_node(nid, (x, y, z))
                if "ENCASTRE" in s.upper():
                    encastre_list.append(nid)
                if state == "instance_node" and current_instance is not None:
                    append_instance_node(current_instance, nid)
                continue

            if state == "tie_table":
                parts = s.replace(",", " ").split()
                if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
                    continue
                slave = int(parts[0])
                masters = [int(tok) for tok in parts[2:] if tok.lstrip("-").isdigit()]
                if not masters:
                    continue
                if slave not in tie_map:
                    tie_map[slave] = []
                    tie_slave_order.append(slave)
                tie_map[slave].extend(masters)
                continue

            if state == "nset_keyword":
                if current_node_set is None:
                    continue
                tokens = [token.strip() for token in s.replace(",", " ").split() if token.strip()]
                if not tokens:
                    continue
                if current_node_set_generate:
                    if len(tokens) < 2:
                        continue
                    if not tokens[0].lstrip("-").isdigit() or not tokens[1].lstrip("-").isdigit():
                        continue
                    start = int(tokens[0])
                    stop = int(tokens[1])
                    step = int(tokens[2]) if len(tokens) >= 3 and tokens[2].lstrip("-").isdigit() else 1
                    if step == 0:
                        raise RuntimeError(f"invalid zero step in .dat node set generate: {path_dat}")
                    for nid in range(start, stop + (1 if step > 0 else -1), step):
                        append_node_set_member(current_node_set, nid)
                    continue
                for token in tokens:
                    if token.lstrip("-").isdigit():
                        append_node_set_member(current_node_set, int(token))
                continue

            if state == "boundary_keyword":
                parts = [part.strip() for part in s.split(",") if part.strip()]
                if len(parts) < 2:
                    parts = [part.strip() for part in s.split() if part.strip()]
                if len(parts) < 2:
                    continue
                if parts[1].upper() != "ENCASTRE":
                    continue
                append_encastre_target(parts[0])

    if not nodes:
        raise RuntimeError(f"no node definitions found in .dat: {path_dat}")
    if not elem_ids_list:
        raise RuntimeError(f"no element definitions found in .dat: {path_dat}")

    node_ids = np.asarray(sorted(nodes.keys()), dtype=np.int32)
    node_xyz = np.asarray([nodes[int(nid)] for nid in node_ids.tolist()], dtype=np.float64)
    encastre_node_ids = (
        np.unique(np.asarray(encastre_list, dtype=np.int32))
        if encastre_list
        else np.empty((0,), dtype=np.int32)
    )

    elem_ids = np.asarray(elem_ids_list, dtype=np.int32)
    elem_props = np.asarray(elem_props_list, dtype=np.int32)
    max_len = max((len(item) for item in elem_types_list), default=1)
    elem_types = np.asarray(elem_types_list, dtype=f"U{max_len}")
    elem_conn_ptr = np.asarray(elem_conn_ptr_list, dtype=np.int32)
    elem_conn_idx = np.asarray(elem_conn_idx_list, dtype=np.int32)

    if tie_slave_order:
        tie_slave_ids = np.asarray(tie_slave_order, dtype=np.int32)
        tie_master_ptr = np.zeros(tie_slave_ids.size + 1, dtype=np.int32)
        masters_flat: list[int] = []
        for idx, slave_id in enumerate(tie_slave_order):
            masters = tie_map.get(slave_id, [])
            unique_master_ids = (
                np.unique(np.asarray(masters, dtype=np.int32)).tolist() if masters else []
            )
            masters_flat.extend(unique_master_ids)
            tie_master_ptr[idx + 1] = len(masters_flat)
        tie_master_idx = np.asarray(masters_flat, dtype=np.int32)
    else:
        tie_slave_ids = np.empty((0,), dtype=np.int32)
        tie_master_ptr = np.zeros((1,), dtype=np.int32)
        tie_master_idx = np.empty((0,), dtype=np.int32)

    instance_names_list = list(instance_nodes.keys())
    if instance_names_list:
        max_name_len = max(len(name) for name in instance_names_list)
        instance_names = np.asarray(instance_names_list, dtype=f"U{max_name_len}")
        instance_node_ptr = np.zeros(len(instance_names_list) + 1, dtype=np.int32)
        flat_nodes: list[int] = []
        for idx, name in enumerate(instance_names_list):
            flat_nodes.extend(instance_nodes[name])
            instance_node_ptr[idx + 1] = len(flat_nodes)
        instance_node_ids = np.asarray(flat_nodes, dtype=np.int32)
    else:
        instance_names = np.empty((0,), dtype="U1")
        instance_node_ptr = np.zeros((1,), dtype=np.int32)
        instance_node_ids = np.empty((0,), dtype=np.int32)

    return AbaqusDatInfo(
        node_ids=node_ids,
        node_xyz=node_xyz,
        encastre_node_ids=encastre_node_ids,
        elem_ids=elem_ids,
        elem_types=elem_types,
        elem_props=elem_props,
        elem_conn_ptr=elem_conn_ptr,
        elem_conn_idx=elem_conn_idx,
        tie_slave_ids=tie_slave_ids,
        tie_master_ptr=tie_master_ptr,
        tie_master_idx=tie_master_idx,
        instance_names=instance_names,
        instance_node_ptr=instance_node_ptr,
        instance_node_ids=instance_node_ids,
    )


def read_dat_nodes(dat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read node coordinates as ``(coords_by_node_id, node_ids)``."""
    info = read_dat_info(dat_path)
    return _build_coords_by_node_id(info.node_ids, info.node_xyz), info.node_ids.astype(np.int64)


def read_dat_elements(dat_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read element connectivity as ``(elem_ids, elem_types, elem_indptr, elem_nodes)``."""
    info = read_dat_info(dat_path)
    return (
        info.elem_ids.astype(np.int64),
        info.elem_types.astype(np.str_),
        info.elem_conn_ptr.astype(np.int64),
        info.elem_conn_idx.astype(np.int64),
    )


def detect_beam_elements(elem_types: np.ndarray) -> np.ndarray:
    """Return a boolean mask for beam-like element types."""
    if elem_types.size == 0:
        return np.empty(0, dtype=bool)
    types = np.asarray(elem_types, dtype=np.str_)
    out = np.zeros(types.size, dtype=bool)
    for idx, value in enumerate(types):
        out[idx] = str(value).strip().upper().startswith("B")
    return out


def compute_mesh_scale_h(
    coords_by_id: np.ndarray,
    elem_types: np.ndarray,
    elem_indptr: np.ndarray,
    elem_nodes: np.ndarray,
    prefer_type: str = "CPE4",
) -> Optional[float]:
    """Estimate a representative mesh spacing ``h`` from element edge lengths."""
    types_u = np.asarray([str(item).strip().upper() for item in elem_types], dtype=object)
    edges = []
    for idx, elem_type in enumerate(types_u):
        if prefer_type and elem_type != prefer_type:
            continue
        start = int(elem_indptr[idx])
        end = int(elem_indptr[idx + 1])
        if end - start != 4:
            continue
        nodes = elem_nodes[start:end]
        pts = coords_by_id[nodes]
        if pts.shape[0] != 4 or not np.all(np.isfinite(pts)):
            continue
        for u, v in ((0, 1), (1, 2), (2, 3), (3, 0)):
            dist = float(np.linalg.norm(pts[u] - pts[v]))
            if dist > 0.0:
                edges.append(dist)
    if not edges:
        return None
    return float(np.median(np.asarray(edges, dtype=np.float64)))
