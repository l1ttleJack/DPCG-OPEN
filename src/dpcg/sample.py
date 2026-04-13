"""Shared sample container for benchmark, analysis, and Abaqus workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class BenchmarkSample:
    sample_id: str
    A: csr_matrix
    b: np.ndarray
    x: np.ndarray | None = None
    l_tensor: Any | None = None
    mask: Any | None = None
    ref: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    coords_by_node_id: np.ndarray | None = None
    node_ids: np.ndarray | None = None
    element_ids: np.ndarray | None = None
    element_types: np.ndarray | None = None
    element_indptr: np.ndarray | None = None
    element_nodes: np.ndarray | None = None
    instance_names: np.ndarray | None = None
    instance_node_ptr: np.ndarray | None = None
    instance_node_ids: np.ndarray | None = None
    free_dof_node_ids: np.ndarray | None = None
    free_dof_labels: np.ndarray | None = None
    free_dof_indices: np.ndarray | None = None
    encastre_node_ids: np.ndarray | None = None
    tie_slave_ids: np.ndarray | None = None
    tie_master_ptr: np.ndarray | None = None
    tie_master_idx: np.ndarray | None = None
    mask_rows: np.ndarray | None = None
    mask_cols: np.ndarray | None = None
    mask_key: np.ndarray | None = None
    diag_inv: np.ndarray | None = None


__all__ = ["BenchmarkSample"]
