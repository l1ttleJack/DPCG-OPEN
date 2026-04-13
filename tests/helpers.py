from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags

from dpcg.io.npz import save_npz_sample


def make_spd_matrix(n: int = 4) -> csr_matrix:
    return diags(
        diagonals=[-1.0 * np.ones(n - 1), 4.0 * np.ones(n), -1.0 * np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )


def write_release_npz_dataset(
    root: Path,
    split_layout: dict[str, list[str]] | None = None,
    n: int = 4,
) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    if split_layout is None:
        split_layout = {
            "train": ["A", "B"],
            "val": ["BH"],
            "iid": ["B"],
            "ood": ["A"],
        }
    matrix = make_spd_matrix(n)
    paths: list[Path] = []
    sample_index = 0
    for split, families in split_layout.items():
        for family in families:
            stem = f"case_{sample_index + 1:04d}_STIF2"
            npz_path = root / f"{stem}.npz"
            rhs = np.arange(1, n + 1, dtype=np.float64) * float(sample_index + 1)
            solution = np.linalg.solve(matrix.toarray(), rhs)
            save_npz_sample(npz_path, matrix, rhs, x=solution, metadata={"sample_index": sample_index})
            meta: dict[str, Any] = {
                "sample_index": sample_index,
                "split": split,
                "family": family,
                "sampling_mode": f"{family.lower()}_mode",
                "template_id": f"template_{family.lower()}",
                "condition_number_est": float(1000.0 + 10.0 * sample_index),
                "condition_ratio_to_ref": float(1.0 + 0.01 * sample_index),
            }
            npz_path.with_suffix(".meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths.append(npz_path)
            sample_index += 1
    return paths
