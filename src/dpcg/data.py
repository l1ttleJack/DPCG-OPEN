"""Minimal NPZ dataset loading for the lightweight DPCG release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from scipy.sparse import coo_matrix, tril
from torch.utils.data import DataLoader, Dataset

from dpcg.utils import coo_to_torch_sparse, inverse_sqrt_diagonal

SUPPORTED_RELEASE_SPLITS = ("train", "val", "iid", "ood")


@dataclass
class DatasetSample:
    """Single cached dataset sample with CPU-side derived tensors."""

    sample_id: str
    m_tensor: Any
    s_tensor: Any
    b_tensor: Any
    l_tensor: Any
    ref: float
    mask_tensor: Any
    metadata: dict[str, Any]
    mask_rows: Any | None = None
    mask_cols: Any | None = None
    mask_key: Any | None = None
    diag_inv: Any | None = None

    def __iter__(self):
        yield self.m_tensor
        yield self.s_tensor
        yield self.b_tensor
        yield self.l_tensor
        yield self.ref
        yield self.mask_tensor

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return (
            self.m_tensor,
            self.s_tensor,
            self.b_tensor,
            self.l_tensor,
            self.ref,
            self.mask_tensor,
        )[index]


def _pick_key(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _parse_npz_arrays(data, source: str):
    indices = _pick_key(data, ["A_indices", "indices", "arr_0"])
    values = _pick_key(data, ["A_values", "values", "arr_1"])
    solutions = _pick_key(data, ["x", "solution", "solutions", "arr_2"])
    rhs = _pick_key(data, ["b", "rhs", "arr_3"])
    if indices is None or values is None or rhs is None:
        raise ValueError(f"Unrecognized npz format: {source}")
    if indices.ndim == 2 and indices.shape[0] == 2:
        row, col = indices[0], indices[1]
    else:
        row, col = indices[:, 0], indices[:, 1]
    return row, col, values, solutions, rhs


def _load_npz_arrays(path: str):
    return _parse_npz_arrays(np.load(path), path)


def iter_npz(folder: str) -> Iterator[tuple[coo_matrix, np.ndarray, np.ndarray]]:
    """Iterate over `.npz` files and yield `(A, x, b)` tuples."""
    for path in sorted(Path(folder).glob("*.npz")):
        row, col, values, solutions, rhs = _load_npz_arrays(str(path))
        yield coo_matrix((values, (row, col)), shape=(len(rhs), len(rhs))), solutions, rhs


def compute_ref_scale(values: np.ndarray) -> float:
    """Match the ref-scaling rule used by the training pipeline."""
    if values.size == 0:
        return 1.0
    max_value = float(np.max(np.abs(values)))
    if not np.isfinite(max_value) or max_value <= 0.0:
        return 1.0
    power = int(np.ceil(np.log10(max_value)) - 2)
    ref = float(10**power)
    return ref if ref > 0.0 else 1.0


def mask_indices_from_lower_triangle(
    l_matrix: coo_matrix,
    mask_percentile: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparsity mask from a lower-triangular matrix."""
    l_coo = l_matrix.tocoo()
    n = int(l_coo.shape[0])
    if l_coo.nnz == 0:
        diag = np.arange(n, dtype=np.int64)
        return diag, diag
    if mask_percentile <= 0.0:
        keep = np.ones(l_coo.nnz, dtype=bool)
    else:
        threshold = float(np.percentile(np.abs(l_coo.data), mask_percentile))
        keep = np.abs(l_coo.data) > threshold
    rows = l_coo.row[keep].astype(np.int64, copy=False)
    cols = l_coo.col[keep].astype(np.int64, copy=False)
    diag = np.arange(n, dtype=np.int64)
    rows = np.concatenate([rows, diag])
    cols = np.concatenate([cols, diag])
    key = rows.astype(np.int64) * np.int64(n) + cols.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    keep_unique = np.concatenate([[True], key_sorted[1:] != key_sorted[:-1]])
    sel = order[keep_unique]
    return rows[sel], cols[sel]


class NpzPreconditionerDataset(Dataset):
    """Sparse NPZ dataset for learned preconditioners."""

    def __init__(
        self,
        data_root: str,
        mask_percentile: float = 50.0,
        cache_samples: bool = True,
        cache_graph: bool = False,
    ):
        del cache_graph
        self.data_root = str(Path(data_root))
        self.entries = load_release_dataset_index(data_root)
        self.files = [entry["npz_path"] for entry in self.entries]
        self.mask_percentile = float(mask_percentile)
        self.cache_samples = bool(cache_samples)
        self._sample_cache: dict[int, DatasetSample] = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_samples and idx in self._sample_cache:
            return self._sample_cache[idx]
        entry = self.entries[idx]
        row, col, values, solutions, rhs = _load_npz_arrays(entry["npz_path"])
        matrix = coo_matrix((values, (row, col)), shape=(len(rhs), len(rhs)))
        l_matrix = tril(matrix).tocoo()

        l_tensor = coo_to_torch_sparse(l_matrix)
        m_tensor = coo_to_torch_sparse(matrix)
        s_tensor = torch.tensor(solutions, dtype=torch.float32).squeeze()
        b_tensor = torch.tensor(rhs, dtype=torch.float32).squeeze()

        ref = compute_ref_scale(values)
        l_tensor = l_tensor / float(ref)
        rows, cols = mask_indices_from_lower_triangle(l_matrix, self.mask_percentile)
        mask_indices = torch.tensor(np.vstack((rows, cols)), dtype=torch.int64)
        mask_values = torch.ones(mask_indices.shape[1], dtype=torch.bool)
        mask_tensor = torch.sparse_coo_tensor(
            mask_indices,
            mask_values,
            size=matrix.shape,
            dtype=torch.bool,
        ).coalesce()
        sample = DatasetSample(
            sample_id=entry["sample_id"],
            m_tensor=m_tensor,
            s_tensor=s_tensor,
            b_tensor=b_tensor,
            l_tensor=l_tensor,
            ref=ref,
            mask_tensor=mask_tensor,
            metadata=dict(entry["metadata"]),
            mask_rows=torch.as_tensor(rows, dtype=torch.int64),
            mask_cols=torch.as_tensor(cols, dtype=torch.int64),
            mask_key=torch.as_tensor(rows * matrix.shape[0] + cols, dtype=torch.int64),
            diag_inv=inverse_sqrt_diagonal(m_tensor, dtype=torch.float32),
        )
        if self.cache_samples:
            self._sample_cache[idx] = sample
        return sample


AbaqusDataset = NpzPreconditionerDataset


def sample_to_half_graph(_batch_or_sample):
    """Graph input is intentionally omitted from the lightweight release."""
    raise RuntimeError("graph inputs are not available in the lightweight SUNet0-only release")


def single_item_collate(batch):
    return batch[0]


def _load_meta_json(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file for dataset sample: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    split = str(payload.get("split", "")).strip().lower()
    if split not in SUPPORTED_RELEASE_SPLITS:
        raise ValueError(
            f"{meta_path}: invalid or missing split={payload.get('split')!r}; "
            f"expected one of {SUPPORTED_RELEASE_SPLITS}"
        )
    payload["split"] = split
    return payload


def load_release_dataset_index(data_root: str) -> list[dict[str, Any]]:
    """Scan a release dataset directory and pair `.npz` with `.meta.json`."""
    root = Path(data_root)
    entries: list[dict[str, Any]] = []
    for npz_path in sorted(root.glob("*.npz")):
        sample_id = npz_path.stem
        meta_path = npz_path.with_suffix(".meta.json")
        metadata = _load_meta_json(meta_path)
        entries.append(
            {
                "npz_path": str(npz_path),
                "meta_path": str(meta_path),
                "sample_id": sample_id,
                "split": metadata["split"],
                "family": metadata.get("family"),
                "sampling_mode": metadata.get("sampling_mode"),
                "template_id": metadata.get("template_id"),
                "condition_number_est": metadata.get("condition_number_est"),
                "condition_ratio_to_ref": metadata.get("condition_ratio_to_ref"),
                "metadata": metadata,
            }
        )
    if not entries:
        raise FileNotFoundError(f"No .npz samples found under {root}")
    return entries


def _limited_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return list(indices)
    if limit < 0:
        raise ValueError("split sample limits must be non-negative")
    return list(indices[:limit])


def _normalize_family_quota_mapping(
    quotas: dict[str, int] | None,
    *,
    split_name: str,
) -> dict[str, int] | None:
    if quotas is None:
        return None
    if not isinstance(quotas, dict):
        raise ValueError(f"{split_name.upper()}_FAMILY_QUOTAS must be a mapping of family -> count")
    normalized: dict[str, int] = {}
    for raw_family, raw_count in quotas.items():
        family = str(raw_family).strip()
        if not family:
            raise ValueError(f"{split_name.upper()}_FAMILY_QUOTAS contains an empty family name")
        if not isinstance(raw_count, Integral):
            raise ValueError(
                f"{split_name.upper()}_FAMILY_QUOTAS[{family!r}] must be a non-negative integer"
            )
        count = int(raw_count)
        if count < 0:
            raise ValueError(
                f"{split_name.upper()}_FAMILY_QUOTAS[{family!r}] must be a non-negative integer"
            )
        normalized[family] = count
    return normalized


def _select_split_indices_by_family_quota(
    dataset: NpzPreconditionerDataset,
    indices: list[int],
    *,
    split_name: str,
    quotas: dict[str, int] | None,
    max_samples: int | None,
) -> list[int]:
    if quotas is None:
        return _limited_indices(indices, max_samples)
    total_quota = int(sum(quotas.values()))
    if max_samples is not None and total_quota > int(max_samples):
        raise ValueError(
            f"{split_name.upper()}_FAMILY_QUOTAS requests {total_quota} samples, "
            f"which exceeds {split_name.upper()}_MAX_SAMPLES_PER_SPLIT={int(max_samples)}"
        )
    available_by_family: dict[str, list[int]] = {}
    for index in indices:
        family = str(dataset.entries[index]["family"] or "unknown")
        available_by_family.setdefault(family, []).append(index)
    selected: list[int] = []
    for family, requested in quotas.items():
        available_indices = available_by_family.get(family, [])
        if requested > len(available_indices):
            raise ValueError(
                f"{split_name.upper()}_FAMILY_QUOTAS[{family!r}] requests {requested} samples, "
                f"but only {len(available_indices)} are available"
            )
        selected.extend(available_indices[:requested])
    return selected


def _family_counts(dataset: NpzPreconditionerDataset, indices: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indices:
        family = str(dataset.entries[index]["family"] or "unknown")
        counts[family] = counts.get(family, 0) + 1
    return counts


def _build_release_manifest(
    dataset: NpzPreconditionerDataset,
    split_indices: dict[str, list[int]],
    *,
    seed: int,
    split_source: str,
    iid_family_quotas: dict[str, int] | None = None,
    ood_family_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    split_samples: dict[str, list[dict[str, Any]]] = {}
    family_counts_by_split: dict[str, dict[str, int]] = {}
    split_counts: dict[str, int] = {}
    for split_name in SUPPORTED_RELEASE_SPLITS:
        indices = split_indices[split_name]
        split_counts[split_name] = int(len(indices))
        family_counts_by_split[split_name] = _family_counts(dataset, indices)
        split_samples[split_name] = [
            {
                "sample_id": dataset.entries[index]["sample_id"],
                "npz_path": dataset.entries[index]["npz_path"],
                "meta_path": dataset.entries[index]["meta_path"],
                "family": dataset.entries[index]["family"],
                "sampling_mode": dataset.entries[index]["sampling_mode"],
                "template_id": dataset.entries[index]["template_id"],
            }
            for index in indices
        ]
    return {
        "seed": int(seed),
        "split_source": split_source,
        "dataset_root": dataset.data_root,
        "mask_percentile": float(dataset.mask_percentile),
        "n_total": int(len(dataset)),
        "split_counts": split_counts,
        "family_counts_by_split": family_counts_by_split,
        "splits": split_samples,
        "iid_family_quotas": None if iid_family_quotas is None else dict(iid_family_quotas),
        "ood_family_quotas": None if ood_family_quotas is None else dict(ood_family_quotas),
    }


def _resolve_split_indices(
    dataset: NpzPreconditionerDataset,
    *,
    train_max_samples: int | None,
    val_max_samples: int | None,
    iid_max_samples: int | None,
    ood_max_samples: int | None,
    iid_family_quotas: dict[str, int] | None,
    ood_family_quotas: dict[str, int] | None,
) -> dict[str, list[int]]:
    split_indices: dict[str, list[int]] = {name: [] for name in SUPPORTED_RELEASE_SPLITS}
    for index, entry in enumerate(dataset.entries):
        split_indices[entry["split"]].append(index)
    return {
        "train": _limited_indices(split_indices["train"], train_max_samples),
        "val": _limited_indices(split_indices["val"], val_max_samples),
        "iid": _select_split_indices_by_family_quota(
            dataset,
            split_indices["iid"],
            split_name="iid",
            quotas=iid_family_quotas,
            max_samples=iid_max_samples,
        ),
        "ood": _select_split_indices_by_family_quota(
            dataset,
            split_indices["ood"],
            split_name="ood",
            quotas=ood_family_quotas,
            max_samples=ood_max_samples,
        ),
    }


def _validate_legacy_split_sizes(
    *,
    requested_train: int | None,
    requested_val: int | None,
    requested_test: int | None,
    split_indices: dict[str, list[int]],
) -> None:
    actual_train = len(split_indices["train"])
    actual_val = len(split_indices["val"])
    actual_iid = len(split_indices["iid"])
    actual_ood = len(split_indices["ood"])
    if requested_train is not None and requested_train > 0 and requested_train != actual_train:
        raise ValueError(
            f"PC_TRAIN={requested_train} does not match the formal train split size {actual_train}"
        )
    if requested_val is not None and requested_val > 0 and requested_val != actual_val:
        raise ValueError(
            f"PC_VAL={requested_val} does not match the formal val split size {actual_val}"
        )
    actual_test = actual_iid + actual_ood
    if requested_test is not None and requested_test > 0 and requested_test != actual_test:
        raise ValueError(
            f"PC_TEST={requested_test} does not match the formal iid+ood sample count {actual_test}"
        )


def build_dataloaders(
    data_root: str,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    seed: int = 42,
    batch_size: int = 1,
    mask_percentile: float = 50.0,
    split_manifest_path: str | None = None,
    return_manifest: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    drop_last: bool = False,
    shuffle_train: bool = True,
    cache_samples: bool = True,
    cache_graph: bool = False,
    split_source: str = "meta_json",
    train_max_samples_per_split: int | None = None,
    val_max_samples_per_split: int | None = None,
    iid_max_samples_per_split: int | None = None,
    ood_max_samples_per_split: int | None = None,
    iid_family_quotas: dict[str, int] | None = None,
    ood_family_quotas: dict[str, int] | None = None,
):
    """Build train/val/iid/ood data loaders from formal release metadata."""
    del cache_graph
    torch.manual_seed(seed)
    if split_source != "meta_json":
        raise ValueError("Only split_source='meta_json' is supported for release datasets")
    dataset = NpzPreconditionerDataset(
        data_root,
        mask_percentile=mask_percentile,
        cache_samples=cache_samples,
    )
    normalized_iid_family_quotas = _normalize_family_quota_mapping(
        iid_family_quotas,
        split_name="iid",
    )
    normalized_ood_family_quotas = _normalize_family_quota_mapping(
        ood_family_quotas,
        split_name="ood",
    )
    split_indices = _resolve_split_indices(
        dataset,
        train_max_samples=train_max_samples_per_split,
        val_max_samples=val_max_samples_per_split,
        iid_max_samples=iid_max_samples_per_split,
        ood_max_samples=ood_max_samples_per_split,
        iid_family_quotas=normalized_iid_family_quotas,
        ood_family_quotas=normalized_ood_family_quotas,
    )
    _validate_legacy_split_sizes(
        requested_train=n_train,
        requested_val=n_val,
        requested_test=n_test,
        split_indices=split_indices,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_data = torch.utils.data.Subset(dataset, split_indices["train"])
    val_data = torch.utils.data.Subset(dataset, split_indices["val"])
    iid_data = torch.utils.data.Subset(dataset, split_indices["iid"])
    ood_data = torch.utils.data.Subset(dataset, split_indices["ood"])
    manifest = _build_release_manifest(
        dataset,
        split_indices,
        seed=seed,
        split_source=split_source,
        iid_family_quotas=normalized_iid_family_quotas,
        ood_family_quotas=normalized_ood_family_quotas,
    )
    if split_manifest_path is not None:
        path = Path(split_manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": single_item_collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_data,
        shuffle=bool(shuffle_train and len(train_data) > 0),
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    iid_loader = DataLoader(iid_data, shuffle=False, **loader_kwargs)
    ood_loader = DataLoader(ood_data, shuffle=False, **loader_kwargs)
    if return_manifest:
        return train_loader, val_loader, iid_loader, ood_loader, manifest
    return train_loader, val_loader, iid_loader, ood_loader
