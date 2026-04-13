from __future__ import annotations

import torch

from dpcg.data import AbaqusDataset, build_dataloaders

from .helpers import write_release_npz_dataset


def test_dataset_sample_shapes_and_dtypes(tmp_path):
    data_root = tmp_path / "dataset"
    write_release_npz_dataset(data_root, n=4)

    dataset = AbaqusDataset(str(data_root), mask_percentile=-1.0, cache_samples=False)
    sample = dataset[0]

    assert sample.m_tensor.is_sparse
    assert sample.l_tensor.is_sparse
    assert sample.s_tensor.dtype == torch.float32
    assert sample.b_tensor.dtype == torch.float32
    assert sample.mask_tensor.dtype == torch.bool
    assert sample.diag_inv.dtype == torch.float32
    assert tuple(sample.m_tensor.shape) == (1, 4, 4)
    assert tuple(sample.l_tensor.shape) == (1, 4, 4)


def test_build_dataloaders_returns_expected_split_counts(tmp_path):
    data_root = tmp_path / "dataset"
    write_release_npz_dataset(
        data_root,
        split_layout={"train": ["A", "B"], "val": ["BH"], "iid": ["A"], "ood": ["B"]},
        n=4,
    )
    manifest_path = tmp_path / "split_manifest.json"

    _train, _val, _iid, _ood, manifest = build_dataloaders(
        data_root=str(data_root),
        n_train=2,
        n_val=1,
        n_test=2,
        mask_percentile=-1.0,
        split_manifest_path=str(manifest_path),
        return_manifest=True,
    )

    assert manifest["split_counts"] == {"train": 2, "val": 1, "iid": 1, "ood": 1}
    assert manifest_path.exists()
