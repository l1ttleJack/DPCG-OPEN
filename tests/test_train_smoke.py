from __future__ import annotations

import json

import pytest

from dpcg.train import train_from_cfg

from .helpers import write_release_npz_dataset


def _make_smoke_cfg(tmp_path):
    data_root = tmp_path / "dataset"
    write_release_npz_dataset(
        data_root,
        split_layout={"train": ["A", "B"], "val": ["BH"], "iid": ["A"], "ood": ["B"]},
        n=4,
    )
    run_dir = tmp_path / "run"
    return {
        "train": {
            "SEED": 42,
            "DEVICE": "cuda",
            "DATA_ROOT": str(data_root),
            "PC_TRAIN": 2,
            "PC_VAL": 1,
            "PC_TEST": 2,
            "EPOCHS": 1,
            "VAL_EPOCH": 1,
            "INI_LR": 1.0e-3,
            "MASK_PERCENTILE": -1.0,
            "DATASET_SPLIT_SOURCE": "meta_json",
            "LOG-DIR_M": str(run_dir),
            "TRAIN_FILE_M": str(run_dir / "best.ckpt"),
            "LAST_CHECKPOINT": str(run_dir / "last.ckpt"),
            "SPLIT_MANIFEST": str(run_dir / "split_manifest.json"),
            "MODEL_KWARGS": {
                "channels": [8, 16, 32, 48, 64],
                "block_depth": 1,
                "use_sparse_head": True,
                "tail_expansion_layers": 1,
                "tail_kernel_size": 3,
            },
            "patience": 5,
            "AMP": False,
            "NUM_WORKERS": 0,
            "SHUFFLE_TRAIN": False,
            "CACHE_SAMPLES": True,
        }
    }


def test_sunet0_train_smoke_writes_outputs(tmp_path):
    pytest.importorskip("torch")
    try:
        import spconv.pytorch  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"spconv unavailable: {exc}")
    try:
        train_from_cfg(_make_smoke_cfg(tmp_path), model_name="sunet0", loss_name="cond_loss")
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "cuda" in str(exc):
            pytest.skip(f"CUDA/spconv training unavailable: {exc}")
        raise

    run_dir = tmp_path / "run"
    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "last.ckpt").exists()
    assert (run_dir / "training_summary.json").exists()
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["loss_name"] == "condition_number_loss_eigs"
    assert summary["split_counts"] == {"train": 2, "val": 1, "iid": 1, "ood": 1}
