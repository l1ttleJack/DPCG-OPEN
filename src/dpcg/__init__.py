"""Lightweight DPCG release package."""

from __future__ import annotations

from .abaqus import build_dataset_from_inp
from .benchmark import run_benchmark as run_gpu_benchmark
from .data import AbaqusDataset
from .io.npz import load_npz_sample
from .losses import CachedConditionNumberLoss, torch_cond_metric
from .models import SUNet0 as SUNet0Tail

__version__ = "0.1.0"
ConditionNumberLoss = CachedConditionNumberLoss

__all__ = [
    "__version__",
    "AbaqusDataset",
    "ConditionNumberLoss",
    "SUNet0Tail",
    "build_dataset_from_inp",
    "load_npz_sample",
    "run_gpu_benchmark",
    "torch_cond_metric",
]
