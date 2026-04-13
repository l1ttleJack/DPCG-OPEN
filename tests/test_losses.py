from __future__ import annotations

import torch

from dpcg.losses import torch_cond_metric


def test_torch_cond_metric_matches_torch_linalg_cond():
    matrix = torch.tensor([[[4.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)

    actual = torch_cond_metric(matrix)
    expected = torch.linalg.cond(matrix)

    assert torch.allclose(actual, expected)
