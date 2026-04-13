"""Training-side torch/spconv helpers."""

from __future__ import annotations

from typing import List

import numpy as np

try:
    import spconv.pytorch as spconv
    import torch
    from torch.sparse import mm as sparse_mm
except Exception:  # pragma: no cover - optional deps
    spconv = None
    torch = None
    sparse_mm = None

from scipy.sparse import coo_matrix

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover - optional dependency
    Data = None

_DIAG_LOG_CLAMP = 6.0
_VALID_DIAG_STRATEGIES = ("learned_exp", "unit_diag", "raw_pred")


def normalize_diag_strategy(diag_strategy: str | None) -> str:
    """Normalize diagonal assembly strategy names."""
    strategy = "learned_exp" if diag_strategy is None else str(diag_strategy).strip().lower()
    if strategy not in _VALID_DIAG_STRATEGIES:
        raise ValueError(
            f"diag_strategy must be one of {_VALID_DIAG_STRATEGIES}, got {diag_strategy!r}"
        )
    return strategy


def squeeze_batch_matrix(matrix: "torch.Tensor") -> "torch.Tensor":  # type: ignore # type: ignore
    """Convert a `(1, n, n)` matrix tensor to `(n, n)` while preserving sparsity."""
    if torch is None:
        raise RuntimeError("torch required")
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch tensor")
    if matrix.dim() == 2:
        return matrix.coalesce() if matrix.is_sparse else matrix
    if matrix.dim() != 3 or int(matrix.size(0)) != 1:
        raise ValueError(
            f"Expected matrix with shape `(1, n, n)` or `(n, n)`, got {tuple(matrix.shape)}"
        )
    if matrix.is_sparse:
        coalesced = matrix.coalesce()
        indices = coalesced.indices()[1:]
        values = coalesced.values()
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.size()[-2:],
            dtype=values.dtype,
            device=values.device,
        ).coalesce()
    return matrix.squeeze(0)


def sparse_diag_values(matrix: "torch.Tensor") -> "torch.Tensor":  # type: ignore
    """Extract the diagonal of a dense or sparse square matrix as a dense vector."""
    if torch is None:
        raise RuntimeError("torch required")
    matrix_2d = squeeze_batch_matrix(matrix)
    n = int(matrix_2d.shape[-1])
    if getattr(matrix_2d, "is_sparse", False):
        coalesced = matrix_2d.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        rows = indices[0].to(torch.int64)
        cols = indices[1].to(torch.int64)
        diag = torch.zeros(n, dtype=values.dtype, device=values.device)
        diag_mask = rows == cols
        if torch.any(diag_mask):
            diag[rows[diag_mask]] = values[diag_mask]
        return diag
    return torch.diagonal(matrix_2d, dim1=-2, dim2=-1)


def inverse_sqrt_diagonal(
    matrix: "torch.Tensor", eps: float | None = None, dtype: "torch.dtype | None" = None
) -> "torch.Tensor":
    """Compute `1 / sqrt(diag(matrix))` for dense or sparse matrices."""
    if torch is None:
        raise RuntimeError("torch required")
    diag = sparse_diag_values(matrix)
    if dtype is not None and diag.dtype != dtype:
        diag = diag.to(dtype=dtype)
    min_value = torch.finfo(diag.dtype).tiny if eps is None else float(eps)
    return torch.rsqrt(torch.clamp(diag, min=min_value))


def ensure_sparse_square(matrix: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    """Convert a dense or sparse square matrix tensor to sparse COO with the target dtype."""
    if torch is None:
        raise RuntimeError("torch required")
    matrix_2d = squeeze_batch_matrix(matrix)
    if not isinstance(matrix_2d, torch.Tensor):
        raise TypeError("matrix must be a torch tensor")
    if getattr(matrix_2d, "is_sparse", False):
        coalesced = matrix_2d.coalesce()
        if coalesced.dtype != dtype:
            coalesced = torch.sparse_coo_tensor(
                coalesced.indices(),
                coalesced.values().to(dtype=dtype),
                size=coalesced.shape,
                device=coalesced.device,
            ).coalesce()
        return coalesced
    dense = matrix_2d.to(dtype=dtype)
    return dense.to_sparse().coalesce()


def dense_masked_to_sparse(matrix: "torch.Tensor", mask) -> "torch.Tensor":
    """Sample dense matrix values on a mask and return a sparse COO tensor."""
    if torch is None:
        raise RuntimeError("torch required")
    if hasattr(mask, "is_sparse") and mask.is_sparse:
        coords = mask.coalesce().indices().to(device=matrix.device, dtype=torch.long)
    else:
        coords = torch.nonzero(mask, as_tuple=False).t().to(device=matrix.device, dtype=torch.long)
    values = matrix[coords[0], coords[1]]
    return torch.sparse_coo_tensor(
        coords,
        values,
        size=matrix.shape,
        dtype=matrix.dtype,
        device=matrix.device,
    ).coalesce()


def assemble_sparse_factor_from_prediction_torch(
    *,
    coords: "torch.Tensor",
    values: "torch.Tensor",
    mask_rows: "torch.Tensor",
    mask_cols: "torch.Tensor",
    mask_key: "torch.Tensor",
    diag_inv: "torch.Tensor",
    shape: tuple[int, int],
    diag_strategy: str = "learned_exp",
    force_unit_diag: bool = True,
    keep_matched_zero: bool = True,
    use_mask_projection: bool = True,
) -> dict[str, "torch.Tensor"]:
    """Project sparse predictions onto a mask support and build a sparse factor tensor."""
    if torch is None:
        raise RuntimeError("torch required")
    if coords.ndim != 2 or int(coords.shape[1]) != 2:
        raise ValueError("prediction coords must have shape (nnz, 2)")
    flat_values = values.reshape(-1)
    if int(coords.shape[0]) != int(flat_values.shape[0]):
        raise ValueError("prediction values/coords length mismatch")

    device = flat_values.device
    dtype = flat_values.dtype
    n = int(shape[0])
    diag_strategy = normalize_diag_strategy(diag_strategy)

    coords = coords.to(device=device, dtype=torch.long)
    diag_inv = diag_inv.to(device=device, dtype=dtype).reshape(-1)
    keep_diag = bool(force_unit_diag or diag_strategy == "unit_diag")

    input_indices = torch.arange(coords.shape[0], device=device, dtype=torch.long)
    if use_mask_projection:
        support_valid = torch.ones(coords.shape[0], dtype=torch.bool, device=device)
    else:
        support_valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < int(n))
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < int(n))
            & (coords[:, 0] >= coords[:, 1])
        )
        coords = coords[support_valid]
        flat_values = flat_values[support_valid]
        input_indices = input_indices[support_valid]

    pred_key = coords[:, 0] * int(n) + coords[:, 1]
    if pred_key.numel() > 0:
        order = torch.argsort(pred_key, stable=True)
        pred_key_sorted = pred_key[order]
        pred_values_sorted = flat_values[order]
        pred_input_indices_sorted = input_indices[order]
        pred_keep = torch.ones_like(pred_key_sorted, dtype=torch.bool)
        pred_keep[1:] = pred_key_sorted[1:] != pred_key_sorted[:-1]
        pred_key_sorted = pred_key_sorted[pred_keep]
        pred_values_sorted = pred_values_sorted[pred_keep]
        pred_input_indices_sorted = pred_input_indices_sorted[pred_keep]
    else:
        pred_key_sorted = torch.empty((0,), dtype=torch.long, device=device)
        pred_values_sorted = torch.empty((0,), dtype=dtype, device=device)
        pred_input_indices_sorted = torch.empty((0,), dtype=torch.long, device=device)

    if use_mask_projection:
        mask_rows = mask_rows.to(device=device, dtype=torch.long).reshape(-1)
        mask_cols = mask_cols.to(device=device, dtype=torch.long).reshape(-1)
        mask_key = mask_key.to(device=device, dtype=torch.long).reshape(-1)
        support_rows = mask_rows
        support_cols = mask_cols
        support_key = mask_key
        support_input_indices = torch.full_like(support_key, -1, dtype=torch.long)
        support_from_prediction = torch.zeros_like(support_key, dtype=torch.bool)
        if pred_key_sorted.numel() > 0:
            positions = torch.searchsorted(pred_key_sorted, mask_key)
            matched = positions < pred_key_sorted.numel()
            if torch.any(matched):
                matched_positions = positions[matched]
                matched_keys = mask_key[matched]
                exact = pred_key_sorted[matched_positions] == matched_keys
                matched_indices = torch.nonzero(matched, as_tuple=False).reshape(-1)
                matched[matched_indices] = exact
                if torch.any(matched):
                    support_input_indices[matched] = pred_input_indices_sorted[positions[matched]]
        else:
            positions = torch.zeros_like(mask_key)
            matched = torch.zeros_like(mask_key, dtype=torch.bool)
        support_from_prediction = matched.clone()
    else:
        support_key = pred_key_sorted
        support_input_indices = pred_input_indices_sorted
        support_from_prediction = torch.ones_like(support_key, dtype=torch.bool)
        if support_key.numel() > 0:
            support_rows = torch.div(support_key, int(n), rounding_mode="floor")
            support_cols = support_key - support_rows * int(n)
            support_pred_values = pred_values_sorted
        else:
            support_rows = torch.empty((0,), dtype=torch.long, device=device)
            support_cols = torch.empty((0,), dtype=torch.long, device=device)
            support_pred_values = torch.empty((0,), dtype=dtype, device=device)
        if keep_diag:
            diag = torch.arange(int(n), dtype=torch.long, device=device)
            diag_key = diag * int(n) + diag
            if support_key.numel() > 0:
                positions = torch.searchsorted(support_key, diag_key)
                diag_present = positions < support_key.numel()
                if torch.any(diag_present):
                    diag_positions = positions[diag_present]
                    diag_present_indices = torch.nonzero(diag_present, as_tuple=False).reshape(-1)
                    diag_present[diag_present_indices] = support_key[diag_positions] == diag_key[diag_present]
            else:
                diag_present = torch.zeros_like(diag_key, dtype=torch.bool)
            if torch.any(~diag_present):
                missing_diag_key = diag_key[~diag_present]
                support_key = torch.cat([support_key, missing_diag_key], dim=0)
                support_input_indices = torch.cat(
                    [
                        support_input_indices,
                        torch.full(
                            (int(missing_diag_key.shape[0]),),
                            -1,
                            dtype=torch.long,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                support_from_prediction = torch.cat(
                    [
                        support_from_prediction,
                        torch.zeros(
                            int(missing_diag_key.shape[0]), dtype=torch.bool, device=device
                        ),
                    ],
                    dim=0,
                )
                support_pred_values = torch.cat(
                    [
                        support_pred_values,
                        torch.zeros(
                            int(missing_diag_key.shape[0]), dtype=dtype, device=device
                        ),
                    ],
                    dim=0,
                )
                support_order = torch.argsort(support_key, stable=True)
                support_key = support_key[support_order]
                support_input_indices = support_input_indices[support_order]
                support_from_prediction = support_from_prediction[support_order]
                support_pred_values = support_pred_values[support_order]
            if support_key.numel() > 0:
                support_rows = torch.div(support_key, int(n), rounding_mode="floor")
                support_cols = support_key - support_rows * int(n)
        elif support_key.numel() > 0:
            support_rows = torch.div(support_key, int(n), rounding_mode="floor")
            support_cols = support_key - support_rows * int(n)
        positions = torch.zeros_like(support_key)
        matched = support_from_prediction.clone()

    diag_mask = support_rows == support_cols
    assembled = torch.zeros(support_key.shape[0], dtype=dtype, device=device)
    jacobian = torch.zeros_like(assembled)
    keep_diag = bool(force_unit_diag or diag_strategy == "unit_diag")

    active = support_from_prediction.clone()
    if diag_strategy == "unit_diag":
        active &= ~diag_mask
    if torch.any(active):
        active_indices = torch.nonzero(active, as_tuple=False).reshape(-1)
        active_diag = diag_mask[active]
        if use_mask_projection:
            matched_values = pred_values_sorted[positions[active]]
        else:
            matched_values = support_pred_values[active]
        if torch.any(~active_diag):
            offdiag_positions = active_indices[~active_diag]
            offdiag_values = matched_values[~active_diag]
            assembled[offdiag_positions] = offdiag_values
            jacobian[offdiag_positions] = 1.0
        if torch.any(active_diag):
            diag_positions = active_indices[active_diag]
            raw_diag = matched_values[active_diag]
            if diag_strategy == "learned_exp":
                clipped_diag = torch.clamp(raw_diag, min=-_DIAG_LOG_CLAMP, max=_DIAG_LOG_CLAMP)
                diag_values = torch.exp(clipped_diag)
                in_bounds = (
                    (raw_diag >= -_DIAG_LOG_CLAMP) & (raw_diag <= _DIAG_LOG_CLAMP)
                ).to(dtype=dtype)
                jacobian_values = diag_values * in_bounds
            elif diag_strategy == "raw_pred":
                diag_values = raw_diag
                jacobian_values = torch.ones_like(raw_diag, dtype=dtype, device=device)
            else:  # pragma: no cover - normalize_diag_strategy guards this
                raise ValueError(f"unsupported diag_strategy: {diag_strategy}")
            assembled[diag_positions] = diag_values
            jacobian[diag_positions] = jacobian_values

    if keep_diag:
        diag_fill_mask = diag_mask if diag_strategy == "unit_diag" else (diag_mask & ~matched)
        if torch.any(diag_fill_mask):
            assembled[diag_fill_mask] = 1.0

    column_scales = diag_inv[support_cols]
    assembled_scaled = assembled * column_scales
    derivative_scales = jacobian * column_scales
    if keep_matched_zero:
        keep = active | (diag_mask if keep_diag else torch.zeros_like(diag_mask))
    else:
        nonzero_mask = assembled_scaled != 0
        keep = (active & nonzero_mask) | (diag_mask if keep_diag else torch.zeros_like(diag_mask))
    sparse_rows = support_rows[keep]
    sparse_cols = support_cols[keep]
    sparse_values = assembled_scaled[keep]
    sparse_tensor = torch.sparse_coo_tensor(
        torch.stack([sparse_rows, sparse_cols], dim=0),
        sparse_values,
        size=shape,
        dtype=dtype,
        device=device,
    ).coalesce()

    active_support_positions = torch.nonzero(active, as_tuple=False).reshape(-1)
    active_pred_input_indices = support_input_indices[active_support_positions]
    return {
        "sparse_tensor": sparse_tensor,
        "sparse_rows": sparse_rows,
        "sparse_cols": sparse_cols,
        "sparse_values": sparse_values,
        "active_pred_input_indices": active_pred_input_indices,
        "active_rows": support_rows[active_support_positions],
        "active_cols": support_cols[active_support_positions],
        "active_scales": derivative_scales[active_support_positions],
    }


def assemble_sparse_factor_from_prediction_numpy(
    *,
    coords,
    values,
    mask_rows,
    mask_cols,
    mask_key,
    diag_inv,
    shape: tuple[int, int],
    diag_strategy: str = "learned_exp",
    force_unit_diag: bool = True,
    keep_matched_zero: bool = True,
    use_mask_projection: bool = True,
):
    """NumPy/SciPy counterpart of sparse factor projection used by the benchmark path."""
    coords = np.asarray(coords, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if coords.ndim != 2 or int(coords.shape[1]) != 2:
        raise ValueError("prediction coords must have shape (nnz, 2)")
    if int(coords.shape[0]) != int(values.shape[0]):
        raise ValueError("prediction values/coords length mismatch")
    diag_inv = np.asarray(diag_inv, dtype=np.float64).reshape(-1)
    n = int(shape[0])
    diag_strategy = normalize_diag_strategy(diag_strategy)
    keep_diag = bool(force_unit_diag or diag_strategy == "unit_diag")

    input_indices = np.arange(coords.shape[0], dtype=np.int64)
    if use_mask_projection:
        support_valid = np.ones(coords.shape[0], dtype=bool)
    else:
        support_valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < int(n))
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < int(n))
            & (coords[:, 0] >= coords[:, 1])
        )
        coords = coords[support_valid]
        values = values[support_valid]
        input_indices = input_indices[support_valid]

    pred_key = coords[:, 0] * np.int64(n) + coords[:, 1]
    order = np.argsort(pred_key, kind="mergesort")
    pred_key_sorted = pred_key[order]
    pred_values_sorted = values[order]
    pred_input_indices_sorted = input_indices[order]
    if pred_key_sorted.size > 0:
        pred_keep = np.ones(pred_key_sorted.shape[0], dtype=bool)
        pred_keep[1:] = pred_key_sorted[1:] != pred_key_sorted[:-1]
        pred_key_sorted = pred_key_sorted[pred_keep]
        pred_values_sorted = pred_values_sorted[pred_keep]
        pred_input_indices_sorted = pred_input_indices_sorted[pred_keep]

    if use_mask_projection:
        mask_rows = np.asarray(mask_rows, dtype=np.int64).reshape(-1)
        mask_cols = np.asarray(mask_cols, dtype=np.int64).reshape(-1)
        mask_key = np.asarray(mask_key, dtype=np.int64).reshape(-1)
        support_rows = mask_rows
        support_cols = mask_cols
        support_key = mask_key
        support_input_indices = np.full(mask_key.shape[0], -1, dtype=np.int64)
        if pred_key_sorted.size > 0:
            positions = np.searchsorted(pred_key_sorted, mask_key)
            matched = positions < pred_key_sorted.size
            if np.any(matched):
                matched_positions = positions[matched]
                matched_keys = mask_key[matched]
                exact = pred_key_sorted[matched_positions] == matched_keys
                matched_indices = np.flatnonzero(matched)
                matched[matched_indices] = exact
                if np.any(matched):
                    support_input_indices[matched] = pred_input_indices_sorted[positions[matched]]
        else:
            positions = np.zeros(mask_key.shape[0], dtype=np.int64)
            matched = np.zeros(mask_key.shape[0], dtype=bool)
        support_from_prediction = matched.copy()
    else:
        support_key = pred_key_sorted
        support_input_indices = pred_input_indices_sorted
        support_from_prediction = np.ones(support_key.shape[0], dtype=bool)
        support_rows = support_key // np.int64(n) if support_key.size > 0 else np.empty(0, dtype=np.int64)
        support_cols = (
            support_key - support_rows * np.int64(n)
            if support_key.size > 0
            else np.empty(0, dtype=np.int64)
        )
        support_pred_values = pred_values_sorted
        if keep_diag:
            diag = np.arange(int(n), dtype=np.int64)
            diag_key = diag * np.int64(n) + diag
            if support_key.size > 0:
                positions = np.searchsorted(support_key, diag_key)
                diag_present = positions < support_key.size
                if np.any(diag_present):
                    matched_positions = positions[diag_present]
                    matched_keys = diag_key[diag_present]
                    exact = support_key[matched_positions] == matched_keys
                    matched_indices = np.flatnonzero(diag_present)
                    diag_present[matched_indices] = exact
            else:
                diag_present = np.zeros(diag_key.shape[0], dtype=bool)
            if np.any(~diag_present):
                missing_diag_key = diag_key[~diag_present]
                support_key = np.concatenate([support_key, missing_diag_key])
                support_input_indices = np.concatenate(
                    [
                        support_input_indices,
                        np.full(missing_diag_key.shape[0], -1, dtype=np.int64),
                    ]
                )
                support_from_prediction = np.concatenate(
                    [
                        support_from_prediction,
                        np.zeros(missing_diag_key.shape[0], dtype=bool),
                    ]
                )
                support_pred_values = np.concatenate(
                    [
                        support_pred_values,
                        np.zeros(missing_diag_key.shape[0], dtype=np.float64),
                    ]
                )
                support_order = np.argsort(support_key, kind="mergesort")
                support_key = support_key[support_order]
                support_input_indices = support_input_indices[support_order]
                support_from_prediction = support_from_prediction[support_order]
                support_pred_values = support_pred_values[support_order]
            support_rows = support_key // np.int64(n)
            support_cols = support_key - support_rows * np.int64(n)
        positions = np.zeros(support_key.shape[0], dtype=np.int64)
        matched = support_from_prediction.copy()

    diag_mask = support_rows == support_cols
    active = support_from_prediction.copy()
    if diag_strategy == "unit_diag":
        active &= ~diag_mask
    assembled = np.zeros(support_key.shape[0], dtype=np.float64)
    if np.any(active):
        active_indices = np.flatnonzero(active)
        active_diag = diag_mask[active]
        if use_mask_projection:
            matched_values = pred_values_sorted[positions[active]]
        else:
            matched_values = support_pred_values[active]
        if np.any(~active_diag):
            assembled[active_indices[~active_diag]] = matched_values[~active_diag]
        if np.any(active_diag):
            raw_diag = matched_values[active_diag]
            if diag_strategy == "learned_exp":
                assembled[active_indices[active_diag]] = np.exp(
                    np.clip(raw_diag, -_DIAG_LOG_CLAMP, _DIAG_LOG_CLAMP)
                )
            elif diag_strategy == "raw_pred":
                assembled[active_indices[active_diag]] = raw_diag
            else:  # pragma: no cover - normalize_diag_strategy guards this
                raise ValueError(f"unsupported diag_strategy: {diag_strategy}")
    if keep_diag:
        diag_fill_mask = diag_mask if diag_strategy == "unit_diag" else np.logical_and(
            diag_mask, ~support_from_prediction
        )
        assembled[diag_fill_mask] = 1.0
    assembled *= diag_inv[support_cols]
    if keep_matched_zero:
        keep = np.logical_or(active, diag_mask) if keep_diag else active
    else:
        nonzero_mask = assembled != 0.0
        keep = (
            np.logical_or(np.logical_and(active, nonzero_mask), diag_mask)
            if keep_diag
            else np.logical_and(active, nonzero_mask)
        )
    matrix = coo_matrix(
        (assembled[keep], (support_rows[keep], support_cols[keep])),
        shape=shape,
    ).tocsr()
    return {
        "matrix": matrix,
        "factor_rows": support_rows[keep],
        "factor_cols": support_cols[keep],
        "factor_values": assembled[keep],
        "active_pred_input_indices": support_input_indices[np.flatnonzero(active)],
        "active_rows": support_rows[np.flatnonzero(active)],
        "active_cols": support_cols[np.flatnonzero(active)],
    }


def spconv_batched_matvec(
    spconv_batch: "spconv.SparseConvTensor", vector_batch: "torch.Tensor", transpose: bool
):
    if spconv is None or torch is None:
        raise RuntimeError("torch/spconv required")
    batch_indices = spconv_batch.indices[:, 0]
    row_indices = spconv_batch.indices[:, 2 if transpose else 1]
    column_indices = spconv_batch.indices[:, 1 if transpose else 2]
    output_batch = torch.zeros_like(vector_batch, device=vector_batch.device)
    updated_features = spconv_batch.features * vector_batch[
        batch_indices, column_indices
    ].unsqueeze(-1)
    spconv_batch = spconv_batch.replace_feature(updated_features)
    for batch_index in range(spconv_batch.batch_size):
        current_batch_mask = batch_indices == batch_index
        current_row_indices = row_indices[current_batch_mask].to(torch.int64)
        current_features = spconv_batch.features[current_batch_mask].squeeze()
        batch_output = torch.zeros(
            vector_batch.shape[-1], device=vector_batch.device, dtype=spconv_batch.features.dtype
        )
        output_batch[batch_index] = batch_output.scatter_reduce(
            dim=0, index=current_row_indices, src=current_features, reduce="sum"
        )
    return output_batch


def spconv_to_coo_list(sparse_tensor: "spconv.SparseConvTensor") -> List[coo_matrix]:
    if spconv is None or torch is None:
        raise RuntimeError("torch/spconv required")
    _indices = sparse_tensor.indices.t()
    _values = sparse_tensor.features.view(-1)
    _batch_size = sparse_tensor.batch_size
    _spatial_shape = sparse_tensor.spatial_shape
    coo_tensor = torch.sparse_coo_tensor(
        indices=_indices, values=_values, size=(_batch_size, *_spatial_shape)
    )
    coolist = []
    for batch_idx in range(_batch_size):
        batch_indices = coo_tensor.indices()[0] == batch_idx
        values = coo_tensor.values()[batch_indices]
        indices = coo_tensor.indices()[1:, batch_indices]
        shape = coo_tensor.size()[1:]
        coolist.append(coo_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape))
    return coolist


def spconv_to_csr_list(sparse_tensor: "spconv.SparseConvTensor"):
    return [coo.tocsr() for coo in spconv_to_coo_list(sparse_tensor)]


def make_spconv_input(
    values: np.ndarray, indices: np.ndarray, shape: list, batch_size=1, device=None
):
    if spconv is None or torch is None:
        raise RuntimeError("torch/spconv required")
    if device is None:
        device = torch.device("cuda")
    batch = {"features": [values], "indices": [indices]}
    features = torch.from_numpy(np.vstack(batch["features"])).to(torch.float64).to(device)
    cor = torch.from_numpy(np.vstack(batch["indices"])).int().to(device)
    cor = torch.cat([torch.full((cor.shape[0], 1), 0, device=device), cor], dim=1)
    return spconv.SparseConvTensor(features, cor, shape, batch_size)


def append_loss_csv(file_path, epoch, loss):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{loss}\n")


def clear_text_file(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")


def spconv_to_torch_sparse(sparse_tensor: "spconv.SparseConvTensor") -> "torch.sparse_coo_tensor":
    if spconv is None or torch is None:
        raise RuntimeError("torch/spconv required")
    _indices = sparse_tensor.indices.t()
    _values = sparse_tensor.features.view(-1)
    _batch_size = sparse_tensor.batch_size
    _spatial_shape = sparse_tensor.spatial_shape
    return torch.sparse_coo_tensor(
        indices=_indices, values=_values, size=(_batch_size, *_spatial_shape)
    )


def torch_sparse_to_spconv(coo_tensor: "torch.sparse_coo_tensor") -> "spconv.SparseConvTensor":
    if spconv is None or torch is None:
        raise RuntimeError("torch/spconv required")
    coalesced = coo_tensor.coalesce()
    device = coalesced.device
    indices = coalesced.indices().transpose(0, 1).contiguous().to(device=device, dtype=torch.int32)
    values = coalesced.values().unsqueeze(1).to(device=device)
    batch_size = coalesced.size()[0]
    spatial_shape = coalesced.size()[-2:]
    return spconv.SparseConvTensor(values, indices, spatial_shape, batch_size)


def ensure_spconv_input(x):
    """Normalize sparse torch input into an spconv sparse tensor."""
    if torch is None:
        raise RuntimeError("torch required")
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            return torch_sparse_to_spconv(x)
        raise TypeError("Expected sparse torch tensor for ML models")
    return x


def sync_cuda(device) -> None:
    """Synchronize the CUDA device when timing GPU work."""
    if torch is not None and device is not None and getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize(device)


def clone_graph_data(graph: "Data") -> "Data":
    """Clone a PyG graph while preserving non-tensor attributes."""
    if hasattr(graph, "clone"):
        return graph.clone()
    if Data is None:
        raise RuntimeError("torch-geometric is required for graph data utilities")
    cloned = Data()
    for key, value in graph.items():
        if hasattr(value, "clone"):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def coo_to_torch_sparse(coo: coo_matrix) -> "torch.sparse_coo_tensor":
    if torch is None:
        raise RuntimeError("torch required")
    row = coo.row
    col = coo.col
    batch_index = np.full((1, len(row)), 0)
    i = torch.LongTensor(np.vstack((batch_index, row, col)))
    val = torch.tensor(coo.data, dtype=torch.float32)
    size = (1, coo.shape[0], coo.shape[1])
    return torch.sparse_coo_tensor(i, val, size, dtype=torch.float32)


def torch_sparse_to_coo(coo_tensor: "torch.sparse_coo_tensor") -> coo_matrix:
    indices = coo_tensor.coalesce().indices().numpy()[1:, :]
    values = coo_tensor.coalesce().values().detach().cpu().numpy()
    shape = coo_tensor.size()[-2:]
    return coo_matrix((values, indices), shape)


def torch_sparse_to_scipy_csr(coo_tensor: "torch.sparse_coo_tensor"):
    """Convert a 2D/3D torch sparse tensor to a SciPy CSR matrix."""
    if torch is None:
        raise RuntimeError("torch required")
    coalesced = coo_tensor.coalesce()
    indices = coalesced.indices().cpu().numpy()
    values = coalesced.values().detach().cpu().numpy()
    shape = coalesced.size()
    if len(shape) == 3:
        return coo_matrix((values, indices[1:, :]), shape=shape[-2:]).tocsr()
    if len(shape) == 2:
        return coo_matrix((values, indices), shape=shape).tocsr()
    raise ValueError(f"Expected sparse tensor with 2 or 3 dims, got shape {tuple(shape)}")


def tensor_matrix_parts(
    matrix: "torch.Tensor",
) -> tuple[int, "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Extract `(n, rows, cols, values)` from a sparse matrix tensor."""
    if torch is None:
        raise RuntimeError("torch required")
    coalesced = matrix.coalesce()
    n = int(matrix.size(-1))
    indices = coalesced.indices().cpu()
    values = coalesced.values().detach().cpu().to(torch.float32)
    rows = indices[-2].to(torch.int64)
    cols = indices[-1].to(torch.int64)
    return n, rows, cols, values


def extract_lower_triangle(sparse_coo_tensor, D=True, OD=True):
    if torch is None:
        raise RuntimeError("torch required")
    indices = sparse_coo_tensor.coalesce().indices()
    values = sparse_coo_tensor.coalesce().values()
    if OD:
        mask = indices[1] == indices[2]
        diagonal_indices = indices[:, mask]
        diagonal_values = values[mask]
        return torch.sparse_coo_tensor(diagonal_indices, diagonal_values, sparse_coo_tensor.size())
    if D:
        mask = indices[1] >= indices[2]
    else:
        mask = indices[1] > indices[2]
    lower_tri_indices = indices[:, mask]
    lower_tri_values = values[mask]
    return torch.sparse_coo_tensor(lower_tri_indices, lower_tri_values, sparse_coo_tensor.size())


def remove_outlier_rows_cols(sparse_tensor, threshold=1e30):
    if torch is None:
        raise RuntimeError("torch required")
    dense_matrix = sparse_tensor.to_dense().squeeze()
    mask = torch.abs(dense_matrix) > threshold
    rows_to_remove = torch.any(mask, dim=1).nonzero().squeeze(1).tolist()
    cols_to_remove = torch.any(mask, dim=0).nonzero().squeeze(1).tolist()
    dense_matrix = dense_matrix[
        [i for i in range(dense_matrix.shape[0]) if i not in rows_to_remove], :
    ]
    dense_matrix = dense_matrix[
        :, [i for i in range(dense_matrix.shape[1]) if i not in cols_to_remove]
    ]
    return dense_matrix, rows_to_remove


def restore_removed_rows_cols(sparse_tensor, removed_data):
    if torch is None:
        raise RuntimeError("torch required")
    dense_matrix = sparse_tensor.to_dense().squeeze()
    for row_idx, cols in removed_data:
        dense_matrix = torch.cat(
            [dense_matrix[:row_idx, :], cols.unsqueeze(0), dense_matrix[row_idx:, :]], dim=0
        )
        dense_matrix = torch.cat(
            [dense_matrix[:, :row_idx], cols.unsqueeze(1), dense_matrix[:, row_idx:]], dim=1
        )
    return dense_matrix


def restore_removed_diagonal(sparse_tensor, removed_data):
    if torch is None:
        raise RuntimeError("torch required")
    dense_matrix = sparse_tensor.to_dense().squeeze()
    for row_idx in removed_data:
        dense_matrix = torch.cat(
            [dense_matrix[:row_idx], torch.tensor(0.0).unsqueeze(0), dense_matrix[row_idx:]]
        )
    return torch.diag(dense_matrix)


def lanczos_extremal_eigs(A, k_min=100, k_max=200, tol=1e-6):
    if torch is None:
        raise RuntimeError("torch required")
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    Q_list = []
    alphas = []
    betas = []

    q = torch.randn(n, dtype=dtype, device=device)
    q = q / torch.norm(q)
    Q_list.append(q)

    prev_max_eig = None
    prev_min_eig = None
    for k in range(k_max):
        v = A @ Q_list[-1]
        if k > 0:
            v = v - betas[-1] * Q_list[-2]
        alpha = torch.dot(Q_list[-1], v)
        v = v - alpha * Q_list[-1]
        beta = torch.norm(v)
        alphas.append(alpha)
        betas.append(beta)
        if beta > 0 and k + 1 < k_max:
            Q_list.append(v / beta)
        if k + 1 >= k_min:
            T = torch.zeros((k + 1, k + 1), dtype=dtype, device=device)
            T += torch.diag(torch.stack(alphas))
            if len(betas) > 1:
                T += torch.diag(torch.stack(betas[:-1]), 1)
                T += torch.diag(torch.stack(betas[:-1]), -1)
            eigvals = torch.linalg.eigh(T)[0]
            if prev_max_eig is not None:
                max_diff = torch.abs(eigvals[-1] - prev_max_eig)
                min_diff = torch.abs(eigvals[0] - prev_min_eig)
                if max_diff < tol and min_diff < tol:
                    break
            prev_max_eig = eigvals[-1]
            prev_min_eig = eigvals[0]
    return prev_min_eig, prev_max_eig


# Backwards-compatible aliases
sparse_matvec_mul = spconv_batched_matvec
sparseconv_to_coolist = spconv_to_coo_list
sparseconv_to_csrlist = spconv_to_csr_list
