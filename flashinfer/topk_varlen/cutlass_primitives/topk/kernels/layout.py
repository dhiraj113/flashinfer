"""Input layout accepted by the kernels, and the view they run on.

The kernels read rows with 16-byte vector loads, so a row must start on a 16-byte boundary
and hold a whole number of vectors: the row stride and the row length are multiples of 16
bytes and the first row is 16-byte aligned.  The stride may exceed the row length (a paged
logits arena, a column-sliced view): the kernels compile the row length ``n`` and a column
offset separately from the tensor's second dimension, which carries the stride, so a strided
input is passed as the compact ``(rows, stride)`` view over its storage and each row is read
from ``offset`` for ``n`` columns.  Stride and offset are part of the compiled specialization,
like ``n``.
"""

from __future__ import annotations

import torch

__all__ = ["check_layout", "arena_view"]


def check_layout(x: torch.Tensor) -> None:
    """Raise unless ``x`` is a 2-D CUDA tensor of 16-byte-aligned, vector-multiple rows."""
    if x.dim() != 2 or not x.is_cuda:
        raise ValueError("expected a 2-D CUDA tensor")
    if x.stride(1) != 1:
        raise ValueError("the inner dimension must be contiguous (stride 1)")
    esize = x.element_size()
    if (x.shape[1] * esize) % 16:
        raise ValueError(
            f"row length must be a multiple of 16 bytes (N={x.shape[1]}, {x.dtype})"
        )
    if x.shape[0] > 1 and (x.stride(0) * esize) % 16:
        raise ValueError(
            f"row stride must be a multiple of 16 bytes (stride={x.stride(0)}, {x.dtype})"
        )
    if x.shape[0] > 1 and x.stride(0) < x.shape[1]:
        raise ValueError("overlapping rows are not supported")
    if x.data_ptr() % 16:
        raise ValueError("rows must start on a 16-byte boundary")


def arena_view(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``(view, offset)``: the compact ``(rows, stride)`` tensor the kernels take and the column
    at which each row starts in it.  ``x`` itself with offset 0 when contiguous.  A view over
    the storage is used when the storage holds ``rows x stride`` elements from the view's
    row-aligned base; otherwise the rows are copied out (correct, one extra pass)."""
    rows, n = x.shape
    if x.is_contiguous():
        return x, 0
    stride = x.stride(0)
    base = x.storage_offset()
    row_base = (
        base - base % stride
    )  # the storage element where x's first row's stride period begins
    offset = base - row_base
    storage_numel = x.untyped_storage().nbytes() // x.element_size()
    if row_base + rows * stride <= storage_numel:
        view = torch.empty(0, dtype=x.dtype, device=x.device).set_(
            x.untyped_storage(), row_base, (rows, stride), (stride, 1)
        )
        return view, offset
    return x.contiguous(), 0
