"""Input layout accepted by the kernels, and the view they run on.

The kernels read rows with 16-byte vector loads, so the layout they run on has every row
starting on a 16-byte boundary: the row stride is a multiple of 16 bytes and the base is
16-byte aligned.  The row length itself may be anything (lengths are arbitrary already; the
row's tail past its last whole vector is read by the scalar tail paths).  The stride may exceed
the row length (a paged logits arena, a column-sliced view): the kernels compile the row
length ``n`` and a column offset separately from the tensor's second dimension, which carries
the stride, so a strided input is passed as the compact ``(rows, stride)`` view over its
storage and each row is read from ``offset`` for ``n`` columns.  Stride and offset are part of
the compiled specialization, like ``n``.

Inputs whose rows are not 16-byte aligned (an odd row length in a contiguous tensor, a slice
at an odd column, a non-unit inner stride) are copied once into a padded arena whose rows are.
That copy is one extra pass over the batch; it keeps the contract total without a scalar-load
variant of every kernel for a layout no model shape produces.
"""

from __future__ import annotations

import torch

__all__ = ["check_layout", "arena_view"]

ALIGN = 16


def check_layout(x: torch.Tensor) -> None:
    """Raise unless ``x`` is a 2-D CUDA tensor (any layout the kernels can take or copy)."""
    if x.dim() != 2 or not x.is_cuda:
        raise ValueError("expected a 2-D CUDA tensor")


def _aligned(x: torch.Tensor) -> bool:
    """Whether the kernels can read ``x`` in place: unit inner stride, 16-byte-aligned base and
    row stride, rows not overlapping."""
    esize = x.element_size()
    if x.stride(1) != 1 or x.data_ptr() % ALIGN:
        return False
    if x.shape[0] > 1 and ((x.stride(0) * esize) % ALIGN or x.stride(0) < x.shape[1]):
        return False
    return True


def arena_view(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``(view, offset)``: the compact ``(rows, stride)`` tensor the kernels take and the column
    at which each row starts in it.  ``x`` itself with offset 0 when contiguous and aligned.
    A view over the storage is used when the storage holds ``rows x stride`` elements from the
    view's row-aligned base; otherwise, and whenever the rows are not 16-byte aligned, the rows
    are copied into a padded arena (correct, one extra pass)."""
    rows, n = x.shape
    if _aligned(x):
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
    per_align = ALIGN // x.element_size()
    stride = (
        -(-n // per_align) * per_align
    )  # the row length rounded up to whole 16-byte vectors
    arena = torch.empty(rows, stride, dtype=x.dtype, device=x.device)
    arena[:, :n].copy_(x)
    return arena, 0
