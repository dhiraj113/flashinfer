"""The top-k router: one entry point, the kernel chosen from the device facts and the problem.

This is what a consumer sees as a single backend.  The rules, each from ledger cells:

1. Rows that fit one CTA's registers (16K fp32, 32K 16-bit): the register-resident kernel.
2. Rows up to eight register slices (128K fp32, 256K 16-bit) in batches whose clusters all
   fit one wave, on any part with clusters: the clustered register-resident kernel.  It won
   or tied the streaming kernel on every measured cell of that grid (B200, k=1024: 32K b=8
   7.87 vs 8.07 us, 32K b=32 8.07 vs 8.69, 128K b=8 8.48 vs 8.90, bf16 64K b=32 8.48 vs
   9.71) except fp32 64K rows in batches under 32 at k <= 1024 (8.48 vs 8.28, 8.28 vs 7.87),
   which stay with the streaming kernel.  A second wave of clusters costs the whole gain
   (128K b=16: 14.8 vs 8.9), hence the occupancy bound.
3. Everything else: the streaming kernel, whose own policy picks the split and merge.

All three share the output contract, so the choice is invisible to the caller.
"""

from __future__ import annotations

import torch

from ...dispatch.device import device_facts, max_active_clusters

from ..kernels.register_cluster import (
    SLICE_ELEMENTS,
    RegisterClusterConfig,
    register_cluster_config_for,
    topk_register_cluster,
)
from ..kernels.register_resident import (
    RegisterConfig,
    register_config_for,
    topk_register,
)
from ..kernels.streaming import StreamingConfig, topk_streaming
from .streaming_policy import streaming_config_for

__all__ = ["REGISTER_MAX_ROW", "choose", "topk"]

REGISTER_MAX_ROW = (
    1024 * 16
)  # words per thread x threads; 16-bit rows hold twice as many


def _cluster_kernel_wins(facts, dtype: torch.dtype, k: int, n: int, rows: int) -> bool:
    # the device's cluster cap is about the streaming kernel's long DSMEM merges; this kernel
    # merges 256 group sums and a few fine bins, and its 8-CTA form beat the alternatives even
    # on consumer Blackwell (RTX 5080 128K b=8: 11.2 us vs 12.4 streaming, 12.0 walk-first)
    if not facts.supports_clusters or k > 4096:
        return False
    if n <= SLICE_ELEMENTS or n > 8 * SLICE_ELEMENTS:
        return False
    splits = next(s for s in (2, 4, 8) if n <= s * SLICE_ELEMENTS)
    if rows > max_active_clusters(facts.index, splits):
        return False
    if (
        dtype == torch.float32
        and 2 * SLICE_ELEMENTS < n <= 4 * SLICE_ELEMENTS
        and rows < 32
        and k <= 1024
    ):
        return False  # the measured exception: fp32 64K rows in small batches
    return True


def choose(
    facts, dtype: torch.dtype, k: int, n: int, rows: int
) -> tuple[str, RegisterConfig | RegisterClusterConfig | StreamingConfig]:
    """(kernel name, configuration) for a batch of ``rows`` rows of ``n`` elements."""
    per_word = 1 if dtype == torch.float32 else 2
    if n <= REGISTER_MAX_ROW * per_word and k <= 4096:
        return "register", register_config_for(facts, dtype, k, n, rows)
    if _cluster_kernel_wins(facts, dtype, k, n, rows):
        return "register_cluster", register_cluster_config_for(facts, dtype, k, n)
    return "streaming", streaming_config_for(facts, dtype, k, n, rows)


def topk(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest elements of each row of ``x`` (rows, N), int32 (rows, k).

    ``lengths`` (rows, int32) limits each row to its first elements; rows shorter than k are
    padded with -1.  NaN ranks above +inf.  Order within a row is unspecified.
    """
    rows, n = x.shape
    kernel, config = choose(device_facts(x.device), x.dtype, k, n, rows)
    if isinstance(config, RegisterConfig):
        return topk_register(x, k, lengths=lengths, config=config, out=out)
    if isinstance(config, RegisterClusterConfig):
        return topk_register_cluster(x, k, lengths=lengths, config=config, out=out)
    return topk_streaming(x, k, lengths=lengths, config=config, out=out)
