"""The top-k router: one entry point, the kernel chosen from the device facts and the problem.

This is what a consumer sees as a single backend.  The rules, each from ledger cells:

1. Rows that fit one CTA's registers (16K fp32, 32K 16-bit): the register-resident kernel.
2. Rows up to eight register slices (128K fp32, 256K 16-bit) in batches whose clusters all
   fit one wave, on any part with clusters: the clustered register-resident kernel.  It won
   or tied the streaming kernel on every measured cell of that grid (B200, k=1024: 32K b=8
   7.87 vs 8.07 us, 32K b=32 8.07 vs 8.69, 128K b=8 8.48 vs 8.90, bf16 64K b=32 8.48 vs
   9.71) except fp32 64K rows in batches under 32 at k <= 1024 on H100 and B200 (8.28 vs 7.87;
   H100 13.7 vs 12.5), which stay with the streaming kernel there; on Rubin and the RTX 5080
   the cluster kernel wins that cell too (6.25 vs 6.42; 8.94 vs 11.0), recorded as the device
   fact ``cheap_small_clusters``.  A second wave of clusters costs the whole gain (128K b=16:
   14.8 vs 8.9), hence the occupancy bound.
3. Large k on rows beyond both register kernels: the census split kernel, exact and flat in k
   (two row passes split across the row's CTAs, every CTA emitting its own winners).  The
   sampled streaming kernel stages about 1.5k survivors and hands the answer to one CTA, so
   it loses once k is a few percent of the row: B200 1M b=8 k=65536 60.3 vs 36.6 us, k=200000
   186.6 vs 37.3, 256K b=8 k=100000 75.9 vs 23.2; it still wins at k=16384 (28.1 vs 32.8) and
   below.  The rule (``_census_split_wins``): k at least N / 24 when the row splits eight ways
   or more, N / 8 otherwise (wide batches split little and the census's two passes per CTA
   cost more: 256K b=64 k=16384 33.3 vs 51.3), and always for k at or above N (identity).
4. Everything else: the streaming kernel, whose own policy picks the split and merge.

All four share the output contract, so the choice is invisible to the caller.
"""

from __future__ import annotations

import torch

from ...dispatch.device import cluster_capacity, device_facts

from ..kernels.census_split import (
    CensusSplitConfig,
    census_split_config_for,
    topk_census_split,
)
from ..kernels.register_cluster import (
    SLICE_ELEMENTS,
    RegisterClusterConfig,
    register_cluster_config_for,
    topk_register_cluster,
)
from ..kernels.register_resident import (
    _DTYPES,
    RegisterConfig,
    register_config_for,
    topk_register,
)
from ..phases.elements import Elements
from ..kernels.streaming import StreamingConfig, topk_streaming
from .streaming_policy import streaming_config_for

__all__ = ["REGISTER_MAX_ROW", "choose", "topk"]


def _census_split_wins(facts, k: int, n: int, rows: int) -> bool:
    if k >= n:
        return True
    splits = census_split_config_for(facts, torch.float32, k, n, rows).splits
    return k * (24 if splits >= 8 else 8) >= n


REGISTER_MAX_ROW = (
    1024 * 16
)  # words per thread x threads; 16-bit rows hold twice as many


def _cluster_kernel_wins(facts, dtype: torch.dtype, k: int, n: int, rows: int) -> bool:
    # the device's cluster cap is about the streaming kernel's long DSMEM merges; this kernel
    # merges 256 group sums and a few fine bins, and its 8-CTA form beat the alternatives even
    # on consumer Blackwell (RTX 5080 128K b=8: 11.2 us vs 12.4 streaming, 12.0 walk-first)
    if not facts.supports_clusters:
        return False
    if n <= SLICE_ELEMENTS or n > 8 * SLICE_ELEMENTS:
        return False
    splits = next(s for s in (2, 4, 8) if n <= s * SLICE_ELEMENTS)
    if rows > cluster_capacity(facts.index, splits):
        return False
    if (
        dtype == torch.float32
        and 2 * SLICE_ELEMENTS < n <= 4 * SLICE_ELEMENTS
        and rows < 32
        and k <= 1024
    ):
        return facts.cheap_small_clusters  # fp32 64K rows in small batches: the streaming kernel wins on H100 and B200 (device fact)
    return True


def choose(
    facts, dtype: torch.dtype, k: int, n: int, rows: int
) -> tuple[str, RegisterConfig | RegisterClusterConfig | StreamingConfig]:
    """(kernel name, configuration) for a batch of ``rows`` rows of ``n`` elements.

    Any k: the register kernels refine an overflowing crossing bin by the radix select over
    its key range, the streaming kernel widens its stage or its split for k beyond the tie
    stage, and a k that nothing holds (or k >= N) takes the exact select for every row.
    Raises ``ValueError`` when no configuration fits the device, so callers can decline up front.
    """
    per_word = 1 if dtype == torch.float32 else 2
    config: RegisterConfig | RegisterClusterConfig | StreamingConfig | CensusSplitConfig
    if n <= REGISTER_MAX_ROW * per_word and k < n:
        kernel, config = "register", register_config_for(facts, dtype, k, n, rows)
    elif k < n and _cluster_kernel_wins(facts, dtype, k, n, rows):
        # one pass from registers beats the census's two at any k (64K b=8 k=32768: 12.9 vs 14.6 us)
        kernel, config = (
            "register_cluster",
            register_cluster_config_for(facts, dtype, k, n),
        )
    elif _census_split_wins(facts, k, n, rows):
        kernel, config = (
            "census_split",
            census_split_config_for(facts, dtype, k, n, rows),
        )
    else:
        kernel, config = "streaming", streaming_config_for(facts, dtype, k, n, rows)
    config.validate(k, Elements.of(_DTYPES[dtype]), facts.shared_memory_optin)
    return kernel, config


def topk(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    next_n: int = 1,
    compress_ratio: int = 1,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest elements of each row of ``x`` (rows, N), int32 (rows, k).

    ``lengths`` (rows // next_n, int32) limits each row to its first elements: row r sees
    ``(lengths[r // next_n] - next_n + r % next_n + 1) // compress_ratio`` of them (both default
    to 1: ``lengths[r]``).  Rows shorter than k are padded with -1.  NaN ranks above +inf.
    Order within a row is unspecified.  ``values`` (rows, k, x's dtype), when given, receives
    the selected elements (-inf in padding).  ``workspace`` (CUDA byte tensor of at least
    ``workspace.workspace_bytes(x, k)`` bytes) makes the call allocation-free and stream-private
    under the caller's control; without it the kernels use per-(device, stream, shape) caches
    (``dispatch/workspace.py``).
    """
    rows, n = x.shape
    kernel, config = choose(device_facts(x.device), x.dtype, k, n, rows)
    extra = {
        "lengths": lengths,
        "out": out,
        "values": values,
        "next_n": next_n,
        "compress_ratio": compress_ratio,
        "workspace": workspace,
    }
    if isinstance(config, RegisterConfig):
        return topk_register(x, k, config=config, **extra)
    if isinstance(config, RegisterClusterConfig):
        return topk_register_cluster(x, k, config=config, **extra)
    if isinstance(config, CensusSplitConfig):
        return topk_census_split(x, k, config=config, **extra)
    return topk_streaming(x, k, config=config, **extra)
