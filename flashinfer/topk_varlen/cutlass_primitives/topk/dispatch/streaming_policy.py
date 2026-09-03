"""Default streaming configuration for a device and a problem: the routing that reads the facts.

Rules, each with its measurement:

* Stage 16K candidates for rows of 1 MB or more, 8K below.  At 1M fp32 elements the tight
  aim is 6144 and its spread reaches 10K survivors on some rows; with an 8K stage one row in
  a batch of 148 overflowed, re-walked (75 us) and set the batch time: 174 us against 105 us
  with the larger stage.  Below 1 MB the aim stays under 3K and 8K never overflows.
* The larger stage needs about 150 KB of shared memory; parts with a 99 KB budget (L40S,
  RTX 5080) keep 8K.
* Programmatic dependent launch and the packed bf16 compare on SM90 and newer; the packed
  fp16 compare everywhere.
* Row splits, cluster merge (SM90+): when one CTA per row leaves SMs idle, the row is shared
  by the largest cluster shape whose clusters all fit one wave (``cluster_capacity``: the
  driver's occupancy query, corrected for small shapes where it under-reports; never plain
  ``SMs // size``: B200 fits 45 clusters of 3, not 49) and whose slices keep at least 4096
  elements.  Shapes above 4 only pay off from 4 MB rows (the wider merge outweighs the shorter
  slice below that; measured on Rubin 256K b=32: 4 -> 6 went 8.6 -> 8.9 us), and below 1 MB
  shape 3 already loses (64K b=64: 7.0 -> 7.6 us), so the allowed set narrows with the row
  size.  When no shape fits one wave, the largest shape whose CTAs fit the SM count is taken
  anyway: its second partial wave costs at most twice a wave, which is never worse than one
  CTA per row (H100 1M b=64: cluster 2 95.7 us, one CTA 155; Rubin 1M b=64: cluster 3 38.0,
  cluster 2 43.8).
* Row splits, slab merge, small batches of long rows on any part: 16 CTAs per row from 1 MB
  and 32 from 4 MB when the grid fits one wave.  The cluster merge caps at 8 CTAs and the slab
  slices are shorter; measured on B200 1M b=8: cluster 8 15.7 us, slab 16 13.8; b=1: cluster 8
  15.6, slab 32 12.2; 256K b=8: cluster 6 10.3, slab 16 10.1.  Between 16-way slabs and 8-way
  slabs sits the 8-CTA cluster where it fits one wave (1M b=12: B200 16.5 vs slab 8 18.3, H100
  29.9 vs 33.4; Rubin 1M b=16: 13.0 vs 15.3).  The slab form cannot re-walk an undershoot, so
  it takes the wide aim.
* Row splits, slab merge, parts without clusters: the largest of 8/4/2 CTAs per row that
  keeps the grid within one wave and slices of at least 4096 elements.  Cluster shapes are
  capped by the device fact ``max_fast_cluster`` (4 on consumer Blackwell, where clusters of 6
  and 8 measured far slower than the slab; 8 on the data-center parts).  Two waves of half
  rows cost more than one wave of whole rows (A100 64K b=64: 2 CTAs per row 38.5 us against
  one CTA 23.3).  Exception from FlashInfer's A100 measurements: rows of 4 MB or more in
  batches up to 64 split four ways even into a second wave, because one CTA streams a 4 MB row
  at ~12 GB/s against ~20 GB/s for shorter slices (1M fp32 b=64: 364 -> 249 us).
* Wide batches (more rows than SMs) run 512-thread CTAs two per SM when the row is at most
  256 KB (512 KB up to two rows per SM).  A 512-thread CTA solves a row 10-60% slower alone,
  so this only pays when the batch runs in waves: measured on B200 at b=256, 64K k=1024
  21.8 -> 18.1 us, 128K 38.2 -> 34.0, ragged 64K k=2048 24.4 -> 20.6.  The stage is 4K for
  k <= 1024 and 8K for k=2048: at k=2048 the tight aim (3072) sits at a 4K stage's cap and
  the survivor spread (up to 3.9K on a 256-row batch) overflowed one row into a re-walk that
  set the batch to 29.5 us.  Above the byte bound the longer per-CTA pass outweighs the
  second wave saved.
"""

from __future__ import annotations

import torch

from ...dispatch.device import DeviceFacts, cluster_capacity

from ..kernels.streaming import StreamingConfig

__all__ = ["streaming_config_for", "splits_for"]

BIG_ROW_BYTES = 1 << 20
MIN_SLICE = 4096


def splits_for(
    facts: DeviceFacts, rows: int, n: int, elem_bytes: int
) -> tuple[int, str]:
    """(CTAs per row, merge medium) for a batch of ``rows`` rows of ``n`` elements."""
    if n < 16384 or rows >= facts.sm_count:
        return 1, "cluster"
    row_bytes = n * elem_bytes
    # long rows in small batches: wide slab splits beat the widest cluster (B200 1M b=8: slab
    # 16 13.8 us, cluster 8 15.7; RTX 5080 1M b=8: slab 8 23.7, cluster 8 45.6)
    if row_bytes >= BIG_ROW_BYTES:
        for s in (32, 16) if row_bytes >= 4 * BIG_ROW_BYTES else (16,):
            if n // s >= MIN_SLICE and rows * s <= facts.sm_count:
                return s, "slab"
        # below 16 CTAs per row the widest cluster beats the 8-way slab (1M b=12: B200 16.5 vs
        # 18.3 us, H100 29.9 vs 33.4; Rubin 1M b=16: 13.0 vs 15.3), and the slab beats a cluster
        # that would need a second wave (B200 1M b=16: slab 8 18.3, cluster 8 30.6)
        if facts.max_fast_cluster >= 8 and rows <= cluster_capacity(facts.index, 8):
            return 8, "cluster"
        if n // 8 >= MIN_SLICE and rows * 8 <= facts.sm_count:
            return 8, "slab"
    if facts.max_fast_cluster >= 2:
        shapes: tuple
        if row_bytes >= 4 * BIG_ROW_BYTES:
            shapes = (8, 6, 4, 3, 2)
        elif row_bytes >= BIG_ROW_BYTES:
            shapes = (6, 4, 3, 2)
        else:
            shapes = (4, 2)
        for s in shapes:
            if (
                s <= facts.max_fast_cluster
                and n // s >= MIN_SLICE
                and rows <= cluster_capacity(facts.index, s)
            ):
                return s, "cluster"
        # no shape fits one wave: a second wave of a few small clusters still beats one CTA per
        # row on half the SMs (H100 1M b=64: cluster 2 95.7 us, one CTA 155; the split rows
        # take at most twice a wave, which is never worse than the unsplit row)
        for s in shapes:
            if (
                s <= facts.max_fast_cluster
                and n // s >= MIN_SLICE
                and rows * s <= facts.sm_count
            ):
                return s, "cluster"
        return 1, "cluster"
    for s in (8, 4, 2):
        if n // s >= MIN_SLICE and rows * s <= facts.sm_count:
            return s, "slab"
    if row_bytes >= 4 * BIG_ROW_BYTES and rows <= 64:
        return 4, "slab"
    return 1, "slab"


def streaming_config_for(
    facts: DeviceFacts, dtype: torch.dtype, k: int, n: int, rows: int = 0
) -> StreamingConfig:
    """The configuration the dispatcher would choose for (device, dtype, k, row length, batch)."""
    elem_bytes = 4 if dtype == torch.float32 else 2
    row_bytes = n * elem_bytes
    threads = 1024
    tie_capacity = max(2 * threads, k)
    packed = dtype == torch.float16 or (
        dtype == torch.bfloat16 and facts.packed_bf16_compare
    )
    splits, merge = splits_for(facts, rows, n, elem_bytes) if rows else (1, "cluster")
    aim = "wide" if (merge == "slab" and splits > 1) else "tight"
    cfg = StreamingConfig(
        threads=threads,
        tie_capacity=tie_capacity,
        pdl=facts.supports_pdl,
        packed_compare=packed,
        splits=splits,
        merge=merge,
        aim=aim,
    )
    row_kb = row_bytes >> 10
    wide_batch = rows > facts.sm_count and (
        row_kb <= 256 or (row_kb <= 512 and rows <= 2 * facts.sm_count)
    )
    if wide_batch and k <= 2048:
        wide_stage = 4096 if k <= 1024 else 8192
        wide = StreamingConfig(
            **{
                **cfg.__dict__,
                "threads": 512,
                "ctas_per_sm": 2,
                "stage": wide_stage,
                "tie_capacity": max(1024, k),
            }
        )
        if 2 * wide.shared_memory_bytes() <= facts.shared_memory_optin:
            return wide
    if row_bytes >= BIG_ROW_BYTES:
        big = StreamingConfig(**{**cfg.__dict__, "stage": 16384})
        if big.shared_memory_bytes() <= facts.shared_memory_optin:
            cfg = big
    return cfg
