"""Rank-k crossing over a histogram that is spread across a cluster, without merging all of it.

Every CTA holds a local ``bins``-bin histogram of its slice.  Merging all bins over DSMEM is
``bins x splits`` remote loads per CTA (3 to 8 us at 4096 bins on B200, latency-bound).  The
crossing only needs the merged counts of the bins above the answer and of the answer's bin,
so it is found in two levels:

1. Each CTA sums its histogram into 256 groups of ``bins / 256`` consecutive bins (a local
   pass over shared memory).  The groups are merged over DSMEM (256 x splits remote loads per
   CTA, one per thread) and the crossing found among them: group G, ``above`` = the merged
   count in groups above G.
2. The ``bins / 256`` fine bins of group G are merged (one thread each) and the crossing is
   found among them by one thread.  The result is identical to a crossing over the fully
   merged histogram.

Every CTA computes the same answer from the same peer data, so no broadcast is needed.
Precondition: a cluster barrier separates the peers' histogram and group writes from this call.
"""

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_warp
from ...device.cluster import peer_load_i32, peer_shared_address
from ...device.memory import load_shared_16

__all__ = ["summarize_groups_256", "cluster_crossing"]


@cute.jit
def summarize_groups_256(s_bins, s_groups, bins: cutlass.Constexpr, tidx):
    """``s_groups[g]`` = sum of the ``bins / 256`` bins of group g, by threads 0..255.  Sixteen-
    byte shared loads; ``s_bins`` 16-byte aligned.  No barrier (the caller's cluster barrier
    publishes the groups to the peers)."""
    per_group = cutlass.const_expr(bins // 256)
    if tidx < 256:
        base = s_bins.toint() + tidx * (per_group * 4)
        total = cutlass.Uint32(0)
        for q in cutlass.range_constexpr(per_group // 4):
            a, b, c, d = load_shared_16(base + q * 16)
            total = total + a + b + c + d
        s_groups[tidx] = total.to(cutlass.Int32)


@cute.jit
def cluster_crossing(
    s_bins,
    s_groups,
    s_merged_groups,
    s_fine,
    s_result,
    k: cutlass.Constexpr,
    bins: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    tidx,
):
    """Publish the rank-k crossing over the cluster's merged histogram to ``s_result[0..2]``
    as ``(bin, above, count in the bin)``.

    ``s_merged_groups``: 256 Int32 scratch; ``s_fine``: ``bins / 256`` Int32 scratch;
    ``s_result``: 4 Int32.  Remote loads: ``splits`` per thread for threads 0..255, then
    ``splits`` for threads 0..per_group-1.  Three block barriers.  Requires 256 or more threads.
    """
    per_group = cutlass.const_expr(bins // 256)
    if tidx < 256:
        addr = s_groups.toint() + tidx * 4
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(splits):
            total = total + peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
        s_merged_groups[tidx] = total
    cute.arch.barrier()
    if tidx < 32:
        g, above_g, _c, _g2, _a2, _c2 = crossing_256_warp(
            s_merged_groups, cutlass.Int32(k), cutlass.Int32(k), tidx, False
        )
        if tidx == 0:
            s_result[2] = g
            s_result[3] = above_g
    cute.arch.barrier()
    group = s_result[2]
    if tidx < per_group:
        addr = s_bins.toint() + (group * per_group + tidx) * 4
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(splits):
            total = total + peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
        s_fine[tidx] = total
    cute.arch.barrier()
    if tidx == 0:  # walk the group's fine bins from the top
        above = s_result[3]
        found = cutlass.Int32(0)
        for j in cutlass.range_constexpr(per_group - 1, -1, -1):
            c = s_fine[j]
            if found == 0:
                if (above < cutlass.Int32(k)) & (
                    (above + c >= cutlass.Int32(k)) | (j == 0)
                ):
                    s_result[0] = group * per_group + j
                    s_result[1] = above
                    s_result[2] = c
                    found = cutlass.Int32(1)
                else:
                    above = above + c
    cute.arch.barrier()
