"""Resolution: from a usable survivor set to the k output indices.

The crossing at rank k over the survivor histogram splits the stage into three: bins above
the crossing are winners outright, the crossing bin holds the ties for the last
``k - above`` slots, bins below are discarded.  One pass over the stage emits winners at a
shared cursor and copies the ties with their exact keys to the tie stage; then the tie select
(ballot for small sets, byte radix otherwise) fills the remainder.

Measured and kept as is: the plain same-address shared atomic per emitted candidate.  Two
warp-aggregated forms (shuffle scan; ballot + popcount) were both slower here (1.1 -> 2.0 and
1.5 us at 64K, k=2048).  The hardware coalesces these atomics well enough.
"""

import cutlass
import cutlass.cute as cute

from ...block.cluster_merge import merge_histograms_256
from ...block.crossing import crossing_256_warp
from ...block.tie_select import tie_select_ballot, tie_select_radix
from ...device.atomics import shared_add
from ...device.cluster import (
    cluster_sync,
    peer_add_i32,
    peer_shared_address,
    peer_store_i32,
)

from .binning import survivor_bin

__all__ = ["emit_and_select", "emit_and_select_cluster"]


@cute.jit
def emit_and_select(
    elems,
    k: cutlass.Constexpr,
    survivors,
    bar,
    scale,
    out_row,
    s_keys,
    s_idx,
    s_hist,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Write the k winners of a usable stage to ``out_row``; return 1, or 0 if the crossing
    bin overflowed the tie stage (the caller then takes the exact fallback).

    Preconditions: ``k <= survivors <= capacity``; ``s_hist`` is the histogram the stage was
    built with, under the same ``bar`` and ``scale``.  ``s_keys`` (>= 512 Int32) is dead after
    the emit and is reused as the radix select's histogram.  ``s_result``: 8 Int32 scratch;
    ``s_slots``: 8 Int32.  Barriers: two, plus the radix select's when taken.
    """
    if tidx < 32:
        b, above, _c, _b2, _a2, _c2 = crossing_256_warp(
            s_hist, cutlass.Int32(k), cutlass.Int32(k), tidx, False
        )
        if tidx == 0:
            s_result[0] = b
            s_result[1] = above
            s_result[6] = cutlass.Int32(0)  # winner cursor
            s_result[7] = cutlass.Int32(0)  # tie cursor
    cute.arch.barrier()
    cut_bin = s_result[0]
    above = s_result[1]
    for t in range(tidx, survivors, threads):
        bits = cutlass.Uint32(s_keys[t])
        b = survivor_bin(elems.value(bits), bar, scale)
        if b > cut_bin:
            p = shared_add(s_result + 6, 1)
            if p < cutlass.Int32(k):
                out_row[p] = s_idx[t]
        else:
            if b == cut_bin:
                e = shared_add(s_result + 7, 1)
                if e < cutlass.Int32(tie_capacity):
                    s_tie_keys[e] = elems.key(bits)
                    s_tie_idx[e] = s_idx[t]
    cute.arch.barrier()
    return _select_ties(
        elems,
        k,
        above,
        s_result[7],
        out_row,
        s_keys,
        s_tie_keys,
        s_tie_idx,
        tie_capacity,
        ballot_limit,
        s_slots,
        s_result,
        tidx,
        threads,
    )


@cute.jit
def _select_ties(
    elems,
    k: cutlass.Constexpr,
    above,
    ties,
    out_row,
    s_scratch,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Fill ``out_row[above, k)`` from the tie stage; 0 if the ties overflowed it."""
    remaining = cutlass.Int32(k) - above
    ok = cutlass.Int32(1)
    if (remaining < 0) | (remaining > ties) | (ties > cutlass.Int32(tie_capacity)):
        ok = cutlass.Int32(0)
    if ok == 1:
        if remaining > 0:
            if ties <= cutlass.Int32(ballot_limit):
                tie_select_ballot(
                    s_tie_keys,
                    s_tie_idx,
                    ties,
                    remaining,
                    out_row,
                    above,
                    tidx,
                    threads,
                )
            else:
                tie_select_radix(
                    s_tie_keys,
                    s_tie_idx,
                    ties,
                    remaining,
                    out_row,
                    above,
                    s_scratch,
                    s_slots,
                    s_result,
                    tidx,
                    threads,
                    elems.key_shifts,
                )
    return ok


@cute.jit
def emit_and_select_cluster(
    elems,
    k: cutlass.Constexpr,
    rank,
    splits: cutlass.Constexpr,
    bar,
    scale,
    out_row,
    s_keys,
    s_idx,
    s_hist,
    s_merged,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_count,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Cluster form of the resolution: the row's stage is spread over ``splits`` CTAs.

    Every CTA merges the peers' survivor histograms over DSMEM (identical result everywhere),
    finds the rank-k crossing, and classifies its own stage: winners go straight to the output
    at a cursor living in rank 0's shared memory, ties are pushed into rank 0's tie stage.
    After the cluster barrier rank 0 alone selects the ties.  Returns ``ok`` (meaningful on
    rank 0; other ranks return 1 and must not run the fallback).  Preconditions: the verdict's
    cluster barrier has passed (every peer's histogram is complete) and every CTA zeroed its
    cursors (``s_result[6..7]``) before that barrier.  Two cluster barriers, two block barriers.
    """
    merge_histograms_256(s_hist, s_merged, splits, tidx)
    cute.arch.barrier()
    if tidx < 32:
        b, above, _c, _b2, _a2, _c2 = crossing_256_warp(
            s_merged, cutlass.Int32(k), cutlass.Int32(k), tidx, False
        )
        if tidx == 0:
            s_result[0] = b
            s_result[1] = above
    cute.arch.barrier()
    cut_bin = s_result[0]
    above = s_result[1]
    root = cutlass.Int32(0)
    winner_cursor = peer_shared_address((s_result + 6).toint(), root)
    tie_cursor = peer_shared_address((s_result + 7).toint(), root)
    tie_keys_root = peer_shared_address(s_tie_keys.toint(), root)
    tie_idx_root = peer_shared_address(s_tie_idx.toint(), root)
    local = s_count[0]
    for t in range(tidx, local, threads):
        bits = cutlass.Uint32(s_keys[t])
        b = survivor_bin(elems.value(bits), bar, scale)
        if b > cut_bin:
            p = peer_add_i32(winner_cursor, cutlass.Int32(1))
            if p < cutlass.Int32(k):
                out_row[p] = s_idx[t]
        else:
            if b == cut_bin:
                e = peer_add_i32(tie_cursor, cutlass.Int32(1))
                if e < cutlass.Int32(tie_capacity):
                    peer_store_i32(
                        tie_keys_root + e * 4, elems.key(bits).bitcast(cutlass.Int32)
                    )
                    peer_store_i32(tie_idx_root + e * 4, s_idx[t])
    cluster_sync()  # every CTA's winners and ties are in; peers may stop touching rank 0 now
    ok = cutlass.Int32(1)
    if rank == 0:
        ok = _select_ties(
            elems,
            k,
            above,
            s_result[7],
            out_row,
            s_keys,
            s_tie_keys,
            s_tie_idx,
            tie_capacity,
            ballot_limit,
            s_slots,
            s_result,
            tidx,
            threads,
        )
    return ok
