"""Slab merge: a row split over CTAs that cannot share memory (no clusters, or more than 8).

Each CTA publishes its stage to its own segment of a global slab, bin-major, with a table of
per-bin prefixes, then adds one to the row's arrival counter.  The CTA that sees the count reach
``splits - 1`` is last: everyone else's data is visible to it (release before the add, acquire
after), and it alone merges the histograms, finds the rank-k bin, copies winners and ties by
range, selects the ties, and resets the counter for the next launch.  No CTA ever waits, so
the grid may run in as many waves as it likes.

Bin-major publish is what makes the last arriver cheap: with every segment sorted into bins
by the prefix table, "everything above bin B" and "everything in bin B" are contiguous ranges,
copied without classifying a single candidate again.  Classifying at the last arriver measured
2.5-3 us per row at 1M elements in the FlashInfer kernel before the change.

Limits of this form: the verdict is known only to the last arriver, after every other CTA has
left, so an undershoot cannot be repaired by a re-walk; the dispatcher pairs the slab merge
with the wide aim, and an undershoot or a stage overflow goes to the exact fallback.

Segment layout per row (Int32 words): ``splits x capacity`` keys, ``splits x capacity``
indices, ``splits x TABLE_WORDS`` tables (``[0]`` staged count or -1 on overflow, ``[1 + b]``
exclusive prefix of bin ``b`` for b in 0..256).  One Int32 arrival counter per row, zero
between launches.  The last arriver rebuilds the merged histogram from prefix differences
(2 x splits L2 loads per bin, all independent); accumulating a row histogram with atomics
instead measured 3-4% worse at 1M and 256K b=8 (docs/measured-worse.md).
"""

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_warp
from ...device.atomics import (
    fence_release_gpu,
    global_add,
    global_store_release,
    shared_add,
)
from ...device.memory import load_global_l2_i32
from ...device.timers import read_clock64
from ...device.warp import warp_inclusive_scan_add

from .binning import survivor_bin
from .resolve import _select_ties

__all__ = ["TABLE_WORDS", "slab_words_per_row", "publish_and_arrive", "merge_slab"]

TABLE_WORDS = 258


def slab_words_per_row(splits: int, capacity: int) -> int:
    return splits * (2 * capacity + TABLE_WORDS)


@cute.jit
def publish_and_arrive(
    elems,
    rank,
    splits: cutlass.Constexpr,
    bar,
    scale,
    capacity: cutlass.Constexpr,
    s_count,
    s_keys,
    s_idx,
    s_hist,
    s_cursor,
    s_result,
    slab_keys,
    slab_idx,
    tables,
    counter,
    tidx,
    threads: cutlass.Constexpr,
):
    """Publish this CTA's stage bin-major into its slab segment and arrive; return 1 on the
    last arriver, 0 elsewhere (block-uniform).

    ``s_cursor``: 256 Int32 scratch (per-bin write cursors).  ``slab_keys``/``slab_idx``:
    row base pointers (this CTA writes ``[rank * capacity, ...)``); ``tables``: row base;
    ``counter``: pointer to the row's arrival word.  Two block barriers plus the fence.
    """
    local = s_count[0]
    seg = rank * cutlass.Int32(capacity)
    table = tables + rank * TABLE_WORDS
    if tidx < 32:  # exclusive prefix of the 256 bins: lane l owns bins 8l..8l+7
        mine = cutlass.Int32(0)
        for j in cutlass.range_constexpr(8):
            mine = mine + s_hist[tidx * 8 + j]
        incl = warp_inclusive_scan_add(mine, tidx)
        run = incl - mine
        for j in cutlass.range_constexpr(8):
            s_cursor[tidx * 8 + j] = run
            table[1 + tidx * 8 + j] = run
            run = run + s_hist[tidx * 8 + j]
        if tidx == 31:
            table[1 + 256] = incl
    if tidx == 0:
        staged = local
        if local > cutlass.Int32(capacity):
            staged = cutlass.Int32(-1)  # overflow: the last arriver takes the fallback
        table[0] = staged
    cute.arch.barrier()
    if local <= cutlass.Int32(capacity):
        for t in range(tidx, local, threads):
            bits = cutlass.Uint32(s_keys[t])
            b = survivor_bin(elems.value(bits), bar, scale)
            pos = seg + shared_add(s_cursor + b, 1)
            slab_keys[pos] = s_keys[t]
            slab_idx[pos] = s_idx[t]
    cute.arch.barrier()
    if tidx == 0:
        fence_release_gpu()
        arrived = global_add(counter, 1)
        s_result[10] = cutlass.Int32(0)
        if arrived == cutlass.Int32(splits - 1):
            s_result[10] = cutlass.Int32(1)
    cute.arch.barrier()
    return s_result[10]


@cute.jit
def _table_word(tables, r, w):
    return load_global_l2_i32((tables + r * TABLE_WORDS + w).toint())


@cute.jit
def merge_slab(
    elems,
    k: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    capacity: cutlass.Constexpr,
    out_row,
    s_scratch,
    s_merged,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_result,
    slab_keys,
    slab_idx,
    tables,
    counter,
    tidx,
    threads: cutlass.Constexpr,
    telemetry: cutlass.Constexpr = False,
):
    """On the last arriver: merge the row and write its k winners; return 1, or 0 when any
    segment overflowed, the row undershot k, or the ties overflowed the stage.

    ``s_scratch``: >= 768 Int32 (the dead stage: range descriptors, then the radix select's
    histogram); ``s_merged``: 256 Int32; ``s_result``: 16 Int32.  Resets the arrival counter.
    Reads slab words with L1-bypassing loads.  Barriers: three plus the tie select's.  With
    ``telemetry`` the sub-phase clocks (gather, ranges, copy, select) land in
    ``s_result[12..15]``.
    """
    mark = cutlass.Int64(0)
    if cutlass.const_expr(telemetry):
        mark = read_clock64()
    cute.arch.fence_acq_rel_gpu()  # acquire side of the arrival handshake
    # (Copying the whole segment tables into shared memory here, to spare the range lookup
    # its own trip to L2, measured worse: 4-8 dependent scalar loads per thread cost 0.9-2.2 us
    # against the 0.34 us the lookup saved.  See docs/measured-worse.md.)
    if tidx < 256:  # merged histogram: sum over segments of prefix[b+1] - prefix[b]
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(splits):
            total = (
                total
                + _table_word(tables, r, 2 + tidx)
                - _table_word(tables, r, 1 + tidx)
            )
        s_merged[tidx] = total
    if tidx < 32:  # per-segment staged counts (lane r) and the overflow / total verdict
        staged = cutlass.Int32(0)
        if tidx < cutlass.Int32(splits):
            staged = _table_word(tables, tidx, 0)
        bad = cutlass.Int32(0)
        if staged < 0:
            bad = cutlass.Int32(1)
            staged = cutlass.Int32(0)
        s_scratch[tidx] = staged
        total = warp_inclusive_scan_add(staged, tidx)
        nbad = warp_inclusive_scan_add(bad, tidx)
        if tidx == 31:
            s_result[8] = total
            s_result[9] = nbad
    cute.arch.barrier()
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[12] = (read_clock64() - mark).to(cutlass.Int32)
        mark = read_clock64()
    ok = cutlass.Int32(1)
    if (s_result[9] != 0) | (s_result[8] < cutlass.Int32(k)):
        ok = cutlass.Int32(0)
    if ok == 1:
        if tidx < 32:
            b, above, _c, _b2, _a2, _c2 = crossing_256_warp(
                s_merged, cutlass.Int32(k), cutlass.Int32(k), tidx, False
            )
            # per-segment ranges: winners [prefix[b+1], staged), ties [prefix[b], prefix[b+1])
            win_src = cutlass.Int32(0)
            win_cnt = cutlass.Int32(0)
            tie_src = cutlass.Int32(0)
            tie_cnt = cutlass.Int32(0)
            if tidx < cutlass.Int32(splits):
                p_b = _table_word(tables, tidx, 1 + b)
                p_b1 = _table_word(tables, tidx, 2 + b)
                win_src = p_b1
                win_cnt = s_scratch[tidx] - p_b1
                tie_src = p_b
                tie_cnt = p_b1 - p_b
            win_incl = warp_inclusive_scan_add(win_cnt, tidx)
            tie_incl = warp_inclusive_scan_add(tie_cnt, tidx)
            s_scratch[32 + tidx] = (
                win_incl - win_cnt
            )  # output offset of segment r's winners
            s_scratch[64 + tidx] = tidx * cutlass.Int32(capacity) + win_src
            s_scratch[96 + tidx] = (
                tie_incl - tie_cnt
            )  # tie-stage offset of segment r's ties
            s_scratch[128 + tidx] = tidx * cutlass.Int32(capacity) + tie_src
            if tidx == 0:
                s_result[0] = b
                s_result[1] = above
            if tidx == 31:
                s_result[2] = win_incl
                s_result[3] = tie_incl
        cute.arch.barrier()
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[13] = (read_clock64() - mark).to(cutlass.Int32)
            mark = read_clock64()
        above = s_result[1]
        n_win = s_result[2]
        n_tie = s_result[3]
        for g in range(tidx, n_win, threads):  # one winner index per thread per step
            r = cutlass.Int32(0)
            for rr in cutlass.range_constexpr(1, splits):
                if g >= s_scratch[32 + rr]:
                    r = cutlass.Int32(rr)
            src = s_scratch[64 + r] + (g - s_scratch[32 + r])
            if g < cutlass.Int32(k):
                out_row[g] = load_global_l2_i32((slab_idx + src).toint())
        for g in range(tidx, n_tie, threads):
            r = cutlass.Int32(0)
            for rr in cutlass.range_constexpr(1, splits):
                if g >= s_scratch[96 + rr]:
                    r = cutlass.Int32(rr)
            src = s_scratch[128 + r] + (g - s_scratch[96 + r])
            if g < cutlass.Int32(tie_capacity):
                s_tie_keys[g] = elems.key(
                    cutlass.Uint32(load_global_l2_i32((slab_keys + src).toint()))
                )
                s_tie_idx[g] = load_global_l2_i32((slab_idx + src).toint())
        cute.arch.barrier()
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[14] = (read_clock64() - mark).to(cutlass.Int32)
            mark = read_clock64()
        ok = _select_ties(
            elems,
            k,
            above,
            n_tie,
            out_row,
            s_scratch + 256,
            s_tie_keys,
            s_tie_idx,
            tie_capacity,
            ballot_limit,
            s_slots,
            s_result,
            tidx,
            threads,
        )
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[15] = (read_clock64() - mark).to(cutlass.Int32)
    if tidx == 0:
        global_store_release(counter.toint(), 0)
    return ok
