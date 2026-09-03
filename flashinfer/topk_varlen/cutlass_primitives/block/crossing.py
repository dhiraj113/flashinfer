"""Rank crossings over a histogram: which bin holds the element of a given rank from the top.

Bins ascend with value.  For a target rank ``t`` (1 = the largest element) the crossing bin
``b`` is the one where the count in bins strictly above ``b`` is below ``t`` and including
``b`` reaches it:  ``above < t <= above + count[b]``.  Every phase of a selection kernel asks
this question: the sample asks it to aim a threshold, the verdict asks it at rank k over the
survivor histogram, the radix fallback asks it once per digit.  If ``t`` exceeds the total
the answer is bin 0 with ``above`` = everything above bin 0 (the caller sees ``above +
count < t`` and knows the histogram ran out).

Three shapes for three sizes of question:

* ``crossing_256_warp``: one warp, 256 bins, two targets, no barrier.  Lane l owns bins
  8l..8l+7 (two 16-byte loads), a warp scan gives each lane the count above its span, and
  the owning lane walks its 8 bins.  Optionally zeroes the histogram on the way out so the
  next phase's clear is free.  Cheaper than a 256-thread scan for this size (measured:
  0.73 us behind one barrier against 1.3 us for 32 warps each redundantly scanning).
* ``crossing_256_block``: 256 threads, one bin each, one named barrier among themselves and
  one full barrier; the form used inside the radix rounds where 256 threads already hold
  the bins.
* ``crossing_wide_pair``: all threads over ``bins`` bins (4096 or 8192), two targets, two
  barriers; for the coarse histograms of the register-resident and radix kernels.

Warp forms return their answer on every lane of the warp.  Block forms publish to a shared
result array and end with a block barrier so every thread can read it.
"""

import cutlass
import cutlass.cute as cute

from ..device.memory import clear_shared_16, load_shared_16
from ..device.warp import (
    warp_broadcast,
    warp_inclusive_scan_add,
    warp_max_u32,
    warp_sum,
)

__all__ = ["crossing_256_warp", "crossing_256_block", "crossing_wide_pair"]


@cute.jit
def _bin_word(v0, v1, v2, v3, v4, v5, v6, v7, j: cutlass.Constexpr):
    return (v0, v1, v2, v3, v4, v5, v6, v7)[j]


@cute.jit
def crossing_256_warp(
    s_hist,
    target_a,
    target_b,
    lane,
    clear: cutlass.Constexpr,
    copies: cutlass.Constexpr = 1,
):
    """Crossing bins of two target ranks over a 256-bin Int32 histogram, by one warp.

    Contract: exactly one warp calls this with all 32 lanes; ``s_hist`` is 16-byte aligned
    and complete (a barrier separates the last increment from this call).  With ``copies``
    above one the histogram is the sum of ``copies`` consecutive 256-bin arrays (a privatized
    histogram; the lanes sum them while loading).  With ``clear`` the histogram (every copy) is
    zero on return (the caller's next barrier publishes the zeros).
    Returns ``(bin_a, above_a, count_a, bin_b, above_b, count_b)`` on every lane.
    Cost: two 16-byte shared loads per copy, one warp scan, 8 steps of compares, three shuffles.
    """
    base_bin = lane * 8
    addr = s_hist.toint() + lane * 32
    v0, v1, v2, v3 = load_shared_16(addr)
    v4, v5, v6, v7 = load_shared_16(addr + 16)
    for c in cutlass.range_constexpr(1, copies):
        u0, u1, u2, u3 = load_shared_16(addr + c * 1024)
        u4, u5, u6, u7 = load_shared_16(addr + c * 1024 + 16)
        v0, v1, v2, v3 = v0 + u0, v1 + u1, v2 + u2, v3 + u3
        v4, v5, v6, v7 = v4 + u4, v5 + u5, v6 + u6, v7 + u7
    mine = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7).to(cutlass.Int32)
    incl = warp_inclusive_scan_add(mine, lane)
    total = warp_broadcast(incl, 31)
    above = total - incl  # count in bins above this lane's span
    hit_a = cutlass.Int32(0)
    hit_b = cutlass.Int32(0)
    above_a = cutlass.Int32(0)
    above_b = cutlass.Int32(0)
    count_a = cutlass.Int32(0)
    count_b = cutlass.Int32(0)
    for j in cutlass.range_constexpr(7, -1, -1):
        c = _bin_word(v0, v1, v2, v3, v4, v5, v6, v7, j).to(cutlass.Int32)
        b = base_bin + j
        if above < target_a:
            if (above + c >= target_a) | (b == 0):
                hit_a = b
                above_a = above
                count_a = c
        if above < target_b:
            if (above + c >= target_b) | (b == 0):
                hit_b = b
                above_b = above
                count_b = c
        above = above + c
    if cutlass.const_expr(clear):
        for c in cutlass.range_constexpr(copies):
            clear_shared_16(addr + c * 1024)
            clear_shared_16(addr + c * 1024 + 16)
    # exactly one lane hit each target (or lane 0 by the bin-0 rule); the bin doubles as the
    # lane address for the broadcast
    bin_a = warp_max_u32(cutlass.Uint32(hit_a)).to(cutlass.Int32)
    bin_b = warp_max_u32(cutlass.Uint32(hit_b)).to(cutlass.Int32)
    return (
        bin_a,
        warp_broadcast(above_a, bin_a >> 3),
        warp_broadcast(count_a, bin_a >> 3),
        bin_b,
        warp_broadcast(above_b, bin_b >> 3),
        warp_broadcast(count_b, bin_b >> 3),
    )


@cute.jit
def crossing_256_block(s_hist, total, target, s_slots, s_result, tidx):
    """Crossing bin of one target rank over a 256-bin Int32 histogram, by threads 0..255.

    ``total`` is the histogram sum (known to the caller from the count that built it).
    ``s_slots``: Int32 shared array of 8 entries.  Publishes ``s_result[0..2] = (bin, above,
    count)`` and ends with a full block barrier; requires at least 256 threads and that every
    thread of the block calls it.  Cost: one warp scan, one named barrier, one full barrier.
    """
    lane = tidx % 32
    warp = tidx // 32
    if tidx < 256:
        c = s_hist[tidx]
        incl = warp_inclusive_scan_add(c, lane)
        if lane == 31:
            s_slots[warp] = incl
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        lower = cutlass.Int32(0)
        if lane < 8:
            if lane < warp:
                lower = s_slots[lane]
        incl = incl + warp_sum(lower)  # count in bins <= mine
        above = total - incl
        if above < target:
            if (above + c >= target) | (tidx == 0):
                s_result[0] = tidx
                s_result[1] = above
                s_result[2] = c
    cute.arch.barrier()


@cute.jit
def crossing_wide_pair(
    s_hist,
    bins: cutlass.Constexpr,
    target_a,
    target_b,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Crossing bins of two target ranks over a ``bins``-bin Int32 histogram, by the whole block.

    Thread t owns bins ``t * items .. (t + 1) * items`` with ``items = bins // threads``.
    ``s_slots``: Int32 shared array of ``warps`` entries.  Targets above the total are clamped
    to it.  Publishes ``s_result[0..2]`` for ``target_a`` and ``s_result[3..5]`` for
    ``target_b`` as (bin, above, count); ends with a block barrier.  Two barriers, one
    16-byte shared load per four bins (values kept for the second pass).
    """
    items = cutlass.const_expr(bins // threads)
    warps = cutlass.const_expr(threads // 32)
    lane = tidx % 32
    warp = tidx // 32
    # the thread's bins in one 16-byte load per four (a scalar load per bin is a four-way
    # bank conflict at this stride), kept in registers for the second pass
    vals: list = []
    if cutlass.const_expr(items % 4 == 0):
        for q in cutlass.range_constexpr(items // 4):
            v0, v1, v2, v3 = load_shared_16(s_hist.toint() + (tidx * items + 4 * q) * 4)
            vals.extend(
                (
                    v0.to(cutlass.Int32),
                    v1.to(cutlass.Int32),
                    v2.to(cutlass.Int32),
                    v3.to(cutlass.Int32),
                )
            )
    else:
        for i in cutlass.range_constexpr(items):
            vals.append(s_hist[tidx * items + i])
    mine = cutlass.Int32(0)
    for i in cutlass.range_constexpr(items):
        mine = mine + vals[i]
    incl = warp_inclusive_scan_add(mine, lane)
    if lane == 31:
        s_slots[warp] = incl
    cute.arch.barrier()
    slot = cutlass.Int32(0)
    lower = cutlass.Int32(0)
    if lane < cutlass.Int32(warps):
        slot = s_slots[lane]
        if lane < warp:
            lower = slot
    total = warp_sum(slot)
    below = warp_sum(lower) + (incl - mine)  # count in bins below this thread's span
    ta = target_a
    if ta > total:
        ta = total
    tb = target_b
    if tb > total:
        tb = total
    for i in cutlass.range_constexpr(items):
        c = vals[i]
        below = below + c
        above = total - below
        if above < ta:
            if above + c >= ta:
                s_result[0] = tidx * items + i
                s_result[1] = above
                s_result[2] = c
        if above < tb:
            if above + c >= tb:
                s_result[3] = tidx * items + i
                s_result[4] = above
                s_result[5] = c
    cute.arch.barrier()
