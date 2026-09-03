"""Sample phase: from ``threads`` vectors of the row to a walk threshold, in three barriers.

1. Every thread loads one 16-byte vector at a row-uniform stride and folds its keys to a
   maximum and a minimum in registers; the block reduces both (barrier 1, which also
   publishes the zeroed histogram).
2. The sampled values are binned into 256 equal-width bins over [min, max] and counted
   (barrier 2).
3. Warp 0 finds the crossing bins of two ranks scaled from the aim and its floor and zeroes
   the histogram on the way out for the filter pass (barrier 3).

The threshold is the lower edge of the aim's crossing bin; the survivor scale maps
[bar, max * span_ext] onto 255 bins so the ~N/4096 elements above the sample maximum do not
all pile into bin 255 (one such pile-up turned a 53 us batch into 320 us before the extension).

Replica-deterministic: every CTA of a split row samples the same positions and gets the same
threshold, so split verdicts agree without communication.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_warp
from ...block.reduce import block_max_min_u32
from ...device.atomics import shared_count
from ...device.memory import load_global_readonly_16
from ...device.timers import read_clock64

__all__ = ["Threshold", "sample_threshold"]


class Threshold(NamedTuple):
    """Output of the sample phase.  ``degenerate`` is 1 when the sample cannot give a
    threshold (constant, NaN or infinite span); the caller then takes the exact fallback."""

    bar: object
    scale: object
    floor_bar: object
    floor_scale: object
    degenerate: object
    sample_max: object


@cute.jit
def _word(w0, w1, w2, w3, j: cutlass.Constexpr):
    return (w0, w1, w2, w3)[j]


@cute.jit
def _scaled_rank(aim, samples: cutlass.Constexpr, length):
    """Rank among the samples corresponding to ``aim`` survivors in the row, in [1, samples]."""
    r = ((cutlass.Int64(aim) * cutlass.Int64(samples)) // cutlass.Int64(length)).to(
        cutlass.Int32
    )
    if r < 1:
        r = cutlass.Int32(1)
    if r > cutlass.Int32(samples):
        r = cutlass.Int32(samples)
    return r


@cute.jit
def sample_threshold(
    elems,
    row_ptr,
    length,
    aim,
    floor_multiple: cutlass.Constexpr,
    span_ext: cutlass.Constexpr,
    s_hist,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    telemetry: cutlass.Constexpr = False,
):
    """Sample the row and return ``(bar, scale, floor_bar, floor_scale, degenerate, max)``.

    Contract: ``length >= per_vector``; ``s_hist`` (256 Int32, 16-byte aligned) is zero on
    exit; ``s_slots`` (2 * warps Uint32) and ``s_result`` (16 Int32) are scratch.  All threads
    call it.  Cost: one vector load per thread, three barriers, one warp crossing.  With
    ``telemetry`` the three sub-phase clocks (load+fold+barrier 1, bin+barrier 2,
    crossing+barrier 3) land in ``s_result[12..14]``.
    """
    mark = cutlass.Int64(0)
    if cutlass.const_expr(telemetry):
        mark = read_clock64()
    per_vector = cutlass.const_expr(elems.per_vector)
    samples = cutlass.const_expr(threads * per_vector)
    n_vectors = length >> cutlass.Int32(elems.log2_per_vector)
    pick = (
        (cutlass.Int64(tidx) * cutlass.Int64(n_vectors)) // cutlass.Int64(threads)
    ).to(cutlass.Int32)
    if pick > n_vectors - 1:
        pick = n_vectors - 1
    w0, w1, w2, w3 = load_global_readonly_16(row_ptr.toint() + cutlass.Int64(pick) * 16)

    kmax = elems.key(elems.bits(w0, 0))
    kmin = kmax
    for j in cutlass.range_constexpr(4):
        for h in cutlass.range_constexpr(elems.per_word):
            if cutlass.const_expr(j > 0 or h > 0):
                key = elems.key(elems.bits(_word(w0, w1, w2, w3, j), h))
                if key > kmax:
                    kmax = key
                if key < kmin:
                    kmin = key
    if tidx < 256:
        s_hist[tidx] = cutlass.Int32(0)
    kmax, kmin = block_max_min_u32(kmax, kmin, s_slots, tidx, threads)  # barrier 1
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[12] = (read_clock64() - mark).to(cutlass.Int32)
        mark = read_clock64()

    smin = elems.value_of_key(kmin)
    smax = elems.value_of_key(kmax)
    span = smax - smin
    degenerate = cutlass.Int32(1)
    if (span > cutlass.Float32(0.0)) & (
        span <= cutlass.Float32(3.0e38)
    ):  # finite, non-empty
        degenerate = cutlass.Int32(0)
    if degenerate == 0:
        to_bin = cutlass.Float32(256.0) / span
        hist_base = s_hist.toint()
        for j in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(elems.per_word):
                v = elems.value(elems.bits(_word(w0, w1, w2, w3, j), h))
                b = ((v - smin) * to_bin).to(cutlass.Int32)
                if b < 0:
                    b = cutlass.Int32(0)
                if b > 255:
                    b = cutlass.Int32(255)
                shared_count(hist_base + b * 4)
    cute.arch.barrier()  # barrier 2
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[13] = (read_clock64() - mark).to(cutlass.Int32)
        mark = read_clock64()

    rank_aim = _scaled_rank(aim, samples, length)
    rank_floor = rank_aim * cutlass.Int32(floor_multiple)
    if rank_floor > cutlass.Int32(samples):
        rank_floor = cutlass.Int32(samples)
    if tidx < 32:
        bin_aim, _a, _c, bin_floor, _a2, _c2 = crossing_256_warp(
            s_hist, rank_aim, rank_floor, tidx, True
        )
        if tidx == 0:
            s_result[0] = bin_aim
            s_result[1] = bin_floor
    cute.arch.barrier()  # barrier 3: bins published, histogram zeroed
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[14] = (read_clock64() - mark).to(cutlass.Int32)

    bin_width = span / cutlass.Float32(256.0)
    bar = smin + s_result[0].to(cutlass.Float32) * bin_width
    floor_bar = smin + s_result[1].to(cutlass.Float32) * bin_width
    scale = cutlass.Float32(0.0)
    span_above = (smax - bar) * cutlass.Float32(span_ext)
    if span_above > cutlass.Float32(0.0):
        scale = cutlass.Float32(255.0) / span_above
    floor_scale = cutlass.Float32(0.0)
    span_floor = (smax - floor_bar) * cutlass.Float32(span_ext)
    if span_floor > cutlass.Float32(0.0):
        floor_scale = cutlass.Float32(255.0) / span_floor
    return bar, scale, floor_bar, floor_scale, degenerate, smax
