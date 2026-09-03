"""Aim policies: how many survivors the sample should aim for.

The sample knows only ``samples`` values out of ``length``.  Aiming the threshold at exactly
k survivors would undershoot on half the rows (each undershoot is a full second pass), so the
aim adds a margin.  Two policies with one signature, selected at Python level:

* ``aim_tight``: ``k + max(k / 2, length / 256)``, plus a statistical floor for wide grids.
  The survivor count above a sample-derived bar is roughly normal with variance
  ``aim * length / samples``; with 32 or more rows in flight the batch waits for its slowest
  row, so the margin is raised to 3.5 sigma wherever the fixed one is below 2.5 sigma
  (k=2048 at 256K to 512K rows).  Measured: removed a bimodal 35/51 us tail at 256K,
  b=148, k=2048 without touching the cells that did not need it.
* ``aim_wide``: ``max(2k, length / 128)``, gvr_2's choice.  Fewer undershoots, more
  candidates to stage and classify.  Kept to measure against.

Both cap the aim at three quarters of the stage so a typical overshoot still fits.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath

__all__ = ["aim_tight", "aim_wide"]


@cute.jit
def _capped(aim, cap: cutlass.Constexpr):
    if aim > cutlass.Int32(cap):
        aim = cutlass.Int32(cap)
    return aim


@cute.jit
def aim_tight(
    k: cutlass.Constexpr,
    length,
    rows,
    samples: cutlass.Constexpr,
    cap: cutlass.Constexpr,
):
    """Target survivor count for a row of ``length`` in a grid of ``rows`` rows."""
    margin = cutlass.Int32(k >> 1)
    by_length = length >> cutlass.Int32(8)
    if by_length > margin:
        margin = by_length
    aim = cutlass.Int32(k) + margin
    if rows >= cutlass.Int32(32):
        per_sample = length.to(cutlass.Float32) / cutlass.Float32(samples)
        margin_f = (aim - cutlass.Int32(k)).to(cutlass.Float32)
        variance = aim.to(cutlass.Float32) * per_sample
        if margin_f * margin_f < cutlass.Float32(6.25) * variance:  # below 2.5 sigma
            zq = cutlass.Float32(3.5) * cmath.sqrt(per_sample)
            root = (
                zq + cmath.sqrt(zq * zq + cutlass.Float32(4.0 * k))
            ) * cutlass.Float32(0.5)
            floor = (root * root).to(cutlass.Int32) + cutlass.Int32(1)
            if floor > aim:
                aim = floor
    return _capped(aim, cap)


@cute.jit
def aim_wide(
    k: cutlass.Constexpr,
    length,
    rows,
    samples: cutlass.Constexpr,
    cap: cutlass.Constexpr,
):
    """Target survivor count: twice k, or length / 128 if larger."""
    aim = cutlass.Int32(2 * k)
    by_length = length >> cutlass.Int32(7)
    if by_length > aim:
        aim = by_length
    return _capped(aim, cap)
