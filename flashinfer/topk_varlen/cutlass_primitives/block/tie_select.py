"""Exact selection among staged candidates that share the threshold bin.

After a histogram names the rank-k bin, the elements above it are winners outright and the
elements in it are ties for the last ``remaining`` output slots.  The stage holds their ordered
keys and indices in shared memory.  Two methods, chosen by the caller from the tie count:

* ``tie_select_ballot`` (up to 128 ties): candidate c's rank is the number of candidates
  greater than it, where equal keys are ordered by ascending index so the order is total and
  the output deterministic.  One ballot + popcount per (warp, candidate) pair, no barrier.
* ``tie_select_radix`` (up to ``4 * threads`` ties): byte-by-byte radix select over the
  keys with a 256-bin histogram per round; candidates whose digit is above the crossing are
  winners, equal digits continue, the last round fills the remainder from exact-equal keys.
  Two barriers per round plus the crossing's.  Output order among exact-equal keys is
  unspecified.

Both write indices to ``out[out_base + rank]`` for ranks below ``remaining``.
"""

import cutlass
import cutlass.cute as cute

from ..device.atomics import shared_add
from .crossing import crossing_256_block

__all__ = ["tie_select_ballot", "tie_select_radix"]


@cute.jit
def _rank_against(ck, ci, tk, ti):
    """Number of lanes whose (key, index) beats (tk, ti): larger key, or equal key and
    smaller index."""
    greater = (ck > tk) | ((ck == tk) & (ci < ti))
    return cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(greater)))


@cute.jit
def tie_select_ballot(
    s_keys, s_idx, n, remaining, out, out_base, tidx, threads: cutlass.Constexpr
):
    """Emit the top ``remaining`` of ``n <= 128`` staged (key, index) pairs, deterministically.

    Warps split the candidates as targets; every lane holds up to four candidates (32, 64,
    96 apart) so that one ballot per held slot counts the beaters of a target.  No barrier,
    no shared writes; the stage is read-only.  Sentinels (key 0, index INT_MAX) lose every
    comparison, so inactive lanes never count.  Cost: ``128 / warps`` targets per warp, up to
    four ballots each.
    """
    warps = cutlass.const_expr(threads // 32)
    lane = tidx % 32
    warp = tidx // 32
    none_key = cutlass.Uint32(0)
    none_idx = cutlass.Int32(0x7FFFFFFF)
    k0 = none_key
    k1 = none_key
    k2 = none_key
    k3 = none_key
    i0 = none_idx
    i1 = none_idx
    i2 = none_idx
    i3 = none_idx
    if lane < n:
        k0 = cutlass.Uint32(s_keys[lane])
        i0 = cutlass.Int32(s_idx[lane])
    if lane + 32 < n:
        k1 = cutlass.Uint32(s_keys[lane + 32])
        i1 = cutlass.Int32(s_idx[lane + 32])
    if lane + 64 < n:
        k2 = cutlass.Uint32(s_keys[lane + 64])
        i2 = cutlass.Int32(s_idx[lane + 64])
    if lane + 96 < n:
        k3 = cutlass.Uint32(s_keys[lane + 96])
        i3 = cutlass.Int32(s_idx[lane + 96])
    for step in cutlass.range_constexpr(128 // warps):
        t = warp + step * warps
        if t < n:
            tk = cutlass.Uint32(s_keys[t])
            ti = cutlass.Int32(s_idx[t])
            rank = _rank_against(k0, i0, tk, ti)
            if n > 32:
                rank = (
                    rank
                    + _rank_against(k1, i1, tk, ti)
                    + _rank_against(k2, i2, tk, ti)
                    + _rank_against(k3, i3, tk, ti)
                )
            if lane == 0:
                if rank < remaining:
                    out[out_base + rank] = ti


@cute.jit
def _digit(key, shift: cutlass.Constexpr):
    return ((key >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)).to(cutlass.Int32)


@cute.jit
def _place(
    active,
    key,
    pos,
    shift: cutlass.Constexpr,
    bucket,
    s_counters,
    filled,
    last_round: cutlass.Constexpr,
):
    """One candidate's fate for this round: above the crossing digit it takes a winner slot;
    below it drops out; equal it continues, and in the last round takes an equal-key slot."""
    if active:
        d = _digit(key, shift)
        if d > bucket:
            pos = shared_add(s_counters, 1)
            active = False
        else:
            if d < bucket:
                active = False
            else:
                if cutlass.const_expr(last_round):
                    pos = filled + shared_add(s_counters + 1, 1)
    return active, pos


@cute.jit
def tie_select_radix(
    s_keys,
    s_idx,
    n,
    remaining,
    out,
    out_base,
    s_hist,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    shifts: cutlass.Constexpr,
):
    """Emit the top ``remaining`` of ``n <= 4 * threads`` staged (key, index) pairs.

    ``shifts``: the digit positions, most significant first ((24, 16, 8, 0) for 32-bit keys,
    (8, 0) for 16-bit).  ``s_hist``: Int32 shared array of 512 entries (two 256-bin rounds
    double-buffered; dead on exit).  ``s_slots``: 8 entries; ``s_result``: 5 entries (3 for
    the crossing, 2 counters).  Every thread holds up to four candidates in named registers
    (``threads`` apart).  Barriers: two per round plus the crossing's two.  Ranks among
    exact-equal keys at the boundary are assigned in atomic order; the set of chosen keys is
    exact.  All threads must call it.
    """
    rounds = cutlass.const_expr(len(shifts))
    s_counters = s_result + 3  # [3] winner slots handed out, [4] equal-key slots
    c0 = tidx
    c1 = tidx + threads
    c2 = tidx + 2 * threads
    c3 = tidx + 3 * threads
    a0 = c0 < n
    a1 = c1 < n
    a2 = c2 < n
    a3 = c3 < n
    k0 = cutlass.Uint32(0)
    k1 = cutlass.Uint32(0)
    k2 = cutlass.Uint32(0)
    k3 = cutlass.Uint32(0)
    i0 = cutlass.Int32(0)
    i1 = cutlass.Int32(0)
    i2 = cutlass.Int32(0)
    i3 = cutlass.Int32(0)
    if a0:
        k0 = cutlass.Uint32(s_keys[c0])
        i0 = cutlass.Int32(s_idx[c0])
    if a1:
        k1 = cutlass.Uint32(s_keys[c1])
        i1 = cutlass.Int32(s_idx[c1])
    if a2:
        k2 = cutlass.Uint32(s_keys[c2])
        i2 = cutlass.Int32(s_idx[c2])
    if a3:
        k3 = cutlass.Uint32(s_keys[c3])
        i3 = cutlass.Int32(s_idx[c3])
    p0 = cutlass.Int32(remaining)  # sentinel: not selected
    p1 = cutlass.Int32(remaining)
    p2 = cutlass.Int32(remaining)
    p3 = cutlass.Int32(remaining)

    if tidx < 256:
        s_hist[tidx] = cutlass.Int32(0)
    if tidx == 0:
        s_counters[0] = cutlass.Int32(0)
        s_counters[1] = cutlass.Int32(0)
    cute.arch.barrier()

    in_play = cutlass.Int32(n)  # candidates whose digits so far equal the crossing's
    remain = cutlass.Int32(remaining)
    for r in cutlass.range_constexpr(rounds):
        shift = cutlass.const_expr(shifts[r])
        this_hist = s_hist + cutlass.const_expr((r % 2) * 256)
        next_hist = s_hist + cutlass.const_expr(((r + 1) % 2) * 256)
        last = cutlass.const_expr(r + 1 == rounds)
        # ``remain`` is block-uniform (derived from shared results), so the guarded barriers
        # are safe; it only skips dead rounds after an early exact resolution
        if remain > 0:
            if a0:
                shared_add(this_hist + _digit(k0, shift), 1)
            if a1:
                shared_add(this_hist + _digit(k1, shift), 1)
            if a2:
                shared_add(this_hist + _digit(k2, shift), 1)
            if a3:
                shared_add(this_hist + _digit(k3, shift), 1)
            if cutlass.const_expr(not last):
                if tidx < 256:
                    next_hist[tidx] = cutlass.Int32(0)
        cute.arch.barrier()
        if remain > 0:
            crossing_256_block(this_hist, in_play, remain, s_slots, s_result, tidx)
            bucket = s_result[0]
            above = s_result[1]
            in_play = s_result[2]
            remain_next = remain - above
            filled = remaining - remain_next  # winners decided so far, all rounds
            a0, p0 = _place(a0, k0, p0, shift, bucket, s_counters, filled, last)
            a1, p1 = _place(a1, k1, p1, shift, bucket, s_counters, filled, last)
            a2, p2 = _place(a2, k2, p2, shift, bucket, s_counters, filled, last)
            a3, p3 = _place(a3, k3, p3, shift, bucket, s_counters, filled, last)
            remain = remain_next
        cute.arch.barrier()

    if p0 < remaining:
        out[out_base + p0] = i0
    if p1 < remaining:
        out[out_base + p1] = i1
    if p2 < remaining:
        out[out_base + p2] = i2
    if p3 < remaining:
        out[out_base + p3] = i3
