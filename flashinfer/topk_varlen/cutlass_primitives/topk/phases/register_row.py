"""Register-resident row: load once, histogram from registers, classify from registers.

For rows that fit the block's registers the whole row is read from memory exactly once.  Each
thread keeps ``words`` 32-bit words (4, 8 or 16: one to four 16-byte vectors) in registers
and every later step reads them from there.  This is gvr_2's register family, the design that
wins below 16K rows: no sample, no aim, no second pass over the row.

Layout: thread t owns vectors ``t + j * threads`` for j in 0..V-1, so consecutive threads
load consecutive 16-byte vectors and the loads coalesce.  Element ``e`` of a thread (fp32:
e = word; 16-bit: e = 2 * word + half) has row index
``(t + (e // (4 * per_word)) * threads) * per_vector + e % (4 * per_word)``.  Vectors past the
row's last one are clamped to it (a harmless duplicate load) and their elements masked by
index, so the load is unconditional and the words stay plain named values.

The per-element code is issue-bound (all of it runs on every element of the row), so its
instruction count is the whole story.  Three measured rules:

* Bins are computed for two elements at a time.  fp32 pairs go through one ``cvt.rn.f16x2``
  and the twiddle runs on both halves in 32-bit arithmetic (five instructions for two keys).
  A scalar fp32 -> fp16 conversion takes a quarter-rate slot per element.
* No ``if`` per element: sixteen traced ``if`` regions in a row serialize the independent
  per-element work.  Validity is arithmetic (out-of-range elements count into a trash bin).
* Bins are kept, packed two per register, from the count to the classify, so the classify is
  an unpack and two compares per element.  The block runs one CTA per SM (64 registers), which
  measured better than two CTAs per SM at 32 registers anyway (8.7 -> 7.3 us at 16K b=64).
* Output positions come from one block scan of per-thread counts, not a shared cursor (1024
  same-address returning atomics measured about 2 us per row), and winners are emitted by a
  bit-walk over the mask (about one iteration per thread).

The coarse bin is the census bin (top 12 bits of the fp16 ordered key, 64 bins per octave).
"""

import cutlass
import cutlass.cute as cute

from ...block.reduce import block_exclusive_scan_i32, block_max_min_u32
from ...device.atomics import shared_count
from ...device.cluster import peer_add_i32, peer_store_i32
from ...device.memory import load_global_readonly_16

from .census import element_bits, pair_key16

__all__ = [
    "load_row_words",
    "count_coarse_bins",
    "classify_from_registers",
    "classify_from_registers_cluster",
    "key_range_in_bin",
]

# The bin count is a compile-time parameter (1024 .. 4096): bin = key16 >> (16 - log2 bins),
# so fewer bins are coarser (fewer per octave) but cost less to zero and to scan.  Expected
# ties in the crossing bin scale with N / bins, so the dispatcher keeps N / bins about 4.
# Out-of-range elements get bin -1: they count into the slot just below the histogram (the
# caller allocates it) and can neither exceed nor equal any crossing bin, so the classify needs
# no extra test.  Packed bins are therefore unpacked with a sign-extending shift.


@cute.jit
def _clamped_vector(row_ptr, v, last_vector):
    vc = v
    if vc > last_vector:
        vc = last_vector
    return load_global_readonly_16(row_ptr.toint() + cutlass.Int64(vc) * 16)


@cute.jit
def load_row_words(
    row_ptr,
    length,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
    log2_per_vector: cutlass.Constexpr,
):
    """The thread's ``words // 4`` vectors of the row as ``words`` Uint32 values.

    ``length >= 1``.  Sixteen-byte read-only loads; vectors past the last one re-load it.
    """
    last_vector = (length - 1) >> cutlass.Int32(log2_per_vector)
    out: list = []
    for j in cutlass.range_constexpr(words // 4):
        w0, w1, w2, w3 = _clamped_vector(row_ptr, tidx + j * threads, last_vector)
        out.extend((w0, w1, w2, w3))
    return tuple(out)


@cute.jit
def _element_index(
    tidx,
    threads: cutlass.Constexpr,
    per_vector: cutlass.Constexpr,
    e: cutlass.Constexpr,
):
    """Row index of the thread's element ``e`` (compile-time e)."""
    return (tidx + cutlass.Int32((e // per_vector) * threads)) * cutlass.Int32(
        per_vector
    ) + cutlass.Int32(e % per_vector)


@cute.jit
def _index_of_bit(tidx, threads: cutlass.Constexpr, per_vector: cutlass.Constexpr, e):
    """Row index of the thread's element ``e`` (dynamic e)."""
    j = e // cutlass.Int32(per_vector)
    return (tidx + j * cutlass.Int32(threads)) * cutlass.Int32(per_vector) + (
        e - j * cutlass.Int32(per_vector)
    )


@cute.jit
def _pair_bins(
    elems,
    wordvals,
    p: cutlass.Constexpr,
    length,
    tidx,
    threads: cutlass.Constexpr,
    bins: cutlass.Constexpr,
):
    """Coarse bins of elements 2p and 2p+1 packed (low half = element 2p), trash bin past the end."""
    shift = cutlass.const_expr(16 - {1024: 10, 2048: 11, 4096: 12}[int(bins)])
    half_mask = cutlass.const_expr((int(bins) - 1) | ((int(bins) - 1) << 16))
    packed = (pair_key16(elems, wordvals, p) >> cutlass.Uint32(shift)) & cutlass.Uint32(
        half_mask
    )
    lo = (packed & cutlass.Uint32(0xFFFF)).to(cutlass.Int32)
    hi = (packed >> cutlass.Uint32(16)).to(cutlass.Int32)
    valid_lo = cutlass.Int32(
        _element_index(tidx, threads, elems.per_vector, 2 * p) < length
    )
    valid_hi = cutlass.Int32(
        _element_index(tidx, threads, elems.per_vector, 2 * p + 1) < length
    )
    lo = (lo + 1) * valid_lo - 1  # -1 past the row's end
    hi = (hi + 1) * valid_hi - 1
    return lo, hi


@cute.jit
def count_coarse_bins(
    elems,
    wordvals,
    length,
    s_bins,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
    bins: cutlass.Constexpr,
):
    """Add every element to its coarse bin in ``s_bins`` (``bins`` Int32 zeroed by the caller,
    with one writable slot just below index 0 for out-of-range elements) and return the bins
    packed two per Int32 (element 2p in the low half).  No barrier, no branches."""
    base = s_bins.toint()
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    packed = []
    for p in cutlass.range_constexpr(pairs):
        lo, hi = _pair_bins(elems, wordvals, p, length, tidx, threads, bins)
        shared_count(base + lo * 4)
        shared_count(base + hi * 4)
        packed.append((lo & cutlass.Int32(0xFFFF)) | (hi << 16))
    return tuple(packed)


_element_bits = element_bits


@cute.jit
def key_range_in_bin(
    elems,
    wordvals,
    packed_bins,
    cut_bin,
    s_slots_u32,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
):
    """(kmin, kmax): the exact key range of this CTA's elements in ``cut_bin``, from registers.

    For the overflow path: the radix select over the bin needs only its key range.  One block
    reduction; ``s_slots_u32``: 2 x warps Uint32.  A CTA with no member returns (0xFFFFFFFF, 0),
    the neutral pair for a further min/max merge.
    """
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    kmax = cutlass.Uint32(0)
    kmin = cutlass.Uint32(0xFFFFFFFF)
    for p in cutlass.range_constexpr(pairs):
        lo = (packed_bins[p] << 16) >> 16
        hi = packed_bins[p] >> 16
        if lo == cut_bin:
            key = elems.key(element_bits(elems, wordvals, 2 * p))
            if key > kmax:
                kmax = key
            if key < kmin:
                kmin = key
        if hi == cut_bin:
            key = elems.key(element_bits(elems, wordvals, 2 * p + 1))
            if key > kmax:
                kmax = key
            if key < kmin:
                kmin = key
    kmax, kmin = block_max_min_u32(kmax, kmin, s_slots_u32, tidx, threads)
    return kmin, kmax


@cute.jit
def _masks(packed_bins, cut_bin, pairs: cutlass.Constexpr):
    """Winner and tie bit masks over the thread's elements from the packed bins."""
    win_mask = cutlass.Int32(0)
    tie_mask = cutlass.Int32(0)
    for p in cutlass.range_constexpr(pairs):
        lo = (
            packed_bins[p] << 16
        ) >> 16  # sign-extending: -1 marks an out-of-range element
        hi = packed_bins[p] >> 16
        win_mask = (
            win_mask
            | (cutlass.Int32(lo > cut_bin) << (2 * p))
            | (cutlass.Int32(hi > cut_bin) << (2 * p + 1))
        )
        tie_mask = (
            tie_mask
            | (cutlass.Int32(lo == cut_bin) << (2 * p))
            | (cutlass.Int32(hi == cut_bin) << (2 * p + 1))
        )
    return win_mask, tie_mask


@cute.jit
def _emit_winners(
    elems, win_mask, win_pos, index_offset, out_row, tidx, threads: cutlass.Constexpr
):
    """Write the row indices of the mask's elements at ``out_row[win_pos ...]`` (bit-walk,
    about one iteration per thread).  ``index_offset`` is the slice's start in the row."""
    while win_mask != 0:
        e = cutlass.Int32(
            cute.arch.popc(
                (win_mask & (cutlass.Int32(0) - win_mask)) - cutlass.Int32(1)
            )
        )
        win_mask = win_mask & (win_mask - cutlass.Int32(1))
        out_row[win_pos] = index_offset + _index_of_bit(
            tidx, threads, elems.per_vector, e
        )
        win_pos = win_pos + 1


@cute.jit
def classify_from_registers(
    elems,
    wordvals,
    packed_bins,
    cut_bin,
    out_row,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    s_slots,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
    bins: cutlass.Constexpr,
):
    """Write every element above ``cut_bin`` to ``out_row`` and stage every element in
    ``cut_bin`` (exact key, index); return ``(winners, ties)`` totals to every thread.

    ``packed_bins`` is what ``count_coarse_bins`` returned.  Positions come from one block
    scan of per-thread (winner, tie) counts packed in one Int32 (16 bits each; at most 32
    elements per thread).  ``s_slots``: warps Int32 scratch.  Two barriers, inside the scan;
    the caller barriers before reading the tie stage.
    """
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    win_mask, tie_mask = _masks(packed_bins, cut_bin, pairs)
    packed = (cutlass.Int32(cute.arch.popc(win_mask)) << 16) | cutlass.Int32(
        cute.arch.popc(tie_mask)
    )
    before, total = block_exclusive_scan_i32(packed, s_slots, tidx, threads)
    _emit_winners(
        elems, win_mask, before >> 16, cutlass.Int32(0), out_row, tidx, threads
    )
    tie_pos = before & cutlass.Int32(0xFFFF)
    if tie_mask != 0:  # rare; the exact key needs the word, so test each position
        for e in cutlass.range_constexpr(2 * pairs):
            if ((tie_mask >> e) & 1) == 1:
                if tie_pos < cutlass.Int32(tie_capacity):
                    s_tie_keys[tie_pos] = elems.key(_element_bits(elems, wordvals, e))
                    s_tie_idx[tie_pos] = _element_index(
                        tidx, threads, elems.per_vector, e
                    )
                tie_pos = tie_pos + 1
    return total >> 16, total & cutlass.Int32(0xFFFF)


@cute.jit
def classify_from_registers_cluster(
    elems,
    wordvals,
    packed_bins,
    cut_bin,
    slice_start,
    out_row,
    tie_keys_root,
    tie_idx_root,
    cursors_root,
    tie_capacity: cutlass.Constexpr,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
):
    """Cluster form: this CTA's winners go to ``out_row`` and its ties to rank 0's tie stage,
    at positions reserved with one remote atomic per CTA on rank 0's cursors.

    ``tie_keys_root``, ``tie_idx_root``, ``cursors_root``: mapped DSMEM addresses of rank 0's
    tie arrays and of its two Int32 cursors (winners, ties), zeroed before the cluster barrier
    that preceded this call.  ``s_result``: 2 Int32 scratch.  Three block barriers.  Row
    indices are ``slice_start`` plus the position in the slice.  Returns nothing; rank 0 reads
    the totals from its cursors after the next cluster barrier.
    """
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    win_mask, tie_mask = _masks(packed_bins, cut_bin, pairs)
    packed = (cutlass.Int32(cute.arch.popc(win_mask)) << 16) | cutlass.Int32(
        cute.arch.popc(tie_mask)
    )
    before, total = block_exclusive_scan_i32(packed, s_slots, tidx, threads)
    if tidx == 0:  # one reservation per CTA on rank 0's cursors
        s_result[0] = peer_add_i32(cursors_root, total >> 16)
        s_result[1] = peer_add_i32(cursors_root + 4, total & cutlass.Int32(0xFFFF))
    cute.arch.barrier()
    _emit_winners(
        elems,
        win_mask,
        s_result[0] + (before >> 16),
        slice_start,
        out_row,
        tidx,
        threads,
    )
    tie_pos = s_result[1] + (before & cutlass.Int32(0xFFFF))
    if tie_mask != 0:
        for e in cutlass.range_constexpr(2 * pairs):
            if ((tie_mask >> e) & 1) == 1:
                if tie_pos < cutlass.Int32(tie_capacity):
                    peer_store_i32(
                        tie_keys_root + tie_pos * 4,
                        elems.key(_element_bits(elems, wordvals, e)).bitcast(
                            cutlass.Int32
                        ),
                    )
                    peer_store_i32(
                        tie_idx_root + tie_pos * 4,
                        slice_start
                        + _element_index(tidx, threads, elems.per_vector, e),
                    )
                tie_pos = tie_pos + 1
