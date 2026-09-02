"""Register-resident top-k for short rows (experimental, walk-first family).

For rows of at most VPT*E*1024 elements (VPT <= 4 vectors of E elements per
thread: 16K fp32, 32K fp16/bf16) the WHOLE row fits in registers of one
1024-thread CTA (16 x 32-bit words per thread).  The streaming pipeline's
fixed costs -- sample, three barriers, split coordination, walk loop,
candidate stage -- exist to avoid re-reading a long row; a short row does
not need any of them.  Two arms, both exact:

HINT ARM (caller hints present).  Hints are a first-class input: the bar is
the minimum exact key over the hinted elements (k gathers, one warp redux,
one barrier).  Every element is then classified against that exact key from
registers: strictly above -> output via a warp-aggregated ticket; equal ->
tie stage.  The counts verify the bar for free: with ``above <= k <=
above + ties`` the top-k is exactly the winners plus ``k - above`` of the
equal-keyed ties (identical values; index tie-break in the tie select), so
the row is done after one barrier and the tie select -- no census, no scan.
Any other outcome (stale/duplicate/invalid hints) falls through to the
census arm, which rewrites every output slot.  (Staging the winners with
their exact keys so a mildly stale bar could finish with one radix select
was built and MEASURED WORSE on B200: the slot->word select chain pushed the
kernel to 64 registers and the good-hint path went 4.6 -> 6.2us, while the
stale path improved only 8.9 -> 8.3us.  Emitting directly is kept.)

CENSUS ARM (hintless, or hint verification failed).
  1. exact coarse histogram (hist_size bins of the 16-bit key prefix)
     tallied from registers                                          -- B1
  2. block-wide crossing at rank k (find_threshold_coarse)           -- B2
  3. classify from registers: bins above the crossing bin -> output via
     a warp-aggregated ticket; the crossing bin -> tie stage        -- B3
  4. exact tie select (ballot rank <= 128, byte-radix <= TIE_CAP); a tie
     flood beyond TIE_CAP takes the fused exact fallback.

Measured on B200 (16K fp32 b=64 k=1024, in-kernel): the per-element
same-address winner ticket cost 2.4-3.3us before warp aggregation; the
row loads are issued before the seqlen load (row-stride vectors are always
in-bounds memory) so the two latencies overlap.  Scope: fp32 / fp16 / bf16,
next_n == compress_ratio == 1, indices only, k <= TIE_CAP, N % E == 0,
N <= regrow_max_n(dtype).  Routed from the walk-first dispatcher below that
width (FLASHINFER_TOPK_WF_REGROW=0 disables; fp32 by default, "all" adds
the 16-bit dtypes, whose 8192-bin census measured slower than the walk
pipeline on B200/H100).
"""

import os

import cutlass
import torch
import cutlass.cute as cute
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.utils.smem_allocator import SmemAllocator

from . import fallback_topk_primitives as _fallback_mod
from . import radix_topk_primitives as _radix_mod
from . import sampled_topk_primitives as _sampled_mod
from . import walkfirst_topk_primitives as _wf_mod
from .fallback_topk_primitives import GatedExactFallback
from .radix_topk_primitives import smem_atomic_add, warp_inclusive_sum
from .sampled_topk_primitives import GCAP
from .walkfirst_topk_primitives import (
    WF_ROW_INTS,
    WalkFirstTopK,
    ld_global_nc_v4_u32,
    warp_max_u32,
)

REGROW_VPT = 4  # vectors per thread held in registers (16 words)


def regrow_max_n(vec_elems: int, nt: int = 1024) -> int:
    return REGROW_VPT * vec_elems * nt  # 16384 fp32, 32768 16-bit


class RegRowTopK(WalkFirstTopK):
    """Register-resident short-row solver; reuses the walk-first / radix
    substrate helpers (keys, coarse bins, crossing find, tie selects)."""

    @cute.jit
    def _rg_slot_idx(self, tidx, bp, lg: cutlass.Constexpr):
        """Element index of register slot ``bp`` (slot = u*E + e) of thread tidx."""
        return (
            (tidx + (bp >> cutlass.Int32(lg)) * cutlass.Int32(self.nt))
            << cutlass.Int32(lg)
        ) + (bp & cutlass.Int32(self.vec_elems - 1))

    @cute.jit
    def _rg_emit_winners(
        self, win, s_count, out_idx_row, tidx, lane, lg: cutlass.Constexpr
    ):
        """Warp-aggregated tickets for this thread's winner mask, then the
        mask walk writes the indices.  ONE shared atomic per warp: the
        per-element same-address atomic serialized k times per row."""
        top_k = cutlass.const_expr(self.top_k)
        cnt = cutlass.Int32(cute.arch.popc(win))
        inc = warp_inclusive_sum(cnt, lane)
        bpos = cutlass.Int32(0)
        if lane == 31:
            if inc != 0:
                bpos = smem_atomic_add(s_count, inc)
        pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
        while win != 0:
            bp = cutlass.Int32(
                cute.arch.popc((win & (cutlass.Int32(0) - win)) - cutlass.Int32(1))
            )
            win = win & (win - cutlass.Int32(1))
            if pos < top_k:
                out_idx_row[pos] = self._rg_slot_idx(tidx, bp, lg)
            pos = pos + cutlass.Int32(1)

    @cute.jit
    def _rg_finish(
        self,
        above,
        nb,
        binb_lo_key,
        binb_hi_key,
        s_tk,
        s_ti,
        s_scratch,
        s_h256,
        s_warp_sums,
        s_misc,
        out_idx_row,
        tidx,
    ):
        """Exact tie select of ``k - above`` among ``nb`` staged ties.
        Caller guarantees 0 <= k - above <= nb <= TIE_CAP."""
        top_k = cutlass.const_expr(self.top_k)
        remaining = cutlass.Int32(top_k) - above
        if remaining > 0:
            if nb <= 128:
                self.tie_select(
                    s_tk,
                    s_ti,
                    nb,
                    above,
                    remaining,
                    s_scratch,
                    s_warp_sums,
                    s_misc,
                    out_idx_row,
                    out_idx_row,  # dummy: no values
                    tidx,
                )
            else:
                self._tie_select_smem_skip_wf(
                    s_tk,
                    s_ti,
                    nb,
                    above,
                    remaining,
                    binb_hi_key,
                    binb_lo_key,
                    s_h256,
                    s_warp_sums,
                    s_misc,
                    out_idx_row,
                    tidx,
                )

    @cute.kernel
    def regrow_topk_kernel(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        hints: cute.Tensor,
        has_hints: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        num_rows_dbg, _, _ = cute.arch.grid_dim()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])
        lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
        epw = cutlass.const_expr(self.vec_elems // 4)
        vpt = cutlass.const_expr(REGROW_VPT)
        nvec_row = cutlass.const_expr(n_cols >> lg)
        lane = tidx % 32
        warp = tidx // 32

        in_ptr = input_data.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator
        hint_ptr = hints.iterator

        smem = SmemAllocator()
        s_h4k = smem.allocate_array(cutlass.Int32, self.hist_size, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tk = smem.allocate_array(cutlass.Uint32, self.tie_cap, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, self.tie_cap, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, self.nw, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        # tie_select's smem scratch (>= 512 ints): the histogram is dead by then
        s_scratch = s_h4k

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * WF_ROW_INTS
        slab_i = slab_k + GCAP
        hint_row = hint_ptr + row64 * top_k

        # PDL (SM90+, compiled out otherwise): wait before the first global read
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()

        # ---- row -> registers, issued BEFORE the seqlen load ----
        # Row-stride vectors are always in-bounds memory (the row buffer is N
        # wide); only the count of VALID elements depends on the length, and
        # that is applied at classify time.  Overlaps the two gmem latencies.
        w = [None] * (4 * vpt)
        for u in cutlass.range_constexpr(vpt):
            vi = tidx + cutlass.Int32(u * self.nt)
            a0 = cutlass.Uint32(0)
            a1 = cutlass.Uint32(0)
            a2 = cutlass.Uint32(0)
            a3 = cutlass.Uint32(0)
            if cutlass.const_expr((u + 1) * self.nt <= nvec_row):
                a0, a1, a2, a3 = ld_global_nc_v4_u32(
                    row_in.toint() + cutlass.Int64(vi) * cutlass.Int64(16)
                )
            else:
                if vi < cutlass.Int32(nvec_row):
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(
                        row_in.toint() + cutlass.Int64(vi) * cutlass.Int64(16)
                    )
            w[4 * u + 0] = a0
            w[4 * u + 1] = a1
            w[4 * u + 2] = a2
            w[4 * u + 3] = a3

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        if top_k >= length:
            for i in range(tidx, top_k, self.nt):
                if i < length:
                    out_idx_row[i] = cutlass.Int32(i)
                else:
                    out_idx_row[i] = cutlass.Int32(-1)
            if tidx == 0:
                st_ptr[row] = cutlass.Int32(0)
                st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
        else:
            n4 = length >> cutlass.Int32(lg)  # full vectors in the valid row
            tail0 = n4 << cutlass.Int32(lg)
            # zero the census + cursors (visible after the first barrier)
            for zz in cutlass.range_constexpr(self.hist_size // self.nt):
                s_h4k[tidx + cutlass.Int32(zz * self.nt)] = cutlass.Int32(0)
            if tidx == 0:
                s_count[0] = cutlass.Int32(0)  # winner ticket
                s_count[4] = cutlass.Int32(0)  # tie cursor

            # ---- HINT ARM: bar = min exact key over the hinted elements ----
            # complement-max idiom: min(key) = ~max(~key); invalid hints
            # contribute ~0 = all-ones complement... i.e. key 0xFFFFFFFF, which
            # never lowers the minimum
            nkmax = cutlass.Uint32(0)
            if has_hints == 1:
                for i in range(tidx, top_k, self.nt):
                    hi = hint_row[i]
                    ok = cutlass.Int32(1)
                    if hi < 0:
                        ok = cutlass.Int32(0)
                    if hi >= length:
                        ok = cutlass.Int32(0)
                    hc = hi
                    if hc < 0:
                        hc = cutlass.Int32(0)
                    if hc > cutlass.Int32(n_cols - 1):
                        hc = cutlass.Int32(n_cols - 1)
                    hk = ~self.exact_key(self.load_scalar(row_in, hc))
                    if ok == 1:
                        if hk > nkmax:
                            nkmax = hk
                nkmax = warp_max_u32(nkmax)
                if lane == 0:
                    s_warp_sums[warp] = nkmax.bitcast(cutlass.Int32)
            cute.arch.barrier()  # B0: zeros (+ hint partials) visible
            solved = cutlass.Int32(0)
            if has_hints == 1:
                bar = ~warp_max_u32(cutlass.Uint32(s_warp_sums[lane]))  # min key
                if bar != cutlass.Uint32(0xFFFFFFFF):  # at least one usable hint
                    win = cutlass.Int32(0)
                    for u in cutlass.range_constexpr(vpt):
                        vi = tidx + cutlass.Int32(u * self.nt)
                        valid = cutlass.Int32(vi < n4)
                        for j in cutlass.range_constexpr(4):
                            for h in cutlass.range_constexpr(epw):
                                bits = self._wf_elem_bits(w[4 * u + j], h)
                                kx = self.exact_key(bits)
                                win = win | (
                                    (cutlass.Int32(kx > bar) & valid)
                                    << cutlass.Int32(u * self.vec_elems + j * epw + h)
                                )
                                if valid == 1:
                                    if kx == bar:
                                        t = smem_atomic_add(s_count + 4, 1)
                                        if t < self.tie_cap:
                                            s_tk[t] = kx
                                            s_ti[t] = (
                                                vi << cutlass.Int32(lg)
                                            ) + cutlass.Int32(j * epw + h)
                    self._rg_emit_winners(win, s_count, out_idx_row, tidx, lane, lg)
                    for i in range(tail0 + tidx, length, self.nt):
                        bits = self.load_scalar(row_in, i)
                        kx = self.exact_key(bits)
                        if kx > bar:
                            p = smem_atomic_add(s_count, 1)
                            if p < top_k:
                                out_idx_row[p] = cutlass.Int32(i)
                        else:
                            if kx == bar:
                                t = smem_atomic_add(s_count + 4, 1)
                                if t < self.tie_cap:
                                    s_tk[t] = kx
                                    s_ti[t] = cutlass.Int32(i)
                    cute.arch.barrier()  # H1: counts final
                    above = s_count[0]
                    nb = s_count[4]
                    ok = cutlass.Int32(1)
                    if above > cutlass.Int32(top_k):
                        ok = cutlass.Int32(0)  # bar too low: stale hints
                    if above + nb < cutlass.Int32(top_k):
                        ok = cutlass.Int32(0)  # bar too high: duplicate/invalid hints
                    if nb > self.tie_cap:
                        ok = cutlass.Int32(0)
                    if ok == 1:
                        self._rg_finish(
                            above,
                            nb,
                            cutlass.Uint32(0),
                            cutlass.Uint32(0xFFFFFFFF),
                            s_tk,
                            s_ti,
                            s_scratch,
                            s_h256,
                            s_warp_sums,
                            s_misc,
                            out_idx_row,
                            tidx,
                        )
                        solved = cutlass.Int32(1)
                    else:
                        # reset cursors for the census arm.  The barrier is
                        # REQUIRED: every thread read above/nb from s_count after
                        # H1, and thread 0 must not zero them until all reads are
                        # done (a late reader would branch on 0 -> divergent
                        # barrier counts).  The census's B1 orders the zeros
                        # before its own cursor atomics.
                        cute.arch.barrier()
                        if tidx == 0:
                            s_count[0] = cutlass.Int32(0)
                            s_count[4] = cutlass.Int32(0)

            ok = cutlass.Int32(1)
            if solved == 0:
                # ---- CENSUS ARM ----
                for u in cutlass.range_constexpr(vpt):
                    vi = tidx + cutlass.Int32(u * self.nt)
                    if vi < n4:
                        for j in cutlass.range_constexpr(4):
                            for h in cutlass.range_constexpr(epw):
                                smem_atomic_add(
                                    s_h4k
                                    + self.coarse_bin(
                                        self._wf_elem_bits(w[4 * u + j], h)
                                    ),
                                    1,
                                )
                for i in range(tail0 + tidx, length, self.nt):
                    smem_atomic_add(
                        s_h4k + self.coarse_bin(self.load_scalar(row_in, i)), 1
                    )
                cute.arch.barrier()  # B1: census complete (also orders the cursor reset)
                # crossing at rank k (ends with a block barrier)               B2
                self.find_threshold_coarse(
                    s_h4k, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
                )
                binb = s_misc[0]
                above = s_misc[1]
                # classify from registers.  fp32: exact FLOAT boundaries of the
                # crossing bin (the substrate's host-verified construction:
                # hi = strict-GT predecessor of bin binb+1's lower bound, lo =
                # bin binb's lower bound; NaN fails both ordered compares and
                # classifies gt, matching the census's +NaN-on-top bins) --
                # two compares per element instead of a quarter-rate fp16
                # conversion plus key ops.  16-bit dtypes keep the integer bins
                # (their coarse bin IS the key, no conversion to save).
                hi_f = cutlass.Float32(0.0)
                lo_f = cutlass.Float32(0.0)
                if cutlass.const_expr(self.is_f32):
                    hi_f = self.coarse_bin_gt_threshold_f32(binb + cutlass.Int32(1))
                    lo_f = self.coarse_bin_lower_bound_f32(binb)
                win = cutlass.Int32(0)
                for u in cutlass.range_constexpr(vpt):
                    vi = tidx + cutlass.Int32(u * self.nt)
                    valid = cutlass.Int32(vi < n4)
                    for j in cutlass.range_constexpr(4):
                        for h in cutlass.range_constexpr(epw):
                            bits = self._wf_elem_bits(w[4 * u + j], h)
                            slot = cutlass.Int32(u * self.vec_elems + j * epw + h)
                            if cutlass.const_expr(self.is_f32):
                                val = bits.bitcast(cutlass.Float32)
                                le_hi = cutlass.Int32(val <= hi_f)
                                win = win | (
                                    ((cutlass.Int32(1) - le_hi) & valid) << slot
                                )
                                if valid == 1:
                                    if le_hi == 1:
                                        if val >= lo_f:
                                            t = smem_atomic_add(s_count + 4, 1)
                                            if t < self.tie_cap:
                                                s_tk[t] = self.exact_key(bits)
                                                s_ti[t] = (
                                                    vi << cutlass.Int32(lg)
                                                ) + cutlass.Int32(j * epw + h)
                            else:
                                bq = self.coarse_bin(
                                    bits
                                )  # distinct name: the fp32 path never defines it
                                win = win | ((cutlass.Int32(bq > binb) & valid) << slot)
                                if valid == 1:
                                    if bq == binb:
                                        t = smem_atomic_add(s_count + 4, 1)
                                        if t < self.tie_cap:
                                            s_tk[t] = self.exact_key(bits)
                                            s_ti[t] = (
                                                vi << cutlass.Int32(lg)
                                            ) + cutlass.Int32(j * epw + h)
                self._rg_emit_winners(win, s_count, out_idx_row, tidx, lane, lg)
                for i in range(tail0 + tidx, length, self.nt):
                    bits = self.load_scalar(row_in, i)
                    b = self.coarse_bin(bits)
                    if b > binb:
                        p = smem_atomic_add(s_count, 1)
                        if p < top_k:
                            out_idx_row[p] = cutlass.Int32(i)
                    else:
                        if b == binb:
                            t = smem_atomic_add(s_count + 4, 1)
                            if t < self.tie_cap:
                                s_tk[t] = self.exact_key(bits)
                                s_ti[t] = cutlass.Int32(i)
                cute.arch.barrier()  # B3: winners written, ties staged
                nb = s_count[4]
                remaining = cutlass.Int32(top_k) - above
                if remaining < 0:
                    ok = cutlass.Int32(0)
                if remaining > nb:
                    ok = cutlass.Int32(0)
                if nb > self.tie_cap:
                    ok = cutlass.Int32(0)
                if ok == 1:
                    # conservative key bounds of the coarse bin (radix skip hints
                    # only: a wider range never changes the result)
                    k_lo = self._wf_coarse_lb_key(binb)
                    k_hi = cutlass.Uint32(0xFFFFFFFF)
                    if binb < cutlass.Int32(self.hist_size - 1):
                        k_hi = self._wf_coarse_lb_key(binb + 1) - cutlass.Uint32(1)
                    self._rg_finish(
                        above,
                        nb,
                        k_lo,
                        k_hi,
                        s_tk,
                        s_ti,
                        s_scratch,
                        s_h256,
                        s_warp_sums,
                        s_misc,
                        out_idx_row,
                        tidx,
                    )
                else:
                    # tie flood (> TIE_CAP in the crossing bin): exact MSD re-solve
                    self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                        row_in,
                        length,
                        out_idx_row,
                        slab_k,
                        slab_i,
                        s_h4k,
                        s_h256,
                        s_warp_sums,
                        s_misc,
                        s_count + 8,
                        s_count + 16,
                        tidx,
                    )
            if tidx == 0:
                st_ptr[row] = cutlass.Int32(1) - ok
                # family tag: 7 = register-resident census arm, 8 = hint arm
                st_ptr[num_rows_dbg + row] = cutlass.Int32(7) + solved

        # PDL: release the dependent grid at the very end (hides the next
        # launch's latency; measured a clean win in the walk-first kernel)
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()

    @cute.jit
    def launch_regrow(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        hints: cute.Tensor,
        has_hints: cutlass.Int32,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.regrow_topk_kernel(
            input_data, seqlen, output_indices, slab, status, hints, has_hints
        ).launch(
            grid=(num_rows, 1, 1),
            block=(self.nt, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
            min_blocks_per_mp=1,
        )


class ProdRegRowTopK(RegRowTopK, GatedExactFallback):
    """Register-resident short-row kernel + fused exact fallback."""


_compiled: dict = {}


def get_regrow_kernel(top_k: int, N: int, dtype=None):
    """Compile (with on-disk caching) the register-resident short-row kernel
    for a (top_k, N, dtype) specialization.  Call as
    kern(input, seqlen, out_indices, slab, status, hints, has_hints) with the
    walk-first slab (rows, WF_ROW_INTS) int32 (only the fused fallback
    touches it) and a (rows, top_k) int32 hints tensor (any contents when
    has_hints == 0)."""
    nt = 1024
    cdt = {
        None: cutlass.Float32,
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[dtype]
    vec_elems = 4 if cdt == cutlass.Float32 else 8
    dt_tag = {cutlass.Float32: "f32", cutlass.Float16: "f16", cutlass.BFloat16: "bf16"}[
        cdt
    ]
    assert N % vec_elems == 0 and regrow_max_n(vec_elems, nt) >= N
    assert top_k <= 2 * nt
    use_pdl = (
        torch.cuda.get_device_capability()[0] >= 9
        and os.environ.get("FLASHINFER_TOPK_WF_PDL") != "0"
    )
    key = (top_k, N, dt_tag, use_pdl)
    if key in _compiled:
        return _compiled[key]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    kern = ProdRegRowTopK(
        dtype=cdt,
        top_k=top_k,
        next_n=1,
        compress_ratio=1,
        return_values=False,
        ctas_per_group=1,
        chunk_elems=0,
        num_sms=148,
        min_blocks_per_mp=0,
        boundary_cls=True,
        approx_ties=True,
        enable_pdl=use_pdl,
        warp_agg=False,
        nt=nt,
    )
    kern.mc_splits = 1
    # inherited helpers read these; none of the walk machinery is compiled
    kern.wf_telemetry = False
    kern.wf_cluster = False
    kern.wf_force_retry = False
    kern.wf_hint_rung = False
    kern.wf_small_n = 0
    kern.lcap = 8192
    sym_rows = cute.sym_int()
    i32 = cutlass.Int32

    def _fk(dt, shape, align=None):
        so = tuple(range(len(shape) - 1, -1, -1))
        if align is None:
            return cute.runtime.make_fake_compact_tensor(dt, shape, stride_order=so)
        return cute.runtime.make_fake_compact_tensor(
            dt, shape, stride_order=so, assumed_align=align
        )

    def _compile_fn():
        return cute.compile(
            kern.launch_regrow,
            _fk(cdt, (sym_rows, N), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, top_k), 16),
            _fk(i32, (sym_rows, WF_ROW_INTS), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, top_k)),
            i32(0),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = build_and_load_cute_dsl_kernel(
        "regrow_topk_primitives",
        f"regrow_v2d_{dt_tag}_k{top_k}_N{N}{'_pdl' if use_pdl else ''}",
        _compile_fn,
        extra_key_files=(
            __file__,
            _wf_mod.__file__,
            _sampled_mod.__file__,
            _radix_mod.__file__,
            _fallback_mod.__file__,
        ),
    )
    _compiled[key] = compiled
    return compiled


def regrow_enabled(dtype) -> bool:
    """Hintless routing policy: fp32 by default; FLASHINFER_TOPK_WF_REGROW=0
    disables, "1"/"all" adds the 16-bit dtypes (their 8192-bin census
    measured a wash on H100 and slower than the walk pipeline on B200 at
    16K, faster on A100/L40S).  With usable caller hints the dispatcher
    routes the 16-bit dtypes here regardless: the hinted arm measured ~2x
    the streaming path at 16K on H100/A100/L40S/5080."""
    v = os.environ.get("FLASHINFER_TOPK_WF_REGROW")
    if v == "0":
        return False
    if dtype == torch.float32:
        return True
    return v in ("1", "all")
