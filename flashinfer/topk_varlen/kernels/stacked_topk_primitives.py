"""Stacked hint->sample top-k on the primitives substrate (experimental).

The three-rung selector: per row, IN ONE KERNEL LAUNCH,
  rung 1 (hint):   min-of-hints pivot T -> ONE boundary-collect [T, T]
                   pass -> count-verify.  Good hints (steady decode)
                   finish here at hint-kernel speed.
  rung 2 (sample): if rung 1's verify fails, the SAME CTA falls through
                   to the sampled-pivot path: 8K-sample gather ->
                   shift-binned bracket -> ONE verify+harvest pass into
                   the smem mirror / gmem slab -> tie select.
  rung 3 (host):   any remaining miss sets status=1 and the caller
                   reruns that row through the standard exact 2-pass
                   histogram backend.

Sequential rungs were chosen over fusing the hint into the sample
bracket after analysis: the hint's win (12.8us vs 26.5us on B200 randn)
comes from skipping the sample+bracket phases AND from the tiny [T, T]
tie set feeding the ballot-arm tie fill -- a fused bracket-tightening
captures neither (its eq stays ~m*N/SMP, still on the radix arm).  The
miss penalty is one wasted row pass, bounded and data-independent.

Rows with no usable hints (all pre_idx < 0) degrade automatically: the
pivot accumulator stays at the max key, verify fails, rung 2 runs.
Callers without hints at all should launch the sampled kernel directly
and skip the wasted pass.

Status tensor telemetry (blocks of num_rows): [0] = status (0 ok,
1 = host fallback), [1] = winning rung (0 degenerate short row, 1 hint,
2 sample), [2] = gt, [3] = eq of the winning rung.

Experimental scope: fp32, next_n == 1, compress_ratio == 1,
return_values=False, single-CTA rows, k <= TIE_CAP, N % 4 == 0.  Not
wired into the public dispatcher; see get_stacked_kernel() and
fi-wt/sm90_probe.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.utils.smem_allocator import SmemAllocator

from . import radix_topk_primitives as _radix_mod
from .radix_topk_primitives import (
    NUM_THREADS,
    NUM_WARPS,
    TIE_CAP,
    _OP_COLLECT,
    ld_global_v4_u32,
    smem_atomic_add,
    smem_red_max_u32,
)
from . import sampled_topk_primitives as _sampled_mod
from .sampled_topk_primitives import GCAP, SMP, SampledPivotTopK


class StackedTopK(SampledPivotTopK):
    """See module docstring.  Rung 1 reuses the parent's boundary-collect
    walker; rung 2 is the sampled-pivot body inherited from
    SampledPivotTopK."""

    @cute.kernel
    def stacked_topk_kernel(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        num_rows_dbg, _, _ = cute.arch.grid_dim()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])

        in_ptr = input_data.iterator
        pi_ptr = pre_idx.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator

        smem = SmemAllocator()
        s_samp = smem.allocate_array(cutlass.Int32, SMP, byte_alignment=128)
        s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        # rung 1 stages its [T, T] ties here; rung 2 reuses the same
        # arrays as the harvest mirror (phases are strictly sequential)
        s_tk = smem.allocate_array(cutlass.Uint32, TIE_CAP, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count_gt = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count_eq = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_dummy = smem.allocate_array(cutlass.Int32, 4, byte_alignment=128)

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        pi_row = pi_ptr + row64 * top_k
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * (2 * GCAP)
        slab_i = slab_k + GCAP

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        u0 = cutlass.Uint32(0)
        i0 = cutlass.Int32(0)

        if top_k >= length:
            for i in range(tidx, top_k, NUM_THREADS):
                if i < length:
                    out_idx_row[i] = cutlass.Int32(i)
                else:
                    out_idx_row[i] = cutlass.Int32(-1)
            if tidx == 0:
                st_ptr[row] = cutlass.Int32(0)
                st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
        else:
            if tidx == 0:
                s_count_gt[0] = cutlass.Int32(0)
                s_count_eq[0] = cutlass.Int32(0)
                s_misc[10] = cutlass.Int32(0)  # red.max(~key) accumulator
            cute.arch.barrier()

            # ================= rung 1: hint pivot =================
            for i in range(tidx, top_k, NUM_THREADS):
                hidx = pi_row[i]
                if hidx >= 0:
                    if hidx < length:
                        b = self.load_scalar(row_in, hidx)
                        smem_red_max_u32(s_misc + 10, ~self.exact_key(b))
            cute.arch.barrier()
            kmin = ~cutlass.Uint32(s_misc[10])
            tbits = self.from_key32(kmin)

            self._stream_row(
                _OP_COLLECT,
                0,
                row_in,
                length,
                i0,
                tidx,
                s_dummy,  # s_hist: untraced in approx boundary collect
                s_count_gt,
                s_count_eq,
                s_tk,
                s_ti,
                s_misc,
                out_idx_row,
                oi_ptr,  # out_val_row dummy (has_values=False)
                i0,  # threshold_bin: unused by boundary_cls arm
                tbits,  # hi boundary = T
                tbits,  # lo boundary = T  ->  tie iff v == T
                u0,
                i0,
                s_misc,  # g_state dummy (MC arms untraced)
            )
            cute.arch.barrier()

            gt = s_count_gt[0]
            eq = s_count_eq[0]
            remaining = top_k - gt
            hint_ok = cutlass.Int32(0)
            if remaining >= 0:
                if remaining <= eq:
                    hint_ok = cutlass.Int32(1)
            if hint_ok == 1:
                # ties are value-equal (single pivot): fill by count
                for t in range(tidx, remaining, NUM_THREADS):
                    out_idx_row[gt + t] = s_ti[t]
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
                    st_ptr[num_rows_dbg + row] = cutlass.Int32(1)
                    st_ptr[num_rows_dbg * 2 + row] = gt
                    st_ptr[num_rows_dbg * 3 + row] = eq
            else:
                # ============= rung 2: sampled pivot =============
                # (block-uniform fall-through; barriers inside are safe)
                if tidx == 0:
                    s_count_gt[0] = cutlass.Int32(0)
                    s_count_eq[0] = cutlass.Int32(0)
                    s_misc[6] = cutlass.Int32(0)  # red.max(key)
                    s_misc[7] = cutlass.Int32(0)  # red.max(~key)
                cute.arch.barrier()

                # ---- sample: coalesced float4 probes -> SMP keys ----
                n4 = length >> cutlass.Int32(2)
                nprobe = cutlass.const_expr(SMP // 4)
                for pp in cutlass.range_constexpr(nprobe // NUM_THREADS):
                    pid = tidx + cutlass.Int32(pp * NUM_THREADS)
                    p4 = cutlass.Int32(
                        (cutlass.Int64(pid) * cutlass.Int64(n4))
                        // cutlass.Int64(nprobe)
                    )
                    if p4 > n4 - 1:
                        p4 = n4 - 1
                    if p4 < 0:
                        p4 = cutlass.Int32(0)
                    w0, w1, w2, w3 = ld_global_v4_u32(
                        row_in.toint() + cutlass.Int64(p4) * cutlass.Int64(16)
                    )
                    k0 = self.exact_key(w0)
                    k1 = self.exact_key(w1)
                    k2 = self.exact_key(w2)
                    k3 = self.exact_key(w3)
                    s_samp[pid * 4 + 0] = k0.bitcast(cutlass.Int32)
                    s_samp[pid * 4 + 1] = k1.bitcast(cutlass.Int32)
                    s_samp[pid * 4 + 2] = k2.bitcast(cutlass.Int32)
                    s_samp[pid * 4 + 3] = k3.bitcast(cutlass.Int32)
                    kmx = k0
                    if k1 > kmx:
                        kmx = k1
                    if k2 > kmx:
                        kmx = k2
                    if k3 > kmx:
                        kmx = k3
                    kmn = k0
                    if k1 < kmn:
                        kmn = k1
                    if k2 < kmn:
                        kmn = k2
                    if k3 < kmn:
                        kmn = k3
                    smem_red_max_u32(s_misc + 6, kmx)
                    smem_red_max_u32(s_misc + 7, ~kmn)
                cute.arch.barrier()

                # ---- bracket: shift-binned sample histogram ----
                ks_i = cutlass.Int32(
                    (cutlass.Int64(top_k) * cutlass.Int64(SMP)) // cutlass.Int64(length)
                )
                if ks_i < 1:
                    ks_i = cutlass.Int32(1)
                m = cutlass.Int32(
                    cmath.sqrt(cutlass.Float32(ks_i) + cutlass.Float32(0.5))
                    * cutlass.Float32(2.5)
                ) + cutlass.Int32(2)
                need_hi = ks_i - m
                if need_hi < 1:
                    need_hi = cutlass.Int32(1)
                need_lo = ks_i + m
                if need_lo > SMP:
                    need_lo = cutlass.Int32(SMP)
                lo_k = ~cutlass.Uint32(s_misc[7])
                hi_k = cutlass.Uint32(s_misc[6])
                k_hi = cutlass.Uint32(0)
                k_lo = cutlass.Uint32(0)
                have = cutlass.Int32(0)
                if lo_k == hi_k:
                    k_hi = lo_k
                    k_lo = lo_k
                    have = cutlass.Int32(1)
                budget = cutlass.Int32(
                    (cutlass.Int64(GCAP) * cutlass.Int64(3 * SMP // 4))
                    // cutlass.Int64(length)
                )
                tight = cutlass.Int32(3) * m + cutlass.Int32(16)
                smem_ok = cutlass.Int32(
                    (cutlass.Int64(TIE_CAP) * cutlass.Int64(SMP))
                    // cutlass.Int64(length)
                )
                if tight < smem_ok:
                    tight = smem_ok
                if budget > tight:
                    budget = tight
                for _round in cutlass.range_constexpr(5):
                    nbins = cutlass.const_expr(4096 if _round == 0 else 256)
                    s_hist = s_h4k if cutlass.const_expr(_round == 0) else s_h256
                    if have == 0:
                        if cutlass.const_expr(_round == 0):
                            for zz in cutlass.range_constexpr(4096 // NUM_THREADS):
                                s_hist[tidx + cutlass.Int32(zz * NUM_THREADS)] = (
                                    cutlass.Int32(0)
                                )
                        else:
                            if tidx < 256:
                                s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()
                        span = cutlass.Int64(hi_k - lo_k) + cutlass.Int64(1)
                        shift = cutlass.Uint32(0)
                        spn = span - cutlass.Int64(1)  # max key offset
                        while spn > cutlass.Int64(nbins - 1):
                            spn = spn >> 1
                            shift = shift + 1
                        for i in range(tidx, SMP, NUM_THREADS):
                            kk = cutlass.Uint32(s_samp[i])
                            if kk >= lo_k:
                                if kk <= hi_k:
                                    bb = cutlass.Int32((kk - lo_k) >> shift)
                                    smem_atomic_add(s_hist + bb, 1)
                        cute.arch.barrier()
                        if cutlass.const_expr(_round == 0):
                            self._wide_dual_find(
                                s_hist, need_hi, need_lo, s_warp_sums, s_misc, tidx
                            )
                        else:
                            if tidx < 32:
                                self._w0_desc_find(s_hist, need_hi, s_misc, tidx)
                                if tidx == 0:
                                    s_misc[2] = s_misc[0]
                                    s_misc[3] = s_misc[1]
                                    s_misc[9] = s_misc[8]
                                cute.arch.sync_warp()
                                self._w0_desc_find(s_hist, need_lo, s_misc, tidx)
                            cute.arch.barrier()
                        b_hi = s_misc[2]
                        a_hi = s_misc[3]
                        b_lo = s_misc[0]
                        a_lo = s_misc[1]
                        c_lo = s_misc[8]
                        ehi_hi = lo_k + cutlass.Uint32(
                            ((cutlass.Int64(b_hi) + 1) << cutlass.Int64(shift))
                            - cutlass.Int64(1)
                        )
                        elo_lo = lo_k + cutlass.Uint32(
                            cutlass.Int64(b_lo) << cutlass.Int64(shift)
                        )
                        br_samples = (a_lo + c_lo) - a_hi
                        if br_samples <= budget:
                            k_hi = ehi_hi
                            k_lo = elo_lo
                            have = cutlass.Int32(1)
                        else:
                            need_hi = need_hi - a_hi
                            need_lo = need_lo - a_hi
                            lo_k = elo_lo
                            hi_k = ehi_hi
                            if lo_k == hi_k:
                                k_hi = lo_k
                                k_lo = lo_k
                                have = cutlass.Int32(1)
                if have == 0:
                    k_hi = hi_k
                    k_lo = lo_k
                t_hi = self.from_key32(k_hi)
                t_lo = self.from_key32(k_lo)

                # ---- one pass: verify + harvest ----
                self._slab_collect(
                    row_in,
                    length,
                    tidx,
                    t_hi,
                    t_lo,
                    s_count_gt,
                    s_count_eq,
                    out_idx_row,
                    slab_k,
                    slab_i,
                    s_tk,
                    s_ti,
                )
                cute.arch.barrier()

                gt2 = s_count_gt[0]
                eq2 = s_count_eq[0]
                remaining2 = top_k - gt2
                ok = cutlass.Int32(0)
                if remaining2 >= 0:
                    if remaining2 <= eq2:
                        if k_hi == k_lo:
                            ok = cutlass.Int32(1)
                            for t in range(tidx, remaining2, NUM_THREADS):
                                out_idx_row[gt2 + t] = s_ti[t]
                        else:
                            if eq2 <= TIE_CAP:
                                ok = cutlass.Int32(1)
                                self.tie_select(
                                    s_tk,
                                    s_ti,
                                    eq2,
                                    gt2,
                                    remaining2,
                                    s_h4k,  # radix arm double-buffers 256x2
                                    s_warp_sums,
                                    s_misc,
                                    out_idx_row,
                                    out_idx_row,  # dummy: has_values False
                                    tidx,
                                )
                            else:
                                if eq2 <= GCAP:
                                    ok = cutlass.Int32(1)
                                    self._slab_select(
                                        slab_k,
                                        slab_i,
                                        eq2,
                                        gt2,
                                        remaining2,
                                        s_h256,
                                        s_warp_sums,
                                        s_misc,
                                        out_idx_row,
                                        tidx,
                                    )
                if cutlass.const_expr(self.fuse_fallback):
                    if ok == 0:
                        # inline exact fallback (see SampledPivotTopK
                        # .fuse_fallback): ok is CTA-uniform here
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
                            s_count_gt + 8,
                            s_count_eq + 8,
                            tidx,
                        )
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(1) - ok
                    st_ptr[num_rows_dbg + row] = cutlass.Int32(2)
                    st_ptr[num_rows_dbg * 2 + row] = gt2
                    st_ptr[num_rows_dbg * 3 + row] = eq2

    @cute.jit
    def launch_stacked(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.stacked_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status
        ).launch(
            grid=(num_rows, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


_compiled: dict = {}


def get_stacked_kernel(top_k: int, N: int):
    """Compile (with on-disk caching) the stacked hint->sample kernel for
    a (top_k, N) specialization.  fp32 / single-CTA / k <= TIE_CAP only.
    Call as kern(input, pre_idx, seqlen, out_indices, slab, status);
    slab is (rows, 2*GCAP) int32, and rows with status != 0 must be
    rerun through an exact backend by the caller."""
    assert top_k <= TIE_CAP, "stacked requires k <= TIE_CAP"
    assert N % 4 == 0, "stacked requires N % 4 == 0 (float4 probes)"
    if (top_k, N) in _compiled:
        return _compiled[(top_k, N)]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    kern = StackedTopK(
        dtype=cutlass.Float32,
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
        enable_pdl=False,
        warp_agg=False,
    )
    sym_rows = cute.sym_int()

    def _compile_fn():
        return cute.compile(
            kern.launch_stacked,
            cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (sym_rows, N), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (sym_rows, 2 * GCAP),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows,), stride_order=(0,)
            ),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = build_and_load_cute_dsl_kernel(
        "stacked_topk_primitives",
        f"stacked_v1_f32_k{top_k}_N{N}",
        _compile_fn,
        extra_key_files=(__file__, _sampled_mod.__file__, _radix_mod.__file__),
    )
    _compiled[(top_k, N)] = compiled
    return compiled
