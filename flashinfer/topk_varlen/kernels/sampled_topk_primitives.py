"""Sampled-pivot top-k on the primitives substrate (experimental, v8).

A GVR2-style single-pass selector rebuilt on the CoarseHistTopKPrimitives
machinery, keeping GVR2's decisive ideas (sample-calibrated threshold
bracket -> ONE fused verify+harvest pass; a 16K-entry gmem candidate SLAB
so a sloppy one-round bracket suffices) while fixing its two measured
weaknesses with parts this codebase already has:

  * duplicate-value collapse: when the bracket degenerates to a single
    value (constant / quantized / two-valued rows -- the inputs where
    gvr_2 measured 108..1340us), the ties are value-equal, so the
    remaining slots are filled BY COUNT from the slab prefix: exact,
    still one pass.  No value-space search is ever attempted on values
    that cannot be split.
  * unbounded fallback: any verification miss (bracket too low/high/wide)
    sets a per-row status flag; the host reruns those rows through the
    standard exact 2-pass backend.  Worst case is therefore bounded at
    ~3 row passes, data-independent.

Algorithm per row (fp32, single CTA, k <= TIE_CAP):
  1. sample 8,192 row values (coalesced float4 probes), staged as ordered
     keys in smem; sample min/max via red.max
  2. bracket: 4096-bin linear-key histogram of the sample + block-scan
     dual crossing-find at ranks ks -+ m (m ~ 2.5*sqrt(ks)+2); round 0's
     fine bins accept in one round on spread data, and the adaptive
     256-bin recursion only fires on duplicate spikes, where it
     collapses to a single exact key (the provable single-value bracket).
  3. ONE streaming pass (subclass slab walker, double-buffered float4):
     v > T_hi -> emit via ticket cursor; T_lo <= v <= T_hi -> (key, idx)
     into the per-row gmem slab, count exact, stores capped at GCAP
  4. verify by counts and finish:
       gt > k                 -> status=1 (bracket too low)
       k - gt > eq            -> status=1 (bracket too high)
       k_hi == k_lo           -> fill k-gt slots by count (equal ties)
       eq <= GCAP             -> exact rank-select over the slab
       else                   -> status=1 (bracket too wide; rare)

Experimental scope: fp32, next_n == 1, compress_ratio == 1,
return_values=False, single-CTA rows (host should route b < ~num_sms or
very long rows elsewhere), k <= 2048, N % 4 == 0.  Not wired into the
public dispatcher; see get_sampled_kernel() and fi-wt/sm90_probe.

Iteration history (measured; do NOT retry the dead ends): v1 exact-rank
rounds with block barriers = ~9us overhead; v2 warp0-only BULK histograms
= 2x worse (one warp scanning the sample has no latency hiding -- the
warp0 discipline is for DECISIONS only); v3-v5 smem-stage variants forced
bracket precision the sample can't cheaply give (eq budget
TIE_CAP*SMP/N vs rank margin 2m).  v6's slab removes that tension.
Phase telemetry (v7) then located the remaining randn overhead: ~12us
in bracket recursion rounds (the 256 linear-key bins are
exponent-coarse, so a tight budget forces 3-4 rounds) and ~4.5us in
the gmem slab-select.  v8/v9 fix both: a 4096-bin round-0 histogram
with SHIFT binning (power-of-2 bins: the per-sample bin is one 32-bit
shift instead of a ~100-instruction Int64 divide, and bin edges are
exact and inclusive, so a width-1 bin provably collapses to the
single-value bracket) plus a one-walk dual crossing-find; and an smem
mirror of the first TIE_CAP harvest entries feeding the parent's
ballot/radix tie_select whenever eq fits.  The acceptance budget
prefers any bracket whose row-side eq fits the mirror.  Measured
(B200): randn 64K accepts in ONE round, bracket 12->4us, epilogue
4.5->3.7us (radix arm; ballot arm 0.35us on duplicate rows).  A
whole-row L2-prefetch overlap during the bracket was tried and is a
measured net loss (see prefetch_l2).
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from . import radix_topk_primitives as _radix_mod
from .radix_topk_primitives import (
    NUM_THREADS,
    NUM_WARPS,
    TIE_CAP,
    CoarseHistTopKPrimitivesKernel,
    ld_global_v4_u32,
    read_globaltimer,
    smem_atomic_add,
    smem_red_max_u32,
    warp_inclusive_sum,
    warp_sum,
)

SMP = 8192  # sample size (staged in smem; rank-find over it)
GCAP = 16384  # per-row gmem candidate-slab capacity (gvr_2's GCAP)


@dsl_user_op
def prefetch_l2(gmem_addr: cutlass.Int64, *, loc=None, ip=None) -> None:
    """Issue an L2 prefetch for the 128B sector at gmem_addr (fire and
    forget).  MEASURED NET LOSS here (do not re-wire without new data):
    priming the whole row into L2 during the bracket phase made every
    probe cell 1-1.5us SLOWER -- the double-buffered collect walker
    already hides HBM latency, so the extra prefetch instructions only
    compete for memory-issue slots.  The overlap trick pays off in
    gvr_2's walker, not in this one."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class SampledPivotTopK(CoarseHistTopKPrimitivesKernel):
    """See module docstring.  Reuses the parent's scan and key helpers;
    the verify+harvest walker and the slab select live here."""

    # Compile-time phase-telemetry switch (plain Python bool, read at
    # trace time): when False (production default) every
    # read_globaltimer() and the status-buffer phase writes (blocks 5-8)
    # are never traced.  Enabled per-specialization by
    # get_prod_kernel(telemetry=True) or FLASHINFER_TOPK_PRIM_TELEMETRY=1.
    sp_telemetry: bool = False
    # Trace-time switch: True (set by get_prod_kernel on the Prod*
    # classes, which inherit GatedExactFallback._fallback_row) fuses the
    # exact fallback INLINE into the kernel's failure path, replacing
    # the separate gated fallback launch (~1.65us of pure launch
    # latency).  False on the bare speculative classes.
    fuse_fallback: bool = False

    @cute.jit
    def _wide_dual_find(self, s_hist, need_hi, need_lo, s_warp_sums, s_misc, tidx):
        """Dual descending crossing search over the 4096-bin round-0
        histogram in ONE walk (find_threshold_wide's block-scan idiom,
        publishing both ranks: 2 block barriers instead of 4).  Slots
        match _w0_desc_find's layout so the consumer is branch-free:
        hi -> s_misc[2] = bin, [3] = above, [9] = cnt;
        lo -> s_misc[0] = bin, [1] = above, [8] = cnt."""
        items = cutlass.const_expr(4096 // NUM_THREADS)
        lane_id = tidx % 32
        warp_id = tidx // 32
        local = cutlass.Int32(0)
        for i in cutlass.range_constexpr(items):
            local = local + s_hist[tidx * items + i]
        incl = warp_inclusive_sum(local, lane_id)
        if lane_id == 31:
            s_warp_sums[warp_id] = incl
        cute.arch.barrier()
        w = cutlass.Int32(0)
        wt = cutlass.Int32(0)
        if lane_id < cutlass.Int32(self.nw):  # only nw warps wrote a slot
            wt = s_warp_sums[lane_id]
            if lane_id < warp_id:
                w = wt
        prev_warps = warp_sum(w)
        hsum = warp_sum(wt)
        nh = need_hi
        if nh > hsum:
            nh = hsum
        nl = need_lo
        if nl > hsum:
            nl = hsum
        prefix = prev_warps + (incl - local)
        for i in cutlass.range_constexpr(items):
            cnt = s_hist[tidx * items + i]
            prefix = prefix + cnt
            above = hsum - prefix
            if above < nh and above + cnt >= nh:
                s_misc[2] = tidx * items + i
                s_misc[3] = above
                s_misc[9] = cnt
            if above < nl and above + cnt >= nl:
                s_misc[0] = tidx * items + i
                s_misc[1] = above
                s_misc[8] = cnt
        cute.arch.barrier()

    @cute.jit
    def _w0_desc_find(self, s_h256, need, s_misc, lane):
        """Warp-0-only descending crossing search over 256 bins: writes
        (bucket, above, bucket_count) to s_misc[0], [1], [8] where
        ``above`` counts elements in bins strictly above the bucket and
        above < need <= above + bucket_count.  Lane l owns bins
        [8l, 8l+8); suffix sums via shuffles, then the crossing lane walks
        its 8 bins top-down.  No block barrier."""
        base = lane * 8
        s8 = cutlass.Int32(0)
        for j in cutlass.range_constexpr(8):
            s8 = s8 + s_h256[base + j]
        suf = cutlass.Int32(s8)  # inclusive suffix sum over lanes >= mine
        for off_ in cutlass.range_constexpr(5):
            off = cutlass.const_expr(1 << off_)
            v = cutlass.Int32(cute.arch.shuffle_sync_down(suf, off))
            if lane + off < 32:
                suf = suf + v
        above_l = suf - s8  # strictly-above-my-lane total
        if above_l < need:
            if above_l + s8 >= need:
                # crossing is in my 8 bins: walk them top-down
                c = cutlass.Int32(above_l)
                for j_ in cutlass.range_constexpr(8):
                    j = cutlass.const_expr(7 - j_)
                    cnt = s_h256[base + j]
                    if c < need:
                        if c + cnt >= need:
                            s_misc[0] = base + j
                            s_misc[1] = c
                            s_misc[8] = cnt
                    c = c + cnt
        cute.arch.sync_warp()

    @cute.jit
    def _slab_elem(
        self,
        w,
        idx,
        thf,
        tlf,
        s_count_gt,
        s_count_eq,
        out_idx_row,
        slab_k,
        slab_i,
        s_tk,
        s_ti,
    ):
        """Verify+harvest classification of one element (fp32 boundary
        semantics identical to the parent's collect arm: NaN fails the
        <= compare and classifies gt, torch parity).  The first TIE_CAP
        harvest entries are mirrored into smem so the epilogue can use
        the parent's ns-scale tie_select when eq fits (the common case);
        the gmem slab remains the complete record for overflow."""
        top_k = cutlass.const_expr(self.top_k)
        val = w.bitcast(cutlass.Float32)
        if val <= thf:
            if val >= tlf:
                c = smem_atomic_add(s_count_eq, 1)
                if c < GCAP:
                    kk = self.exact_key(w)
                    slab_k[c] = kk.bitcast(cutlass.Int32)
                    slab_i[c] = cutlass.Int32(idx)
                    if c < TIE_CAP:
                        s_tk[c] = kk
                        s_ti[c] = cutlass.Int32(idx)
        else:
            pos = smem_atomic_add(s_count_gt, 1)
            if pos < top_k:
                out_idx_row[pos] = cutlass.Int32(idx)

    @cute.jit
    def _slab_collect(
        self,
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
    ):
        """The one streaming pass: double-buffered float4 walk (the
        parent's A/B idiom); rows are 16B aligned (N % 4 == 0) so there is
        no prologue; a scalar tail covers length % 4."""
        thf = t_hi.bitcast(cutlass.Float32)
        tlf = t_lo.bitcast(cutlass.Float32)
        addr = row_in.toint()
        lv = length >> cutlass.Int32(2)
        stride16 = cutlass.Int64(NUM_THREADS * 16)
        w0 = cutlass.Uint32(0)
        w1 = cutlass.Uint32(0)
        w2 = cutlass.Uint32(0)
        w3 = cutlass.Uint32(0)
        m0 = cutlass.Uint32(0)
        m1 = cutlass.Uint32(0)
        m2 = cutlass.Uint32(0)
        m3 = cutlass.Uint32(0)
        v = cutlass.Int32(tidx)
        va = addr + cutlass.Int64(tidx) * 16
        if v < lv:
            w0, w1, w2, w3 = ld_global_v4_u32(va)
        while v < lv:
            v2 = v + NUM_THREADS
            if v2 < lv:
                m0, m1, m2, m3 = ld_global_v4_u32(va + stride16)
            base = v * 4
            for j in cutlass.range_constexpr(4):
                self._slab_elem(
                    (w0, w1, w2, w3)[j],
                    base + j,
                    thf,
                    tlf,
                    s_count_gt,
                    s_count_eq,
                    out_idx_row,
                    slab_k,
                    slab_i,
                    s_tk,
                    s_ti,
                )
            va = va + stride16 + stride16
            v = v2 + NUM_THREADS
            if v < lv:
                w0, w1, w2, w3 = ld_global_v4_u32(va)
            if v2 < lv:
                base2 = v2 * 4
                for j in cutlass.range_constexpr(4):
                    self._slab_elem(
                        (m0, m1, m2, m3)[j],
                        base2 + j,
                        thf,
                        tlf,
                        s_count_gt,
                        s_count_eq,
                        out_idx_row,
                        slab_k,
                        slab_i,
                        s_tk,
                        s_ti,
                    )
        tail_base = lv * 4
        for i in range(tidx, length - tail_base, NUM_THREADS):
            self._slab_elem(
                self.load_scalar(row_in, tail_base + i),
                tail_base + i,
                thf,
                tlf,
                s_count_gt,
                s_count_eq,
                out_idx_row,
                slab_k,
                slab_i,
                s_tk,
                s_ti,
            )

    @cute.jit
    def _slab_select(
        self,
        slab_k,
        slab_i,
        eq,
        gt,
        remaining,
        s_h256,
        s_warp_sums,
        s_misc,
        out_idx_row,
        tidx,
    ):
        """Exact selection of ``remaining`` winners among the eq slab
        candidates: 4-round byte-radix rank-find of the pivot key at
        descending rank ``remaining``, then one emit pass over the slab
        (strictly-above -> ticketed slots right after the gt block;
        pivot-equal fill the rest by count -- equal keys, any subset
        exact)."""
        prefix = cutlass.Uint32(0)
        pmask = cutlass.Uint32(0)
        need = cutlass.Int32(remaining)
        total = cutlass.Int32(eq)
        for r_ in cutlass.range_constexpr(4):
            sh = cutlass.const_expr(24 - 8 * r_)
            if tidx < 256:
                s_h256[tidx] = cutlass.Int32(0)
            cute.arch.barrier()
            for i in range(tidx, eq, self.nt):
                kk = cutlass.Uint32(slab_k[i])
                if (kk & pmask) == prefix:
                    smem_atomic_add(
                        s_h256
                        + cutlass.Int32(
                            (kk >> cutlass.Uint32(sh)) & cutlass.Uint32(0xFF)
                        ),
                        1,
                    )
            cute.arch.barrier()
            self.scan256_and_find(s_h256, total, need, s_warp_sums, s_misc, tidx)
            bucket = s_misc[0]
            above = s_misc[1]
            cnt = s_misc[2]
            cute.arch.barrier()
            prefix = prefix | (cutlass.Uint32(bucket) << cutlass.Uint32(sh))
            pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(sh))
            need = need - above
            total = cnt
        # winners strictly above the pivot take (remaining - need) slots
        # right after the gt block; pivot-equal candidates fill the last
        # ``need`` by count.
        if tidx == 0:
            s_misc[10] = cutlass.Int32(0)
            s_misc[11] = cutlass.Int32(0)
        cute.arch.barrier()
        nab = remaining - need
        pfx64 = cutlass.Int64(cutlass.Uint32(prefix))  # unambiguous compare
        for i in range(tidx, eq, self.nt):
            kk = cutlass.Uint32(slab_k[i])
            if cutlass.Int64(kk) > pfx64:
                p = smem_atomic_add(s_misc + 10, 1)
                out_idx_row[gt + p] = slab_i[i]
            else:
                if cutlass.Int64(kk) == pfx64:
                    e = smem_atomic_add(s_misc + 11, 1)
                    if e < need:
                        out_idx_row[gt + nab + e] = slab_i[i]

    @cute.kernel
    def sampled_topk_kernel(
        self,
        input_data: cute.Tensor,
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
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator

        smem = SmemAllocator()
        s_samp = smem.allocate_array(cutlass.Int32, SMP, byte_alignment=128)
        # 4096-bin round-1 histogram: 16x finer than 256 bins, so spread
        # data accepts the bracket in ONE round (each extra round costs
        # ~3.5us of block barriers -- measured).  Recursion rounds (rare;
        # duplicate spikes) reuse its first 256 bins.
        s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        # smem mirror of the first TIE_CAP harvest entries: when eq fits
        # (the common case) the epilogue is the parent's ns-scale
        # tie_select instead of gmem slab rounds (measured 4-5us).
        s_tk = smem.allocate_array(cutlass.Uint32, TIE_CAP, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count_gt = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count_eq = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * (2 * GCAP)
        slab_i = slab_k + GCAP

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        if top_k >= length:
            for i in range(tidx, top_k, NUM_THREADS):
                if i < length:
                    out_idx_row[i] = cutlass.Int32(i)
                else:
                    out_idx_row[i] = cutlass.Int32(-1)
            if tidx == 0:
                st_ptr[row] = cutlass.Int32(0)
        else:
            tel = cutlass.const_expr(self.sp_telemetry)
            # pre-defined: the DSL's staged-control-flow rewriter requires
            # names used inside an if region to exist beforehand, even
            # under a const-false condition (dead movs in production)
            ts0 = cutlass.Int64(0)
            ts1 = cutlass.Int64(0)
            ts2 = cutlass.Int64(0)
            ts3 = cutlass.Int64(0)
            ts4 = cutlass.Int64(0)
            if tel:
                ts0 = read_globaltimer()
            if tidx == 0:
                s_count_gt[0] = cutlass.Int32(0)
                s_count_eq[0] = cutlass.Int32(0)
                s_misc[6] = cutlass.Int32(0)  # red.max(key)
                s_misc[7] = cutlass.Int32(0)  # red.max(~key)
            cute.arch.barrier()

            # ---- 1. sample: coalesced float4 probes -> SMP ordered keys ----
            n4 = length >> cutlass.Int32(2)
            nprobe = cutlass.const_expr(SMP // 4)  # float4 probes total
            for pp in cutlass.range_constexpr(nprobe // NUM_THREADS):
                pid = tidx + cutlass.Int32(pp * NUM_THREADS)
                p4 = cutlass.Int32(
                    (cutlass.Int64(pid) * cutlass.Int64(n4)) // cutlass.Int64(nprobe)
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
                # sample key min/max (red.max on key and ~key), published by
                # the same barrier as the staging
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
            if tel:
                ts1 = read_globaltimer()

            # ---- 2. bracket: linear-key sample histogram + warp-0 dual
            # crossing-find; the slab budget makes round 1 accept almost
            # always, and the recursion only fires on duplicate spikes,
            # collapsing to a single exact key (the provable single-value
            # bracket the duplicate cure needs).
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
            lo_k = ~cutlass.Uint32(s_misc[7])  # sample min key
            hi_k = cutlass.Uint32(s_misc[6])  # sample max key
            k_hi = cutlass.Uint32(0)
            k_lo = cutlass.Uint32(0)
            have = cutlass.Int32(0)
            if lo_k == hi_k:  # degenerate sample (constant row)
                k_hi = lo_k
                k_lo = lo_k
                have = cutlass.Int32(1)
            # acceptance threshold: the row-side eq scales as
            # br_samples * N / SMP.  Two forces: never overflow the slab
            # (hard cap, 3/4 safety), and PREFER a tight bracket when one
            # more cheap histogram round buys it -- a fat accepted bracket
            # makes the slab-select epilogue walk 10K+ gmem entries
            # (measured +10us on randn), while a round costs ~1us.
            budget = cutlass.Int32(
                (cutlass.Int64(GCAP) * cutlass.Int64(3 * SMP // 4))
                // cutlass.Int64(length)
            )
            tight = cutlass.Int32(3) * m + cutlass.Int32(16)
            # a bracket whose row-side eq fits the smem tie mirror keeps
            # the epilogue on the ns-scale tie_select, so accepting up to
            # that is FREE -- prefer it over an extra ~2us histogram round.
            smem_ok = cutlass.Int32(
                (cutlass.Int64(TIE_CAP) * cutlass.Int64(SMP)) // cutlass.Int64(length)
            )
            if tight < smem_ok:
                tight = smem_ok
            if budget > tight:
                budget = tight
            rounds_run = cutlass.Int32(0)
            for _round in cutlass.range_constexpr(5):
                # round 0 uses 4096 bins (16x finer -> spread data accepts
                # in ONE round; each extra round costs ~3.5us of block
                # barriers -- measured); recursion rounds use 256 bins.
                nbins = cutlass.const_expr(4096 if _round == 0 else 256)
                s_hist = s_h4k if cutlass.const_expr(_round == 0) else s_h256
                if have == 0:  # block-uniform: barriers inside are safe
                    rounds_run = rounds_run + 1
                    if cutlass.const_expr(_round == 0):
                        for zz in cutlass.range_constexpr(4096 // NUM_THREADS):
                            s_hist[tidx + cutlass.Int32(zz * NUM_THREADS)] = (
                                cutlass.Int32(0)
                            )
                    else:
                        if tidx < 256:
                            s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()
                    # shift binning: bins are power-of-2 key ranges, so the
                    # per-sample bin is one 32-bit shift (the Int64
                    # multiply-divide alternative is a ~100-instruction
                    # subroutine, 8x per thread) and bin edges are EXACT
                    # (never exclude a member key, collapse to width 1).
                    # Uses (spn <= nbins) bins, > nbins/2 unless the range
                    # itself is small.
                    span = cutlass.Int64(hi_k - lo_k) + cutlass.Int64(1)
                    # bound the MAX BIN INDEX (span-1 >> shift), not the
                    # span: for span = 2^k + 1 the halving loop understates
                    # by one and the top key writes one bin past the
                    # histogram (smem corruption; bisected on two_values).
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
                        # 4096-bin dual crossing-find, one block-scan walk
                        # (2 barriers); publishes the same s_misc slots as
                        # the warp-0 path below
                        self._wide_dual_find(
                            s_hist, need_hi, need_lo, s_warp_sums, s_misc, tidx
                        )
                    else:
                        if tidx < 32:
                            # dual crossing-find (warp-only, no block barrier)
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
                    # bin b covers keys [lo_k + (b << shift),
                    # lo_k + ((b+1) << shift) - 1]: exact INCLUSIVE edges
                    # (shift binning), so a width-1 bin collapses to
                    # lo_k == hi_k (the single-value bracket the duplicate
                    # cure needs).  Computed in Int64 then truncated like
                    # the additions themselves (mod 2^32).
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
                        # recurse into the COMBINED range [b_lo .. b_hi];
                        # both ranks re-base by the samples above the new
                        # range (a_hi)
                        need_hi = need_hi - a_hi
                        need_lo = need_lo - a_hi
                        lo_k = elo_lo
                        hi_k = ehi_hi
                        if lo_k == hi_k:  # collapsed to one exact key
                            k_hi = lo_k
                            k_lo = lo_k
                            have = cutlass.Int32(1)
            if have == 0:
                k_hi = hi_k
                k_lo = lo_k
            t_hi = self.from_key32(k_hi)  # fp32 bit patterns of the bracket
            t_lo = self.from_key32(k_lo)
            if tel:
                ts2 = read_globaltimer()

            # ---- 3. the one pass: verify + harvest into the slab ----
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
            if tel:
                ts3 = read_globaltimer()

            # ---- 4. count-verify + finish (smem when eq fits, slab
            # otherwise) ----
            gt = s_count_gt[0]
            eq = s_count_eq[0]
            remaining = top_k - gt
            ok = cutlass.Int32(0)
            if remaining >= 0:
                if remaining <= eq:
                    if k_hi == k_lo:
                        # single-value bracket: ties are value-equal ->
                        # fill by count from the smem mirror (exact; the
                        # duplicate-value cure).  remaining <= k <=
                        # TIE_CAP, so the mirror always covers it.
                        ok = cutlass.Int32(1)
                        for t in range(tidx, remaining, NUM_THREADS):
                            out_idx_row[gt + t] = s_ti[t]
                    else:
                        if eq <= TIE_CAP:
                            # common case: the whole tie set is mirrored
                            # in smem -> the parent's ballot/radix select
                            # (the gmem slab-select costs 4-5us; measured)
                            ok = cutlass.Int32(1)
                            # s_h4k as scratch: the radix arm double-
                            # buffers 256-bin histograms (needs >= 512)
                            self.tie_select(
                                s_tk,
                                s_ti,
                                eq,
                                gt,
                                remaining,
                                s_h4k,
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                out_idx_row,  # dummy: has_values is False
                                tidx,
                            )
                        else:
                            if eq <= GCAP:
                                ok = cutlass.Int32(1)
                                self._slab_select(
                                    slab_k,
                                    slab_i,
                                    eq,
                                    gt,
                                    remaining,
                                    s_h256,
                                    s_warp_sums,
                                    s_misc,
                                    out_idx_row,
                                    tidx,
                                )
            if cutlass.const_expr(self.fuse_fallback):
                if ok == 0:
                    # inline exact fallback: ok derives from CTA-uniform
                    # smem reads, so the block barriers inside are safe;
                    # s_count slots 8+ are free (harvest uses 0)
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
            if tel:
                ts4 = read_globaltimer()
            if tidx == 0:
                st_ptr[row] = cutlass.Int32(1) - ok
                # debug telemetry (experimental kernel): rows are cheap
                st_ptr[num_rows_dbg + row] = gt
                st_ptr[num_rows_dbg * 2 + row] = eq
                st_ptr[num_rows_dbg * 3 + row] = k_hi.bitcast(cutlass.Int32)
                st_ptr[num_rows_dbg * 4 + row] = k_lo.bitcast(cutlass.Int32)
                if tel:
                    # phase times (ns, thread 0's view): sample, bracket,
                    # collect pass, epilogue
                    st_ptr[num_rows_dbg * 5 + row] = (ts1 - ts0).to(cutlass.Int32)
                    st_ptr[num_rows_dbg * 6 + row] = (ts2 - ts1).to(cutlass.Int32)
                    st_ptr[num_rows_dbg * 7 + row] = (ts3 - ts2).to(cutlass.Int32)
                    st_ptr[num_rows_dbg * 8 + row] = (ts4 - ts3).to(cutlass.Int32)
                st_ptr[num_rows_dbg * 9 + row] = rounds_run

    @cute.jit
    def launch_sampled(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.sampled_topk_kernel(
            input_data, seqlen, output_indices, slab, status
        ).launch(
            grid=(num_rows, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


_compiled: dict = {}


def get_sampled_kernel(top_k: int, N: int):
    """Compile (with on-disk caching) the sampled-pivot kernel for a
    (top_k, N) specialization.  fp32 / single-CTA / k <= TIE_CAP only.
    The caller supplies a per-row slab tensor of shape (rows, 2*GCAP)
    int32 (keys in [0, GCAP), indices in [GCAP, 2*GCAP))."""
    assert top_k <= TIE_CAP, "sampled-pivot requires k <= TIE_CAP"
    assert N % 4 == 0, "sampled-pivot requires N % 4 == 0 (float4 probes)"
    if (top_k, N) in _compiled:
        return _compiled[(top_k, N)]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    kern = SampledPivotTopK(
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
            kern.launch_sampled,
            cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (sym_rows, N), stride_order=(1, 0), assumed_align=16
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
        "sampled_topk_primitives",
        f"sampled_v8_f32_k{top_k}_N{N}",
        _compile_fn,
        extra_key_files=(__file__, _radix_mod.__file__),
    )
    _compiled[(top_k, N)] = compiled
    return compiled
