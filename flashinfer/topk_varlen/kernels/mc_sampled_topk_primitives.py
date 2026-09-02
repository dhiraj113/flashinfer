"""Multi-CTA (row-split) speculative top-k kernels (experimental) -- the
row-splitting the B200 comparison identified as gvr_2's remaining
structural advantage, rebuilt on the primitives substrate for BOTH
speculative pivot sources:

  * McSampledTopK -- self-sampling bracket (hintless), and
  * McHintTopK   -- min-of-hints [T, T] pivot (the hint rung), whose
    pivot costs one k-element gather, so row-splitting pays off hardest.

Each row is solved by S CTAs (grid = (rows, S)):

  1. every CTA of the row REDUNDANTLY derives the pivot/bracket (the
     sample probe positions and the hint gather are deterministic, so all
     replicas agree; replica reads hit L2).  No pre-collect inter-CTA
     synchronization exists at all.
  2. staged split harvest: CTA s walks its 16B-aligned slice, staging
     sure winners and bracket members in SMEM via CTA-scope atomics, then
     makes <= 3 device-scope atomics TOTAL to reserve output tickets and
     slab slots.  (The per-hit gmem-ticket variant serialized ~3K
     same-address L2 atomics per row and measured 3-4x slower -- do not
     revert.)  Entries a slice cannot stage (local > TIE_CAP) leave
     slots < count, which the epilogue detects and fails to the exact
     fallback rather than risking a hole.
  3. last-arriver epilogue: each CTA release-adds an arrival counter; the
     last one (acquire) count-verifies and finishes from the slab
     (fill-by-count for single-key brackets / smem-gathered tie_select /
     slab_select), writes status + telemetry, and self-resets the row's
     counters (CUDA-graph replay safe).  No CTA ever waits: deadlock is
     impossible by construction.

Split policy lives host-side (topk_varlen._mc_splits_for): SINGLE WAVE
ONLY -- rows * S must not exceed the SM count, because a second wave
re-pays the replicated pivot derivation (measured: S=4 at b=64 on 148
SMs was slower than not splitting).

mc_state is (rows, 8) int32, zero-initialised at first allocation:
[0]=gt, [1]=eq count, [2]=eq slab slots, [3]=arrive.  Scope matches the
single-CTA kernels: fp32, next_n == compress_ratio == 1,
return_values=False, k <= TIE_CAP, N % 4 == 0 (sampled) / any N (hint).
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.utils.smem_allocator import SmemAllocator

from . import radix_topk_primitives as _radix_mod
from . import sampled_topk_primitives as _sampled_mod
from . import fallback_topk_primitives as _fallback_mod2
from .fallback_topk_primitives import GatedExactFallback
from .radix_topk_primitives import (
    NUM_THREADS,
    NUM_WARPS,
    TIE_CAP,
    gmem_atomic_add,
    ld_global_v4_u32,
    mc_barrier,
    smem_atomic_add,
    smem_red_max_u32,
)
from .sampled_topk_primitives import GCAP, SampledPivotTopK


# Per-CTA eq-candidate STAGING capacity, decoupled from TIE_CAP (the
# tie-SELECT capacity): at 1M rows eq/S is 1.3-4.7K per CTA, and the
# old TIE_CAP (2048) stage silently dropped entries -> slots < eq ->
# every degraded-hint row fell through to the ~700us exact re-solve.
MC_EQ_CAP = 4096


class _McCommon(SampledPivotTopK):
    """Shared machinery: staged split harvest + last-arriver epilogue."""

    # set by the factory before compile; consumed as const_expr
    mc_splits: int = 2
    # MC bracket sample size.  The sample is REPLICATED per CTA (replica
    # reads hit L2).  4096 was too coarse at DSv4-scale rows: at 1M the
    # kth element's sample-rank sigma widened brackets to 2.6-9.4K
    # elements, overflowing the per-CTA eq stage on most rows.
    mc_smp: int = 8192

    @cute.jit
    def _tie_select_smem_skip(
        self,
        s_tk,
        s_ti,
        eq,
        gt,
        remaining,
        k_hi,
        k_lo,
        s_h256,
        s_warp_sums,
        s_misc,
        out_idx_row,
        tidx,
    ):
        """Byte-radix rank-select over the smem tie stage, SKIPPING rounds
        whose byte is shared by every candidate: all keys lie in
        [k_lo, k_hi], so byte r is uniform iff k_hi >> sh == k_lo >> sh
        (this byte AND all higher bits agree -- monotone, so only leading
        rounds skip).  A round-0-accepted bracket is ~span/4096 wide,
        making 1-2 of the 4 rounds degenerate on spread data (~0.8us
        each in block barriers)."""
        prefix = cutlass.Uint32(0)
        pmask = cutlass.Uint32(0)
        need = cutlass.Int32(remaining)
        total = cutlass.Int32(eq)
        for r_ in cutlass.range_constexpr(4):
            sh = cutlass.const_expr(24 - 8 * r_)
            if (k_hi >> cutlass.Uint32(sh)) == (k_lo >> cutlass.Uint32(sh)):
                # byte shared by construction: seed it, no histogram
                prefix = prefix | (k_lo & (cutlass.Uint32(0xFF) << cutlass.Uint32(sh)))
                pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(sh))
            else:
                if tidx < 256:
                    s_h256[tidx] = cutlass.Int32(0)
                cute.arch.barrier()
                for i in range(tidx, eq, NUM_THREADS):
                    kk = cutlass.Uint32(s_tk[i])
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
        if tidx == 0:
            s_misc[10] = cutlass.Int32(0)
            s_misc[11] = cutlass.Int32(0)
        cute.arch.barrier()
        nab = remaining - need
        for i in range(tidx, eq, NUM_THREADS):
            kk = cutlass.Uint32(s_tk[i])
            if kk > prefix:
                p = smem_atomic_add(s_misc + 10, 1)
                out_idx_row[gt + p] = s_ti[i]
            else:
                if kk == prefix:
                    e = smem_atomic_add(s_misc + 11, 1)
                    if e < need:
                        out_idx_row[gt + nab + e] = s_ti[i]

    @cute.jit
    def _mc_elem(self, w, idx, thf, tlf, s_count, s_tk, s_ti, s_gt):
        """Classify one element into the CTA-local smem stages (NaN -> gt,
        torch parity, same as the single-CTA collect).  A gt-stage
        overflow raises the DOOM flag (s_count[2]): the row's gt already
        exceeds any k <= TIE_CAP, so the walk may abort early -- the
        harvest poisons the reported count so a partial scan can never
        falsely pass verification."""
        val = w.bitcast(cutlass.Float32)
        if val <= thf:
            if val >= tlf:
                c = smem_atomic_add(s_count + 1, 1)
                if c < MC_EQ_CAP:
                    s_tk[c] = self.exact_key(w)
                    s_ti[c] = cutlass.Int32(idx)
        else:
            c = smem_atomic_add(s_count, 1)
            if c < TIE_CAP:  # TIE_CAP >= top_k: enough when the row succeeds
                s_gt[c] = cutlass.Int32(idx)
            else:
                s_count[2] = cutlass.Int32(1)  # doom: abort + poison

    @cute.jit
    def _mc_harvest(
        self,
        row_in,
        start,
        cnt,
        thf,
        tlf,
        tidx,
        s_count,
        s_tk,
        s_ti,
        s_gt,
        s_misc,
        mc_cnt,
        out_idx_row,
        slab_k,
        slab_i,
    ):
        """Staged slice harvest: smem staging + <= 3 gmem atomics per CTA
        (on mc_cnt[0]=gt, [1]=eq count, [2]=eq slots).  Boundary semantics
        match the single-CTA collect (NaN -> gt).  A doomed slice (gt
        stage overflow) aborts the walk early and poisons the reported gt
        by top_k+1, so partial scans always fail verification."""
        top_k = cutlass.const_expr(self.top_k)
        if tidx == 0:
            s_count[0] = cutlass.Int32(0)  # local gt
            s_count[1] = cutlass.Int32(0)  # local eq
            s_count[2] = cutlass.Int32(0)  # doom flag
        cute.arch.barrier()
        # float4 double-buffered walk over the slice (start % 4 == 0 by the
        # chunk rounding; scalar tail covers cnt % 4).  The scalar variant
        # measured ~2x less bandwidth, erasing the split's gain at S=2.
        addr = row_in.toint() + cutlass.Int64(start) * 4
        lv = cnt >> cutlass.Int32(2)
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
            if s_count[2] != 0:  # doomed: abort the walk (count poisoned)
                v = cutlass.Int32(lv)
            else:
                v2 = v + NUM_THREADS
                if v2 < lv:
                    m0, m1, m2, m3 = ld_global_v4_u32(va + stride16)
                base = start + v * 4
                for j in cutlass.range_constexpr(4):
                    self._mc_elem(
                        (w0, w1, w2, w3)[j],
                        base + j,
                        thf,
                        tlf,
                        s_count,
                        s_tk,
                        s_ti,
                        s_gt,
                    )
                va = va + stride16 + stride16
                v = v2 + NUM_THREADS
                if v < lv:
                    w0, w1, w2, w3 = ld_global_v4_u32(va)
                if v2 < lv:
                    base2 = start + v2 * 4
                    for j in cutlass.range_constexpr(4):
                        self._mc_elem(
                            (m0, m1, m2, m3)[j],
                            base2 + j,
                            thf,
                            tlf,
                            s_count,
                            s_tk,
                            s_ti,
                            s_gt,
                        )
        tail_base = lv * 4
        for i in range(tidx, cnt - tail_base, NUM_THREADS):
            if s_count[2] == 0:
                idx = start + tail_base + i
                self._mc_elem(
                    self.load_scalar(row_in, idx),
                    idx,
                    thf,
                    tlf,
                    s_count,
                    s_tk,
                    s_ti,
                    s_gt,
                )
        cute.arch.barrier()
        local_gt = s_count[0]
        local_eq = s_count[1]
        if s_count[2] != 0:
            # poisoned: partial counts must never pass verification
            local_gt = local_gt + cutlass.Int32(top_k + 1)
        gt_staged = local_gt
        if gt_staged > TIE_CAP:
            gt_staged = cutlass.Int32(TIE_CAP)
        eq_staged = local_eq
        if eq_staged > MC_EQ_CAP:
            eq_staged = cutlass.Int32(MC_EQ_CAP)
        if tidx == 0:
            s_misc[10] = cutlass.Int32(0)  # gt ticket base
            s_misc[11] = cutlass.Int32(0)  # eq slab base
            if local_gt > 0:
                s_misc[10] = cutlass.Int32(gmem_atomic_add(mc_cnt + 0, local_gt))
            if local_eq > 0:
                gmem_atomic_add(mc_cnt + 1, local_eq)  # exact count
                s_misc[11] = cutlass.Int32(gmem_atomic_add(mc_cnt + 2, eq_staged))
        cute.arch.barrier()
        gt_base = s_misc[10]
        eq_base = s_misc[11]
        for t in range(tidx, gt_staged, NUM_THREADS):
            p = gt_base + t
            if p < top_k:
                out_idx_row[p] = s_gt[t]
        for t in range(tidx, eq_staged, NUM_THREADS):
            p = eq_base + t
            if p < GCAP:
                slab_k[p] = s_tk[t].bitcast(cutlass.Int32)
                slab_i[p] = s_ti[t]

    @cute.jit
    def _mc_epilogue(
        self,
        k_hi,
        k_lo,
        tidx,
        row,
        num_rows_dbg,
        s_tk,
        s_ti,
        s_h4k,
        s_h256,
        s_warp_sums,
        s_misc,
        mc_row,
        out_idx_row,
        slab_k,
        slab_i,
        st_ptr,
        row_in,
        length,
        s_count,
    ):
        """Last-arriver election + count-verify + finish + self-reset.
        A failed row re-solves INLINE via _fallback_row (fused; the
        gated fallback kernel's empty pass measured a flat ~1.65us of
        pure second-launch latency -- see walkfirst)."""
        S = cutlass.const_expr(self.mc_splits)
        cute.arch.barrier()  # publish this CTA's writes (mc_barrier idiom)
        if tidx == 0:
            old = cute.arch.atomic_add(
                mc_row + 3, cutlass.Int32(1), sem="release", scope="gpu"
            )
            s_misc[11] = cutlass.Int32(0)
            if old == cutlass.Int32(S - 1):
                cute.arch.load(mc_row + 3, cutlass.Int32, sem="acquire", scope="gpu")
                s_misc[11] = cutlass.Int32(1)
        cute.arch.barrier()
        if s_misc[11] == 1:
            gt = cutlass.Int32(mc_row[0])
            eq = cutlass.Int32(mc_row[1])
            slots = cutlass.Int32(mc_row[2])
            ok = self._mc_finish(
                k_hi,
                k_lo,
                gt,
                eq,
                slots,
                tidx,
                s_tk,
                s_ti,
                s_h4k,
                s_h256,
                s_warp_sums,
                s_misc,
                out_idx_row,
                slab_k,
                slab_i,
            )
            if ok == 0:
                # inline exact fallback: CTA-uniform ok (uniform smem/gmem
                # reads), so the block barriers inside are safe; epilogue
                # rows always have length > top_k.  s_count slots 8/16
                # are free (harvest uses 0/1/2).
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
                st_ptr[num_rows_dbg + row] = gt
                st_ptr[num_rows_dbg * 2 + row] = eq
                st_ptr[num_rows_dbg * 3 + row] = k_hi.bitcast(cutlass.Int32)
                st_ptr[num_rows_dbg * 4 + row] = k_lo.bitcast(cutlass.Int32)
                mc_row[0] = cutlass.Int32(0)
                mc_row[1] = cutlass.Int32(0)
                mc_row[2] = cutlass.Int32(0)
                mc_row[3] = cutlass.Int32(0)

    @cute.jit
    def _mc_finish(
        self,
        k_hi,
        k_lo,
        gt,
        eq,
        slots,
        tidx,
        s_tk,
        s_ti,
        s_h4k,
        s_h256,
        s_warp_sums,
        s_misc,
        out_idx_row,
        slab_k,
        slab_i,
    ):
        """Count-verify + finish from the slab; returns ok (1 = row done)."""
        top_k = cutlass.const_expr(self.top_k)
        remaining = top_k - gt
        ok = cutlass.Int32(0)
        if remaining >= 0:
            if remaining <= eq:
                if k_hi == k_lo:
                    # ties key-equal: any subset exact; dense slab prefix
                    # covers remaining (staged_total >= min(eq, TIE_CAP)
                    # >= remaining)
                    if remaining <= slots:
                        ok = cutlass.Int32(1)
                        for t in range(tidx, remaining, NUM_THREADS):
                            out_idx_row[gt + t] = slab_i[t]
                else:
                    if eq <= TIE_CAP and slots == eq:
                        ok = cutlass.Int32(1)
                        for t in range(tidx, eq, NUM_THREADS):
                            s_tk[t] = cutlass.Uint32(slab_k[t])
                            s_ti[t] = slab_i[t]
                        cute.arch.barrier()
                        if eq <= 128:
                            # parent's warp-ballot paths (no radix rounds)
                            self.tie_select(
                                s_tk,
                                s_ti,
                                eq,
                                gt,
                                remaining,
                                s_h4k,  # radix arm needs >= 512 scratch
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                out_idx_row,  # dummy: has_values False
                                tidx,
                            )
                        else:
                            self._tie_select_smem_skip(
                                s_tk,
                                s_ti,
                                eq,
                                gt,
                                remaining,
                                k_hi,
                                k_lo,
                                s_h256,
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                tidx,
                            )
                    else:
                        if eq <= GCAP and slots == eq:
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
        return ok


class McSampledTopK(_McCommon):
    """Row-split self-sampling selector (see module docstring)."""

    @cute.jit
    def _sample_bracket(
        self, row_in, length, tidx, s_samp, s_h4k, s_h256, s_warp_sums, s_misc
    ):
        """Sample + shift-binned bracket (same logic as the single-CTA
        kernel at mc_smp sample size); deterministic, so every replica CTA
        agrees.  s_h4k is zeroed here BEFORE the sampling barrier (disjoint
        smem), saving one block barrier per call."""
        top_k = cutlass.const_expr(self.top_k)
        smp = cutlass.const_expr(self.mc_smp)
        if tidx == 0:
            s_misc[6] = cutlass.Int32(0)
            s_misc[7] = cutlass.Int32(0)
        for zz in cutlass.range_constexpr(4096 // NUM_THREADS):
            s_h4k[tidx + cutlass.Int32(zz * NUM_THREADS)] = cutlass.Int32(0)
        # one barrier orders BOTH inits (accumulators + round-0 histogram)
        # before the sampling loop's red.max: without it, thread 0's zeroing
        # races other warps' contributions and the bracket goes
        # NONDETERMINISTIC (observed: run-to-run varying gt/eq)
        cute.arch.barrier()
        n4 = length >> cutlass.Int32(2)
        nprobe = cutlass.const_expr(smp // 4)
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

        ks_i = cutlass.Int32(
            (cutlass.Int64(top_k) * cutlass.Int64(smp)) // cutlass.Int64(length)
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
        if need_lo > smp:
            need_lo = cutlass.Int32(smp)
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
            (cutlass.Int64(GCAP) * cutlass.Int64(3 * smp // 4)) // cutlass.Int64(length)
        )
        tight = cutlass.Int32(3) * m + cutlass.Int32(16)
        smem_ok = cutlass.Int32(
            (cutlass.Int64(TIE_CAP) * cutlass.Int64(smp)) // cutlass.Int64(length)
        )
        if tight < smem_ok:
            tight = smem_ok
        if budget > tight:
            budget = tight
        for _round in cutlass.range_constexpr(5):
            nbins = cutlass.const_expr(4096 if _round == 0 else 256)
            s_hist = s_h4k if cutlass.const_expr(_round == 0) else s_h256
            if have == 0:
                if cutlass.const_expr(_round != 0):
                    # round 0's s_h4k was zeroed before the sampling
                    # barrier (disjoint smem): no extra barrier for it
                    if tidx < 256:
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()
                span = cutlass.Int64(hi_k - lo_k) + cutlass.Int64(1)
                shift = cutlass.Uint32(0)
                spn = span - cutlass.Int64(1)
                while spn > cutlass.Int64(nbins - 1):
                    spn = spn >> 1
                    shift = shift + 1
                for i in range(tidx, smp, NUM_THREADS):
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
        return k_hi, k_lo

    @cute.kernel
    def mc_sampled_topk_kernel(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, sl, _ = cute.arch.block_idx()
        num_rows_dbg, _, _ = cute.arch.grid_dim()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])
        S = cutlass.const_expr(self.mc_splits)

        in_ptr = input_data.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator
        mc_ptr = mc_state.iterator

        smem = SmemAllocator()
        s_samp = smem.allocate_array(cutlass.Int32, self.mc_smp, byte_alignment=128)
        s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tk = smem.allocate_array(cutlass.Uint32, MC_EQ_CAP, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, MC_EQ_CAP, byte_alignment=128)
        s_gt = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * (2 * GCAP)
        slab_i = slab_k + GCAP
        mc_row = mc_ptr + row64 * 8

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        if top_k >= length:
            if sl == 0:
                for i in range(tidx, top_k, NUM_THREADS):
                    if i < length:
                        out_idx_row[i] = cutlass.Int32(i)
                    else:
                        out_idx_row[i] = cutlass.Int32(-1)
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
                    st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
        else:
            k_hi, k_lo = self._sample_bracket(
                row_in, length, tidx, s_samp, s_h4k, s_h256, s_warp_sums, s_misc
            )
            thf = self.from_key32(k_hi).bitcast(cutlass.Float32)
            tlf = self.from_key32(k_lo).bitcast(cutlass.Float32)
            chunk = ((length + cutlass.Int32(S) - 1) // cutlass.Int32(S) + 3) & ~3
            start = cutlass.Int32(sl) * chunk
            cnt = length - start
            if cnt > chunk:
                cnt = chunk
            if cnt < 0:
                cnt = cutlass.Int32(0)
            self._mc_harvest(
                row_in,
                start,
                cnt,
                thf,
                tlf,
                tidx,
                s_count,
                s_tk,
                s_ti,
                s_gt,
                s_misc,
                mc_row,
                out_idx_row,
                slab_k,
                slab_i,
            )
            self._mc_epilogue(
                k_hi,
                k_lo,
                tidx,
                row,
                num_rows_dbg,
                s_tk,
                s_ti,
                s_h4k,
                s_h256,
                s_warp_sums,
                s_misc,
                mc_row,
                out_idx_row,
                slab_k,
                slab_i,
                st_ptr,
                row_in,
                length,
                s_count,
            )

    @cute.jit
    def launch_mc(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.mc_sampled_topk_kernel(
            input_data, seqlen, output_indices, slab, status, mc_state
        ).launch(
            grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


class McHintTopK(_McCommon):
    """Row-split hint-pivot selector: min-of-hints [T, T] pivot (replicated
    gather, ~k reads per CTA) + the shared staged split harvest.  The
    single-key epilogue arm (k_hi == k_lo always here) fills by count."""

    @cute.kernel
    def mc_hint_topk_kernel(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, sl, _ = cute.arch.block_idx()
        num_rows_dbg, _, _ = cute.arch.grid_dim()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])
        S = cutlass.const_expr(self.mc_splits)

        in_ptr = input_data.iterator
        pi_ptr = pre_idx.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator
        mc_ptr = mc_state.iterator

        smem = SmemAllocator()
        s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tk = smem.allocate_array(cutlass.Uint32, MC_EQ_CAP, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, MC_EQ_CAP, byte_alignment=128)
        s_gt = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        pi_row = pi_ptr + row64 * top_k
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * (2 * GCAP)
        slab_i = slab_k + GCAP
        mc_row = mc_ptr + row64 * 8

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        if top_k >= length:
            if sl == 0:
                for i in range(tidx, top_k, NUM_THREADS):
                    if i < length:
                        out_idx_row[i] = cutlass.Int32(i)
                    else:
                        out_idx_row[i] = cutlass.Int32(-1)
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
                    st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
        else:
            # replicated pivot: min over the hinted values (deterministic)
            if tidx == 0:
                s_misc[10] = cutlass.Int32(0)  # red.max(~key) accumulator
            cute.arch.barrier()
            for i in range(tidx, top_k, NUM_THREADS):
                hidx = pi_row[i]
                if hidx >= 0:
                    if hidx < length:
                        b = self.load_scalar(row_in, hidx)
                        smem_red_max_u32(s_misc + 10, ~self.exact_key(b))
            cute.arch.barrier()
            kmin = ~cutlass.Uint32(s_misc[10])
            tf = self.from_key32(kmin).bitcast(cutlass.Float32)

            chunk = ((length + cutlass.Int32(S) - 1) // cutlass.Int32(S) + 3) & ~3
            start = cutlass.Int32(sl) * chunk
            cnt = length - start
            if cnt > chunk:
                cnt = chunk
            if cnt < 0:
                cnt = cutlass.Int32(0)
            self._mc_harvest(
                row_in,
                start,
                cnt,
                tf,
                tf,
                tidx,
                s_count,
                s_tk,
                s_ti,
                s_gt,
                s_misc,
                mc_row,
                out_idx_row,
                slab_k,
                slab_i,
            )
            self._mc_epilogue(
                kmin,
                kmin,
                tidx,
                row,
                num_rows_dbg,
                s_tk,
                s_ti,
                s_h4k,
                s_h256,
                s_warp_sums,
                s_misc,
                mc_row,
                out_idx_row,
                slab_k,
                slab_i,
                st_ptr,
                row_in,
                length,
                s_count,
            )

    @cute.jit
    def launch_mc(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.mc_hint_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status, mc_state
        ).launch(
            grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


class McStackedTopK(McSampledTopK):
    """Row-split two-rung selector: hint [T, T] rung (with early-abort on
    poisoned hints) -> full-group verdict -> sampled-bracket rung.  The
    inter-CTA verdict uses mc_barrier, which is spin-based and safe here
    ONLY because the launch policy guarantees a single co-resident wave
    (rows * S <= SM count).  Counter blocks: rung 1 uses mc_row[0..3],
    rung 2 uses mc_row[4..7]."""

    @cute.kernel
    def mc_stacked_topk_kernel(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, sl, _ = cute.arch.block_idx()
        num_rows_dbg, _, _ = cute.arch.grid_dim()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])
        S = cutlass.const_expr(self.mc_splits)

        in_ptr = input_data.iterator
        pi_ptr = pre_idx.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator
        mc_ptr = mc_state.iterator

        smem = SmemAllocator()
        s_samp = smem.allocate_array(cutlass.Int32, self.mc_smp, byte_alignment=128)
        s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tk = smem.allocate_array(cutlass.Uint32, MC_EQ_CAP, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, MC_EQ_CAP, byte_alignment=128)
        s_gt = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        pi_row = pi_ptr + row64 * top_k
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * (2 * GCAP)
        slab_i = slab_k + GCAP
        mc_row = mc_ptr + row64 * 8

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        if top_k >= length:
            if sl == 0:
                for i in range(tidx, top_k, NUM_THREADS):
                    if i < length:
                        out_idx_row[i] = cutlass.Int32(i)
                    else:
                        out_idx_row[i] = cutlass.Int32(-1)
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
                    st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
        else:
            chunk = ((length + cutlass.Int32(S) - 1) // cutlass.Int32(S) + 3) & ~3
            start = cutlass.Int32(sl) * chunk
            cnt = length - start
            if cnt > chunk:
                cnt = chunk
            if cnt < 0:
                cnt = cutlass.Int32(0)

            # ---- rung 1: min-of-hints [T, T] (replicated pivot) ----
            if tidx == 0:
                s_misc[10] = cutlass.Int32(0)
            cute.arch.barrier()
            for i in range(tidx, top_k, NUM_THREADS):
                hidx = pi_row[i]
                if hidx >= 0:
                    if hidx < length:
                        b = self.load_scalar(row_in, hidx)
                        smem_red_max_u32(s_misc + 10, ~self.exact_key(b))
            cute.arch.barrier()
            kmin = cutlass.Uint32(~cutlass.Uint32(s_misc[10]))
            tf = self.from_key32(kmin).bitcast(cutlass.Float32)
            self._mc_harvest(
                row_in,
                start,
                cnt,
                tf,
                tf,
                tidx,
                s_count,
                s_tk,
                s_ti,
                s_gt,
                s_misc,
                mc_row,
                out_idx_row,
                slab_k,
                slab_i,
            )
            # full-group verdict (spin barrier: single-wave co-residency
            # guaranteed by the launch policy)
            mc_barrier(mc_row + 3, cutlass.Int32(S), tidx)
            gt = cutlass.Int32(mc_row[0])
            eq = cutlass.Int32(mc_row[1])
            slots = cutlass.Int32(mc_row[2])
            remaining = top_k - gt
            ok1 = cutlass.Int32(0)
            if remaining >= 0:
                if remaining <= eq:
                    if remaining <= slots:
                        ok1 = cutlass.Int32(1)
            # second, NON-SPINNING arrival round: the last CTA to finish
            # reading resets the rung-1 counters.  Resetting from slice 0
            # directly would race a slow sibling's post-barrier read and
            # could split the verdict (one CTA entering rung 2 alone would
            # deadlock its spin barrier).
            cute.arch.barrier()  # all my threads' reads precede my add
            if tidx == 0:
                old2 = cute.arch.atomic_add(
                    mc_row + 3, cutlass.Int32(1), sem="release", scope="gpu"
                )
                if old2 == cutlass.Int32(2 * S - 1):
                    mc_row[0] = cutlass.Int32(0)
                    mc_row[1] = cutlass.Int32(0)
                    mc_row[2] = cutlass.Int32(0)
                    mc_row[3] = cutlass.Int32(0)
            if ok1 == 1:
                # hint verified: single-key fill; slice 0 finishes
                if sl == 0:
                    for t in range(tidx, remaining, NUM_THREADS):
                        out_idx_row[gt + t] = slab_i[t]
                    if tidx == 0:
                        st_ptr[row] = cutlass.Int32(0)
                        st_ptr[num_rows_dbg + row] = cutlass.Int32(1)  # rung
                        st_ptr[num_rows_dbg * 2 + row] = gt
                        st_ptr[num_rows_dbg * 3 + row] = eq
            else:
                # ---- rung 2: sampled bracket (counters mc_row[4..7]) ----
                k_hi, k_lo = self._sample_bracket(
                    row_in,
                    length,
                    tidx,
                    s_samp,
                    s_h4k,
                    s_h256,
                    s_warp_sums,
                    s_misc,
                )
                thf = self.from_key32(k_hi).bitcast(cutlass.Float32)
                tlf = self.from_key32(k_lo).bitcast(cutlass.Float32)
                self._mc_harvest(
                    row_in,
                    start,
                    cnt,
                    thf,
                    tlf,
                    tidx,
                    s_count,
                    s_tk,
                    s_ti,
                    s_gt,
                    s_misc,
                    mc_row + 4,
                    out_idx_row,
                    slab_k,
                    slab_i,
                )
                mc_barrier(mc_row + 7, cutlass.Int32(S), tidx)
                if sl == 0:
                    gt2 = cutlass.Int32(mc_row[4])
                    eq2 = cutlass.Int32(mc_row[5])
                    slots2 = cutlass.Int32(mc_row[6])
                    ok2 = self._mc_finish(
                        k_hi,
                        k_lo,
                        gt2,
                        eq2,
                        slots2,
                        tidx,
                        s_tk,
                        s_ti,
                        s_h4k,
                        s_h256,
                        s_warp_sums,
                        s_misc,
                        out_idx_row,
                        slab_k,
                        slab_i,
                    )
                    if ok2 == 0:
                        # inline exact fallback (see _mc_epilogue): slice 0
                        # is CTA-uniform and the sole post-barrier finisher
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
                        st_ptr[row] = cutlass.Int32(1) - ok2
                        st_ptr[num_rows_dbg + row] = cutlass.Int32(2)  # rung
                        st_ptr[num_rows_dbg * 2 + row] = gt2
                        st_ptr[num_rows_dbg * 3 + row] = eq2
                        # rung-2 counters: slice 0 is the sole post-barrier
                        # reader, so its reset cannot race anyone
                        mc_row[4] = cutlass.Int32(0)
                        mc_row[5] = cutlass.Int32(0)
                        mc_row[6] = cutlass.Int32(0)
                        mc_row[7] = cutlass.Int32(0)

    @cute.jit
    def launch_mc(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.mc_stacked_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status, mc_state
        ).launch(
            grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


class ProdMcSampledTopK(McSampledTopK, GatedExactFallback):
    """MC sampled kernel + gated exact fallback, one compiled launcher."""

    @cute.jit
    def launch_prod(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        # No fallback launch: the exact fallback is FUSED into the MC
        # epilogue (_mc_epilogue / stacked rung 2) -- the gated fallback
        # kernel's empty pass measured a flat ~1.65us of launch latency.
        self.mc_sampled_topk_kernel(
            input_data, seqlen, output_indices, slab, status, mc_state
        ).launch(
            grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


class ProdMcStackedTopK(McStackedTopK, GatedExactFallback):
    """MC stacked kernel + gated exact fallback, one compiled launcher."""

    @cute.jit
    def launch_prod(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.mc_stacked_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status, mc_state
        ).launch(
            grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )  # fallback fused in-kernel (see ProdMcSampledTopK)


class ProdMcHintTopK(McHintTopK, GatedExactFallback):
    """MC hint kernel + gated exact fallback, one compiled launcher."""

    @cute.jit
    def launch_prod(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.mc_hint_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status, mc_state
        ).launch(
            grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )  # fallback fused in-kernel (see ProdMcSampledTopK)


_compiled: dict = {}


def _mc_factory(cls, top_k, splits):
    kern = cls(
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
    kern.mc_splits = splits
    return kern


def _fk(dt, shape, align=None):
    so = tuple(range(len(shape) - 1, -1, -1))
    if align is None:
        return cute.runtime.make_fake_compact_tensor(dt, shape, stride_order=so)
    return cute.runtime.make_fake_compact_tensor(
        dt, shape, stride_order=so, assumed_align=align
    )


def get_mc_kernel(kind: str, top_k: int, N: int, splits: int):
    """Compile (with on-disk caching) the fused MC speculative + gated
    fallback launcher.  kind is 'sampled' or 'hint' (hint additionally
    takes pre_idx).  Caller supplies mc_state (rows, 8) int32
    ZERO-INITIALISED at first use (self-resetting afterwards)."""
    assert top_k <= TIE_CAP
    assert splits in (2, 4, 8, 16)
    if kind in ("sampled", "stacked"):
        assert N % 4 == 0
    key = (kind, top_k, N, splits)
    if key in _compiled:
        return _compiled[key]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    cls = {
        "sampled": ProdMcSampledTopK,
        "hint": ProdMcHintTopK,
        "stacked": ProdMcStackedTopK,
    }[kind]
    kern = _mc_factory(cls, top_k, splits)
    sym_rows = cute.sym_int()
    f32, i32 = cutlass.Float32, cutlass.Int32

    def _compile_fn():
        args = [_fk(f32, (sym_rows, N), 16)]
        if kind in ("hint", "stacked"):
            args.append(_fk(i32, (sym_rows, top_k), 16))
        args += [
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, top_k), 16),
            _fk(i32, (sym_rows, 2 * GCAP), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, 8), 16),
        ]
        return cute.compile(
            kern.launch_prod,
            *args,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = build_and_load_cute_dsl_kernel(
        "mc_sampled_topk_primitives",
        f"mc_{kind}_v2_f32_k{top_k}_N{N}_S{splits}",
        _compile_fn,
        extra_key_files=(
            __file__,
            _sampled_mod.__file__,
            _radix_mod.__file__,
            _fallback_mod2.__file__,
        ),
    )
    _compiled[key] = compiled
    return compiled
