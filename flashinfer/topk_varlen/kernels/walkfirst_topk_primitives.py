"""Walk-first top-k on the primitives substrate (experimental) -- a full
replication of gvr_2's streaming architecture, plus two things it lacks:
row-splitting at k > 1024 and an exact fallback.

gvr_2's structural speed (measured and source-analyzed) comes from
inverting the decide-first order our other selectors use: it never pays a
deliberation phase before the walk.

  1. PRE-SCALARS (~1us, deterministic): a 1024-element register sample ->
     smem min/max -> one 256-bin value histogram of the SAMPLE -> a
     provisional threshold TF at the bin whose sample rank targets
     ``aim ~ 2.5k`` survivors (conservatively LOW edge: overshooting the
     candidate count is recoverable, undershooting k is not), and a
     survivor-bin scale SC = 256 / (smax - TF).
  2. ONE FUSED WALK: element v survives iff NOT (v <= TF) (NaN survives,
     torch parity).  Each survivor is staged into the CTA-local candidate
     buffer AND binned into a CTA-local 256-bin survivor histogram --
     both per-element costs are a couple of instructions, no barriers.
     Per CTA at slice end: one gmem atomic reserves a slab range (bulk,
     never per-hit -- the serialization lesson), one bulk copy stages
     out, 256 red.adds merge the survivor histogram into the row's gmem
     histogram (kept in the slab's unused tail).
  3. POST-WALK SELECT (last-arriver CTA, no spins anywhere): with cnt
     survivors, k <= cnt <= SCAP and slots == cnt required (else status
     -> the gated exact fallback re-solves; duplicate floods land there
     by design in v1).  The row's survivor histogram locates the bin B
     containing descending rank k AMONG CANDIDATES; candidates in bins
     above B are winners; rank ties WITHIN bin B (a value range, not a
     single value) get an exact key-space sub-select (ballot for small
     sets, byte-radix with degenerate-round skipping otherwise).
     Exactness never depends on TF or bin quality: candidates are
     exactly {v : !(v <= TF)} with an exact count, and the sub-select is
     key-exact -- TF/SC only steer performance.

mc_state: [0]=cand count, [1]=slab slots, [2]=arrive.  The row's
256-bin survivor histogram lives at slab_k[2*GCAP .. 2*GCAP+256), i.e.
the tail of the WF_ROW_INTS-wide slab row (zeroed by the epilogue's
self-reset; zero-init at first allocation).  Scope:
fp32, next_n == compress_ratio == 1, return_values=False, k <= TIE_CAP,
N % 4 == 0.  Not in the auto heuristic.
"""

import os

import cutlass
import torch
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from . import fallback_topk_primitives as _fallback_mod
from . import gvr2_topk_decode as _gvr2_mod
from . import radix_topk_primitives as _radix_mod
from . import sampled_topk_primitives as _sampled_mod
from .fallback_topk_primitives import GatedExactFallback
from .radix_topk_primitives import (
    gmem_atomic_add,
    gmem_red_add,
    read_clock64,
    smem_atomic_add,
    smem_red_max_u32,
    warp_inclusive_sum,
    warp_sum,
)
from .gvr2_topk_decode import (
    RES_B,
    RES_B2,
    _atom_shared_cluster_add_i32,
    _cluster_sync_aligned,
    _ld_shared_cluster_i32,
    _mapa_shared_cluster,
    _pin_i32,
    _pin_i64,
    _st_shared_cluster_i32,
    lds128_i32,
    smem_atom_i32_128,
    sts128_i32,
    warp_incl_scan_add,
    warp_max_u32,
)
from .sampled_topk_primitives import GCAP, SampledPivotTopK

# DSv4-scale constants (sized for rows up to 1M): the original 64-128K
# era values (SCAP 8192 / LCAP 4096 / PSAMP 1024) collapsed at >=512K --
# the bracket rode on ~4 effective samples, candidates overflowed SCAP,
# and nearly every row fell back (382us at 1M vs gvr_2's 83).
SCAP = 16384  # candidate capacity == GCAP (full slab prefix)
# short-row arm threshold: rows at or below this length skip the
# walk pipeline entirely (see the SHORT-ROW ARM comment in the kernel)
# census-arm cutoff.  Was 16384 when the long pipeline had an ~11us fixed
# floor; the cluster/DSMEM epilogue dropped that floor enough that the walk
# pipeline now beats the census arm on 8-16K rows (B200 16K b=64 graph:
# k=512 7.24 vs 7.45, k=1024 8.06 vs 8.47).  Override:
# FLASHINFER_TOPK_WF_SMALL_N.
WF_SMALL_N = 8192
LCAP = 8192  # per-CTA local candidate stage (pairs); smem ~83KB total
# doubled stage where the device has the shared memory (>= LCAP_BIG_SMEM
# bytes per block): ~147KB total, still 1 CTA/SM (registers pin occupancy),
# and the overshoot re-walk class disappears (see get_walkfirst_kernel)
LCAP_BIG = 16384
LCAP_BIG_SMEM = 160 * 1024
PSAMP = 4096  # pre-scalar sample size (one float4 per thread)
# wf slab row layout: [0..GCAP) keys, [GCAP..2*GCAP) idx, then the
# 256-int row survivor histogram (SCAP == GCAP leaves no prefix slack)
WF_SMAX = 32  # max row split supported by the gmem path
WF_TBL = 260  # per-CTA publish table: base, staged, prefix[0..256], pad
WF_ROW_INTS = 2 * GCAP + 256 + WF_SMAX * WF_TBL
# survivor-histogram span past the sample max (see the sample block)
WF_SPAN_EXT = 1.5

from cutlass._mlir import ir  # noqa: E402
from cutlass._mlir.extras import types as T  # noqa: E402


@dsl_user_op
def ld_global_nc_v4_u32(gmem_addr: cutlass.Int64, *, loc=None, ip=None):
    """128-bit NON-COHERENT vectorized load: ld.global.nc.v4.b32.

    SASS comparison against gvr_2's streaming kernel showed its walk loads
    all compile to LDG.E.128.CONSTANT (the read-only data path, which
    sustains far more outstanding loads than the coherent LDG pipeline);
    ours were plain LDG.E.128 and moved ~1.6x fewer bytes/us.  The row is
    strictly read-only during the kernel, so .nc is safe here."""
    st = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [0])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [1])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [2])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [3])),
    )


# Shared binning arithmetic: MUST be textually identical between wf_bin_u8
# (epilogue emit) and wf_emit_pred (walk histogram) -- any rounding
# difference between the two shifts the rank accounting.  NaN -> bin 255
# (torch sorts NaN on top; the saturating convert alone would send it to 0).
_WF_BIN_PTX = (
    "sub.rn.f32 d, {v}, {tf};\n"
    "mul.rn.f32 m, d, {sc};\n"
    "cvt.rzi.u32.f32 {b}, m;\n"
    "min.u32 {b}, {b}, 255;\n"
    "setp.neu.f32 q, {v}, {v};\n"
    "@q mov.u32 {b}, 255;\n"
)


@dsl_user_op
def wf_bin_u8(
    vf: cutlass.Float32, tf: cutlass.Float32, sc: cutlass.Float32, *, loc=None, ip=None
):
    """Survivor bin in [0,255] via the exact _WF_BIN_PTX sequence."""
    res = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Float32(vf).ir_value(loc=loc, ip=ip),
            cutlass.Float32(tf).ir_value(loc=loc, ip=ip),
            cutlass.Float32(sc).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .f32 d, m;\n.reg .pred q;\n"
        + _WF_BIN_PTX.format(v="$1", tf="$2", sc="$3", b="$0")
        + "}",
        "=r,f,f,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(res)


@dsl_user_op
def wf_pack_tf16(tf: cutlass.Float32, bf16: bool, *, loc=None, ip=None):
    """fp32 walk threshold -> both halves of a 32-bit word holding the
    LARGEST 16-bit-grid value <= tf (cvt.rm), so ``x <= tf16`` in the
    packed compare equals ``f32(x) <= tf`` for every grid value x."""
    ty = "bf16" if bf16 else "f16"
    res = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(tf).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rm." + ty + ".f32 h, $1;\nmov.b32 $0, {h, h};\n}",
        "=r,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(res)


@dsl_user_op
def wf_dead_v4_16(
    dead: cutlass.Int32,
    w0: cutlass.Uint32,
    w1: cutlass.Uint32,
    w2: cutlass.Uint32,
    w3: cutlass.Uint32,
    tf2: cutlass.Uint32,
    base: int,
    bf16: bool,
    *,
    loc=None,
    ip=None,
):
    """Classify one 16-byte vector of 8 packed 16-bit elements: sets dead
    bit ``base + e`` for every element with x <= tf (4 packed setp + 8
    predicated ORs; element e = 2*word + half, low half first)."""
    ty = "bf16x2" if bf16 else "f16x2"
    body = "{\n.reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\nmov.b32 $0, $1;\n"
    for j in range(4):
        # PTX spells the two-predicate destination as ``p|q``
        body += f"setp.le.{ty} p{2 * j}|p{2 * j + 1}, ${2 + j}, $6;\n"
    for e in range(8):
        body += f"@p{e} or.b32 $0, $0, 0x{(1 << (base + e)) & 0xFFFFFFFF:08x};\n"
    body += "}"
    res = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int32(dead).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w3).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(tf2).ir_value(loc=loc, ip=ip),
        ],
        body,
        "=r,r,r,r,r,r,r",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(res)


class WalkFirstTopK(SampledPivotTopK):
    """See module docstring.  Inherits key helpers and the tie selects."""

    mc_splits: int = 2
    # Per-instance capacity knobs (defaults = the 1024-thread family; the
    # k<=1024 512-thread family halves them -- see get_walkfirst_kernel):
    lcap: int = LCAP
    psamp: int = PSAMP
    # Compile-time phase-telemetry switch (plain Python bool, read at
    # trace time): when False (production default) every
    # read_clock64() and the status-buffer phase writes (blocks 5-9)
    # are never traced -- zero instructions in the compiled kernel.
    # Enabled per-specialization by get_walkfirst_kernel(telemetry=True)
    # or FLASHINFER_TOPK_WF_TELEMETRY=1 (used by phase_walkfirst.py).
    wf_telemetry: bool = False
    # Cluster epilogue (SM90+, S in {2,4,8}): the S CTAs of a row form one
    # hardware cluster and coordinate through DSMEM instead of the gmem
    # slab + release/acquire arrival.  Set by get_walkfirst_kernel from
    # the device capability; FLASHINFER_TOPK_WF_CLUSTER=0 disables.
    wf_cluster: bool = False
    # TEST ONLY (FLASHINFER_TOPK_WF_FORCE_RETRY=1): compile the retry-
    # capable paths with aim < k so the T3 rung fires on every row
    wf_force_retry: bool = False
    # Hint rung: with caller hints, the smallest hinted value tightens the
    # sample-derived walk threshold (never replaces the sample).  Compiled
    # in only with FLASHINFER_TOPK_USE_HINTS=1 (see get_walkfirst_kernel);
    # off by default because realistic stale hints cost more than they save.
    wf_hint_rung: bool = False
    # Tail-safe aim floor for grids of >= 32 rows (see the aim computation);
    # FLASHINFER_TOPK_WF_TAILSAFE=0 is the A/B switch
    wf_tail_safe: bool = True
    # short-row arm cutoff (rows at or below this length take the census
    # arm; above it, the walk pipeline).  Compile-time; the factory can
    # override via FLASHINFER_TOPK_WF_SMALL_N for A/B experiments.
    wf_small_n: int = WF_SMALL_N
    # packed-pair 16-bit walk classify (setp.le.f16x2 / .bf16x2); the
    # factory enables it for fp16 everywhere and bf16 on sm_90+
    wf_pair16: bool = False

    @cute.jit
    def _wf_f32(self, bits):
        """Element bits (dtype pattern in the low bits of a Uint32) -> f32."""
        if cutlass.const_expr(self.is_f32):
            return bits.bitcast(cutlass.Float32)
        else:
            return self.value_from_bits(bits).to(cutlass.Float32)

    @cute.jit
    def _wf_val_of_key(self, key):
        """Ordered key -> f32 value (bijective per dtype)."""
        return self.value_from_key(key).to(cutlass.Float32)

    @cute.jit
    def _wf_key_of_f32(self, f):
        """Ordered key of an f32 threshold in this dtype's key space.  Used
        only for radix-round skip bounds (a wider range skips fewer rounds,
        never changes the result), so the f32->16-bit rounding is safe."""
        if cutlass.const_expr(self.is_f32):
            return self.to_key32(f.bitcast(cutlass.Uint32))
        else:
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                hb = f.to(cutlass.Float16).bitcast(cutlass.Uint16).to(cutlass.Uint32)
            else:
                hb = f.to(cutlass.BFloat16).bitcast(cutlass.Uint16).to(cutlass.Uint32)
            return self.to_key16(hb)

    @cute.jit
    def _wf_coarse_lb_key(self, b):
        """Exact key lower bound of coarse bin ``b`` (census arm skip bounds).
        fp32: the fp16-midpoint construction; 16-bit dtypes: coarse keys ARE
        the 16-bit keys, so the bound is simply b << coarse_shift."""
        if cutlass.const_expr(self.is_f32):
            return self.exact_key(
                self.coarse_bin_lower_bound_f32(b).bitcast(cutlass.Uint32)
            )
        else:
            return cutlass.Uint32(b) << cutlass.Uint32(self.coarse_shift)

    @cute.jit
    def _wf_elem_bits(self, w, h: cutlass.Constexpr):
        """Element ``h`` (0..EPW-1) of a loaded 32-bit word."""
        if cutlass.const_expr(self.is_f32):
            return w
        else:
            return (w >> cutlass.Uint32(16 * h)) & cutlass.Uint32(0xFFFF)

    @cute.jit
    def _wf_bin(self, val, tf_f, sc):
        """Survivor bin.  MUST be the single binning function for both the
        walk histogram and the epilogue emit: any disagreement between the
        two shifts the rank accounting.  NaN handled explicitly (a
        saturating float->int convert sends NaN to 0, i.e. ranked BOTTOM,
        breaking torch parity where NaN sorts on top).  Implemented as
        the shared _WF_BIN_PTX asm sequence so the rounding is
        bit-identical with the walk's wf_emit_pred by construction."""
        return wf_bin_u8(val, tf_f, sc)

    @cute.jit
    def _wf_short_row(
        self,
        row_in,
        length,
        out_idx_row,
        slab_k,
        slab_i,
        s_cv,
        s_h256,
        s_tk,
        s_ti,
        s_warp_sums,
        s_misc,
        s_count,
        tidx,
    ):
        """Direct solve for one short row (length <= WF_SMALL_N), smem
        only: the parent's 4096-bin COARSE histogram (fp16-key bins --
        an 8-bit ordered-key hist was tried first and measured WORSE
        than the plain MSD re-solve: its top byte is an EXPONENT
        histogram, so randn's crossing bin held thousands of ties and
        always overflowed into the fallback) -> rank bin -> emit
        winners above the bin + stage bin ties in smem -> key-exact
        tie select seeded with the bin's lower-bound keys.  Two L2-hot
        gmem passes, no slab traffic -- the census structure that
        makes radix_primitives the short-row winner, built from this
        class's inherited helpers.  Tie overflow (> TIE_CAP in one
        coarse bin, e.g. constant rows) falls back to the exact MSD
        re-solve.  s_cv doubles as the 4096-bin histogram (dead until
        the tie-select scratch phase consumes the hist)."""
        top_k = cutlass.const_expr(self.top_k)
        # coarse histogram: hist_size bins (4096 fp32 / 8192 16-bit), aliased
        # into the dead LCAP+4-int candidate stage
        s_h4k = s_cv
        for zz in cutlass.range_constexpr(self.hist_size // self.nt):
            s_h4k[tidx + cutlass.Int32(zz * self.nt)] = cutlass.Int32(0)
        if tidx == 0:
            s_count[0] = cutlass.Int32(0)  # emit ticket
            s_count[4] = cutlass.Int32(0)  # tie cursor
        cute.arch.barrier()
        # float4 census (the scalar form left ~1us on the 16K-length
        # boundary vs radix_primitives' vectorized stream)
        lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
        epw = cutlass.const_expr(self.vec_elems // 4)  # elements per 32-bit word
        n4 = length >> cutlass.Int32(lg)
        for i in range(tidx, n4, self.nt):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(row_in.toint() + cutlass.Int64(i) * 16)
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(epw):
                    smem_atomic_add(
                        s_h4k
                        + self.coarse_bin(self._wf_elem_bits((w0, w1, w2, w3)[j], h)),
                        1,
                    )
        for i in range((n4 << cutlass.Int32(lg)) + tidx, length, self.nt):
            smem_atomic_add(s_h4k + self.coarse_bin(self.load_scalar(row_in, i)), 1)
        cute.arch.barrier()
        # crossing at rank top_k: bin order ascends with value
        # (coarse_bin is monotone); above = count in strictly greater bins
        self.find_threshold_coarse(
            s_h4k, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
        )
        # no post-find barrier: find_threshold_coarse ends with a
        # full-block barrier, the classify below touches disjoint smem
        # (cursors zeroed and fenced at arm entry), and s_misc[0..1] are
        # not rewritten until the tie select's own fenced machinery
        binb = s_misc[0]
        above = s_misc[1]
        for i in range(tidx, n4, self.nt):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(row_in.toint() + cutlass.Int64(i) * 16)
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(epw):
                    bits = self._wf_elem_bits((w0, w1, w2, w3)[j], h)
                    b = self.coarse_bin(bits)
                    eoff = cutlass.Int32(j * epw + h)
                    if b > binb:
                        p = smem_atomic_add(s_count, 1)
                        if p < top_k:
                            out_idx_row[p] = (i << cutlass.Int32(lg)) + eoff
                    else:
                        if b == binb:
                            t = smem_atomic_add(s_count + 4, 1)
                            if t < self.tie_cap:
                                s_tk[t] = self.exact_key(bits)
                                s_ti[t] = (i << cutlass.Int32(lg)) + eoff
        for i in range((n4 << cutlass.Int32(lg)) + tidx, length, self.nt):
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
        cute.arch.barrier()
        nb = s_count[4]
        remaining = top_k - above
        ok = cutlass.Int32(1)
        if remaining < 0:
            ok = cutlass.Int32(0)
        if remaining > nb:
            ok = cutlass.Int32(0)
        if nb > self.tie_cap:
            ok = cutlass.Int32(0)
        if ok == 1:
            if remaining > 0:
                if nb <= 128:
                    self.tie_select(
                        s_tk,
                        s_ti,
                        nb,
                        above,
                        remaining,
                        s_cv,  # >= 512 ints scratch
                        s_warp_sums,
                        s_misc,
                        out_idx_row,
                        out_idx_row,  # dummy: has_values False
                        tidx,
                    )
                else:
                    # conservative key bounds of the coarse bin (skip-
                    # round hints only: a wider range just skips fewer
                    # radix rounds, never changes the result)
                    k_lo = self._wf_coarse_lb_key(binb)
                    k_hi = cutlass.Uint32(0xFFFFFFFF)  # top bin: NaN keys
                    if binb < cutlass.Int32(self.hist_size - 1):
                        k_hi = self._wf_coarse_lb_key(binb + 1) - cutlass.Uint32(1)
                    self._tie_select_smem_skip_wf(
                        s_tk,
                        s_ti,
                        nb,
                        above,
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
            self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                row_in,
                length,
                out_idx_row,
                slab_k,
                slab_i,
                s_cv,
                s_h256,
                s_warp_sums,
                s_misc,
                s_count + 8,
                s_count + 16,
                tidx,
            )

    @cute.jit
    def _wf_elem(self, w, idx, tf_f, sc, s_count, s_hist, s_cv, s_ci):
        """Scalar classify+emit for TAIL elements only (at most 3 per
        row; the hot path is the mask walk in wf_topk_kernel).  Survivor
        iff NOT (v <= TF): NaN survives (torch parity).  Overflow lands
        in the trash slot LCAP; slots < count still routes the row to
        the exact fallback."""
        val = self._wf_f32(w)
        if not (val <= tf_f):
            c = smem_atomic_add(s_count, 1)
            ps = c
            if ps > self.lcap:
                ps = cutlass.Int32(self.lcap)  # trash slot (IMNMX)
            s_cv[ps] = w.bitcast(cutlass.Int32)
            s_ci[ps] = cutlass.Int32(idx)
            cute.arch.red(
                s_hist + self._wf_bin(val, tf_f, sc),
                cutlass.Int32(1),
                op="add",
                dtype="s32",
                sem="relaxed",
                scope="cta",
            )

    @cute.jit
    def _wf_scan_cross0_2t(self, s_hist, target, target2, tidx, s_res):
        """256-bin warp-0-only suffix scan (gvr_2's scan_cross0 idiom,
        zero=True, two=True): publishes the crossing bins for ``target``
        (-> s_res[RES_B]) and ``target2`` (-> s_res[RES_B2]) in one
        barrier-free warp pass and ZEROES the histogram on the way out --
        the walk phase's survivor-histogram clear is folded in for free.
        HOLD path (2 vectors/lane); caller pays exactly one barrier."""
        if tidx < cutlass.Int32(32):
            lane = tidx
            atom = smem_atom_i32_128()
            hbase = s_hist.toint()
            frag0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            frag1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            lds128_i32(atom, hbase, lane * cutlass.Int32(32), frag0)
            lds128_i32(atom, hbase, lane * cutlass.Int32(32) + 16, frag1)
            sm = (
                frag0[0]
                + frag0[1]
                + frag0[2]
                + frag0[3]
                + frag1[0]
                + frag1[1]
                + frag1[2]
                + frag1[3]
            )
            w = warp_incl_scan_add(sm, lane)
            tot = cute.arch.shuffle_sync(w, cutlass.Int32(31))
            after = tot - w  # bins strictly above my span
            base = lane * cutlass.Int32(8)
            zeros = cute.make_rmem_tensor((4,), cutlass.Int32)
            for j in cutlass.range_constexpr(4):
                zeros[j] = cutlass.Int32(0)
            for q in cutlass.range_constexpr(1, -1, -1):
                vv = frag1
                if cutlass.const_expr(q == 0):
                    vv = frag0
                for j in cutlass.range_constexpr(3, -1, -1):
                    cq = vv[j]
                    gb = base + cutlass.Int32(4 * q + j)
                    cross = cutlass.Int32(0)
                    if after < target:
                        if (after + cq) >= target:
                            cross = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross = cutlass.Int32(1)
                    if cross != cutlass.Int32(0):
                        s_res[RES_B] = gb
                    cross2 = cutlass.Int32(0)
                    if after < target2:
                        if (after + cq) >= target2:
                            cross2 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross2 = cutlass.Int32(1)
                    if cross2 != cutlass.Int32(0):
                        s_res[RES_B2] = gb
                    after = after + cq
                sts128_i32(atom, zeros, hbase, lane * cutlass.Int32(32) + q * 16)

    @cute.jit
    def _wf_walk_slice(
        self,
        row_in,
        start,
        cnt_sl,
        tf_f,
        sc,
        degen,
        s_count,
        s_h256,
        s_cv,
        s_ci,
        tidx,
    ):
        """One staging pass over this CTA's slice: zero the local count +
        survivor histogram, then the mask walk (see the comment inside).
        Factored out so the T3 retry rung can re-run it with the floor
        threshold; ends with a block barrier (counts/hist visible)."""
        if tidx == 0:
            s_count[0] = cutlass.Int32(0)  # local candidates
        if tidx < 256:
            s_h256[tidx] = cutlass.Int32(0)  # local survivor hist
        cute.arch.barrier()
        if degen == 0:
            # ---- MASK WALK (gvr_2's structure, adopted after two
            # measured dead ends: v10 unconditional-side-effects was
            # 1.4-3x worse at ~3-6% survivor rates; v11's monolithic
            # predicated-asm emit lost ptxas's warp aggregation and
            # scheduling and ran ~20% behind the branchy v9).
            # Per iteration each thread loads 4 float4s (16 elements)
            # and classifies them into a 16-bit survivor mask -- one
            # FSETP + OR per element, no memory side effects, nothing
            # for ptxas to fence.  The staging reservation is warp-
            # aggregated ONCE per iteration (popc + shfl scan + a
            # single lane-31 atomic), then a divergent bit-walk emits
            # only actual survivors, RELOADING each value with a
            # scalar load: holding all 16 floats live across the emit
            # spills, and the reload hits L1/L2 (gvr_2's measured
            # lesson, replicated here).  Dead-mask form (~dead) keeps
            # one compare per element with NaN surviving (NaN <= TF
            # is false).  The loop trip count is CTA-uniform (full
            # iterations unguarded; the boundary iteration clamps
            # addresses and masks validity) so the warp collectives
            # are always converged.
            # gvr_2's _pin discipline (D:1266): opaque identity movs stop
            # NVVM rematerializing the invariants' defining chains (param
            # base + start mul/add, the lv division ladder) inside the scf
            # while-region -- at the 64-register wall the rematerialized
            # chains cost registers the walk needs.  (PRIME-LATE tile-0
            # register priming was built on top of these pins and STILL
            # regressed -- 16 live primes exceed this kernel's budget even
            # with pinned invariants; the machinery was removed.  Do not
            # retry priming without a whole-kernel register redesign.)
            lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
            epw = cutlass.const_expr(self.vec_elems // 4)  # elements per word
            addr = _pin_i64(
                row_in.toint() + cutlass.Int64(start) * cutlass.Int64(self.elem_bytes)
            )
            lv = cnt_sl >> cutlass.Int32(lg)
            lvm1 = _pin_i32(lv - 1)
            stride16 = cutlass.Int64(self.nt * 16)
            lane = tidx % 32
            # 4*nt float4s per iteration
            it_sh = cutlass.const_expr((4 * self.nt).bit_length() - 1)
            n_it = _pin_i32(
                (lv + cutlass.Int32(4 * self.nt - 1)) >> cutlass.Int32(it_sh)
            )
            n_full = _pin_i32(lv >> cutlass.Int32(it_sh))
            it = cutlass.Int32(0)
            vbase = cutlass.Int32(tidx)
            va = addr + cutlass.Int64(tidx) * 16
            tf2 = cutlass.Uint32(0)
            if cutlass.const_expr(self.wf_pair16):
                tf2 = wf_pack_tf16(tf_f, self.dtype == cutlass.BFloat16)
            while it < n_it:
                dead = cutlass.Int32(0)
                # all 4*E element bits valid (E=4 -> 0xFFFF; E=8 -> all 32)
                valid = cutlass.Int32(0xFFFF)
                if cutlass.const_expr(self.vec_elems == 8):
                    valid = cutlass.Int32(-1)
                if it < n_full:  # CTA-uniform: no bounds checks at all
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(va)
                    b0, b1, b2, b3 = ld_global_nc_v4_u32(va + stride16)
                    c0, c1, c2, c3 = ld_global_nc_v4_u32(va + stride16 * 2)
                    d0, d1, d2, d3 = ld_global_nc_v4_u32(va + stride16 * 3)
                    # (An L2 prefetch of the next tile here -- 4 x
                    # prefetch.global.L2, no registers -- was built and
                    # MEASURED: walk 5.26 -> 5.48us at 64K.  The walk takes
                    # the same ~1us per 64KB tile with 8 CTAs or 148 on the
                    # GPU, so it is issue-bound on the classify + survivor
                    # bit-walk, not waiting on HBM; prefetching only adds
                    # instructions.  Removed.)
                    if cutlass.const_expr(self.wf_pair16):
                        bf = cutlass.const_expr(self.dtype == cutlass.BFloat16)
                        dead = wf_dead_v4_16(dead, a0, a1, a2, a3, tf2, 0, bf)
                        dead = wf_dead_v4_16(dead, b0, b1, b2, b3, tf2, 8, bf)
                        dead = wf_dead_v4_16(dead, c0, c1, c2, c3, tf2, 16, bf)
                        dead = wf_dead_v4_16(dead, d0, d1, d2, d3, tf2, 24, bf)
                    else:
                        for j in cutlass.range_constexpr(4):
                            for h in cutlass.range_constexpr(epw):
                                e = j * epw + h
                                fa = self._wf_f32(
                                    self._wf_elem_bits((a0, a1, a2, a3)[j], h)
                                )
                                fb = self._wf_f32(
                                    self._wf_elem_bits((b0, b1, b2, b3)[j], h)
                                )
                                fc = self._wf_f32(
                                    self._wf_elem_bits((c0, c1, c2, c3)[j], h)
                                )
                                fd = self._wf_f32(
                                    self._wf_elem_bits((d0, d1, d2, d3)[j], h)
                                )
                                dead = dead | (cutlass.Int32(fa <= tf_f) << e)
                                dead = dead | (
                                    cutlass.Int32(fb <= tf_f) << (self.vec_elems + e)
                                )
                                dead = dead | (
                                    cutlass.Int32(fc <= tf_f)
                                    << (2 * self.vec_elems + e)
                                )
                                dead = dead | (
                                    cutlass.Int32(fd <= tf_f)
                                    << (3 * self.vec_elems + e)
                                )
                else:  # boundary: clamped addresses + validity bits
                    valid = cutlass.Int32(0)
                    for uu in cutlass.range_constexpr(4):
                        vi = vbase + cutlass.Int32(uu * self.nt)
                        ic = vi
                        if ic > lvm1:
                            ic = lvm1  # clamp (IMNMX); load is harmless
                        e0, e1, e2, e3 = ld_global_nc_v4_u32(
                            addr + cutlass.Int64(ic) * 16
                        )
                        if vi < lv:
                            valid = valid | (
                                cutlass.Int32((1 << self.vec_elems) - 1)
                                << (uu * self.vec_elems)
                            )
                            if cutlass.const_expr(self.wf_pair16):
                                dead = wf_dead_v4_16(
                                    dead,
                                    e0,
                                    e1,
                                    e2,
                                    e3,
                                    tf2,
                                    uu * 8,
                                    self.dtype == cutlass.BFloat16,
                                )
                            else:
                                for j in cutlass.range_constexpr(4):
                                    for h in cutlass.range_constexpr(epw):
                                        fv = self._wf_f32(
                                            self._wf_elem_bits((e0, e1, e2, e3)[j], h)
                                        )
                                        dead = dead | (
                                            cutlass.Int32(fv <= tf_f)
                                            << (uu * self.vec_elems + j * epw + h)
                                        )
                M = (~dead) & valid
                # warp-aggregated reservation: one atomic per warp
                cnt = cutlass.Int32(cute.arch.popc(M))
                inc = warp_inclusive_sum(cnt, lane)
                bpos = cutlass.Int32(0)
                if lane == 31:
                    if inc != 0:
                        bpos = smem_atomic_add(s_count, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                # survivor bit-walk (executes ~cnt times, cnt ~ 0-2)
                while M != 0:
                    bp = cutlass.Int32(
                        cute.arch.popc((M & (cutlass.Int32(0) - M)) - cutlass.Int32(1))
                    )
                    M = M & (M - cutlass.Int32(1))
                    fi = vbase + (bp >> cutlass.Int32(lg)) * cutlass.Int32(self.nt)
                    eidx = (
                        start
                        + (fi << cutlass.Int32(lg))
                        + (bp & cutlass.Int32(self.vec_elems - 1))
                    )
                    wbits = self.load_scalar(row_in, eidx)
                    ps = pos
                    if ps > self.lcap:
                        ps = cutlass.Int32(self.lcap)  # trash slot (IMNMX)
                    s_cv[ps] = wbits.bitcast(cutlass.Int32)
                    s_ci[ps] = eidx
                    cute.arch.red(
                        s_h256 + self._wf_bin(self._wf_f32(wbits), tf_f, sc),
                        cutlass.Int32(1),
                        op="add",
                        dtype="s32",
                        sem="relaxed",
                        scope="cta",
                    )
                    pos = pos + cutlass.Int32(1)
                vbase = vbase + cutlass.Int32(4 * self.nt)
                va = va + stride16 * 4
                it = it + cutlass.Int32(1)
            tail_base = lv * cutlass.Int32(self.vec_elems)
            for i in range(tidx, cnt_sl - tail_base, self.nt):
                idx = start + tail_base + i
                self._wf_elem(
                    self.load_scalar(row_in, idx),
                    idx,
                    tf_f,
                    sc,
                    s_count,
                    s_h256,
                    s_cv,
                    s_ci,
                )
        cute.arch.barrier()

    @cute.kernel
    def wf_topk_kernel(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        hints: cute.Tensor,
        has_hints: cutlass.Int32,
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
        hp_ptr = hints.iterator

        smem = SmemAllocator()
        # local candidate stage only (the epilogue reads candidates from
        # the gmem slab directly): LCAP pairs + 1 trash slot each (the
        # branchless overflow sink)
        s_cv = smem.allocate_array(cutlass.Int32, self.lcap + 4, byte_alignment=128)
        s_ci = smem.allocate_array(cutlass.Int32, self.lcap + 4, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tk = smem.allocate_array(cutlass.Uint32, self.tie_cap, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, self.tie_cap, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, self.nw, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)
        # cluster-merged row histogram (the per-CTA s_h256 must stay
        # intact while peers read it, so the merge lands here)
        s_hm = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        out_idx_row = oi_ptr + row64 * top_k
        slab_k = sl_ptr + row64 * WF_ROW_INTS
        slab_i = slab_k + GCAP
        slab_h = slab_k + 2 * GCAP  # row survivor histogram: 256 ints
        slab_t = slab_h + 256  # per-CTA publish tables (S > 1 gmem path)
        mc_row = mc_ptr + row64 * 8
        tel_k = cutlass.const_expr(self.wf_telemetry)
        # Sub-phase telemetry uses ONE running mark (ts_m): each mark writes
        # its delta straight into the status buffer instead of holding its
        # own timestamp.  The DSL traces even a const-false `if tel:` region
        # and carries every name assigned inside it as a region result, so
        # each extra Int64 timestamp is a live value in PRODUCTION too --
        # eight of them measured +0.3us at 256K b=1 on the register-bound
        # S=16 kernel.  Blocks: 16 entry->PDL, 17 PDL->ts0, 10-13 sample
        # sub-phases, 19 aim math, 21 tie select.
        ts_m = cutlass.Int64(0)
        if tel_k:
            ts_m = read_clock64()
        # PDL (SM90+, compiled out otherwise): the launch/prologue above may
        # overlap the previous kernel in the stream; wait before the first
        # global read (seqlen, hints, inputs, and this kernel's own self-
        # resetting slab/mc_state from the previous launch).
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()
        if tel_k:
            if tidx == 0:
                st_ptr[num_rows_dbg * 16 + row] = (read_clock64() - ts_m).to(
                    cutlass.Int32
                )
            ts_m = read_clock64()

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)

        if top_k >= length:
            if sl == 0:
                for i in range(tidx, top_k, self.nt):
                    if i < length:
                        out_idx_row[i] = cutlass.Int32(i)
                    else:
                        out_idx_row[i] = cutlass.Int32(-1)
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
                    st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
        else:
            # ---- SHORT-ROW ARM (per-row runtime gate, gvr_2's varlen
            # idiom: adapt the ALGORITHM per row under a frozen grid).
            # The long pipeline has a ~11us fixed floor (pre-sample,
            # k-derived candidate epilogue, publish/arrival, launch)
            # that does not scale with length: at length N/8 it barely
            # drops while the digit-family cost scales with length
            # (varlen sweep: radix_primitives 5.6us vs our 11.1 vs
            # gvr_2 7.5 on B200 64K-short).  For short rows slice 0
            # solves the row DIRECTLY with the exact MSD radix select
            # (~2 L2-hot passes at these lengths); other slices exit.
            # CTA-uniform (length is per-row), so barriers inside are
            # safe.
            if length <= cutlass.Int32(cutlass.const_expr(self.wf_small_n)):
                if sl == 0:
                    self._wf_short_row(
                        row_in,
                        length,
                        out_idx_row,
                        slab_k,
                        slab_i,
                        s_cv,
                        s_h256,
                        s_tk,
                        s_ti,
                        s_warp_sums,
                        s_misc,
                        s_count,
                        tidx,
                    )
                    # the overflow escape's harvest may dirty the slab's
                    # row-histogram tail; the long path requires zeros on
                    # entry (its publish red.adds accumulate)
                    for t in range(tidx, 256, self.nt):
                        slab_h[t] = cutlass.Int32(0)
                    if tidx == 0:
                        st_ptr[row] = cutlass.Int32(0)
                        st_ptr[num_rows_dbg + row] = cutlass.Int32(4)  # arm tag
            else:
                tel = cutlass.const_expr(self.wf_telemetry)
                # pre-define timestamps: the DSL's staged-control-flow rewriter
                # requires names assigned inside an if region to exist
                # beforehand, even under a const-false condition, and carries
                # them as region results (see ts_m at the kernel top) -- keep
                # this set minimal
                ts0 = cutlass.Int64(0)
                ts1 = cutlass.Int64(0)
                ts2 = cutlass.Int64(0)
                ts3 = cutlass.Int64(0)
                ts3a = cutlass.Int64(0)
                ts3b = cutlass.Int64(0)
                ts4 = cutlass.Int64(0)
                if tel:
                    ts0 = read_clock64()
                    if tidx == 0:  # block 17: PDL released -> ts0 (seqlen + geometry)
                        st_ptr[num_rows_dbg * 17 + row] = (ts0 - ts_m).to(cutlass.Int32)
                    ts_m = ts0
                # slice geometry (walk + sample share it)
                chunk = (
                    (length + cutlass.Int32(S) - 1) // cutlass.Int32(S)
                    + cutlass.Int32(self.vec_elems - 1)
                ) & ~cutlass.Int32(self.vec_elems - 1)
                start = cutlass.Int32(sl) * chunk
                cnt_sl = length - start
                if cnt_sl > chunk:
                    cnt_sl = chunk
                if cnt_sl < 0:
                    cnt_sl = cutlass.Int32(0)
                # ---- 0. HINT RUNG (stacked's rung 1 on this substrate) ----
                # With caller hints, TF is one key-step BELOW the smallest
                # hinted value (the walk keeps v > TF, so v == min_hint
                # must survive: the k hinted values are then all
                # survivors -> cand >= k by construction, never an
                # undershoot) and the survivor scale spans up to the
                # largest hinted value.  ~k/nt gathers + 2 barriers
                # replace the ~1.9us sample phase.  Stale hints (a rank-
                # (k+64) intruder) just widen cand by ~64; garbage hints
                # overflow the cand gate and take the exact fallback; NaN
                # or fewer than k in-range hints (varlen truncation) skip
                # the rung.  Replica-deterministic (identical gather on
                # every CTA).
                #
                # PLAUSIBILITY GATE.  "Garbage hints overflow the cand gate
                # and take the exact fallback" was a 6-8x cliff on the
                # multi-CTA forms (identity hints at 1M: 683us vs 90us
                # hintless on B200), because only the S==1 form has an
                # overflow rung and the gmem form (S>=8) cannot re-walk
                # after its last-arriver verdict.  So the hint is qualified
                # BEFORE the walk: every thread loads one vector at the same
                # row-uniform stride the fused sample below uses and counts
                # keys >= the smallest hinted key; the block sum scaled to
                # the row estimates the survivor count.  A previous-step
                # top-k puts ~k survivors there (about 4 sample hits at 1M,
                # k=1024, 4096 samples); garbage puts most of the row
                # (thousands of hits).  Reject above lcap/2 -- far above
                # any stale-but-real hint, far below any garbage one -- and
                # the row runs the sample path exactly as if hintless.
                # Identical positions on every CTA of the row, so the
                # verdict is replica-uniform; cost is one v4 load + one
                # warp redux + one barrier on top of the k gathers.
                #
                # REVISED (hints are not always correct): the hint no longer
                # REPLACES the sample -- it TIGHTENS it.  The fused sample
                # always runs; with k distinct in-range hints, the smallest
                # hinted value is <= the true k-th value (k distinct elements
                # cannot all rank above k), so one key-step below it is a
                # threshold that can never undershoot, and max(sample_tf,
                # hint_tf) can only shrink the candidate set.  An oracle hint
                # pins cand == k (the stage/classify saving the old rung had);
                # a stale hint lands below the sample's aim and changes
                # nothing; garbage is ignored -- so a hinted call can never be
                # slower than the hintless one by more than the k gathers,
                # which are issued alongside the sample's vector load and
                # folded through the sample's own barrier (no extra barriers).
                # Measured before this revision: a 90%-overlap stale hint cost
                # +2-3us over hintless (rung 2.9us + plausibility reject +
                # sample 2.1us), identity hints +1-2.5us.
                tf_f = cutlass.Float32(0.0)
                sc = cutlass.Float32(0.0)
                degen = cutlass.Int32(0)
                tf3 = cutlass.Float32(0.0)
                sc3 = cutlass.Float32(0.0)
                if True:  # (kept indentation of the former hint_ok == 0 arm)
                    # ---- 1. FUSED SAMPLE (gvr_2's streaming-main structure:
                    # 3 block barriers total, replica-deterministic, with walk
                    # tile 0's latency exposed at walk start).  The
                    # earlier separate _pre_scalars paid 5-6 barriers and two
                    # serial reduction rounds; the pieces below only work as a
                    # WHOLE (measured: priming or the warp fold bolted onto
                    # the old structure each regressed).  ----
                    lgk = cutlass.const_expr(self.vec_elems.bit_length() - 1)
                    epwk = cutlass.const_expr(self.vec_elems // 4)
                    n4s = length >> cutlass.Int32(lgk)
                    p4 = cutlass.Int32(
                        (cutlass.Int64(tidx) * cutlass.Int64(n4s))
                        // cutlass.Int64(self.nt)
                    )
                    if p4 > n4s - 1:
                        p4 = n4s - 1
                    if p4 < 0:
                        p4 = cutlass.Int32(0)
                    # hint gathers: index loads first so their latency overlaps
                    # the sample's vector load; the dependent value loads follow.
                    # Per-warp partials go to owned s_hm slots (free until the
                    # cluster epilogue): [0,32) max key, [32,64) max ~key
                    # (= min key), [64,96) in-range count.  No zero-init round.
                    hg_max = cutlass.Uint32(0)
                    hg_nmax = cutlass.Uint32(0)
                    hg_cnt = cutlass.Int32(0)
                    if cutlass.const_expr(self.wf_hint_rung):
                        if has_hints == cutlass.Int32(1):
                            hints_row = hp_ptr + row64 * top_k
                            for ht in range(tidx, top_k, self.nt):
                                hidx = hints_row[ht]
                                if hidx >= 0 and hidx < length:
                                    hk = self.exact_key(self.load_scalar(row_in, hidx))
                                    if hk > hg_max:
                                        hg_max = hk
                                    if ~hk > hg_nmax:
                                        hg_nmax = ~hk
                                    hg_cnt = hg_cnt + cutlass.Int32(1)
                    w0, w1, w2, w3 = ld_global_nc_v4_u32(
                        row_in.toint() + cutlass.Int64(p4) * cutlass.Int64(16)
                    )
                    # register key fold -> warp redux (fkey space) -> per-warp
                    # partials (owned slots: no zero-init round)
                    kk = self.exact_key(self._wf_elem_bits(w0, 0))
                    nk = ~kk
                    for j_ in cutlass.range_constexpr(4):
                        for h_ in cutlass.range_constexpr(epwk):
                            if cutlass.const_expr(j_ > 0 or h_ > 0):
                                k2 = self.exact_key(
                                    self._wf_elem_bits((w0, w1, w2, w3)[j_], h_)
                                )
                                if k2 > kk:
                                    kk = k2
                                n2 = ~k2
                                if n2 > nk:
                                    nk = n2
                    kk = warp_max_u32(kk)
                    nk = warp_max_u32(nk)
                    if tel:  # block 10: sample vector loaded + folded
                        if tidx == 0:
                            st_ptr[num_rows_dbg * 10 + row] = (
                                read_clock64() - ts_m
                            ).to(cutlass.Int32)
                        ts_m = read_clock64()
                    lane_s = tidx % 32
                    warp_s = tidx // 32
                    if lane_s == 0:
                        s_warp_sums[warp_s] = kk.bitcast(cutlass.Int32)
                        s_count[warp_s] = nk.bitcast(cutlass.Int32)
                    if cutlass.const_expr(self.wf_hint_rung):
                        if has_hints == cutlass.Int32(1):
                            hg_max = warp_max_u32(hg_max)
                            hg_nmax = warp_max_u32(hg_nmax)
                            hg_cnt = warp_sum(hg_cnt)
                            if lane_s == 0:
                                s_hm[warp_s] = hg_max.bitcast(cutlass.Int32)
                                s_hm[32 + warp_s] = hg_nmax.bitcast(cutlass.Int32)
                                s_hm[64 + warp_s] = hg_cnt
                    if tidx < 256:
                        s_h256[tidx] = cutlass.Int32(0)
                    if tel:
                        if tidx == 0:
                            s_misc[3] = cutlass.Int32(0)  # max per-thread walk time
                            s_misc[5] = cutlass.Int32(0)  # 0x7FFFFFFF - min walk time
                    cute.arch.barrier()  # B1: partials + hist zeros
                    # cross-warp fold: ONE lane-indexed load + ONE redux per
                    # value (a 32-iteration serial loop here was measured slow)
                    kmax = warp_max_u32(cutlass.Uint32(s_warp_sums[lane_s]))
                    kminc = warp_max_u32(cutlass.Uint32(s_count[lane_s]))
                    kmin = ~kminc
                    if tel:  # block 11: B1 + cross-warp fold
                        if tidx == 0:
                            st_ptr[num_rows_dbg * 11 + row] = (
                                read_clock64() - ts_m
                            ).to(cutlass.Int32)
                        ts_m = read_clock64()
                    smin_f = self._wf_val_of_key(kmin)
                    smax_f = self._wf_val_of_key(kmax)
                    span = smax_f - smin_f
                    degen = cutlass.Int32(0)
                    if kmin == kmax:
                        degen = cutlass.Int32(1)
                    if span <= cutlass.Float32(0.0):
                        degen = cutlass.Int32(1)
                    if degen == 0:
                        sc0 = cutlass.Float32(256.0) / span
                        for j_ in cutlass.range_constexpr(4):
                            for h_ in cutlass.range_constexpr(epwk):
                                vj = self._wf_f32(
                                    self._wf_elem_bits((w0, w1, w2, w3)[j_], h_)
                                )
                                bs = cutlass.Int32((vj - smin_f) * sc0)
                                if bs < 0:
                                    bs = cutlass.Int32(0)
                                if bs > 255:
                                    bs = cutlass.Int32(255)
                                smem_atomic_add(s_h256 + bs, 1)
                    cute.arch.barrier()  # B2: sample histogram
                    if tel:  # block 12: sample histogram + B2
                        if tidx == 0:
                            st_ptr[num_rows_dbg * 12 + row] = (
                                read_clock64() - ts_m
                            ).to(cutlass.Int32)
                        ts_m = read_clock64()
                    retry_cap = cutlass.const_expr(
                        self.mc_splits == 1 or (self.wf_cluster and self.mc_splits <= 4)
                    )
                    if cutlass.const_expr(retry_cap and self.wf_force_retry):
                        aim = cutlass.Int32(top_k) >> cutlass.Int32(1)
                        if aim < 1:
                            aim = cutlass.Int32(1)
                    elif cutlass.const_expr(retry_cap):
                        margin = cutlass.Int32(top_k) >> cutlass.Int32(1)
                        lm = length >> cutlass.Int32(8)
                        if lm > margin:
                            margin = lm
                        aim = cutlass.Int32(top_k) + margin
                        # Tail-safe floor for wide batches.  The count above
                        # the sample-derived bar is ~ N(aim, sqrt(aim*len/P)),
                        # so a fixed k/2 margin is only ~2.3 sigma at 256K
                        # k=2048 (r_aim = 48 samples): ~1% of rows undershoot
                        # and take the T3 re-walk (+16us).  With >= 32 rows in
                        # flight the wall time is the SLOWEST row, so most
                        # batches paid that tail (256K b=148 k=2048 measured
                        # bimodal 35 / 51us).  Raise the aim to z = 3.5 sigma:
                        # aim - k >= z*sqrt(aim*len/P)  <=>  sqrt(aim) >=
                        # (z*q + sqrt(z^2 q^2 + 4k)) / 2 with q = sqrt(len/P).
                        # Applied ONLY where the fixed margin is thin (below
                        # 2.5 sigma: k=2048 at 256K-512K, where N/256 < k/2);
                        # elsewhere (z >= 2.8 at 256K k=1024, 4.6 at 64K
                        # k=2048, 3.3 at 1M k=2048) the aim is unchanged, so
                        # no cell outside the band pays anything (a blanket
                        # 3.5-sigma floor measured +2-4% median on the wide
                        # cells it did not need to touch).  Small grids keep
                        # the lean aim (expected retry cost ~1% x 16us).
                        if cutlass.const_expr(self.wf_tail_safe):
                            if num_rows_dbg >= cutlass.Int32(32):
                                lenf = cutlass.Float32(length) / cutlass.Float32(
                                    self.psamp
                                )
                                mf = cutlass.Float32(aim - cutlass.Int32(top_k))
                                var = (
                                    cutlass.Float32(aim) * lenf
                                )  # sigma^2 of the count
                                if mf * mf < cutlass.Float32(6.25) * var:  # z < 2.5
                                    q = cmath.sqrt(lenf)
                                    zq = cutlass.Float32(3.5) * q
                                    s_ = (
                                        zq
                                        + cmath.sqrt(
                                            zq * zq + cutlass.Float32(4.0 * top_k)
                                        )
                                    ) * cutlass.Float32(0.5)
                                    aim_stat = cutlass.Int32(s_ * s_) + cutlass.Int32(1)
                                    if aim_stat > aim:
                                        aim = aim_stat
                    else:
                        aim = cutlass.Int32(top_k) * cutlass.Int32(2)
                        la = length >> cutlass.Int32(7)
                        if la > aim:
                            aim = la
                    if aim > cutlass.Int32(3 * SCAP // 4):
                        aim = cutlass.Int32(3 * SCAP // 4)
                    # ... and to the per-CTA smem stage: each CTA stages
                    # ~aim/S candidates into lcap slots.  Unclamped, N >= 2M
                    # aimed above the stage on EVERY row (9216 > 8192 at
                    # 2M), turning the whole batch into re-walks (S == 1)
                    # or exact fallbacks (S > 1) -- 3-4x wall time.
                    aim_cap = cutlass.Int32(
                        cutlass.const_expr(3 * self.lcap * self.mc_splits // 4)
                    )
                    if aim > aim_cap:
                        aim = aim_cap
                    r_aim = cutlass.Int32(
                        (cutlass.Int64(aim) * cutlass.Int64(self.psamp))
                        // cutlass.Int64(length)
                    )
                    if r_aim < 1:
                        r_aim = cutlass.Int32(1)
                    if r_aim > self.psamp:
                        r_aim = cutlass.Int32(self.psamp)
                    r3 = r_aim * cutlass.Int32(2)
                    if r3 > self.psamp:
                        r3 = cutlass.Int32(self.psamp)
                    # ONE warp-0 pass: both ladder crossings + free re-zero of
                    # s_h256 for the walk's survivor histogram.  (Two
                    # alternatives were built and MEASURED WORSE or equal on
                    # B200 at 64K: a 256-thread scan with a named barrier
                    # (0.73us, same) and a redundant per-warp scan without the
                    # B3 barrier (1.3us: 32 warps issuing the same ~900-cycle
                    # dependent chain saturate the 4 schedulers, so idling 31
                    # warps behind one barrier is the cheaper shape).)
                    self._wf_scan_cross0_2t(s_h256, r_aim, r3, tidx, s_misc)
                    cute.arch.barrier()  # B3: scan publish + zeros
                    if tel:  # block 13: crossing scan + B3
                        if tidx == 0:
                            st_ptr[num_rows_dbg * 13 + row] = (
                                read_clock64() - ts_m
                            ).to(cutlass.Int32)
                        ts_m = read_clock64()
                    bkt = s_misc[RES_B]
                    bkt3 = s_misc[RES_B2]
                    w_bin = span / cutlass.Float32(256.0)
                    tf_f = smin_f + cutlass.Float32(bkt) * w_bin
                    tf3 = smin_f + cutlass.Float32(bkt3) * w_bin
                    if degen == 1:
                        tf_f = smin_f  # unused: the walk is skipped under degen (cand = 0 -> fallback)
                        tf3 = smin_f
                    # survivor-hist span 1.5x past the SAMPLE max: the top
                    # bin saturates, and the ~N/4096 elements above the
                    # sample max (>k on ~2% of fp32 rows) otherwise pile
                    # into bin 255 -- when the rank-k crossing lands there
                    # and the pile exceeds the tie stage the row takes the
                    # exact fallback (measured: one row turned a 53us
                    # batch into 320us).  1.5x wider bins cost tens of
                    # extra ties in the rank-k bin; no cliff.
                    # ---- hint tightening (see the note above the sample) ----
                    # k distinct in-range hints -> one key-step below the
                    # smallest hinted value never undershoots; take it when it
                    # is above the sample's threshold and below the sample max
                    # (the survivor scale must keep a positive span).  Only the
                    # walk threshold moves; the T3 floor stays the sample's.
                    if cutlass.const_expr(self.wf_hint_rung):
                        if has_hints == cutlass.Int32(1):
                            if degen == 0:
                                nwarps_h = cutlass.const_expr(self.nt // 32)
                                hv_max = cutlass.Uint32(0)
                                hv_nmax = cutlass.Uint32(0)
                                hv_cnt = cutlass.Int32(0)
                                if lane_s < nwarps_h:
                                    hv_max = cutlass.Uint32(s_hm[lane_s])
                                    hv_nmax = cutlass.Uint32(s_hm[32 + lane_s])
                                    hv_cnt = s_hm[64 + lane_s]
                                hv_max = warp_max_u32(hv_max)
                                hv_min = ~warp_max_u32(hv_nmax)
                                hv_cnt = warp_sum(hv_cnt)
                                if hv_cnt == top_k and hv_min != hv_max and hv_min != 0:
                                    h_tf = self._wf_val_of_key(
                                        hv_min - cutlass.Uint32(1)
                                    )
                                    if h_tf > tf_f:
                                        if h_tf < smax_f:
                                            tf_f = h_tf
                    sspan = (smax_f - tf_f) * WF_SPAN_EXT
                    sc = cutlass.Float32(0.0)
                    if sspan > cutlass.Float32(0.0):
                        sc = cutlass.Float32(255.0) / sspan
                    sspan3 = (smax_f - tf3) * WF_SPAN_EXT
                    sc3 = cutlass.Float32(0.0)
                    if sspan3 > cutlass.Float32(0.0):
                        sc3 = cutlass.Float32(255.0) / sspan3
                if tel:
                    ts1 = read_clock64()
                    if tidx == 0:  # block 19: aim / threshold arithmetic
                        st_ptr[num_rows_dbg * 19 + row] = (ts1 - ts_m).to(cutlass.Int32)
                self._wf_walk_slice(
                    row_in,
                    start,
                    cnt_sl,
                    tf_f,
                    sc,
                    degen,
                    s_count,
                    s_h256,
                    s_cv,
                    s_ci,
                    tidx,
                )
                if tel:
                    ts2 = read_clock64()
                    # per-thread walk-time spread (straggler diagnosis)
                    wdt = cutlass.Uint32((ts2 - ts1).to(cutlass.Int32))
                    smem_red_max_u32(s_misc + 3, wdt)
                    smem_red_max_u32(s_misc + 5, cutlass.Uint32(0x7FFFFFFF) - wdt)

                if cutlass.const_expr(self.wf_cluster and S > 1):
                    # ---- CLUSTER EPILOGUE (the S CTAs of a row = one hw
                    # cluster).  DSMEM replaces the whole gmem coordination
                    # layer: counts and histograms are read from peer smem
                    # (one mapa per rank), classification is DISTRIBUTED --
                    # each CTA classifies the candidates it staged, winners
                    # go straight to the gmem output at a DSMEM cursor on
                    # rank 0, bin ties are DSMEM-staged into rank 0's tie
                    # arrays -- and rank 0 alone runs the key-exact tie
                    # sub-select or the fused fallback.  Phase telemetry
                    # showed the gmem path pays a flat ~1.8us hist gather
                    # + ~1us slab emit + arrival skew after the slowest
                    # CTA; DSMEM removes all three.  No SCAP bound here:
                    # capacity is per-CTA lcap only.  All locals are
                    # cl_-prefixed: a const-eliminated branch still
                    # registers its names, and reusing the gmem branch's
                    # names would break its staged-control-flow joins.
                    cl_local = s_count[0]
                    cl_failo = cutlass.Int32(0)
                    if cl_local > self.lcap:
                        cl_failo = cutlass.Int32(1)
                    if tidx == 0:
                        s_misc[12] = cl_local
                        s_misc[13] = cl_failo
                        s_misc[10] = cutlass.Int32(0)  # winner cursor (rank 0)
                        s_count[4] = cutlass.Int32(0)  # tie cursor (rank 0)
                    _cluster_sync_aligned()
                    if tel:
                        ts3 = read_clock64()
                        ts3a = ts3
                        ts3b = ts3
                    # replicated peer reads: identical result on every CTA,
                    # so cl_ok is cluster-uniform and the barriers inside
                    # the guarded region are safe
                    cl_cand = cutlass.Int32(0)
                    cl_failc = cutlass.Int32(0)
                    for cl_r in cutlass.range_constexpr(S):
                        cl_pm = _mapa_shared_cluster(s_misc, cutlass.Int32(cl_r))
                        cl_cand += _ld_shared_cluster_i32(cl_pm + cutlass.Int32(12 * 4))
                        cl_failc += _ld_shared_cluster_i32(
                            cl_pm + cutlass.Int32(13 * 4)
                        )
                    cl_ok = cutlass.Int32(0)
                    if degen == 0 and cl_failc == 0 and cl_cand >= top_k:
                        cl_ok = cutlass.Int32(1)
                    # ---- T3 retry rung (gvr_2 ladder, cluster form) ----
                    # Undershoot with clean staging: the tight aim missed k.
                    # The verdict is cluster-uniform (merged counts + degen
                    # are identical on every CTA), so every CTA re-walks at
                    # the T3 floor together -- the barriers inside the walk
                    # and the extra cluster sync cannot deadlock.  tf/sc are
                    # rebound so the epilogue's classify matches the
                    # retried histogram (single-binning-function rule).
                    cl_retry = cutlass.Int32(0)
                    if degen == 0 and cl_failc == 0 and cl_cand < top_k:
                        cl_retry = cutlass.Int32(1)
                    if cl_retry == 1:
                        tf_f = tf3
                        sc = sc3
                        self._wf_walk_slice(
                            row_in,
                            start,
                            cnt_sl,
                            tf_f,
                            sc,
                            degen,
                            s_count,
                            s_h256,
                            s_cv,
                            s_ci,
                            tidx,
                        )
                        cl_local = s_count[0]
                        cl_failo = cutlass.Int32(0)
                        if cl_local > self.lcap:
                            cl_failo = cutlass.Int32(1)
                        if tidx == 0:
                            s_misc[12] = cl_local
                            s_misc[13] = cl_failo
                        _cluster_sync_aligned()
                        cl_cand = cutlass.Int32(0)
                        cl_failc = cutlass.Int32(0)
                        for cl_r in cutlass.range_constexpr(S):
                            cl_pm = _mapa_shared_cluster(s_misc, cutlass.Int32(cl_r))
                            cl_cand += _ld_shared_cluster_i32(
                                cl_pm + cutlass.Int32(12 * 4)
                            )
                            cl_failc += _ld_shared_cluster_i32(
                                cl_pm + cutlass.Int32(13 * 4)
                            )
                        if degen == 0 and cl_failc == 0 and cl_cand >= top_k:
                            cl_ok = cutlass.Int32(1)
                    cl_binb = cutlass.Int32(0)
                    cl_above = cutlass.Int32(0)
                    if cl_ok == 1:
                        if tidx < 256:
                            cl_hv = cutlass.Int32(0)
                            for cl_r in cutlass.range_constexpr(S):
                                cl_ph = _mapa_shared_cluster(
                                    s_h256, cutlass.Int32(cl_r)
                                )
                                cl_hv += _ld_shared_cluster_i32(
                                    cl_ph + tidx * cutlass.Int32(4)
                                )
                            s_hm[tidx] = cl_hv
                        cute.arch.barrier()
                        self.scan256_and_find(
                            s_hm,
                            cl_cand,
                            cutlass.Int32(top_k),
                            s_warp_sums,
                            s_misc,
                            tidx,
                        )
                        # no post-scan barrier: scan256_and_find ends
                        # with a full-block barrier and the classify's
                        # smem traffic (DSMEM cursors zeroed before the
                        # first cluster sync, fresh s_tk/s_ti) is disjoint
                        cl_binb = s_misc[0]
                        cl_above = s_misc[1]
                        if tel:
                            ts3a = read_clock64()
                        # distributed classify of this CTA's OWN candidates
                        cl_wcur = _mapa_shared_cluster(
                            s_misc, cutlass.Int32(0)
                        ) + cutlass.Int32(10 * 4)
                        cl_tcur = _mapa_shared_cluster(
                            s_count, cutlass.Int32(0)
                        ) + cutlass.Int32(4 * 4)
                        cl_tk0 = _mapa_shared_cluster(s_tk, cutlass.Int32(0))
                        cl_ti0 = _mapa_shared_cluster(s_ti, cutlass.Int32(0))
                        for cl_t in range(tidx, cl_local, self.nt):
                            cl_wv = cutlass.Uint32(s_cv[cl_t])
                            cl_val = self._wf_f32(cl_wv)
                            cl_b = self._wf_bin(cl_val, tf_f, sc)
                            if cl_b > cl_binb:
                                cl_p = _atom_shared_cluster_add_i32(
                                    cl_wcur, cutlass.Int32(1)
                                )
                                if cl_p < top_k:
                                    out_idx_row[cl_p] = s_ci[cl_t]
                            else:
                                if cl_b == cl_binb:
                                    cl_e = _atom_shared_cluster_add_i32(
                                        cl_tcur, cutlass.Int32(1)
                                    )
                                    if cl_e < self.tie_cap:
                                        _st_shared_cluster_i32(
                                            cl_tk0 + cl_e * cutlass.Int32(4),
                                            self.exact_key(cl_wv).bitcast(
                                                cutlass.Int32
                                            ),
                                        )
                                        _st_shared_cluster_i32(
                                            cl_ti0 + cl_e * cutlass.Int32(4),
                                            s_ci[cl_t],
                                        )
                    _cluster_sync_aligned()
                    if tel:
                        ts3b = read_clock64()
                    if sl == 0:
                        if cl_ok == 1:  # CTA-uniform on rank 0
                            cl_nb = s_count[4]
                            cl_rem = cutlass.Int32(top_k) - cl_above
                            if cl_rem < 0:
                                cl_ok = cutlass.Int32(0)
                            if cl_rem > cl_nb:
                                cl_ok = cutlass.Int32(0)
                            if cl_nb > self.tie_cap:
                                cl_ok = cutlass.Int32(0)
                            if cl_ok == 1:
                                if cl_rem > 0:
                                    if cl_nb <= 128:
                                        self.tie_select(
                                            s_tk,
                                            s_ti,
                                            cl_nb,
                                            cl_above,
                                            cl_rem,
                                            s_cv,  # >= 512 ints scratch
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            out_idx_row,  # dummy: no values
                                            tidx,
                                        )
                                    else:
                                        # exact key bounds of the rank-k
                                        # value bin (one-bin float margin,
                                        # as in the gmem epilogue)
                                        cl_bw = cutlass.Float32(1.0)
                                        if sc > cutlass.Float32(0.0):
                                            cl_bw = cutlass.Float32(1.0) / sc
                                        cl_blo = (
                                            tf_f + cutlass.Float32(cl_binb - 1) * cl_bw
                                        )
                                        cl_bhi = cl_blo + cl_bw * (cutlass.Float32(3.0))
                                        cl_klo = self._wf_key_of_f32(cl_blo)
                                        cl_khi = self._wf_key_of_f32(cl_bhi)
                                        if cl_binb == 255:
                                            cl_khi = cutlass.Uint32(0xFFFFFFFF)
                                        self._tie_select_smem_skip_wf(
                                            s_tk,
                                            s_ti,
                                            cl_nb,
                                            cl_above,
                                            cl_rem,
                                            cl_khi,
                                            cl_klo,
                                            s_h256,
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            tidx,
                                        )
                        if cl_ok == 0:
                            # inline exact fallback (CTA-uniform cl_ok)
                            self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                                row_in,
                                length,
                                out_idx_row,
                                slab_k,
                                slab_i,
                                s_cv,
                                s_h256,
                                s_warp_sums,
                                s_misc,
                                s_count + 8,
                                s_count + 16,
                                tidx,
                            )
                        if tel:
                            ts4 = read_clock64()
                        if tidx == 0:
                            st_ptr[row] = cutlass.Int32(1) - cl_ok
                            st_ptr[num_rows_dbg + row] = cutlass.Int32(6)
                            st_ptr[num_rows_dbg * 2 + row] = cl_cand
                            st_ptr[num_rows_dbg * 3 + row] = cl_failc
                            if tel:
                                st_ptr[num_rows_dbg * 5 + row] = (ts1 - ts0).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 6 + row] = (ts2 - ts1).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 7 + row] = (ts4 - ts3b).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 8 + row] = (ts3a - ts3).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 9 + row] = (ts3b - ts3a).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 14 + row] = (ts3 - ts2).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 15 + row] = s_count[4]  # ties
                else:
                    # bulk publish: one range reservation + copies + 256 red.adds.
                    # S == 1 SHORT-CIRCUIT: the sole CTA is its own last arriver
                    # and s_cv/s_ci/s_h256 already hold the full row's survivors
                    # and histogram, so the slab round trip, the gmem histogram
                    # merge, and the arrival protocol are all skipped (measured
                    # ~1.5-2us of gmem atomics + copies + gather at b >= 128).
                    local = s_count[0]
                    staged = local
                    if staged > self.lcap:
                        staged = cutlass.Int32(self.lcap)
                    if cutlass.const_expr(S > 1):
                        # BIN-MAJOR PUBLISH.  Exclusive prefix of this CTA's
                        # 256-bin survivor histogram (warp 0, 8 bins/lane) ->
                        # s_hm; the per-CTA table [base, staged, prefix[0..256]]
                        # goes to the slab; then the staged candidates are
                        # scattered by bin (s_hm doubles as the per-bin cursor).
                        # The last arriver then copies bins > B as winners and
                        # bin B as ties by RANGE -- the per-candidate classify
                        # it used to run measured 2.5-3us per row (1M b <= 8).
                        if tidx == 0:
                            s_misc[10] = cutlass.Int32(0)
                            if local > 0:
                                gmem_atomic_add(mc_row + 0, local)  # exact count
                                s_misc[10] = cutlass.Int32(
                                    gmem_atomic_add(mc_row + 1, staged)
                                )
                        if tidx < 32:
                            bm_sum = cutlass.Int32(0)
                            for bm_j in cutlass.range_constexpr(8):
                                bm_sum += s_h256[tidx * 8 + bm_j]
                            bm_inc = warp_inclusive_sum(bm_sum, tidx)
                            bm_run = (
                                bm_inc - bm_sum
                            )  # exclusive prefix of this lane's 8 bins
                            for bm_j in cutlass.range_constexpr(8):
                                s_hm[tidx * 8 + bm_j] = bm_run
                                bm_run += s_h256[tidx * 8 + bm_j]
                            if tidx == 31:
                                s_misc[14] = bm_inc  # total survivors (== local)
                        cute.arch.barrier()
                        base = s_misc[10]
                        bm_tbl = slab_t + cutlass.Int32(sl) * WF_TBL
                        if tidx < 256:
                            bm_tbl[2 + tidx] = s_hm[tidx]
                        if tidx == 0:
                            bm_tbl[0] = base
                            bm_tbl[1] = staged
                            bm_tbl[2 + 256] = s_misc[14]
                        for t in range(tidx, staged, self.nt):
                            wv_t = cutlass.Uint32(s_cv[t])
                            b_t = self._wf_bin(self._wf_f32(wv_t), tf_f, sc)
                            pos_t = smem_atomic_add(s_hm + b_t, 1)
                            p = base + pos_t
                            if p < SCAP:
                                slab_k[p] = s_cv[t]
                                slab_i[p] = s_ci[t]
                        if tidx < 256:
                            hv = s_h256[tidx]
                            if hv > 0:
                                gmem_red_add(slab_h + tidx, hv)

                    # ---- 3. last-arriver epilogue (no spins anywhere) ----
                    cute.arch.barrier()
                    if tel:
                        ts3 = read_clock64()
                    if tidx == 0:
                        s_misc[11] = cutlass.Int32(0)
                        if cutlass.const_expr(S > 1):
                            # acq_rel: release orders this CTA's publish before
                            # its arrival; acquire makes every earlier arriver's
                            # publish visible to the last one -- ONE round trip
                            old = cute.arch.atomic_add(
                                mc_row + 2, cutlass.Int32(1), sem="acq_rel", scope="gpu"
                            )
                            if old == cutlass.Int32(S - 1):
                                s_misc[11] = cutlass.Int32(1)
                        else:
                            s_misc[11] = cutlass.Int32(1)
                    cute.arch.barrier()
                    if s_misc[11] == 1:
                        cand = local
                        slots = staged  # < cand iff s_cv overflowed => fallback
                        if cutlass.const_expr(S == 1):
                            # ---- T3 retry rung (gvr_2 ladder, S==1 form):
                            # CTA-uniform verdict; tf/sc rebound so the
                            # classify matches the retried histogram
                            if degen == 0 and cand < top_k and slots == cand:
                                tf_f = tf3
                                sc = sc3
                                self._wf_walk_slice(
                                    row_in,
                                    start,
                                    cnt_sl,
                                    tf_f,
                                    sc,
                                    degen,
                                    s_count,
                                    s_h256,
                                    s_cv,
                                    s_ci,
                                    tidx,
                                )
                                cand = s_count[0]
                                slots = cand
                                if slots > self.lcap:
                                    slots = cutlass.Int32(self.lcap)
                            # ---- overflow rung (S==1 form): the aim
                            # OVERSHOT the staging capacity (slots < cand).
                            # s_h256 is the complete survivor histogram
                            # (the walk bins every survivor, staged or
                            # not), so the rank-lcap crossing names a
                            # higher threshold with < lcap survivors by
                            # construction; re-walk there when it still
                            # covers k.  Before this rung a ~1.7x sample
                            # overshoot (3/256 fp32 rows at 1M, k=1024)
                            # cost the ~500us fused fallback per row.
                            if degen == 0 and slots < cand:
                                # rank lcap-64, not lcap: the retried walk's
                                # bin-edge rounding can move a few elements,
                                # and a landing at lcap+3 is another overflow
                                # (seen once on L40S at 1M b=256)
                                self.scan256_and_find(
                                    s_h256,
                                    cand,
                                    cutlass.Int32(self.lcap - 64),
                                    s_warp_sums,
                                    s_misc,
                                    tidx,
                                )
                                ob = s_misc[0]
                                oabove = s_misc[1]
                                if oabove >= top_k and ob < 254:
                                    # raise tf to bin ob's upper edge; keep
                                    # the 255-bin scale over [tf', smax]
                                    obw = cutlass.Float32(ob + 1)
                                    tf_f = tf_f + obw / sc
                                    sc = (
                                        sc
                                        * cutlass.Float32(255.0)
                                        / (cutlass.Float32(255.0) - obw)
                                    )
                                    self._wf_walk_slice(
                                        row_in,
                                        start,
                                        cnt_sl,
                                        tf_f,
                                        sc,
                                        degen,
                                        s_count,
                                        s_h256,
                                        s_cv,
                                        s_ci,
                                        tidx,
                                    )
                                    cand = s_count[0]
                                    slots = cand
                                    if slots > self.lcap:
                                        slots = cutlass.Int32(self.lcap)
                        if cutlass.const_expr(S > 1):
                            cand = cutlass.Int32(mc_row[0])
                            slots = cutlass.Int32(mc_row[1])
                            # histogram gather issued with the counts (one
                            # gmem round trip; final under the acquire above).
                            # (Copying all S publish tables into smem here was
                            # measured WORSE: 8 serialized loads per thread,
                            # +2.3us on the last arriver's chain.)
                            if tidx < 256:
                                s_h256[tidx] = cutlass.Int32(slab_h[tidx])
                            cute.arch.barrier()
                        if tel:
                            ts3a = ts3  # epilogue sub-phase marks (telemetry)
                            ts3b = ts3
                        ok = cutlass.Int32(0)
                        if (
                            degen == 0
                            and cand >= top_k
                            and cand <= SCAP
                            and slots == cand
                        ):
                            ok = cutlass.Int32(1)
                            # find rank-k bin over the row survivor histogram
                            # (at S == 1 s_h256 IS the row histogram; at S > 1
                            # it was gathered above)
                            self.scan256_and_find(
                                s_h256,
                                cand,
                                cutlass.Int32(top_k),
                                s_warp_sums,
                                s_misc,
                                tidx,
                            )
                            # no post-scan barrier: scan256_and_find
                            # ends with a full-block barrier; the cursor
                            # zeroing below is fenced by its own barrier
                            binb = s_misc[0]
                            above = s_misc[1]
                            if tel:
                                ts3a = read_clock64()  # hist gather + scan done
                            # emit winners above bin B; stage bin-B ties for the
                            # exact key-space sub-select
                            if tidx == 0:
                                s_misc[10] = cutlass.Int32(0)  # winner cursor
                                s_count[4] = cutlass.Int32(0)  # tie cursor
                            cute.arch.barrier()
                            if cutlass.const_expr(S == 1):
                                # candidates never left shared memory: classify
                                # straight from s_cv/s_ci with one same-address
                                # shared atomic per winner / tie.  Two warp-
                                # aggregated forms were built here and MEASURED
                                # SLOWER on B200 (64K, k=2048, ~3.1k candidates):
                                # a 5-step shuffle scan per warp (1.1 -> 2.0us)
                                # and ballot + popc + lane-0 atomic + shuffle
                                # (1.1 -> 1.5us).  The hardware already
                                # coalesces same-address shared atomics well
                                # enough that the extra collectives and the
                                # uniform-trip-count loop cost more than they
                                # save; keep the plain form.
                                for t in range(tidx, cand, self.nt):
                                    wv = cutlass.Uint32(s_cv[t])
                                    val = self._wf_f32(wv)
                                    b = self._wf_bin(val, tf_f, sc)
                                    if b > binb:
                                        p = smem_atomic_add(s_misc + 10, 1)
                                        if p < top_k:
                                            out_idx_row[p] = s_ci[t]
                                    else:
                                        if b == binb:
                                            t2 = smem_atomic_add(s_count + 4, 1)
                                            if t2 < self.tie_cap:
                                                s_tk[t2] = self.exact_key(wv)
                                                s_ti[t2] = s_ci[t]
                            else:
                                # RANGE COPY from the bin-major slab.  Per CTA r:
                                # winners = [base_r + pre_r[B+1], base_r + staged_r),
                                # ties    = [base_r + pre_r[B],   base_r + pre_r[B+1]).
                                # Lane r < S of warp 0 loads its CTA's four table
                                # words (one round trip), a warp scan gives the
                                # output offsets, then every thread copies one
                                # winner index / one tie (key + index) per step.
                                bm_win = cutlass.Int32(0)
                                bm_tie = cutlass.Int32(0)
                                bm_wsrc = cutlass.Int32(0)
                                bm_tsrc = cutlass.Int32(0)
                                if tidx < 32:
                                    if tidx < cutlass.Int32(S):
                                        bm_t = (
                                            slab_t + tidx * WF_TBL
                                        )  # lane r reads CTA r's table
                                        bm_base = bm_t[0]
                                        bm_stg = bm_t[1]
                                        bm_pb = bm_t[2 + binb]
                                        bm_pb1 = bm_t[3 + binb]
                                        bm_wsrc = bm_base + bm_pb1
                                        bm_win = bm_stg - bm_pb1
                                        bm_tsrc = bm_base + bm_pb
                                        bm_tie = bm_pb1 - bm_pb
                                        if bm_win < 0:
                                            bm_win = cutlass.Int32(0)
                                        if bm_tie < 0:
                                            bm_tie = cutlass.Int32(0)
                                    bm_winc = warp_inclusive_sum(bm_win, tidx)
                                    bm_tinc = warp_inclusive_sum(bm_tie, tidx)
                                    # per-CTA descriptors -> smem (s_hm is free here)
                                    s_hm[tidx] = bm_winc - bm_win  # winner out offset
                                    s_hm[32 + tidx] = bm_win
                                    s_hm[64 + tidx] = bm_wsrc
                                    s_hm[96 + tidx] = (
                                        bm_tinc - bm_tie
                                    )  # tie stage offset
                                    s_hm[128 + tidx] = bm_tie
                                    s_hm[160 + tidx] = bm_tsrc
                                    if tidx == 31:
                                        s_hm[192] = bm_winc  # total winners
                                        s_hm[193] = bm_tinc  # total ties
                                cute.arch.barrier()
                                bm_nw = s_hm[192]
                                bm_nt = s_hm[193]
                                # winners: one index per thread per step
                                for g in range(tidx, bm_nw, self.nt):
                                    bm_r = cutlass.Int32(0)
                                    for rr in cutlass.range_constexpr(S):
                                        if g >= s_hm[rr]:
                                            bm_r = cutlass.Int32(rr)
                                    if g < top_k:
                                        out_idx_row[g] = slab_i[
                                            s_hm[64 + bm_r] + (g - s_hm[bm_r])
                                        ]
                                # ties: key + index into the tie stage
                                for g in range(tidx, bm_nt, self.nt):
                                    bm_r = cutlass.Int32(0)
                                    for rr in cutlass.range_constexpr(S):
                                        if g >= s_hm[96 + rr]:
                                            bm_r = cutlass.Int32(rr)
                                    if g < self.tie_cap:
                                        bm_src = s_hm[160 + bm_r] + (
                                            g - s_hm[96 + bm_r]
                                        )
                                        s_tk[g] = self.exact_key(
                                            cutlass.Uint32(slab_k[bm_src])
                                        )
                                        s_ti[g] = slab_i[bm_src]
                                if tidx == 0:
                                    s_misc[10] = bm_nw
                                    s_count[4] = bm_nt
                            cute.arch.barrier()
                            if tel:
                                ts3b = read_clock64()  # emit pass done
                            nb = s_count[4]  # bin-B members (== hist[binb] if fit)
                            remaining = top_k - above
                            if tel:
                                ts_m = read_clock64()  # tie select start
                            if remaining < 0 or remaining > nb or nb > self.tie_cap:
                                ok = cutlass.Int32(0)  # bin-B overflow: fallback
                            else:
                                if remaining > 0:
                                    if nb <= 128:
                                        self.tie_select(
                                            s_tk,
                                            s_ti,
                                            nb,
                                            above,
                                            remaining,
                                            s_cv,  # >= 512 ints scratch
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            out_idx_row,  # dummy: has_values False
                                            tidx,
                                        )
                                    else:
                                        # bin-B members' VALUES lie in one narrow
                                        # bin, so their KEYS share high bytes
                                        # (exact_key is monotone): pass the bin's
                                        # key bounds so the leading radix rounds
                                        # skip (~0.8us of barriers each)
                                        # one-bin safety margin each side: the
                                        # members were binned by (val-tf)*sc
                                        # truncation, whose rounding can land a
                                        # member marginally outside the
                                        # reconstructed edge floats
                                        binw = cutlass.Float32(1.0)
                                        if sc > cutlass.Float32(0.0):
                                            binw = cutlass.Float32(1.0) / sc
                                        blo_f = tf_f + cutlass.Float32(binb - 1) * binw
                                        bhi_f = blo_f + binw * cutlass.Float32(3.0)
                                        kmin_b = self._wf_key_of_f32(blo_f)
                                        kmax_b = self._wf_key_of_f32(bhi_f)
                                        if binb == 255:
                                            kmax_b = cutlass.Uint32(
                                                0xFFFFFFFF
                                            )  # NaN top
                                        self._tie_select_smem_skip_wf(
                                            s_tk,
                                            s_ti,
                                            nb,
                                            above,
                                            remaining,
                                            kmax_b,
                                            kmin_b,
                                            s_h256,
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            tidx,
                                        )
                            if tel:  # block 21: tie select alone
                                if tidx == 0:
                                    st_ptr[num_rows_dbg * 21 + row] = (
                                        read_clock64() - ts_m
                                    ).to(cutlass.Int32)
                        if ok == 0:
                            # ---- inline exact fallback (fused; no 2nd launch) ----
                            # CTA-uniform (ok derives from uniform smem/gmem reads),
                            # so the block barriers inside are safe.  smem aliasing:
                            # s_cv (LCAP+4 >= 4096 ints, dead in the failure path)
                            # becomes the 4096-bin key histogram; s_count slots
                            # 8/16 become the gt/eq ticket counters.  Harvest may
                            # write slab_k[SCAP..], overlapping the row-histogram
                            # tail -- dead here, and the self-reset below re-zeroes
                            # it AFTER this call (order is load-bearing: the next
                            # call's red.adds need zeros).
                            self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                                row_in,
                                length,
                                out_idx_row,
                                slab_k,
                                slab_i,
                                s_cv,
                                s_h256,
                                s_warp_sums,
                                s_misc,
                                s_count + 8,
                                s_count + 16,
                                tidx,
                            )
                        if tel:
                            ts4 = read_clock64()
                        if tidx == 0:
                            st_ptr[row] = cutlass.Int32(1) - ok
                            st_ptr[num_rows_dbg + row] = cutlass.Int32(
                                5 if S == 1 else 3  # family tag (5 = S1 direct)
                            )
                            st_ptr[num_rows_dbg * 2 + row] = cand
                            st_ptr[num_rows_dbg * 3 + row] = slots
                            if tel:
                                # phase telemetry (last-arriver CTA's own view, ns):
                                # pre / walk / select+resets / hist+scan / emit pass
                                st_ptr[num_rows_dbg * 5 + row] = (ts1 - ts0).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 6 + row] = (ts2 - ts1).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 7 + row] = (ts4 - ts3b).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 8 + row] = (ts3a - ts3).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 9 + row] = (ts3b - ts3a).to(
                                    cutlass.Int32
                                )
                                # epilogue wait: walk end -> last-arriver verdict
                                st_ptr[num_rows_dbg * 14 + row] = (ts3 - ts2).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 15 + row] = s_count[4]  # ties
                                st_ptr[num_rows_dbg * 18 + row] = (
                                    read_clock64() - ts4
                                ).to(cutlass.Int32)  # select end -> status writes done
                                # per-thread walk time: max / min across the CTA
                                st_ptr[num_rows_dbg * 4 + row] = s_misc[3]
                                st_ptr[num_rows_dbg * 20 + row] = cutlass.Int32(
                                    cutlass.Uint32(0x7FFFFFFF)
                                    - cutlass.Uint32(s_misc[5])
                                )
                            # self-reset: counters + the row histogram (S == 1
                            # never touched mc_row or slab_h -- nothing to reset)
                            if cutlass.const_expr(S > 1):
                                mc_row[0] = cutlass.Int32(0)
                                mc_row[1] = cutlass.Int32(0)
                                mc_row[2] = cutlass.Int32(0)
                        if cutlass.const_expr(S > 1):
                            for t in range(tidx, 256, self.nt):
                                slab_h[t] = cutlass.Int32(0)

        # PDL: release the dependent grid only at the very end.  Releasing
        # after the walk let the next launch's CTAs co-schedule during our
        # epilogue and measured +2-3% at 256K b=64; at the end PDL still hides
        # the next launch's latency (measured -0.4..-0.6us on most cells).
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()

    @cute.jit
    def _tie_select_smem_skip_wf(
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
        """Byte-radix rank-select over the smem tie stage (full 4 rounds;
        k_hi/k_lo passed as full-range sentinels here, so no rounds skip
        -- kept as a separate name to avoid perturbing the MC module's
        compiled kernels)."""
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
        for i in range(tidx, eq, self.nt):
            kk = cutlass.Uint32(s_tk[i])
            if kk > prefix:
                p = smem_atomic_add(s_misc + 10, 1)
                out_idx_row[gt + p] = s_ti[i]
            else:
                if kk == prefix:
                    e = smem_atomic_add(s_misc + 11, 1)
                    if e < need:
                        out_idx_row[gt + nab + e] = s_ti[i]


class ProdWalkFirstTopK(WalkFirstTopK, GatedExactFallback):
    """Walk-first kernel + gated exact fallback, one compiled launcher."""

    @cute.jit
    def launch_prod(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        hints: cute.Tensor,
        has_hints: cutlass.Int32,
        stream,
    ):
        num_rows = input_data.shape[0]
        # min_blocks_per_mp=1: __launch_bounds__(1024, 1).  SASS inspection
        # showed that WITHOUT a bound, ptxas' occupancy heuristic squeezed
        # this kernel to REG:32 with a 24-byte SPILL stack (targeting
        # 2 blocks/SM) -- LDL/STL in the walk loop.  Declaring 1 block/SM
        # intended frees the full 64-register budget for the quad-buffered
        # walker.  (min_blocks_per_mp=2 is the same squeeze, explicitly;
        # measured slower.)
        if cutlass.const_expr(self.wf_cluster and self.mc_splits > 1):
            # hw cluster along Y: the S slices of a row share DSMEM
            self.wf_topk_kernel(
                input_data,
                seqlen,
                output_indices,
                slab,
                status,
                mc_state,
                hints,
                has_hints,
            ).launch(
                grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
                block=(self.nt, 1, 1),
                cluster=(1, cutlass.const_expr(self.mc_splits), 1),
                stream=stream,
                use_pdl=self.enable_pdl,
                min_blocks_per_mp=(1 if self.nt == 1024 else 2),
            )
        else:
            self.wf_topk_kernel(
                input_data,
                seqlen,
                output_indices,
                slab,
                status,
                mc_state,
                hints,
                has_hints,
            ).launch(
                grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
                block=(self.nt, 1, 1),
                stream=stream,
                use_pdl=self.enable_pdl,
                min_blocks_per_mp=(1 if self.nt == 1024 else 2),
            )
        # No fallback launch: the exact fallback is FUSED into the wf
        # kernel's last-arriver epilogue (a failed row re-solves inline).
        # Per-kernel profiling showed the always-launched gate kernel
        # cost a flat ~1.65us -- the entire b=16 deficit vs gvr_2.


_compiled: dict = {}


def get_walkfirst_kernel(
    top_k: int, N: int, splits: int, telemetry: bool = False, dtype=None
):
    """Compile (with on-disk caching) the walk-first kernel + gated
    fallback for a (top_k, N, splits) specialization.  mc_state (rows, 8)
    int32 AND the slab must be zero-initialised at first use (the row
    histogram lives in the slab tail and self-resets afterwards).

    telemetry=True (or FLASHINFER_TOPK_WF_TELEMETRY=1) compiles the
    phase-instrumented variant (globaltimer reads + status blocks 5-9);
    production kernels carry no instrumentation."""
    telemetry = (
        telemetry
        or os.environ.get("FLASHINFER_TOPK_WF_TELEMETRY") == "1"
        or os.environ.get("FLASHINFER_TOPK_PRIM_TELEMETRY") == "1"
    )
    # nt=512 was built and MEASURED as a k<=1024 family: no win anywhere
    # (walk phase identical -- the 2-CTAs/SM shape buys no bandwidth --
    # while publish/arrival overhead doubles), and tie_cap = 2*nt = 1024
    # overflows on randn's threshold-adjacent value bins (4/64 rows fell
    # back at 1M).  gvr_2's small-k edge is its leaner fused pipeline,
    # not block shape.  The parameterization stays as an experimentation
    # hook; production uses 1024 threads at every k.
    #
    # Wide batches (b > SMs) are the case where block shape WOULD matter: a
    # 1024-thread CTA at 64 regs fills the register file, so one CTA per SM
    # and a 256-row batch on 148 SMs runs as two waves, while gvr_2 routes
    # those batches to 256/512-thread CTAs and wins 1.3-1.6x at 32K-128K.
    # Measured 2026-09-02: forcing nt=512 here is NOT a usable shortcut --
    # the walk/cluster/fallback paths carry 1024-thread assumptions beyond
    # tie_cap (wrong results at k=512 S=2 without any fallback, mass
    # fallbacks above 32K), so a 512-thread variant is a real port.
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
    assert top_k <= max(2 * nt, 2048) and N % vec_elems == 0
    assert splits in (1, 2, 3, 4, 6, 8, 16, 32)
    # cs=8 was built and MEASURED WORSE (B200 1M b=16: 18.1 -> 28.3us):
    # an 8-CTA cluster must pack into one GPC, and at 1 CTA/SM occupancy
    # the scheduler strands SMs waiting to co-place whole clusters.  cs=2
    # is a clear win (64K b=64 12.98 -> 10.73) and cs=4 neutral-positive;
    # S=8/16 keep the gmem slab + release/acquire arrival path.
    # DSMEM cluster shapes: 2 and 4 (measured on B200), 3 and 6 (one-wave
    # shapes the dispatcher picks for long rows on 208-SM parts; measured on
    # Rubin: 1M b=64 S3 43.2 -> 38.2us, 1M b=32 S6 24.4 -> 22.3us).  8 is
    # experimental only (FLASHINFER_TOPK_WF_CLUSTERX=1): measured worse
    # everywhere on both parts.
    _cl_sizes: tuple[int, ...] = (2, 3, 4, 6)
    if os.environ.get("FLASHINFER_TOPK_WF_CLUSTERX") == "1":
        _cl_sizes = (2, 3, 4, 6, 8)
    use_cluster = (
        splits in _cl_sizes
        and os.environ.get("FLASHINFER_TOPK_WF_CLUSTER") != "0"
        and torch.cuda.get_device_capability()[0] >= 9
    )
    small_n = int(os.environ.get("FLASHINFER_TOPK_WF_SMALL_N", WF_SMALL_N))
    force_retry_env = os.environ.get("FLASHINFER_TOPK_WF_FORCE_RETRY") == "1"
    hint_rung_env = os.environ.get("FLASHINFER_TOPK_USE_HINTS") == "1"
    # packed-pair 16-bit classify: f16x2 setp exists on every supported
    # arch, bf16x2 setp needs sm_90 (older arches keep the scalar path);
    # FLASHINFER_TOPK_WF_PAIR16=0 is the A/B switch
    pair16 = (
        cdt == cutlass.Float16
        or (cdt == cutlass.BFloat16 and torch.cuda.get_device_capability()[0] >= 9)
    ) and os.environ.get("FLASHINFER_TOPK_WF_PAIR16") != "0"
    key = (
        top_k,
        N,
        splits,
        telemetry,
        nt,
        use_cluster,
        small_n,
        force_retry_env,
        hint_rung_env,
        dt_tag,
        pair16,
        os.environ.get("FLASHINFER_TOPK_WF_LCAP"),
        os.environ.get("FLASHINFER_TOPK_WF_TAILSAFE"),
        os.environ.get("FLASHINFER_TOPK_WF_PDL"),
        torch.cuda.get_device_capability()[0],
    )
    if key in _compiled:
        return _compiled[key]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    # programmatic dependent launch: griddepcontrol is SM90+; A/B switch
    # FLASHINFER_TOPK_WF_PDL=0
    use_pdl = (
        torch.cuda.get_device_capability()[0] >= 9
        and os.environ.get("FLASHINFER_TOPK_WF_PDL") != "0"
    )
    kern = ProdWalkFirstTopK(
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
    kern.mc_splits = splits
    # the tie stage must hold a full rank-k bin regardless of block shape
    # (the base class sizes it 2*nt; the 512-thread shape needs >= k)
    kern.tie_cap = max(2 * nt, top_k)
    kern.wf_telemetry = telemetry
    kern.wf_cluster = use_cluster
    force_retry = os.environ.get("FLASHINFER_TOPK_WF_FORCE_RETRY") == "1"
    kern.wf_force_retry = force_retry
    # Hints are OFF by default (FLASHINFER_TOPK_USE_HINTS=1 enables): with
    # realistic previous-step hints (85-90% overlap) the k value gathers cost
    # 0.6-1.4us and the tightened threshold saves less, so a hinted call
    # measured slower than hintless everywhere except 1M b=64 (-5%).  Exact
    # hints win up to 1.4us at 16K, but a caller cannot know its hint is
    # exact, and "never slower than hintless" is the contract we keep.
    hint_rung = os.environ.get("FLASHINFER_TOPK_USE_HINTS") == "1"
    kern.wf_hint_rung = hint_rung
    tail_safe = os.environ.get("FLASHINFER_TOPK_WF_TAILSAFE") != "0"
    kern.wf_tail_safe = tail_safe
    kern.wf_small_n = int(os.environ.get("FLASHINFER_TOPK_WF_SMALL_N", WF_SMALL_N))
    # candidate stage: 16384 entries when the device's opt-in shared memory
    # per block allows (~147KB kernel: B200/H100 228KB, A100 164KB), else 8192
    # (L40S, RTX 5080 ~100KB).  FLASHINFER_TOPK_WF_LCAP=8192|16384 overrides.
    from ...utils import get_shared_bytes_per_block_optin

    # Only the kernels that can overflow get the big stage: single-worker
    # rows (S == 1) at N >= 1M.  Split-row and short-row kernels keep 8192:
    # they never overflow, and the larger shared-memory carveout costs
    # ~0.4us per row (less L1 for the survivor re-reads) -- 3-4% at 64K.
    lcap_sel = LCAP
    if (
        splits == 1
        and N >= (1 << 20)
        and get_shared_bytes_per_block_optin(torch.device("cuda")) >= LCAP_BIG_SMEM
    ):
        lcap_sel = LCAP_BIG
    if os.environ.get("FLASHINFER_TOPK_WF_LCAP") in ("8192", "16384"):
        lcap_sel = int(os.environ["FLASHINFER_TOPK_WF_LCAP"])
    kern.lcap = lcap_sel if nt == 1024 else 4096
    kern.psamp = vec_elems * nt  # one 16B probe per thread
    kern.wf_pair16 = pair16
    sym_rows = cute.sym_int()
    f32, i32 = cdt, cutlass.Int32  # f32 alias = logits dtype

    def _fk(dt, shape, align=None):
        so = tuple(range(len(shape) - 1, -1, -1))
        if align is None:
            return cute.runtime.make_fake_compact_tensor(dt, shape, stride_order=so)
        return cute.runtime.make_fake_compact_tensor(
            dt, shape, stride_order=so, assumed_align=align
        )

    def _compile_fn():
        return cute.compile(
            kern.launch_prod,
            _fk(f32, (sym_rows, N), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, top_k), 16),
            _fk(i32, (sym_rows, WF_ROW_INTS), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, 8), 16),
            _fk(i32, (sym_rows, top_k), 16),
            cutlass.Int32(0),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = build_and_load_cute_dsl_kernel(
        "walkfirst_topk_primitives",
        f"wf_v1_{dt_tag}_k{top_k}_N{N}_S{splits}"
        f"{'_nt512' if nt == 512 else ''}{'_tel' if telemetry else ''}"
        f"{'_clu' if use_cluster else ''}"
        f"{'' if small_n == WF_SMALL_N else f'_sn{small_n}'}"
        f"{'_fr' if force_retry else ''}{'' if hint_rung else '_nh'}"
        f"{'_np' if (cdt != cutlass.Float32 and not pair16) else ''}"
        f"{'_L16k' if kern.lcap == LCAP_BIG else ''}"
        f"{'' if tail_safe else '_nts'}{'_pdl' if use_pdl else ''}",
        _compile_fn,
        extra_key_files=(
            __file__,
            _sampled_mod.__file__,
            _radix_mod.__file__,
            _fallback_mod.__file__,
            _gvr2_mod.__file__,  # DSMEM/cluster op set
        ),
    )
    _compiled[key] = compiled
    return compiled
