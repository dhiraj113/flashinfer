"""Status-gated exact fallback for the experimental primitives selectors.

The production-plumbing companion to the sampled / hint / stacked kernels:
instead of a host status readback (a device->host sync every call, and a
CUDA-graph capture breaker), the dispatcher ALWAYS enqueues this kernel
right after the speculative one.  Each CTA reads its row's status word
and exits in nanoseconds when the row already succeeded (the overwhelming
common case); a failed row is re-solved from scratch, exactly, on device.
The whole two-launch sequence is sync-free and CUDA-graph capturable.

Algorithm for a failed row: MSD radix-select with 12-bit digits over the
ordered fp32 key space (the DIGIT family -- certainty by construction):

  1. up to 3 histogram passes: 4096 shift-binned key buckets over the
     current candidate range, descending crossing-find at the remaining
     rank, recurse into the crossing bucket.  32-bit keys / 12 bits per
     round means round 3's buckets are single keys, so the loop PROVABLY
     terminates at either a bucket of <= GCAP candidates or a width-1
     (single-key) bucket -- no data-dependent escape hatch needed.
  2. one harvest pass with KEY-space compares (total order, so +/-inf and
     NaN are unambiguous; NaN keys sort above +inf, matching the torch
     ordering the other backends produce): key > bucket_hi -> emit via
     ticket cursor, key in bucket -> (key, idx) into the gmem slab.
  3. finish: single-key bucket -> fill by count (ties value-equal);
     else exact byte-radix rank-select over the <= GCAP slab entries.

Worst case is 4 row passes, data-independent.  Scope matches the
speculative kernels: fp32, next_n == compress_ratio == 1, k <= TIE_CAP,
single CTA per row.
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator

from . import hint_topk_primitives as _hint_mod
from . import sampled_topk_primitives as _sampled_mod
from . import stacked_topk_primitives as _stacked_mod
from .hint_topk_primitives import HintPivotTopK
from . import radix_topk_primitives as _radix_mod
from .radix_topk_primitives import (
    NUM_THREADS,
    NUM_WARPS,
    TIE_CAP,
    smem_atomic_add,
)
from .sampled_topk_primitives import GCAP, SampledPivotTopK
from .stacked_topk_primitives import StackedTopK


class GatedExactFallback(SampledPivotTopK):
    """See module docstring.  Inherits the slab select and key helpers."""

    @cute.kernel
    def fallback_topk_kernel(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])

        st_ptr = status.iterator
        # the gate: rows the speculative kernel solved cost one gmem read
        if st_ptr[row] != 0:
            in_ptr = input_data.iterator
            seq_ptr = seqlen.iterator
            oi_ptr = output_indices.iterator
            sl_ptr = slab.iterator

            smem = SmemAllocator()
            s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
            s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
            s_warp_sums = smem.allocate_array(
                cutlass.Int32, NUM_WARPS, byte_alignment=128
            )
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

            # a failed speculative row always has length > top_k (the
            # degenerate arm writes status = 0), but guard anyway
            if top_k >= length:
                for i in range(tidx, top_k, NUM_THREADS):
                    if i < length:
                        out_idx_row[i] = cutlass.Int32(i)
                    else:
                        out_idx_row[i] = cutlass.Int32(-1)
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
            else:
                self._fallback_row(
                    row_in,
                    length,
                    out_idx_row,
                    slab_k,
                    slab_i,
                    s_h4k,
                    s_h256,
                    s_warp_sums,
                    s_misc,
                    s_count_gt,
                    s_count_eq,
                    tidx,
                )
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)

    @cute.jit
    def _fallback_row(
        self,
        row_in,
        length,
        out_idx_row,
        slab_k,
        slab_i,
        s_h4k,
        s_h256,
        s_warp_sums,
        s_misc,
        s_count_gt,
        s_count_eq,
        tidx,
    ):
        """Exact MSD radix-select re-solve of one failed row (see module
        docstring).  Requires length > top_k and a whole CTA executing
        under a CTA-uniform condition (block barriers inside).  s_h4k
        needs hist_size ints (4096 for fp32, 8192 for the 16-bit dtypes)
        -- callers may alias any dead smem array of at least that size
        (the fused walk-first epilogue passes its LCAP+4 candidate stage).
        Also used inline by wf_topk_kernel so a failed row costs no
        second kernel launch."""
        top_k = cutlass.const_expr(self.top_k)
        # ---- MSD radix-select: <= 3 key-histogram passes ----
        lo_k = cutlass.Uint32(0)
        hi_k = cutlass.Uint32(0xFFFFFFFF)
        need = cutlass.Int32(top_k)  # descending rank within range
        done = cutlass.Int32(0)
        for _round in cutlass.range_constexpr(3):
            if done == 0:  # block-uniform: barriers inside are safe
                # Histogram geometry MUST follow the substrate's hist_size:
                # find_threshold_wide scans hist_items = hist_size / nt bins
                # per thread (fp32: 4096 bins; 16-bit dtypes: 8192).  A
                # hardcoded 4096 here left bins 4096..8191 un-zeroed for
                # 16-bit rows -- and s_h4k aliases the walk's candidate
                # stage, so any row that staged > 4096 survivors fed the
                # find stale survivor VALUES as counts (garbage crossing,
                # empty harvest, zero-filled output).
                for zz in cutlass.range_constexpr(self.hist_size // self.nt):
                    s_h4k[tidx + cutlass.Int32(zz * self.nt)] = cutlass.Int32(0)
                cute.arch.barrier()
                # shift binning, max bin index bounded (span=2^k+1
                # would otherwise write one past the histogram)
                span = cutlass.Int64(hi_k - lo_k) + cutlass.Int64(1)
                shift = cutlass.Uint32(0)
                spn = span - cutlass.Int64(1)
                while spn > cutlass.Int64(self.hist_size - 1):
                    spn = spn >> 1
                    shift = shift + 1
                # DSL signedness landmine: lo_k/hi_k are loop-carried
                # across dynamic regions and get re-wrapped SIGNED, so
                # `kk <= hi_k` with hi_k = 0xFFFFFFFF lowered as `kk <= -1`
                # -- true only for keys with the top bit set (fp32 keys of
                # POSITIVE floats), silently dropping every fp16/bf16 key
                # and every fp32 negative-float key from the histogram
                # (nothing published, stale s_misc, garbage bucket).
                # Re-assert unsignedness at every use site.
                # Range test + bin in Int64: unsigned 32-bit compares on
                # loop-carried keys were STILL lowered signed in some
                # kernel instantiations even after Uint32 re-assertion
                # (S==1 fp16 landed on bucket 4095); Int64 zero-extension
                # is proven reliable here (the shift derivation above
                # depends on it), so the compares become unambiguous.
                lo64 = cutlass.Int64(cutlass.Uint32(lo_k))
                span64 = cutlass.Int64(cutlass.Uint32(hi_k)) - lo64
                sh64 = cutlass.Int64(shift)
                for i in range(tidx, length, self.nt):
                    d = (
                        cutlass.Int64(
                            cutlass.Uint32(self.exact_key(self.load_scalar(row_in, i)))
                        )
                        - lo64
                    )
                    if d >= cutlass.Int64(0):
                        if d <= span64:
                            bb = cutlass.Int32(d >> sh64)
                            smem_atomic_add(s_h4k + bb, 1)
                cute.arch.barrier()
                # descending crossing at rank ``need`` (ends with a
                # block barrier; publishes bin/above/cnt)
                self.find_threshold_wide(
                    s_h4k,
                    cutlass.Int32(0),  # advisory; walk uses own sum
                    need,
                    s_warp_sums,
                    s_misc,
                    tidx,
                )
                bkt = s_misc[0]
                above = s_misc[1]
                cnt = s_misc[2]
                cute.arch.barrier()
                lo_k2 = lo_k + cutlass.Uint32(
                    cutlass.Int64(bkt) << cutlass.Int64(shift)
                )
                hi_k2 = lo_k + cutlass.Uint32(
                    ((cutlass.Int64(bkt) + 1) << cutlass.Int64(shift))
                    - cutlass.Int64(1)
                )
                need = need - above
                lo_k = lo_k2
                hi_k = hi_k2
                if cnt <= GCAP:
                    done = cutlass.Int32(1)
                if lo_k == hi_k:  # single key: fill-by-count cures it
                    done = cutlass.Int32(1)
        # round 3 buckets are single keys (32-bit key, 12+12+8 bits),
        # so done == 1 here unconditionally

        # ---- harvest pass: KEY-space compares (NaN-unambiguous) ----
        if tidx == 0:
            s_count_gt[0] = cutlass.Int32(0)
            s_count_eq[0] = cutlass.Int32(0)
        cute.arch.barrier()
        lo64 = cutlass.Int64(cutlass.Uint32(lo_k))
        hi64 = cutlass.Int64(cutlass.Uint32(hi_k))
        for i in range(tidx, length, self.nt):
            kk = cutlass.Uint32(self.exact_key(self.load_scalar(row_in, i)))
            k64 = cutlass.Int64(kk)
            if k64 > hi64:
                pos = smem_atomic_add(s_count_gt, 1)
                if pos < top_k:
                    out_idx_row[pos] = cutlass.Int32(i)
            else:
                if k64 >= lo64:
                    c = smem_atomic_add(s_count_eq, 1)
                    if c < GCAP:
                        slab_k[c] = kk.bitcast(cutlass.Int32)
                        slab_i[c] = cutlass.Int32(i)
        cute.arch.barrier()

        # ---- finish ----
        gt = s_count_gt[0]
        eq = s_count_eq[0]
        remaining = top_k - gt
        if lo_k == hi_k:
            # single-key bucket: candidates are key-equal, any
            # subset is exact; remaining <= k <= GCAP <= stored
            for t in range(tidx, remaining, self.nt):
                out_idx_row[gt + t] = slab_i[t]
        else:
            # cnt <= GCAP guaranteed by the loop's exit condition
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

    @cute.jit
    def launch_fallback(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.fallback_topk_kernel(
            input_data, seqlen, output_indices, slab, status
        ).launch(
            grid=(num_rows, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


# ---------------------------------------------------------------------------
# Production launchers: speculative kernel + gated fallback fused into ONE
# compiled call (one FFI roundtrip per top_k_varlen call, like gvr_2's
# bind-once host module), sync-free and CUDA-graph capturable.
# ---------------------------------------------------------------------------


class ProdHintTopK(HintPivotTopK, GatedExactFallback):
    """Hint rung + gated exact fallback in one compiled launcher."""

    @cute.jit
    def launch_prod(
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
        # No fallback launch: the exact fallback is FUSED into the
        # speculative kernel's failure path (fuse_fallback=True) -- the
        # gated kernel's empty pass measured ~1.65us of launch latency.
        self.hint_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status
        ).launch(grid=(num_rows, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


class ProdStackedTopK(StackedTopK, GatedExactFallback):
    """Stacked hint->sample rungs + gated exact fallback, one launcher."""

    @cute.jit
    def launch_prod(
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
            grid=(num_rows, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )  # fallback fused in-kernel (see ProdHintTopK)


class ProdSampledTopK(GatedExactFallback):
    """Sampled kernel + gated exact fallback, one launcher."""

    @cute.jit
    def launch_prod(
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
            grid=(num_rows, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )  # fallback fused in-kernel (see ProdHintTopK)


def _prod_ctor(cls, top_k):
    return cls(
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


def _fake(dt, shape, align=None):
    if align is None:
        return cute.runtime.make_fake_compact_tensor(
            dt, shape, stride_order=tuple(range(len(shape) - 1, -1, -1))
        )
    return cute.runtime.make_fake_compact_tensor(
        dt,
        shape,
        stride_order=tuple(range(len(shape) - 1, -1, -1)),
        assumed_align=align,
    )


_prod_compiled: dict = {}


def get_prod_kernel(kind: str, top_k: int, N: int, telemetry: bool = False):
    """Compile (with on-disk caching) the fused speculative+fallback
    launcher.  kind is 'hint', 'stacked', or 'sampled'; hint/stacked take
    (input, pre_idx, seqlen, out_indices, slab, status), sampled drops
    pre_idx.  One FFI call per invocation; sync-free; CUDA-graph safe.

    telemetry=True (or FLASHINFER_TOPK_PRIM_TELEMETRY=1) compiles the
    phase-instrumented sampled variant (globaltimer reads + status blocks
    5-8); production kernels carry no instrumentation."""
    telemetry = telemetry or os.environ.get("FLASHINFER_TOPK_PRIM_TELEMETRY") == "1"
    assert top_k <= TIE_CAP
    key = (kind, top_k, N, telemetry)
    if key in _prod_compiled:
        return _prod_compiled[key]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    cls = {
        "hint": ProdHintTopK,
        "stacked": ProdStackedTopK,
        "sampled": ProdSampledTopK,
    }[kind]
    kern = _prod_ctor(cls, top_k)
    kern.sp_telemetry = telemetry
    kern.fuse_fallback = True
    sym_rows = cute.sym_int()
    f32, i32 = cutlass.Float32, cutlass.Int32

    def _compile_fn():
        args = [_fake(f32, (sym_rows, N), 16)]
        if kind != "sampled":
            args.append(_fake(i32, (sym_rows, top_k), 16))
        args += [
            _fake(i32, (sym_rows,)),
            _fake(i32, (sym_rows, top_k), 16),
            _fake(i32, (sym_rows, 2 * GCAP), 16),
            _fake(i32, (sym_rows,)),
        ]
        return cute.compile(
            kern.launch_prod,
            *args,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = build_and_load_cute_dsl_kernel(
        "prod_topk_primitives",
        f"prod_{kind}_v1_f32_k{top_k}_N{N}{'_tel' if telemetry else ''}",
        _compile_fn,
        extra_key_files=(
            __file__,
            _sampled_mod.__file__,
            _hint_mod.__file__,
            _stacked_mod.__file__,
            _radix_mod.__file__,
        ),
    )
    _prod_compiled[key] = compiled
    return compiled


_compiled: dict = {}


def get_fallback_kernel(top_k: int, N: int):
    """Compile (with on-disk caching) the gated exact fallback for a
    (top_k, N) specialization.  Launch it immediately after a speculative
    kernel on the same stream, with the same slab/status tensors."""
    assert top_k <= TIE_CAP, "fallback requires k <= TIE_CAP"
    if (top_k, N) in _compiled:
        return _compiled[(top_k, N)]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    kern = GatedExactFallback(
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
            kern.launch_fallback,
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
        "fallback_topk_primitives",
        f"fallback_v1_f32_k{top_k}_N{N}",
        _compile_fn,
        extra_key_files=(__file__, _sampled_mod.__file__, _radix_mod.__file__),
    )
    _compiled[(top_k, N)] = compiled
    return compiled
