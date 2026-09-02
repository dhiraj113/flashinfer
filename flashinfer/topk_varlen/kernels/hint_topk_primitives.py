"""Hint-pivot top-k on the primitives substrate (experimental, "GVR1
modern").

The GVR V1 ALGORITHM (hint-derived pivot, guess-verify-refine) rebuilt on
the CoarseHistTopKPrimitives machinery, so the algorithm runs at the same
implementation quality as the histogram and sampled-pivot kernels.  The
research probe behind it showed the original V1 kernel's slowness was
almost entirely implementation, not algorithm: with oracle hints this
kernel matches gvr_2-class times and dominates flood/duplicate rows
(13-40us EXACT on B200 where gvr_2 measured 108us-1.3ms).

Algorithm per row (fp32, single CTA):
  1. guess: gather the k hinted values (parallel strided loads),
     block-reduce the MIN via ordered-key red.max on the inverted key
     -> pivot T.  Invalid hints (idx < 0 or >= length) are skipped; an
     all-invalid hint row degenerates to T = +max-key and fails verify
     into the status fallback.
  2. verify + harvest: ONE streaming pass (the parent's boundary-classify
     collect walker), boundaries [T, T]: v > T (or NaN) -> emit via
     ticket cursor; v == T -> stage (key, idx) + count; else nothing.
  3. count-verify: ok iff gt <= k <= gt + eq.  ok -> fill the remaining
     slots from staged ties (all value-EQUAL, so any subset is exact);
     else write status=1 and the host reruns that row through the
     standard exact 2-pass backend.  Worst case is therefore bounded at
     ~3 row passes, data-independent -- unlike original V1, a bad hint
     can never produce a wrong result, only the fallback.

Hint semantics: `pre_idx` holds candidate indices for THIS call (e.g. the
previous decode step's top-k).  The pivot is min-of-hints, so ONE
corrupted hint pointing at a tiny value drags T below the true
threshold, gt grows past k, verify fails, and the row falls back: zero
corruption tolerance in the happy path (measured), but hints only
steer -- exactness never depends on them.

Experimental scope: fp32, next_n == 1, compress_ratio == 1,
return_values=False, single-CTA rows, k <= TIE_CAP.  Not wired into the
public dispatcher; see get_hint_kernel() and fi-wt/sm90_probe.
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator

from . import radix_topk_primitives as _radix_mod
from .radix_topk_primitives import (
    NUM_THREADS,
    NUM_WARPS,
    TIE_CAP,
    _OP_COLLECT,
    CoarseHistTopKPrimitivesKernel,
    smem_red_max_u32,
)
from .sampled_topk_primitives import GCAP


class HintPivotTopK(CoarseHistTopKPrimitivesKernel):
    """See module docstring.  Reuses the parent's walker and key helpers;
    only the pivot derivation and the count-verify epilogue live here."""

    # Trace-time switch (see SampledPivotTopK.fuse_fallback): True on the
    # Prod class fuses the exact fallback inline into the failure path.
    fuse_fallback: bool = False

    @cute.kernel
    def hint_topk_kernel(
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
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])

        in_ptr = input_data.iterator
        pi_ptr = pre_idx.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator

        smem = SmemAllocator()
        s_dummy = smem.allocate_array(cutlass.Int32, 4, byte_alignment=128)
        s_tie_keys = smem.allocate_array(cutlass.Uint32, TIE_CAP, byte_alignment=128)
        s_tie_idx = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count_gt = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count_eq = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        # scratch for the fused inline fallback (smem does not gate
        # occupancy here: 1024-thread CTAs are register-bound to 1/SM)
        s_h4k = smem.allocate_array(cutlass.Int32, 4096, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)

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
        else:
            if tidx == 0:
                s_count_gt[0] = cutlass.Int32(0)
                s_count_eq[0] = cutlass.Int32(0)
                s_misc[10] = cutlass.Int32(0)  # red.max(~key) accumulator
            cute.arch.barrier()

            # ---- 1. guess: pivot = min over the hinted values ----
            for i in range(tidx, top_k, NUM_THREADS):
                hidx = pi_row[i]
                if hidx >= 0:
                    if hidx < length:
                        b = self.load_scalar(row_in, hidx)
                        smem_red_max_u32(s_misc + 10, ~self.exact_key(b))
            cute.arch.barrier()
            kmin = ~cutlass.Uint32(s_misc[10])
            tbits = self.from_key32(kmin)  # pivot as fp32 bit pattern

            # ---- 2. verify + harvest: ONE pass, boundaries [T, T] ----
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
                s_tie_keys,
                s_tie_idx,
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

            # ---- 3. count-verify + tie fill ----
            gt = s_count_gt[0]
            eq = s_count_eq[0]
            remaining = top_k - gt
            ok = cutlass.Int32(0)
            if remaining >= 0:
                if remaining <= eq:
                    ok = cutlass.Int32(1)
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
            if ok == 1:
                # staged = min(eq, TIE_CAP) >= remaining (remaining <= k <=
                # TIE_CAP); ties are value-equal so any subset is exact
                for t in range(tidx, remaining, NUM_THREADS):
                    out_idx_row[gt + t] = s_tie_idx[t]

    @cute.jit
    def launch_hint(
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
        self.hint_topk_kernel(
            input_data, pre_idx, seqlen, output_indices, slab, status
        ).launch(
            grid=(num_rows, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


_compiled: dict = {}


def get_hint_kernel(top_k: int, N: int):
    """Compile (with on-disk caching) the hint-pivot kernel for a
    (top_k, N) specialization.  fp32 / single-CTA / k <= TIE_CAP only.
    Call as kern(input, pre_idx, seqlen, out_indices, status); rows with
    status != 0 must be rerun through an exact backend by the caller."""
    assert top_k <= TIE_CAP, "hint-pivot requires k <= TIE_CAP"
    if (top_k, N) in _compiled:
        return _compiled[(top_k, N)]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    kern = HintPivotTopK(
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
        approx_ties=True,  # slims collect; exact here (ties value-equal)
        enable_pdl=False,
        warp_agg=False,
    )
    sym_rows = cute.sym_int()

    def _compile_fn():
        return cute.compile(
            kern.launch_hint,
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
        "hint_topk_primitives",
        f"hint_v1_f32_k{top_k}_N{N}",
        _compile_fn,
        extra_key_files=(__file__, _radix_mod.__file__),
    )
    _compiled[(top_k, N)] = compiled
    return compiled
