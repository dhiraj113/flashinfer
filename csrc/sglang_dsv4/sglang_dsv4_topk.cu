/*
 * Vendored sglang DeepSeek-V4 ragged top-k, wired as a FlashInfer JIT module.
 *
 * Device code is copied VERBATIM from sglang (Apache-2.0):
 *   python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh  (ragged kernel +
 *   helpers; the paged/cluster/plan machinery is intentionally not vendored)
 *   python/sglang/kernels/jit/include/sgl_kernel/*.cuh      (support headers,
 *   vendored unmodified under ./sgl_kernel/)
 * upstream commit 7f27bf4708 (2026-08).  Only this host launcher (TVM-FFI
 * TensorView + cudaLaunchKernelEx w/ PDL) is FlashInfer-specific.
 *
 * Purpose: apples-to-apples benchmarking of the `radix_primitives` backend
 * against the exact sglang kernel under the same top_k_varlen contract
 * (fp32 scores, per-row lengths, local column indices out, -1 padded).
 */

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

// sglang's utils.cuh defines its own CHECK_CUDA; FlashInfer's tvm_ffi_utils.h
// redefines it.  The sglang macro is not used below, so drop it.
#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif

#include "../tvm_ffi_utils.h"

namespace sglang {

namespace impl = device::topk;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)

struct TopKRaggedParams {
  float* __restrict__ scores;  // NOTE: may write
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ out_offsets;
  int32_t* __restrict__ topk_indices;
  int64_t score_stride;
  uint32_t topk;
};

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

/**
 * \brief Ragged (prefill) top-k: select inside a per-row window, emit indices
 * rebased onto the flattened KV.  (Verbatim from sglang topk_v2.cuh; see the
 * upstream file for the full commentary on the in-place masking of the <= 3
 * columns ahead of unaligned windows -- with row_starts == nullptr every
 * window starts at column 0 and no masking write occurs.)
 */
template <bool kPDL>
TOPK_KERNEL void topk_ragged_kernel(const __grid_constant__ TopKRaggedParams params) {
  device::enable_smem_spilling();
  constexpr uint32_t kVecSize = impl::TopKStreaming::kVecSize;
  const auto bx = blockIdx.x;
  // issue all metadata prefetch ahead of time
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[bx]);
  const auto offset = params.out_offsets[bx];
  const auto row_start = params.row_starts == nullptr ? 0u : params.row_starts[bx];
  const auto topk = params.topk;
  const auto out = params.topk_indices + bx * static_cast<int64_t>(topk);

  if (seq_len <= topk) {
    device::PDLWaitPrimary<kPDL>();
    for_each_item(topk, [&](uint32_t tx, uint32_t) {
      out[tx] = tx < seq_len ? static_cast<int32_t>(tx) + offset : -1;  // note: need offset
    });
    return;
  }

  const auto rem = row_start % kVecSize;
  const auto score = params.scores + bx * params.score_stride;
  if (rem != 0) {
    // The mask has to land after the indexer has retired
    // Otherwise it may be accidentally overwritten by DG upstream
    device::PDLWaitPrimary<kPDL>();
    static_assert(kVecSize <= kBlockSize, "not enough threads ");
    if (const auto tx = threadIdx.x; tx < rem) {
      score[row_start - rem + tx] = -std::numeric_limits<float>::max();
    }
  }

  const auto problem = impl::TopKProblem{
      .in = score + (row_start - rem),
      .out = out,
      .page_table = nullptr,  // unused
      .topk = topk,
      .seq_len = seq_len + rem,
      .page_bits = 1,  // unused
      .bias = offset - static_cast<int32_t>(rem),
  };
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= Register2::kMaxSeqLen) {
    Register2::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= Register4::kMaxSeqLen) {
    Register4::forward<kPDL>(problem, &smem);
  } else {
    Streaming::forward<kPDL>(problem, &smem);
  }
  // PDL trigger secondary at the end the block typically has no use, so ignore it
}

}  // namespace sglang

using tvm::ffi::TensorView;

void sglang_dsv4_topk_ragged(TensorView scores, TensorView lengths, TensorView out_offsets,
                             TensorView out_indices, bool enable_pdl) {
  CHECK_INPUT(scores);
  CHECK_INPUT(lengths);
  CHECK_INPUT(out_offsets);
  CHECK_INPUT(out_indices);
  CHECK_DIM(2, scores);       // (num_rows, N)
  CHECK_DIM(1, lengths);      // (num_rows,) int32, per-ROW effective lengths
  CHECK_DIM(1, out_offsets);  // (num_rows,) int32, added to emitted indices
  CHECK_DIM(2, out_indices);  // (num_rows, top_k)
  TVM_FFI_ICHECK_EQ(scores.dtype(), dl_float32) << "sglang dsv4 topk supports fp32 scores only";
  TVM_FFI_ICHECK_EQ(lengths.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(out_offsets.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(out_indices.dtype(), dl_int32);

  const int64_t num_rows = scores.size(0);
  const int64_t n_cols = scores.size(1);
  const int64_t top_k = out_indices.size(1);
  TVM_FFI_ICHECK_EQ(lengths.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(out_offsets.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(out_indices.size(0), num_rows);
  TVM_FFI_ICHECK(top_k > 0 && top_k <= sglang::kMaxTopK)
      << "top_k must be in (0, " << sglang::kMaxTopK << "]";
  TVM_FFI_ICHECK_EQ(n_cols % 4, 0)
      << "score row stride must be a multiple of 4 (16-byte vectorized load)";

  cudaSetDevice(scores.device().device_id);
  const cudaStream_t stream = get_stream(scores.device());

  const sglang::TopKRaggedParams params{
      static_cast<float*>(scores.data_ptr()),
      static_cast<const int32_t*>(lengths.data_ptr()),
      nullptr,  // row_starts: every window starts at column 0
      static_cast<const int32_t*>(out_offsets.data_ptr()),
      static_cast<int32_t*>(out_indices.data_ptr()),
      n_cols,
      static_cast<uint32_t>(top_k),
  };

  cudaLaunchConfig_t config;
  config.gridDim = static_cast<unsigned int>(num_rows);
  config.blockDim = sglang::kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaError_t status;
  if (enable_pdl) {
    status = cudaLaunchKernelEx(&config, sglang::topk_ragged_kernel<true>, params);
  } else {
    status = cudaLaunchKernelEx(&config, sglang::topk_ragged_kernel<false>, params);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sglang_dsv4_topk_ragged launch failed: " << cudaGetErrorString(status);
}
