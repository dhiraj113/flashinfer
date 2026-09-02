/*
 * TVM-FFI binding for the vendored sglang DeepSeek-V4 ragged top-k.
 */
#include "../tvm_ffi_utils.h"

using tvm::ffi::TensorView;

void sglang_dsv4_topk_ragged(TensorView scores, TensorView lengths, TensorView out_offsets,
                             TensorView out_indices, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sglang_dsv4_topk_ragged, sglang_dsv4_topk_ragged);
