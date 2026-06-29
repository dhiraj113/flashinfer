"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import torch
from enum import Enum
from typing import List, Literal, Optional

from .cudnn_utils import (
    cudnn,
    CUDNN_AVAILABLE,
    UIDs,
    _check_cudnn_availability,
    _check_cudnn_fp4_availability,
    _is_cublas_fp4_available_in_cudnn,
    _cudnn_available_or_raise_for_backend,
    _get_cudnn_handle,
    _get_cudnn_workspace_size,
    _torch_data_type_to_cudnn_data_type,
    _get_bf16_3d_shape_stride,
    _get_real_fp4_shape_from_packed_uint8,
    _expand_block_scale_tensor_shape,
)
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..utils import (
    get_native_fp4_dtype,
    supported_compute_capability,
)

# ---------------------------------------------------------------------------
# cuDNN FP4 GEMM + bias graph (nvfp4 / mxfp4)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_fp4_bias_graph(
    a_shape,
    a_stride,
    b_shape,
    b_stride,
    a_descale_shape,
    a_descale_stride,
    b_descale_shape,
    b_descale_stride,
    bias_shape,
    bias_stride,
    ab_type,
    o_type,
    block_size,
    device,
    alpha_is_not_none,
    use_nvfp4,
    policy=None,
):
    """Build cuDNN graph: out = (A @ B) * [alpha] + bias  (FP4 inputs)."""
    _check_cudnn_fp4_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

        a_cudnn_tensor = graph.tensor(
            name="a", dim=a_shape, stride=a_stride, data_type=ab_type
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=b_shape, stride=b_stride, data_type=ab_type
        )
        block_descale_a_cudnn_tensor = graph.tensor(
            name="block_descale_a",
            dim=a_descale_shape,
            stride=a_descale_stride,
            data_type=scale_type,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        block_descale_b_cudnn_tensor = graph.tensor(
            name="block_descale_b",
            dim=b_descale_shape,
            stride=b_descale_stride,
            data_type=scale_type,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        dequant_a_tensor = graph.block_scale_dequantize(
            a_cudnn_tensor,
            block_descale_a_cudnn_tensor,
            block_size=[1, block_size],
            name="dequant_a",
        )
        dequant_a_tensor.set_data_type(cudnn.data_type.FLOAT)
        dequant_b_tensor = graph.block_scale_dequantize(
            b_cudnn_tensor,
            block_descale_b_cudnn_tensor,
            block_size=[block_size, 1],
            name="dequant_b",
        )
        dequant_b_tensor.set_data_type(cudnn.data_type.FLOAT)
        c_tensor = graph.matmul(
            dequant_a_tensor,
            dequant_b_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="gemm",
        )
        c_tensor.set_data_type(cudnn.data_type.FLOAT)

        c_pre_bias = c_tensor

        if alpha_is_not_none:
            global_scale_cudnn_tensor = graph.tensor(
                name="global_scale",
                dim=(1, 1, 1),
                stride=(1, 1, 1),
                data_type=cudnn.data_type.FLOAT,
            )
            c_pre_bias = graph.mul(
                name="scale_mul",
                a=c_tensor,
                b=global_scale_cudnn_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            c_pre_bias.set_data_type(cudnn.data_type.FLOAT)
            global_scale_cudnn_tensor.set_uid(UIDs.ALPHA_UID.value)

        bias_cudnn_tensor = graph.tensor(
            name="bias",
            dim=list(bias_shape),
            stride=list(bias_stride),
            data_type=o_type,
        )
        c_output_tensor = graph.add(
            name="bias_add",
            a=c_pre_bias,
            b=bias_cudnn_tensor,
        )
        bias_cudnn_tensor.set_uid(UIDs.BIAS_UID.value)
        c_output_tensor.set_name("c_final").set_output(True).set_data_type(o_type)

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        block_descale_a_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_A_UID.value)
        block_descale_b_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_B_UID.value)
        c_output_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])

        # WAR: alpha not supported by cuBLAS backend (eng0) in older cuDNN
        if alpha_is_not_none and not _is_cublas_fp4_available_in_cudnn():
            graph.deselect_engines(["eng0"])

        graph.check_support()
        graph.build_plans(policy)

        return graph


def execute_cudnn_gemm_fp4_bias_graph(
    graph, a, b, a_descale, b_descale, alpha, bias, out, workspace_buffer, tactic: int = -1
):
    """Execute FP4 GEMM + bias cuDNN graph."""
    variant_pack = {
        UIDs.A_UID.value: a.view(get_native_fp4_dtype()),
        UIDs.B_UID.value: b.view(get_native_fp4_dtype()),
        UIDs.BLOCK_DESCALE_A_UID.value: a_descale,
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.BIAS_UID.value: bias,
        UIDs.O_UID.value: out,
    }
    if alpha is not None:
        variant_pack[UIDs.ALPHA_UID.value] = alpha.view(torch.float)

    if tactic >= graph.get_execution_plan_count():
        tactic = -1

    workspace_size = _get_cudnn_workspace_size(graph, tactic)
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    stream = torch.cuda.current_stream(a.device)
    if tactic == -1:
        graph.execute(variant_pack, workspace_buffer, handle=_get_cudnn_handle(a.device, stream))
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace_buffer, tactic, handle=_get_cudnn_handle(a.device, stream)
        )


# ---------------------------------------------------------------------------
# gemm_<fused_op> API convention
#
# Public fused-GEMM APIs follow the naming pattern  gemm_<op>  where <op> is
# the element-wise or epilogue operation fused after the matrix multiply:
#
#   gemm_bias   – out = A @ B + bias          (gemm_bias.py)
#   gemm_norm   – out = rmsnorm(A @ B)        (future)
#   gemm_gelu   – out = gelu(A @ B)           (future)
#   gemm_silu   – out = silu(A @ B)           (future)
#
# Each gemm_<op> function follows the same structure:
#   1. @backend_requirement(backend_checks={...}, common_check=..., heuristic_func=...)
#   2. @flashinfer_api(trace=<op>_trace)
#   3. Unified signature: (a, b, *op_args, a_scale, b_scale, a_descale, b_descale,
#                          alpha, fp4_dtype, out_dtype, out, backend)
#   4. Dispatch via AutoTuner → TunableRunner subclass → cuDNN (or other) graph
#
# Internal symbols use the prefix _cudnn_gemm_<op>_  (e.g. _cudnn_gemm_bias_dense).
# Tuning config constants are named _GEMM_<OP>_TUNING_CONFIG.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# gemm_bias: fused GEMM + bias (cuDNN backend, all float dtypes)
# ---------------------------------------------------------------------------


class FP4Type(Enum):
    """FP4 quantization variant for :func:`gemm_bias`.

    Attributes
    ----------
    NVFP4: NV native FP4, block_size=16, descale dtype ``uint8``.
    MXFP4: MX FP4, block_size=32, descale dtype ``uint8`` (fp8_e8m0).
    """

    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"

    @property
    def block_size(self) -> int:
        return 16 if self == FP4Type.NVFP4 else 32

    @property
    def use_nvfp4(self) -> bool:
        return self == FP4Type.NVFP4


_GEMM_BIAS_DENSE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_GEMM_BIAS_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_GEMM_BIAS_OUTPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in _GEMM_BIAS_FP8_DTYPES


_GEMM_BIAS_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (-2,),  # M dimension
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            9,  # out tensor at index 9 in [a, b, bias, a_scale, b_scale, a_descale, b_descale, alpha, fp4_dtype, out, workspace]
            -2,
            lambda shapes: shapes[0][-2],
        ),
    ),
)


def _get_3d_bias_shape_stride(bias: torch.Tensor):
    """Expand bias tensor to 3D for use in cuDNN graphs.

    Supports:
    - 1D [N]       → (1, 1, N)
    - 3D [B, 1, N] → unchanged
    """
    if bias.dim() == 1:
        n = bias.shape[0]
        p = bias.stride(0)
        return (1, 1, n), (1, 1, p)
    elif bias.dim() == 3:
        return tuple(bias.shape), tuple(bias.stride())
    else:
        raise ValueError(
            f"bias must be 1D [N] or 3D [B, 1, N], got {bias.dim()}D with shape {bias.shape}"
        )


# ---------------------------------------------------------------------------
# cuDNN graph: dense GEMM + bias (BF16 / FP16 / FP32)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_bias_graph(
    a_shape,
    a_stride,
    b_shape,
    b_stride,
    bias_shape,
    bias_stride,
    ab_dtype,
    o_dtype,
    device,
    policy=None,
):
    """Build cuDNN graph for: out = a @ b + bias (dense float dtypes)."""
    _check_cudnn_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a", dim=list(a_shape), stride=list(a_stride), data_type=ab_dtype
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=list(b_shape), stride=list(b_stride), data_type=ab_dtype
        )
        c_cudnn_tensor = graph.matmul(
            name="matmul",
            A=a_cudnn_tensor,
            B=b_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_cudnn_tensor.set_data_type(cudnn.data_type.FLOAT)

        bias_cudnn_tensor = graph.tensor(
            name="bias",
            dim=list(bias_shape),
            stride=list(bias_stride),
            data_type=o_dtype,
        )
        out_cudnn_tensor = graph.add(
            name="bias_add",
            a=c_cudnn_tensor,
            b=bias_cudnn_tensor,
        )
        out_cudnn_tensor.set_name("out").set_output(True).set_data_type(o_dtype)

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        bias_cudnn_tensor.set_uid(UIDs.BIAS_UID.value)
        out_cudnn_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(policy)

        return graph


def execute_cudnn_gemm_bias_graph(graph, a, b, bias, out, workspace, tactic: int = -1):
    """Execute dense GEMM + bias cuDNN graph."""
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.BIAS_UID.value: bias,
        UIDs.O_UID.value: out,
    }

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    if tactic >= graph.get_execution_plan_count():
        tactic = -1

    workspace_size = _get_cudnn_workspace_size(graph, tactic)
    if workspace.numel() < workspace_size:
        workspace.resize_(workspace_size)

    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
    else:
        graph.execute_plan_at_index(variant_pack, workspace, tactic, handle=cudnn_handle)


# ---------------------------------------------------------------------------
# cuDNN graph: FP8 GEMM + bias (per-tensor scales + bias epilogue)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_fp8_bias_graph(
    a_shape,
    a_stride,
    b_shape,
    b_stride,
    bias_shape,
    bias_stride,
    a_type,
    b_type,
    o_type,
    device,
    policy=None,
):
    """Build cuDNN graph for: out = (a @ b) * a_scale * b_scale + bias (FP8)."""
    _check_cudnn_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a", dim=list(a_shape), stride=list(a_stride), data_type=a_type
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=list(b_shape), stride=list(b_stride), data_type=b_type
        )
        a_scale_cudnn_tensor = graph.tensor(
            name="a_scale",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        b_scale_cudnn_tensor = graph.tensor(
            name="b_scale",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        c_cudnn_tensor = graph.matmul(
            name="matmul",
            A=a_cudnn_tensor,
            B=b_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_cudnn_tensor.set_data_type(cudnn.data_type.FLOAT)

        c_after_a_scale = graph.mul(
            name="scale_mul_a",
            a=c_cudnn_tensor,
            b=a_scale_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_after_a_scale.set_data_type(cudnn.data_type.FLOAT)
        c_after_b_scale = graph.mul(
            name="scale_mul_b",
            a=c_after_a_scale,
            b=b_scale_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_after_b_scale.set_data_type(cudnn.data_type.FLOAT)

        bias_cudnn_tensor = graph.tensor(
            name="bias",
            dim=list(bias_shape),
            stride=list(bias_stride),
            data_type=o_type,
        )
        out_cudnn_tensor = graph.add(
            name="bias_add",
            a=c_after_b_scale,
            b=bias_cudnn_tensor,
        )
        out_cudnn_tensor.set_name("out").set_output(True).set_data_type(o_type)

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        a_scale_cudnn_tensor.set_uid(UIDs.A_SCALE_UID.value)
        b_scale_cudnn_tensor.set_uid(UIDs.B_SCALE_UID.value)
        bias_cudnn_tensor.set_uid(UIDs.BIAS_UID.value)
        out_cudnn_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(policy)

        return graph


def execute_cudnn_gemm_fp8_bias_graph(
    graph, a, b, a_scale, b_scale, bias, out, workspace, tactic: int = -1
):
    """Execute FP8 GEMM + bias cuDNN graph."""
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.A_SCALE_UID.value: a_scale,
        UIDs.B_SCALE_UID.value: b_scale,
        UIDs.BIAS_UID.value: bias,
        UIDs.O_UID.value: out,
    }

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    if tactic >= graph.get_execution_plan_count():
        tactic = -1

    workspace_size = _get_cudnn_workspace_size(graph, tactic)
    if workspace.numel() < workspace_size:
        workspace.resize_(workspace_size)

    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
    else:
        graph.execute_plan_at_index(variant_pack, workspace, tactic, handle=cudnn_handle)


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------


def _cudnn_gemm_bias_dense(workspace, a, b, bias, out, tactic=-1):
    """Dispatch dense (non-FP8) GEMM + bias through cuDNN."""
    _check_cudnn_availability()

    a_shape, a_stride = _get_bf16_3d_shape_stride(a)
    b_shape, b_stride = _get_bf16_3d_shape_stride(b)
    bias_shape, bias_stride = _get_3d_bias_shape_stride(bias)

    if tactic == -1:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    else:
        policy = cudnn.build_plan_policy.ALL

    graph = build_cudnn_gemm_bias_graph(
        a_shape,
        a_stride,
        b_shape,
        b_stride,
        bias_shape,
        bias_stride,
        _torch_data_type_to_cudnn_data_type(a.dtype),
        _torch_data_type_to_cudnn_data_type(out.dtype),
        a.device,
        policy=policy,
    )
    execute_cudnn_gemm_bias_graph(graph, a, b, bias, out, workspace, tactic=tactic)
    return out


def _cudnn_gemm_bias_fp8(workspace, a, b, a_scale, b_scale, bias, out, tactic=-1):
    """Dispatch FP8 GEMM + bias through cuDNN."""
    _check_cudnn_availability()

    a_shape, a_stride = _get_bf16_3d_shape_stride(a)
    b_shape, b_stride = _get_bf16_3d_shape_stride(b)
    bias_shape, bias_stride = _get_3d_bias_shape_stride(bias)

    if tactic == -1:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    else:
        policy = cudnn.build_plan_policy.ALL

    graph = build_cudnn_gemm_fp8_bias_graph(
        a_shape,
        a_stride,
        b_shape,
        b_stride,
        bias_shape,
        bias_stride,
        _torch_data_type_to_cudnn_data_type(a.dtype),
        _torch_data_type_to_cudnn_data_type(b.dtype),
        _torch_data_type_to_cudnn_data_type(out.dtype),
        a.device,
        policy=policy,
    )

    # Ensure scale tensors are 3D FP32 scalars for cuDNN
    def _to_3d_float_scalar(t):
        return t.float().reshape(1, 1, 1)

    execute_cudnn_gemm_fp8_bias_graph(
        graph,
        a,
        b,
        _to_3d_float_scalar(a_scale),
        _to_3d_float_scalar(b_scale),
        bias,
        out,
        workspace,
        tactic=tactic,
    )
    return out


def _cudnn_gemm_fp4_bias(
    workspace, a, b, a_descale, b_descale, alpha, bias, out, fp4_dtype: "FP4Type", tactic=-1
):
    """Execute FP4 GEMM + bias via cuDNN."""
    m_shape, m_stride = _get_real_fp4_shape_from_packed_uint8(a)
    n_shape, n_stride = _get_real_fp4_shape_from_packed_uint8(b)
    bias_shape, bias_stride = _get_3d_bias_shape_stride(bias)

    batch = m_shape[0]
    a_descale_shape, a_descale_stride = _expand_block_scale_tensor_shape(a_descale, batch)
    b_descale_shape, b_descale_stride = _expand_block_scale_tensor_shape(b_descale, batch)

    ab_type = cudnn.data_type.FP4_E2M1
    o_type = _torch_data_type_to_cudnn_data_type(out.dtype)

    graph = build_cudnn_gemm_fp4_bias_graph(
        tuple(m_shape), tuple(m_stride),
        tuple(n_shape), tuple(n_stride),
        tuple(a_descale_shape), tuple(a_descale_stride),
        tuple(b_descale_shape), tuple(b_descale_stride),
        tuple(bias_shape), tuple(bias_stride),
        ab_type, o_type,
        fp4_dtype.block_size,
        a.device,
        alpha_is_not_none=(alpha is not None),
        use_nvfp4=fp4_dtype.use_nvfp4,
    )
    execute_cudnn_gemm_fp4_bias_graph(
        graph, a, b, a_descale, b_descale, alpha, bias, out, workspace, tactic=tactic
    )
    return out


def _cudnn_gemm_bias_runner():
    class CudnnMmBiasRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, bias, a_scale, b_scale, a_descale, b_descale, alpha, fp4_dtype, out, _ = inputs
            return (a.dtype, b.dtype, out.dtype, a_scale is not None, fp4_dtype)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a, b, bias, a_scale, b_scale, a_descale, b_descale, alpha, fp4_dtype, out, _ = inputs
            bias_shape, bias_stride = _get_3d_bias_shape_stride(bias)

            if a_descale is not None:
                # FP4 path
                m_shape, m_stride = _get_real_fp4_shape_from_packed_uint8(a)
                n_shape, n_stride = _get_real_fp4_shape_from_packed_uint8(b)
                batch = m_shape[0]
                a_descale_shape, a_descale_stride = _expand_block_scale_tensor_shape(
                    a_descale, batch
                )
                b_descale_shape, b_descale_stride = _expand_block_scale_tensor_shape(
                    b_descale, batch
                )
                graph = build_cudnn_gemm_fp4_bias_graph(
                    tuple(m_shape), tuple(m_stride),
                    tuple(n_shape), tuple(n_stride),
                    tuple(a_descale_shape), tuple(a_descale_stride),
                    tuple(b_descale_shape), tuple(b_descale_stride),
                    tuple(bias_shape), tuple(bias_stride),
                    cudnn.data_type.FP4_E2M1,
                    _torch_data_type_to_cudnn_data_type(out.dtype),
                    fp4_dtype.block_size, a.device,
                    alpha_is_not_none=(alpha is not None),
                    use_nvfp4=fp4_dtype.use_nvfp4,
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
            elif _is_fp8(a.dtype):
                a_shape, a_stride = _get_bf16_3d_shape_stride(a)
                b_shape, b_stride = _get_bf16_3d_shape_stride(b)
                graph = build_cudnn_gemm_fp8_bias_graph(
                    a_shape, a_stride, b_shape, b_stride,
                    bias_shape, bias_stride,
                    _torch_data_type_to_cudnn_data_type(a.dtype),
                    _torch_data_type_to_cudnn_data_type(b.dtype),
                    _torch_data_type_to_cudnn_data_type(out.dtype),
                    a.device,
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
            else:
                a_shape, a_stride = _get_bf16_3d_shape_stride(a)
                b_shape, b_stride = _get_bf16_3d_shape_stride(b)
                graph = build_cudnn_gemm_bias_graph(
                    a_shape, a_stride, b_shape, b_stride,
                    bias_shape, bias_stride,
                    _torch_data_type_to_cudnn_data_type(a.dtype),
                    _torch_data_type_to_cudnn_data_type(out.dtype),
                    a.device,
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
            return list(range(graph.get_execution_plan_count()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, bias, a_scale, b_scale, a_descale, b_descale, alpha, fp4_dtype, out, workspace_buffer = inputs
            if a_descale is not None:
                _cudnn_gemm_fp4_bias(
                    workspace_buffer, a, b, a_descale, b_descale, alpha, bias, out,
                    fp4_dtype, tactic=tactic
                )
            elif _is_fp8(a.dtype):
                _cudnn_gemm_bias_fp8(
                    workspace_buffer, a, b, a_scale, b_scale, bias, out, tactic=tactic
                )
            else:
                _cudnn_gemm_bias_dense(workspace_buffer, a, b, bias, out, tactic=tactic)
            return out

    return CudnnMmBiasRunner()


# ---------------------------------------------------------------------------
# gemm_bias requirement / check / heuristic functions
# ---------------------------------------------------------------------------


@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 110, 120, 121])
def _cudnn_gemm_bias_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    a_descale: Optional[torch.Tensor] = None,
    b_descale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    fp4_dtype: "FP4Type" = FP4Type.NVFP4,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "auto"] = "cudnn",
):
    """cuDNN backend requirement for gemm_bias.

    The three exclusive input groups and their required/forbidden optional args:

      Dense  (BF16/FP16/FP32, SM80+):
        a_scale=None, b_scale=None, a_descale=None, b_descale=None

      FP8  (float8_e4m3fn / float8_e5m2, SM89+):
        a_scale≠None, b_scale≠None          — both per-tensor scales required
        a_descale=None, b_descale=None       — block descales must be absent

      FP4  (NVFP4 / MXFP4 uint8, SM100+):
        a_descale≠None, b_descale≠None       — both block descales required
        a_scale=None, b_scale=None           — per-tensor scales must be absent
        fp4_dtype                 — selects NVFP4 (block=16) or MXFP4 (block=32)

    Returns False for any mismatched combination so that the backend dispatcher
    can report "no suitable backend" rather than a confusing internal error.
    """
    fp8 = _is_fp8(a.dtype)
    is_uint8 = a.dtype == torch.uint8

    # ── FP4 path ────────────────────────────────────────────────────────────
    # Either both block descales are present or neither; mixing is not supported.
    if a_descale is not None or b_descale is not None:
        if a_descale is None or b_descale is None:
            return False  # one-sided block descale
        if a_scale is not None or b_scale is not None:
            return False  # FP4 and per-tensor scales are mutually exclusive
        if not is_uint8:
            return False  # FP4 inputs must be packed uint8
        _check_cudnn_fp4_availability()  # raises with a clear message if SM/cuDNN too old
        return True

    # ── FP8 path ────────────────────────────────────────────────────────────
    # Either both per-tensor scales are present or neither; mixing is not supported.
    if a_scale is not None or b_scale is not None:
        if a_scale is None or b_scale is None:
            return False  # one-sided per-tensor scale
        if not fp8:
            return False  # per-tensor scales require FP8 inputs
        return _cudnn_available_or_raise_for_backend(backend)

    # ── Dense path ──────────────────────────────────────────────────────────
    # No scales of any kind.  FP8 / uint8 inputs without scales are not supported
    # here (they belong to the paths above).
    if fp8 or is_uint8:
        return False
    return _cudnn_available_or_raise_for_backend(backend)


def _check_gemm_bias_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    a_descale: Optional[torch.Tensor] = None,
    b_descale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    fp4_dtype: "FP4Type" = FP4Type.NVFP4,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "auto"] = "cudnn",
):
    fp4 = a_descale is not None
    fp8 = _is_fp8(a.dtype)

    if fp4:
        # FP4 path: a/b are packed uint8 (2 FP4 values per byte).
        if a.dtype != torch.uint8 or b.dtype != torch.uint8:
            raise ValueError(
                f"For FP4 gemm_bias (a_descale provided), a and b must be uint8 (packed FP4), "
                f"got a={a.dtype}, b={b.dtype}."
            )
        if a_scale is not None or b_scale is not None:
            raise ValueError(
                "a_scale and b_scale must be None for FP4 inputs; use a_descale and b_descale."
            )
        if b_descale is None:
            raise ValueError("b_descale is required when a_descale is provided.")

        # Packed uint8: 2 FP4 values per byte → logical K = 2 * K_packed.
        k_packed = a.shape[-1]
        if k_packed < 1:
            raise ValueError(
                f"gemm_bias FP4 requires at least 1 packed K byte (logical K >= 2), got K_packed={k_packed}."
            )
    else:
        # cuDNN segfaults when K=1 due to TMA alignment requirements.
        # FP4 K is in units of 2 (packed), so this check doesn't apply there.
        k = a.shape[-1]
        if k < 2:
            raise ValueError(
                f"gemm_bias requires K >= 2 for the cuDNN backend (got K={k}). "
                "cuDNN's matmul TMA alignment requires at least 2 elements in the K dimension."
            )

        # Input dtype validation
        if not fp8 and a.dtype not in _GEMM_BIAS_DENSE_DTYPES:
            raise ValueError(
                f"Unsupported a dtype {a.dtype} for gemm_bias. "
                f"Supported: bfloat16, float16, float32, float8_e4m3fn, float8_e5m2, "
                f"or uint8 (packed FP4, requires a_descale)."
            )
        if not fp8 and b.dtype != a.dtype:
            raise ValueError(
                f"a and b must have the same dtype for dense gemm_bias, got a={a.dtype}, b={b.dtype}."
            )
        if fp8 and b.dtype not in _GEMM_BIAS_FP8_DTYPES:
            raise ValueError(
                f"When a is FP8, b must also be FP8, got b={b.dtype}."
            )

        # FP8 scale requirements
        if fp8:
            if a_scale is None or b_scale is None:
                raise ValueError(
                    "a_scale and b_scale are required when a/b are FP8 tensors."
                )
        else:
            if a_scale is not None or b_scale is not None:
                raise ValueError(
                    "a_scale and b_scale must be None for non-FP8 inputs."
                )

    # Output dtype
    resolved_out_dtype = out_dtype if out_dtype is not None else (
        torch.bfloat16 if (fp8 or fp4) else a.dtype
    )
    if resolved_out_dtype not in _GEMM_BIAS_OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported out_dtype {resolved_out_dtype}. "
            "Supported: bfloat16, float16, float32."
        )

    # Bias dtype must match output dtype
    if bias.dtype != resolved_out_dtype:
        raise ValueError(
            f"bias dtype {bias.dtype} must match out_dtype {resolved_out_dtype}."
        )

    # Shape: a [m, k] or [b, m, k]; b [k, n] col-major or [b, k, n]
    if a.dim() not in (2, 3):
        raise ValueError(f"a must be 2D or 3D, got {a.dim()}D with shape {a.shape}.")
    if b.dim() not in (2, 3):
        raise ValueError(f"b must be 2D or 3D, got {b.dim()}D with shape {b.shape}.")
    if a.dim() != b.dim():
        raise ValueError(
            f"a and b must have the same number of dimensions, got {a.dim()}D and {b.dim()}D."
        )

    # Bias shape check.
    # For FP4: b has packed shape [k//2, n], so b.shape[-1] = N directly (same as non-FP4).
    if bias.dim() not in (1, 3):
        raise ValueError(
            f"bias must be 1D [N] or 3D [B, 1, N], got {bias.dim()}D with shape {bias.shape}."
        )
    n = b.shape[-1]
    if bias.dim() == 1 and bias.shape[0] != n:
        raise ValueError(
            f"bias size {bias.shape[0]} does not match b output dim N={n}."
        )
    if bias.dim() == 3:
        if bias.shape[1] != 1:
            raise ValueError(
                f"3D bias must have shape [B, 1, N] (M=1), got {bias.shape}."
            )
        if bias.shape[2] != n:
            raise ValueError(
                f"3D bias N dim {bias.shape[2]} != b output dim N={n}."
            )

    if out is not None:
        expected_shape = (*a.shape[:-1], n)
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {out.shape} does not match expected {expected_shape}."
            )
        if out.dtype != resolved_out_dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match out_dtype {resolved_out_dtype}."
            )

    return True


def _heuristic_func_gemm_bias(
    suitable_backends: List[str],
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    a_descale: Optional[torch.Tensor] = None,
    b_descale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    fp4_dtype: "FP4Type" = FP4Type.NVFP4,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "auto"] = "cudnn",
):
    heuristic_backends = []
    if CUDNN_AVAILABLE and "cudnn" in suitable_backends:
        heuristic_backends.append("cudnn")
    return heuristic_backends
