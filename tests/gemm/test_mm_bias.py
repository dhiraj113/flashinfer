"""Tests for mm_bias: fused GEMM + bias (cuDNN backend, all float dtypes)."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bias
from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.utils import get_compute_capability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp8_quantize(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    """Quantize tensor to FP8 and return (quantized, scale)."""
    max_val = x.float().abs().max()
    if dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:  # e5m2
        fp8_max = 57344.0
    scale = max_val / fp8_max
    if scale == 0:
        scale = torch.tensor(1.0)
    x_fp8 = x.float().div(scale).clamp(-fp8_max, fp8_max).to(dtype)
    return x_fp8, scale


def _skip_if_no_cudnn():
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN is not available on this system.")


def _skip_if_no_fp8():
    cc = get_compute_capability(torch.device("cuda"))
    cc_num = cc[0] * 10 + cc[1]
    if cc_num < 89:
        pytest.skip(f"FP8 requires SM89+, got SM{cc_num}.")


def _skip_if_no_dense_mm_bias():
    _skip_if_no_cudnn()
    cc = get_compute_capability(torch.device("cuda"))
    cc_num = cc[0] * 10 + cc[1]
    if not mm_bias.is_compute_capability_supported(cc_num):
        pytest.skip(f"mm_bias not supported on SM{cc_num}.")


# ---------------------------------------------------------------------------
# BF16 GEMM + bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 8, 32, 64])
@pytest.mark.parametrize("n", [1024, 2048, 4096])
@pytest.mark.parametrize("k", [1024, 2048])
def test_mm_bias_bf16(m, n, k):
    _skip_if_no_dense_mm_bias()

    torch.manual_seed(42)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    reference = F.linear(a, b.T.contiguous(), bias)
    out = mm_bias(a, b, bias)

    assert out.shape == (m, n), f"Expected shape ({m}, {n}), got {out.shape}"
    assert out.dtype == torch.bfloat16

    cos_sim = F.cosine_similarity(reference.reshape(-1).float(), out.reshape(-1).float(), dim=0)
    assert cos_sim > 0.99, f"BF16 mm_bias cos_sim={cos_sim:.4f} too low"


# ---------------------------------------------------------------------------
# FP16 GEMM + bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 16, 64])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [1024, 2048])
def test_mm_bias_fp16(m, n, k):
    _skip_if_no_dense_mm_bias()

    torch.manual_seed(7)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.float16)

    reference = F.linear(a.float(), b.T.contiguous().float(), bias.float()).half()
    out = mm_bias(a, b, bias)

    assert out.shape == (m, n)
    assert out.dtype == torch.float16

    cos_sim = F.cosine_similarity(reference.reshape(-1).float(), out.reshape(-1).float(), dim=0)
    assert cos_sim > 0.99, f"FP16 mm_bias cos_sim={cos_sim:.4f} too low"


# ---------------------------------------------------------------------------
# FP32 GEMM + bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 8, 32])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [512, 1024])
def test_mm_bias_fp32(m, n, k):
    _skip_if_no_dense_mm_bias()

    torch.manual_seed(3)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, k, device="cuda", dtype=torch.float32).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.float32)

    reference = F.linear(a, b.T.contiguous(), bias)
    out = mm_bias(a, b, bias)

    assert out.shape == (m, n)
    assert out.dtype == torch.float32

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.999, f"FP32 mm_bias cos_sim={cos_sim:.6f} too low"


# ---------------------------------------------------------------------------
# FP8 GEMM + bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("a_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_bias_fp8(m, n, k, a_dtype, out_dtype):
    _skip_if_no_fp8()
    _skip_if_no_cudnn()

    cc = get_compute_capability(torch.device("cuda"))
    cc_num = cc[0] * 10 + cc[1]
    if not mm_bias.is_compute_capability_supported(cc_num):
        pytest.skip(f"mm_bias not supported on SM{cc_num}.")

    torch.manual_seed(11)
    a_fp32 = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b_fp32 = torch.randn(n, k, device="cuda", dtype=torch.float32)

    a_fp8, a_scale = _fp8_quantize(a_fp32, a_dtype)
    b_fp8, b_scale = _fp8_quantize(b_fp32, a_dtype)
    b_fp8_col_major = b_fp8.T.contiguous()

    bias = torch.randn(n, device="cuda", dtype=out_dtype)

    a_scale_t = a_scale.detach().clone().to(device="cuda", dtype=torch.float32) if isinstance(a_scale, torch.Tensor) else torch.tensor(float(a_scale), device="cuda", dtype=torch.float32)
    b_scale_t = b_scale.detach().clone().to(device="cuda", dtype=torch.float32) if isinstance(b_scale, torch.Tensor) else torch.tensor(float(b_scale), device="cuda", dtype=torch.float32)

    out = mm_bias(
        a_fp8,
        b_fp8_col_major,
        bias,
        a_scale=a_scale_t,
        b_scale=b_scale_t,
        out_dtype=out_dtype,
    )

    # Reference: dequantize and compute in float32
    a_deq = a_fp8.float() * a_scale
    b_deq = b_fp8.float() * b_scale
    reference = (a_deq @ b_deq.T + bias.float()).to(out_dtype)

    assert out.shape == (m, n), f"Expected shape ({m}, {n}), got {out.shape}"
    assert out.dtype == out_dtype

    cos_sim = F.cosine_similarity(reference.reshape(-1).float(), out.reshape(-1).float(), dim=0)
    assert cos_sim > 0.99, f"FP8 mm_bias cos_sim={cos_sim:.4f} too low"


# ---------------------------------------------------------------------------
# Output dtype overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mm_bias_bf16_output_dtype(out_dtype):
    """BF16 inputs with different output dtypes."""
    _skip_if_no_dense_mm_bias()

    m, n, k = 16, 512, 256
    torch.manual_seed(5)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=out_dtype)

    out = mm_bias(a, b, bias, out_dtype=out_dtype)
    assert out.dtype == out_dtype

    reference = (F.linear(a.float(), b.T.contiguous().float(), bias.float())).to(out_dtype)
    cos_sim = F.cosine_similarity(reference.reshape(-1).float(), out.reshape(-1).float(), dim=0)
    assert cos_sim > 0.99, f"out_dtype={out_dtype} cos_sim={cos_sim:.4f} too low"


# ---------------------------------------------------------------------------
# Pre-allocated output tensor
# ---------------------------------------------------------------------------


def test_mm_bias_preallocated_out():
    _skip_if_no_dense_mm_bias()

    m, n, k = 32, 512, 256
    torch.manual_seed(9)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    returned = mm_bias(a, b, bias, out=out)

    assert returned is out or returned.data_ptr() == out.data_ptr(), \
        "mm_bias should write to the pre-allocated out tensor"
    assert out.shape == (m, n)

    reference = F.linear(a, b.T.contiguous(), bias)
    cos_sim = F.cosine_similarity(reference.reshape(-1).float(), out.reshape(-1).float(), dim=0)
    assert cos_sim > 0.99


# ---------------------------------------------------------------------------
# Autotuning
# ---------------------------------------------------------------------------


def test_mm_bias_autotuning():
    _skip_if_no_dense_mm_bias()

    m, n, k = 64, 1024, 512
    torch.manual_seed(13)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    reference = F.linear(a, b.T.contiguous(), bias)
    with autotune():
        out = mm_bias(a, b, bias)

    cos_sim = F.cosine_similarity(reference.reshape(-1).float(), out.reshape(-1).float(), dim=0)
    assert cos_sim > 0.99


# ---------------------------------------------------------------------------
# 3D batched input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("m", [8, 16])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [512, 1024])
def test_mm_bias_batched(batch, m, n, k):
    _skip_if_no_dense_mm_bias()

    torch.manual_seed(17)
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, n, k, device="cuda", dtype=torch.bfloat16).transpose(-2, -1).contiguous()
    # 3D bias [batch, 1, n]
    bias = torch.randn(batch, 1, n, device="cuda", dtype=torch.bfloat16)

    out = mm_bias(a, b, bias)

    assert out.shape == (batch, m, n), f"Expected ({batch}, {m}, {n}), got {out.shape}"
    assert out.dtype == torch.bfloat16

    # Reference using torch.bmm
    reference = torch.bmm(a, b) + bias
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99, f"batched mm_bias cos_sim={cos_sim:.4f} too low"


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


def test_mm_bias_invalid_dtype_raises():
    _skip_if_no_dense_mm_bias()

    m, n, k = 8, 64, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.float16)  # dtype mismatch

    with pytest.raises(ValueError):
        mm_bias(a, b, bias)


def test_mm_bias_fp8_missing_scale_raises():
    _skip_if_no_fp8()
    _skip_if_no_cudnn()

    m, n, k = 8, 64, 64
    a_fp32 = torch.randn(m, k, device="cuda")
    a_fp8, _ = _fp8_quantize(a_fp32)
    b_fp32 = torch.randn(n, k, device="cuda")
    b_fp8, _ = _fp8_quantize(b_fp32)
    b_fp8_col = b_fp8.T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="a_scale and b_scale are required"):
        mm_bias(a_fp8, b_fp8_col, bias, out_dtype=torch.bfloat16)


def test_mm_bias_non_fp8_with_scale_raises():
    _skip_if_no_dense_mm_bias()

    m, n, k = 8, 64, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    spurious_scale = torch.tensor(1.0, device="cuda")

    with pytest.raises(ValueError, match="must be None"):
        mm_bias(a, b, bias, a_scale=spurious_scale)


def test_mm_bias_wrong_bias_shape_raises():
    _skip_if_no_dense_mm_bias()

    m, n, k = 8, 64, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n + 1, device="cuda", dtype=torch.bfloat16)  # wrong N

    with pytest.raises(ValueError):
        mm_bias(a, b, bias)


# ---------------------------------------------------------------------------
# is_compute_capability_supported / is_backend_supported
# ---------------------------------------------------------------------------


def test_mm_bias_capability_checks():
    cc = get_compute_capability(torch.device("cuda"))
    cc_num = cc[0] * 10 + cc[1]

    supported = mm_bias.is_compute_capability_supported(cc_num)
    # At least SM80 supports dense, SM89+ supports FP8
    if cc_num >= 80:
        assert supported, f"mm_bias should be supported on SM{cc_num}"

    assert mm_bias.is_backend_supported("cudnn") == CUDNN_AVAILABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
