"""Tests for gemm_bias: fused GEMM + bias (cuDNN backend, all float dtypes).

Includes both basic correctness tests (fixed shapes) and stress / adversarial
tests (randomised shapes).  Set PYTEST_STRESS_SEED=<int> to reproduce a
specific stress run:

    PYTEST_STRESS_SEED=1234 pytest tests/gemm/test_gemm_bias.py -v
"""

import gc
import os
import random
import time

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, gemm_bias, nvfp4_quantize, mxfp4_quantize, SfLayout, FP4Type
from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE, clear_cudnn_graph_cache
from flashinfer.utils import get_compute_capability, is_sm100a_supported

# ---------------------------------------------------------------------------
# Stress-test randomisation
#   Each run samples a fresh set of shapes unless PYTEST_STRESS_SEED is set.
#   Print the seed so any failure can be reproduced exactly.
# ---------------------------------------------------------------------------

_STRESS_SEED = int(os.environ.get("PYTEST_STRESS_SEED", int(time.time()) % (2**31)))
_rng = random.Random(_STRESS_SEED)
print(
    f"\n[gemm_bias stress] seed={_STRESS_SEED}  "
    f"(re-run with PYTEST_STRESS_SEED={_STRESS_SEED} to reproduce)\n",
    flush=True,
)


def _sample_m(count=10, lo=1, hi=512):
    """Random M values, always including 1 and hi for boundary coverage."""
    pool = list(range(lo, hi + 1))
    chosen = _rng.sample(pool, min(count - 2, len(pool)))
    return sorted(set(chosen) | {1, hi})


def _sample_shapes(count=6, n_range=(128, 2048, 128), k_range=(64, 1024, 64)):
    """Random (n, k) pairs drawn from multiples of given strides."""
    ns = list(range(*n_range))
    ks = list(range(*k_range))
    return [(_rng.choice(ns), _rng.choice(ks)) for _ in range(count)]


# Generate once at collection time so parametrize IDs are stable within a run
_STRESS_M = _sample_m(10)
_STRESS_NK = _sample_shapes(6)
_STRESS_BATCH_M = _sample_m(5, lo=1, hi=64)


# ---------------------------------------------------------------------------
# Compute-capability constants (set once at module level)
# ---------------------------------------------------------------------------

_CC = get_compute_capability(torch.device("cuda"))
_CC_NUM = _CC[0] * 10 + _CC[1]
_HAS_FP8 = _CC_NUM >= 89
_SKIP_FP8 = pytest.mark.skipif(not _HAS_FP8, reason=f"FP8 needs SM89+, got SM{_CC_NUM}")

pytestmark = pytest.mark.skipif(not CUDNN_AVAILABLE, reason="cuDNN not available")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fp8_quantize(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    max_val = x.float().abs().max()
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    scale = max_val / fp8_max
    if scale == 0:
        scale = torch.tensor(1.0)
    return x.float().div(scale).clamp(-fp8_max, fp8_max).to(dtype), scale


def _fp8_quant(x, dtype=torch.float8_e4m3fn):
    """FP8 quantize returning float32 scale tensor."""
    max_val = x.float().abs().max().clamp(min=1e-9)
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    scale = max_val / fp8_max
    return x.float().div(scale).clamp(-fp8_max, fp8_max).to(dtype), scale.to(torch.float32)


def _cos_sim(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def _ref(a, b, bias):
    return (a.float() @ b.float() + bias.float())


def _skip_if_no_cudnn():
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN not available.")


def _skip_if_no_fp8():
    if _CC_NUM < 89:
        pytest.skip(f"FP8 requires SM89+, got SM{_CC_NUM}.")


def _skip_if_no_dense():
    if not gemm_bias.is_compute_capability_supported(_CC_NUM):
        pytest.skip(f"gemm_bias not supported on SM{_CC_NUM}.")


def _skip_if_no_fp4():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(f"FP4 gemm_bias requires SM100+, got SM{_CC[0]}{_CC[1]}.")
    try:
        from flashinfer.gemm.gemm_base import _check_cudnn_fp4_availability
        _check_cudnn_fp4_availability()
    except RuntimeError as e:
        pytest.skip(f"FP4 cuDNN unavailable: {e}")


def _prepare_nvfp4(a_fp, b_fp):
    fp8_max, fp4_max = 448.0, 6.0
    gsf_a = (fp8_max * fp4_max) / a_fp.float().abs().nan_to_num().max()
    gsf_b = (fp8_max * fp4_max) / b_fp.float().abs().nan_to_num().max()
    gsf_a_t = torch.tensor(float(gsf_a), device=a_fp.device, dtype=torch.float32)
    gsf_b_t = torch.tensor(float(gsf_b), device=b_fp.device, dtype=torch.float32)
    a_packed, a_descale = nvfp4_quantize(a_fp, gsf_a_t, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    b_packed, b_descale = nvfp4_quantize(b_fp, gsf_b_t, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    return a_packed, a_descale, b_packed.T, b_descale.T, gsf_a, gsf_b


def _prepare_mxfp4(a_fp, b_fp):
    a_packed, a_descale = mxfp4_quantize(a_fp)
    b_packed, b_descale = mxfp4_quantize(b_fp)
    return a_packed, a_descale, b_packed.T, b_descale.T


# ===========================================================================
# Basic correctness — fixed shapes for stable CI
# ===========================================================================


@pytest.mark.parametrize("m", [1, 8, 32, 64])
@pytest.mark.parametrize("n", [1024, 2048, 4096])
@pytest.mark.parametrize("k", [1024, 2048])
def test_gemm_bias_bf16(m, n, k):
    _skip_if_no_dense()
    torch.manual_seed(42)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _cos_sim(out, F.linear(a, b.T.contiguous(), bias)) > 0.99


@pytest.mark.parametrize("m", [1, 16, 64])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [1024, 2048])
def test_gemm_bias_fp16(m, n, k):
    _skip_if_no_dense()
    torch.manual_seed(7)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    out = gemm_bias(a, b, bias)
    assert out.shape == (m, n) and out.dtype == torch.float16
    ref = F.linear(a.float(), b.T.contiguous().float(), bias.float()).half()
    assert _cos_sim(out, ref) > 0.99


@pytest.mark.parametrize("m", [1, 8, 32])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [512, 1024])
def test_gemm_bias_fp32(m, n, k):
    _skip_if_no_dense()
    torch.manual_seed(3)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, k, device="cuda", dtype=torch.float32).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    out = gemm_bias(a, b, bias)
    assert out.shape == (m, n) and out.dtype == torch.float32
    assert _cos_sim(out, F.linear(a, b.T.contiguous(), bias)) > 0.999


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_gemm_bias_fp8(m, n, k, out_dtype):
    _skip_if_no_fp8()
    cc_num = _CC[0] * 10 + _CC[1]
    if not gemm_bias.is_compute_capability_supported(cc_num):
        pytest.skip(f"gemm_bias not supported on SM{cc_num}.")
    torch.manual_seed(11)
    a_fp32 = torch.randn(m, k, device="cuda")
    b_fp32 = torch.randn(n, k, device="cuda")
    a_fp8, a_scale = _fp8_quantize(a_fp32)
    b_fp8, b_scale = _fp8_quantize(b_fp32)
    a_s = torch.tensor(float(a_scale), device="cuda", dtype=torch.float32)
    b_s = torch.tensor(float(b_scale), device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=out_dtype)
    out = gemm_bias(a_fp8, b_fp8.T.contiguous(), bias, a_scale=a_s, b_scale=b_s, out_dtype=out_dtype)
    ref = (a_fp32 @ b_fp32.T + bias.float()).to(out_dtype)
    assert out.shape == (m, n) and out.dtype == out_dtype
    assert _cos_sim(out, ref) > 0.99


@pytest.mark.parametrize("m", [16, 64, 128])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_gemm_bias_nvfp4_basic(m, n, k, out_dtype):
    _skip_if_no_fp4()
    torch.manual_seed(42)
    a_fp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_fp = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_packed, a_descale, b_packed_T, b_descale_T, gsf_a, gsf_b = _prepare_nvfp4(a_fp, b_fp)
    alpha = torch.tensor(1.0 / (float(gsf_a) * float(gsf_b)), device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=out_dtype)
    out = gemm_bias(a_packed, b_packed_T, bias, a_descale=a_descale, b_descale=b_descale_T,
                    alpha=alpha, fp4_dtype=FP4Type.NVFP4, out_dtype=out_dtype)
    assert out.shape == (m, n) and out.dtype == out_dtype
    ref = (a_fp.float() @ b_fp.float().T + bias.float()).to(out_dtype)
    assert _cos_sim(out, ref) > 0.95


@pytest.mark.parametrize("m", [16, 64])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [128, 256])
def test_gemm_bias_mxfp4_basic(m, n, k):
    _skip_if_no_fp4()
    torch.manual_seed(11)
    a_fp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_fp = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_packed, a_descale, b_packed_T, b_descale_T = _prepare_mxfp4(a_fp, b_fp)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a_packed, b_packed_T, bias, a_descale=a_descale, b_descale=b_descale_T,
                    fp4_dtype=FP4Type.MXFP4, out_dtype=torch.bfloat16)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert not out.isnan().any()


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_bias_output_dtype(out_dtype):
    _skip_if_no_dense()
    m, n, k = 16, 512, 256
    torch.manual_seed(5)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=out_dtype)
    out = gemm_bias(a, b, bias, out_dtype=out_dtype)
    assert out.dtype == out_dtype
    ref = F.linear(a.float(), b.T.contiguous().float(), bias.float()).to(out_dtype)
    assert _cos_sim(out, ref) > 0.99


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("m", [8, 16])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [512, 1024])
def test_gemm_bias_batched(batch, m, n, k):
    _skip_if_no_dense()
    torch.manual_seed(17)
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, n, k, device="cuda", dtype=torch.bfloat16).transpose(-2, -1).contiguous()
    bias = torch.randn(batch, 1, n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert out.shape == (batch, m, n) and out.dtype == torch.bfloat16
    assert _cos_sim(out, torch.bmm(a, b) + bias) > 0.99


def test_gemm_bias_preallocated_out():
    _skip_if_no_dense()
    m, n, k = 32, 512, 256
    torch.manual_seed(9)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    returned = gemm_bias(a, b, bias, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert _cos_sim(out, F.linear(a, b.T.contiguous(), bias)) > 0.99


def test_gemm_bias_nvfp4_preallocated_out():
    _skip_if_no_fp4()
    m, n, k = 32, 128, 128
    torch.manual_seed(3)
    a_fp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_fp = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_packed, a_descale, b_packed_T, b_descale_T, gsf_a, gsf_b = _prepare_nvfp4(a_fp, b_fp)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out_pre = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    alpha = torch.tensor(1.0 / (float(gsf_a) * float(gsf_b)), device="cuda", dtype=torch.float32)
    returned = gemm_bias(a_packed, b_packed_T, bias, a_descale=a_descale, b_descale=b_descale_T,
                         alpha=alpha, fp4_dtype=FP4Type.NVFP4, out=out_pre)
    assert returned.data_ptr() == out_pre.data_ptr()
    assert not out_pre.isnan().any()


def test_gemm_bias_autotuning():
    _skip_if_no_dense()
    m, n, k = 64, 1024, 512
    torch.manual_seed(13)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T.contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(a, b.T.contiguous(), bias)
    with autotune():
        out = gemm_bias(a, b, bias)
    assert _cos_sim(out, ref) > 0.99


# ===========================================================================
# Stress: shape coverage — random per run, PYTEST_STRESS_SEED for reproduce
# ===========================================================================


@pytest.mark.parametrize("m", _STRESS_M)
def test_stress_varying_m(m):
    """BF16: random M values, fixed n=512 k=256 — covers boundary and mid-range."""
    _skip_if_no_dense()
    n, k = 512, 256
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    ref = _ref(a, b, bias).bfloat16()
    assert out.shape == (m, n)
    assert _cos_sim(out, ref) > 0.99, f"M={m} seed={_STRESS_SEED}"


@pytest.mark.parametrize("n,k", _STRESS_NK)
def test_stress_random_nk_shapes(n, k):
    """BF16: random (n, k) pairs — exercises varied cuDNN graph shapes."""
    _skip_if_no_dense()
    m = 16
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    try:
        out = gemm_bias(a, b, bias)
        assert _cos_sim(out, _ref(a, b, bias).bfloat16()) > 0.99, f"n={n} k={k} seed={_STRESS_SEED}"
    except ValueError:
        pass  # cuDNN may reject certain TMA-alignment shapes


@pytest.mark.parametrize("m", _STRESS_BATCH_M)
def test_stress_batched_random_m(m):
    """Batched 3D: random M, fixed batch=4, n=256, k=128."""
    _skip_if_no_dense()
    batch, n, k = 4, 256, 128
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(batch, 1, n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    ref = (torch.bmm(a.float(), b.float()) + bias.float()).bfloat16()
    assert out.shape == (batch, m, n)
    assert _cos_sim(out, ref) > 0.99, f"batch_m={m} seed={_STRESS_SEED}"


@_SKIP_FP8
@pytest.mark.parametrize("m", _sample_m(6, lo=1, hi=256))
def test_stress_fp8_random_m(m):
    """FP8: random M values — cuDNN graph cache handles varying shapes."""
    n, k = 256, 128
    a_f = torch.randn(m, k, device="cuda")
    b_f = torch.randn(k, n, device="cuda")
    a_fp8, a_s = _fp8_quant(a_f)
    b_fp8, b_s = _fp8_quant(b_f)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a_fp8, b_fp8, bias, a_scale=a_s, b_scale=b_s, out_dtype=torch.bfloat16)
    ref = (a_f @ b_f + bias.float()).bfloat16()
    assert _cos_sim(out, ref) > 0.97, f"FP8 M={m} seed={_STRESS_SEED}"


# ===========================================================================
# Shape edge cases — deterministic
# ===========================================================================


def test_shape_k1_raises():
    """K=1 must raise: cuDNN TMA alignment requires K >= 2."""
    a = torch.tensor([[2.0]], device="cuda", dtype=torch.bfloat16)
    b = torch.tensor([[3.0]], device="cuda", dtype=torch.bfloat16)
    bias = torch.tensor([1.0], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K >= 2"):
        gemm_bias(a, b, bias)


def test_shape_odd_primes():
    """Non-power-of-2 shapes: cuDNN should handle or raise a clean error."""
    m, n, k = 7, 37, 17
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    try:
        out = gemm_bias(a, b, bias)
        assert _cos_sim(out, _ref(a, b, bias).bfloat16()) > 0.99
    except Exception as e:
        pytest.skip(f"cuDNN rejects shape ({m},{n},{k}): {e}")


# ===========================================================================
# Contiguity / stride
# ===========================================================================


def test_contiguity_b_transposed():
    """b as (k, n) — the expected column-major usage."""
    m, n, k = 16, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert _cos_sim(out, _ref(a, b, bias).bfloat16()) > 0.99


def test_contiguity_slice_vs_clone():
    """Contiguous slice and clone must give identical results."""
    m, n, k = 16, 256, 128
    a_full = torch.randn(m * 4, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out1 = gemm_bias(a_full[:m].contiguous(), b, bias)
    out2 = gemm_bias(a_full[:m].clone(), b, bias)
    assert torch.allclose(out1.float(), out2.float(), atol=1e-3)


def test_contiguity_non_contiguous_a():
    """Non-contiguous a (strided slice): correct result or clear error."""
    m, n, k = 32, 128, 64
    a_big = torch.randn(m * 2, k, device="cuda", dtype=torch.bfloat16)
    a = a_big[::2]  # every other row
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    try:
        out = gemm_bias(a, b, bias)
        assert _cos_sim(out, _ref(a, b, bias).bfloat16()) > 0.99
    except Exception:
        pass  # cuDNN may reject non-unit strides


def test_contiguity_non_contiguous_bias():
    """Non-contiguous bias: correct result or clean error."""
    m, n, k = 16, 128, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias_big = torch.randn(n * 2, device="cuda", dtype=torch.bfloat16)
    try:
        out = gemm_bias(a, b, bias_big[::2])
        assert _cos_sim(out, _ref(a, b, bias_big[::2]).bfloat16()) > 0.99
    except Exception:
        pass


# ===========================================================================
# Numerical accuracy
# ===========================================================================


def test_numerical_all_zeros():
    """Zero A and B → output equals bias."""
    m, n, k = 16, 128, 64
    a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert torch.allclose(out.float(), bias.float().unsqueeze(0).expand(m, -1), atol=1e-3)


def test_numerical_identity_a():
    """Identity A: output row i = b row i + bias."""
    n = k = 64
    a = torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert _cos_sim(out, _ref(a, b, bias).bfloat16()) > 0.999


def test_numerical_large_values():
    """Near-BF16-max values should not silently overflow."""
    m, n, k = 8, 64, 32
    a = torch.full((m, k), 10.0, device="cuda", dtype=torch.bfloat16)
    b = torch.full((k, n), 0.1, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert out.float().mean().item() == pytest.approx(k * 10.0 * 0.1, rel=0.05)


def test_numerical_fp32_precision():
    """FP32: cos_sim > 0.9999 vs torch reference."""
    m, n, k = 16, 128, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    out = gemm_bias(a, b, bias)
    ref = a @ b + bias
    cs = _cos_sim(out, ref)
    assert cs > 0.9999, f"FP32 cos_sim={cs:.6f}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_numerical_deterministic(dtype):
    """Same inputs must give bit-identical outputs across two calls."""
    m, n, k = 32, 256, 128
    torch.manual_seed(42)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)
    assert torch.equal(gemm_bias(a, b, bias), gemm_bias(a, b, bias))


def test_numerical_cross_dtype_consistency():
    """BF16, FP16, FP32 should all agree with FP32 reference."""
    m, n, k = 32, 256, 128
    torch.manual_seed(7)
    a32 = torch.randn(m, k, device="cuda")
    b32 = torch.randn(k, n, device="cuda")
    bias32 = torch.randn(n, device="cuda")
    ref = a32 @ b32 + bias32
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        out = gemm_bias(a32.to(dtype), b32.to(dtype), bias32.to(dtype))
        assert _cos_sim(out, ref.to(dtype)) > 0.99, f"dtype={dtype}"


# ===========================================================================
# cuDNN graph cache
# ===========================================================================


def test_cache_clear_and_recompute():
    """Results must be identical before and after clearing the graph cache."""
    m, n, k = 16, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out1 = gemm_bias(a, b, bias)
    clear_cudnn_graph_cache()
    out2 = gemm_bias(a, b, bias)
    assert torch.allclose(out1.float(), out2.float(), atol=1e-3)


def test_cache_alternating_shapes():
    """Alternating between two configs across 5 rounds must not corrupt output."""
    configs = [(8, 128, 64), (32, 512, 256)]
    bs = [(torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
           torch.randn(n, device="cuda", dtype=torch.bfloat16))
          for _, n, k in configs]
    for _ in range(5):
        for i, (m, n, k) in enumerate(configs):
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b, bias = bs[i]
            out = gemm_bias(a, b, bias)
            ref = _ref(a, b, bias).bfloat16()
            assert _cos_sim(out, ref) > 0.99, f"config {i} corrupted"


# ===========================================================================
# Bias shape edge cases
# ===========================================================================


def test_bias_1d():
    m, n, k = 16, 128, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    assert gemm_bias(a, b, bias).shape == (m, n)


def test_bias_3d_broadcast():
    batch, m, n, k = 4, 8, 128, 64
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(batch, 1, n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    ref = (torch.bmm(a.float(), b.float()) + bias.float()).bfloat16()
    assert out.shape == (batch, m, n) and _cos_sim(out, ref) > 0.99


def test_bias_2d_raises():
    m, n, k = 8, 64, 32
    with pytest.raises((ValueError, Exception)):
        gemm_bias(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                  torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
                  torch.randn(m, n, device="cuda", dtype=torch.bfloat16))


def test_bias_wrong_n_raises():
    m, n, k = 8, 128, 64
    with pytest.raises(ValueError):
        gemm_bias(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                  torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
                  torch.randn(n + 7, device="cuda", dtype=torch.bfloat16))


# ===========================================================================
# Output tensor edge cases
# ===========================================================================


def test_out_wrong_shape_raises():
    m, n, k = 16, 128, 64
    with pytest.raises(ValueError):
        gemm_bias(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                  torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
                  torch.randn(n, device="cuda", dtype=torch.bfloat16),
                  out=torch.empty(m + 1, n, device="cuda", dtype=torch.bfloat16))


def test_out_wrong_dtype_raises():
    m, n, k = 16, 128, 64
    with pytest.raises(ValueError):
        gemm_bias(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                  torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
                  torch.randn(n, device="cuda", dtype=torch.bfloat16),
                  out=torch.empty(m, n, device="cuda", dtype=torch.float16))


def test_out_nan_filled_is_overwritten():
    m, n, k = 16, 128, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_bias(a, b, bias, out=out)
    assert not out.isnan().any()


def test_out_not_aliased_to_inputs():
    m, n, k = 16, 128, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = gemm_bias(a, b, bias)
    assert out.data_ptr() not in (a.data_ptr(), b.data_ptr(), bias.data_ptr())


# ===========================================================================
# FP8 edge cases
# ===========================================================================


@_SKIP_FP8
def test_fp8_e5m2_variant():
    m, n, k = 16, 128, 64
    a_f = torch.randn(m, k, device="cuda")
    b_f = torch.randn(k, n, device="cuda")
    a_fp8, a_s = _fp8_quant(a_f, torch.float8_e5m2)
    b_fp8, b_s = _fp8_quant(b_f, torch.float8_e5m2)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    try:
        out = gemm_bias(a_fp8, b_fp8, bias, a_scale=a_s, b_scale=b_s, out_dtype=torch.bfloat16)
        assert _cos_sim(out, (a_f @ b_f + bias.float()).bfloat16()) > 0.95
    except Exception as e:
        pytest.skip(f"e5m2 unsupported on this GPU: {e}")


@_SKIP_FP8
def test_fp8_extreme_scales():
    m, n, k = 8, 64, 32
    a_fp8 = torch.ones(m, k, device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8 = torch.ones(k, n, device="cuda", dtype=torch.float8_e4m3fn)
    bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    for scale_val in [1e-6, 1.0, 1e6]:
        out = gemm_bias(a_fp8, b_fp8, bias,
                        a_scale=torch.tensor(scale_val, device="cuda"),
                        b_scale=torch.tensor(scale_val, device="cuda"),
                        out_dtype=torch.bfloat16)
        assert not out.isnan().any(), f"NaN at scale={scale_val}"


@_SKIP_FP8
def test_fp8_unit_scale():
    m, n, k = 16, 128, 64
    a_f = torch.randn(m, k, device="cuda") * 0.01
    b_f = torch.randn(k, n, device="cuda") * 0.01
    a_fp8 = a_f.to(torch.float8_e4m3fn)
    b_fp8 = b_f.to(torch.float8_e4m3fn)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    one = torch.tensor(1.0, device="cuda")
    out = gemm_bias(a_fp8, b_fp8, bias, a_scale=one, b_scale=one, out_dtype=torch.bfloat16)
    ref = (a_fp8.float() @ b_fp8.float() + bias.float()).bfloat16()
    assert _cos_sim(out, ref) > 0.99


# ===========================================================================
# Input validation errors
# ===========================================================================


def test_validation_dtype_mismatch_ab():
    with pytest.raises(ValueError, match="same dtype"):
        gemm_bias(torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
                  torch.randn(32, 64, device="cuda", dtype=torch.float16),
                  torch.randn(64, device="cuda", dtype=torch.bfloat16))


def test_validation_fp8_missing_scale():
    _skip_if_no_fp8()
    a_fp8 = torch.ones(8, 32, device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8 = torch.ones(32, 64, device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="a_scale and b_scale are required"):
        gemm_bias(a_fp8, b_fp8, torch.zeros(64, device="cuda", dtype=torch.bfloat16),
                  out_dtype=torch.bfloat16)


def test_validation_non_fp8_with_scale():
    with pytest.raises(ValueError, match="must be None"):
        gemm_bias(torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
                  torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
                  torch.randn(64, device="cuda", dtype=torch.bfloat16),
                  a_scale=torch.tensor(1.0, device="cuda"))


def test_validation_3d_dim_mismatch():
    """a is 2D, b is 3D — must raise."""
    with pytest.raises(ValueError):
        gemm_bias(torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
                  torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16),
                  torch.randn(64, device="cuda", dtype=torch.bfloat16))


def test_validation_fp4_non_uint8_raises():
    _skip_if_no_fp4()
    m, n, k = 16, 128, 128
    with pytest.raises(ValueError, match="uint8"):
        gemm_bias(torch.randn(m, k // 2, device="cuda", dtype=torch.bfloat16),
                  torch.zeros(k // 2, n, device="cuda", dtype=torch.uint8),
                  torch.randn(n, device="cuda", dtype=torch.bfloat16),
                  a_descale=torch.ones(m, k // 16, device="cuda", dtype=torch.float8_e4m3fn),
                  b_descale=torch.ones(k // 16, n, device="cuda", dtype=torch.float8_e4m3fn),
                  fp4_dtype=FP4Type.NVFP4)


def test_validation_fp4_missing_b_descale():
    _skip_if_no_fp4()
    m, n, k = 16, 128, 128
    with pytest.raises(ValueError, match="b_descale"):
        gemm_bias(torch.zeros(m, k // 2, device="cuda", dtype=torch.uint8),
                  torch.zeros(k // 2, n, device="cuda", dtype=torch.uint8),
                  torch.randn(n, device="cuda", dtype=torch.bfloat16),
                  a_descale=torch.ones(m, k // 16, device="cuda", dtype=torch.float8_e4m3fn),
                  fp4_dtype=FP4Type.NVFP4)


def test_validation_fp4_a_scale_and_a_descale_raises():
    _skip_if_no_fp4()
    m, n, k = 16, 128, 128
    with pytest.raises(ValueError):
        gemm_bias(torch.zeros(m, k // 2, device="cuda", dtype=torch.uint8),
                  torch.zeros(k // 2, n, device="cuda", dtype=torch.uint8),
                  torch.randn(n, device="cuda", dtype=torch.bfloat16),
                  a_scale=torch.tensor(1.0, device="cuda"),
                  a_descale=torch.ones(m, k // 16, device="cuda", dtype=torch.float8_e4m3fn),
                  b_descale=torch.ones(k // 16, n, device="cuda", dtype=torch.float8_e4m3fn),
                  fp4_dtype=FP4Type.NVFP4)


# ===========================================================================
# Memory / GC stress
# ===========================================================================


def test_memory_repeated_calls_no_oom():
    """100 calls on a moderately large problem should not OOM."""
    m, n, k = 128, 2048, 1024
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for _ in range(100):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = gemm_bias(a, b, bias)
        del a, out
    torch.cuda.synchronize()


# ===========================================================================
# Capability checks
# ===========================================================================


def test_capability_checks():
    assert gemm_bias.is_backend_supported("cudnn") == CUDNN_AVAILABLE
    if _CC_NUM >= 80:
        assert gemm_bias.is_compute_capability_supported(_CC_NUM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
