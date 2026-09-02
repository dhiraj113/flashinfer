"""Register-resident short-row kernel of the walk-first primitives backend.

Rows of at most 16K fp32 (32K fp16/bf16) elements route to
``kernels/regrow_topk_primitives.py`` (whole row in registers, exact coarse
census or, with caller hints, a verified exact-key bar).  This suite checks
exactness against torch.topk on the valid prefix of every row (values as a
multiset, NaN ranked on top) across dtypes, adversarial value patterns,
lengths (full / random / shorter than k) and hint qualities (none, oracle,
stale, garbage, duplicate, all -1), and asserts via the status buffer that the
register-resident kernel really ran and that oracle hints take the hint arm.
"""

import os

import pytest
import torch

import flashinfer
from flashinfer.topk_varlen.topk_varlen import _experimental_prim_buffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

_DEV = torch.device("cuda:0")


def _exact(x, seq, oi, K):
    xf = x.float()
    for r in range(x.shape[0]):
        n = int(seq[r])
        m = min(K, n)
        if m < K:
            assert bool((oi[r][m:] == -1).all()), f"row {r}: padding not -1"
        if m == 0:
            continue
        idx = oi[r][:m]
        assert not bool((idx < 0).any()) and int(idx.max()) < n, (
            f"row {r}: index out of range"
        )
        assert torch.unique(idx).numel() == m, f"row {r}: duplicate index"
        got = torch.sort(xf[r][idx.long()], descending=True).values
        ref = torch.topk(xf[r][:n], m).values
        assert bool(((got == ref) | (got.isnan() & ref.isnan())).all()), (
            f"row {r}: values differ"
        )


def _pattern(name, b, N, dt, g):
    if name == "randn":
        return (torch.randn(b, N, device=_DEV, generator=g) * 2).to(dt)
    if name == "const":
        return torch.full((b, N), 1.5, device=_DEV, dtype=dt)
    if name == "nan_inf":
        x = (torch.randn(b, N, device=_DEV, generator=g) * 2).to(dt)
        x[:, ::97] = float("nan")
        x[:, 5::101] = float("inf")
        x[:, 7::103] = float("-inf")
        return x
    if name == "quant":  # many exact ties
        return (
            (torch.randn(b, N, device=_DEV, generator=g) * 2).to(torch.float16).to(dt)
        )
    raise ValueError(name)


def _hints(mode, x, seq, K, g):
    b, N = x.shape
    if mode == "none":
        return None
    if mode in ("oracle", "stale"):
        xm = torch.where(
            torch.arange(N, device=_DEV)[None] < seq[:, None],
            x.float(),
            torch.tensor(float("-inf"), device=_DEV),
        )
        order = torch.argsort(xm, dim=1, descending=True)
        if mode == "oracle" or K + 64 >= N:
            return order[:, :K].int().contiguous()
        return (
            torch.cat([order[:, : K - 1], order[:, K + 63 : K + 64]], dim=1)
            .int()
            .contiguous()
        )
    if mode == "garbage":
        return torch.randint(-5, N + 5, (b, K), device=_DEV, generator=g).int()
    if mode == "dup":
        return torch.full((b, K), 3, dtype=torch.int32, device=_DEV)
    if mode == "neg":
        return torch.full((b, K), -1, dtype=torch.int32, device=_DEV)
    raise ValueError(mode)


_DTYPES = [torch.float32]
if os.environ.get("FLASHINFER_TOPK_WF_REGROW") in ("1", "all"):
    _DTYPES += [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("N", [64, 4096, 16000, 16384])
@pytest.mark.parametrize("pattern", ["randn", "const", "nan_inf", "quant"])
@pytest.mark.parametrize("K", [1, 512, 2048])
@pytest.mark.parametrize("length", ["full", "uniform", "tiny"])
def test_regrow_exact_hintless(dtype, N, pattern, K, length):
    g = torch.Generator(device=_DEV)
    g.manual_seed(N * 7 + K)
    b = 3 if N >= 16000 else 149
    x = _pattern(pattern, b, N, dtype, g)
    if length == "full":
        seq = torch.full((b,), N, dtype=torch.int32, device=_DEV)
    elif length == "uniform":
        seq = torch.randint(0, N + 1, (b,), device=_DEV, generator=g).int()
    else:
        seq = torch.randint(0, min(N, K + 2) + 1, (b,), device=_DEV, generator=g).int()
    o = torch.full((b, K), -7, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(x, seq, K, out_indices=o, backend="walkfirst_primitives")
    torch.cuda.synchronize()
    _exact(x, seq, o, K)
    _, st = _experimental_prim_buffers(b, _DEV)
    fam = st[b : 2 * b]
    if K < N:
        # every row solved by the register-resident kernel (7 census / 8 hint arm)
        assert bool(((fam == 7) | (fam == 8) | (seq <= K)).all()), fam.tolist()


@pytest.mark.parametrize("hint_mode", ["oracle", "stale", "garbage", "dup", "neg"])
@pytest.mark.parametrize("N", [4096, 16384])
@pytest.mark.parametrize("pattern", ["randn", "quant", "nan_inf"])
@pytest.mark.parametrize("K", [512, 1024, 2048])
def test_regrow_exact_with_hints(hint_mode, N, pattern, K):
    g = torch.Generator(device=_DEV)
    g.manual_seed(N * 3 + K + len(hint_mode))
    b = 64
    x = _pattern(pattern, b, N, torch.float32, g)
    seq = torch.randint(K + 1, N + 1, (b,), device=_DEV, generator=g).int()
    seq[0] = N
    hints = _hints(hint_mode, x, seq, K, g)
    o = torch.full((b, K), -7, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(
        x, seq, K, out_indices=o, backend="walkfirst_primitives", pre_idx=hints
    )
    torch.cuda.synchronize()
    _exact(x, seq, o, K)
    _, st = _experimental_prim_buffers(b, _DEV)
    fam = st[b : 2 * b]
    assert bool(((fam == 7) | (fam == 8)).all()), fam.tolist()
    if hint_mode == "oracle" and pattern == "randn":
        # oracle hints on distinct values verify the bar: the hint arm must engage
        assert bool((fam == 8).all()), fam.tolist()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("hint_mode", ["oracle", "stale", "garbage"])
@pytest.mark.parametrize("N", [16384, 32768])
def test_regrow_16bit_with_hints(dtype, hint_mode, N):
    """16-bit rows route to the register kernel only when hints are usable."""
    g = torch.Generator(device=_DEV)
    g.manual_seed(N + len(hint_mode))
    b, K = 32, 1024
    x = _pattern("randn", b, N, dtype, g)
    seq = torch.randint(K + 1, N + 1, (b,), device=_DEV, generator=g).int()
    hints = _hints(hint_mode, x, seq, K, g)
    o = torch.full((b, K), -7, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(
        x, seq, K, out_indices=o, backend="walkfirst_primitives", pre_idx=hints
    )
    torch.cuda.synchronize()
    _exact(x, seq, o, K)
    _, st = _experimental_prim_buffers(b, _DEV)
    fam = st[b : 2 * b]
    assert bool(((fam == 7) | (fam == 8)).all()), fam.tolist()


def test_regrow_disabled_falls_back_to_walk(monkeypatch):
    """FLASHINFER_TOPK_WF_REGROW=0 keeps the streaming pipeline (family tags != 7/8)."""
    monkeypatch.setenv("FLASHINFER_TOPK_WF_REGROW", "0")
    b, N, K = 4, 16384, 512
    x = torch.randn(b, N, device=_DEV) * 2
    seq = torch.full((b,), N, dtype=torch.int32, device=_DEV)
    o = torch.empty(b, K, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(x, seq, K, out_indices=o, backend="walkfirst_primitives")
    torch.cuda.synchronize()
    _exact(x, seq, o, K)
    _, st = _experimental_prim_buffers(b, _DEV)
    fam = st[b : 2 * b]
    assert not bool(((fam == 7) | (fam == 8)).any()), fam.tolist()
