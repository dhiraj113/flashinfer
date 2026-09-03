"""
Copyright (c) 2025 by FlashInfer team.

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

"""Correctness tests for flashinfer.top_k_varlen.

On Blackwell (sm_100+) with ``pre_idx`` supplied the GVR fast path runs.
On other hardware the radix fallback is used; most tests still execute.

Test matrix
-----------
test_basic_decode             — dtype × top_k × N × batch; works on all GPUs
test_return_values            — return_values=True correctness
test_next_n                   — next_n=2 (V3.2 speculative-decode stride)
test_compress_ratio           — compress_ratio=4 (DSv4 KV compression)
test_preallocated_outputs     — pre-allocated out_indices / out_values
test_large_batch              — stress: large batch × long rows
test_repeated_calls           — same inputs twice → same top-K set
test_no_pre_idx_selects_radix — pre_idx=None → a radix backend (never GVR), correct
test_lb_config_validation     — GvrTopKLBConfig bad args raise at construction
test_load_balance_modes       — True/False GVR paths correct
test_gvr_row_width_alignment  — GVR rejects non-vec-aligned N
test_radix_cutlass_*          — masked CUTLASS radix (any GPU) coverage
test_auto_gvr_knobs_256bit_alignment_gate  — 256-bit gated on 32B N alignment
test_lb_256bit_misaligned_no_crash  — N=4104 LB regression (latent crash fixed)
test_auto_gvr_knobs_shape_aware  — auto() picks shape-appropriate launch config

radix (CuTe DSL) backend — Blackwell only
-----------------------------------------
test_radix_basic              — single-CTA correctness across dtype/K/batch
test_radix_multi_cta_regime   — ctas_per_group > 1 (SMEM split + small-batch fan-out;
                                covers the N=131072 SMEM-overflow regression)
test_radix_next_n / _compress_ratio / _return_values / _preallocated_outputs
test_varlen_ragged            — distinct per-row seq_lens (radix + radix_cutlass)
test_seq_len_equals_top_k     — degenerate seq_len == top_k selects all valid indices

Cross-cutting
-------------
test_cuda_graph_radix_multi_cta — capture/replay incl. fresh-data replay (row_states guard)
test_cuda_graph_gvr           — GVR under CUDA graph
test_backend_heuristic_priority — auto priority gvr > radix > radix_cutlass
test_cross_backend_value_consistency — all backends select the same value multiset
test_unknown_backend_rejected — unregistered / pre-rename backend names rejected
test_input_validation         — 1-D logits / non-int32 seq_lens rejected
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk_varlen.kernels.config import GvrTopKLBConfig
    from flashinfer.cute_dsl.utils import is_cute_dsl_available
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False
    GvrTopKLBConfig = None

pytestmark = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE, reason="flashinfer not installed"
)


# True only on Blackwell (sm_100+) with nvidia-cutlass-dsl installed.
# Use the public is_backend_supported() method exposed by @backend_requirement.
def _gvr_hw_supported() -> bool:
    if not torch.cuda.is_available() or not _FLASHINFER_AVAILABLE:
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    cc = major * 10 + minor
    return (
        flashinfer.top_k_varlen.is_backend_supported("gvr", cc)
        and is_cute_dsl_available()
    )


_IS_BLACKWELL = _gvr_hw_supported()

requires_blackwell = pytest.mark.skipif(
    not _IS_BLACKWELL,
    reason="GVR fast path requires Blackwell (sm_100+) and nvidia-cutlass-dsl",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(num_rows, N, top_k, dtype, seed, next_n=1, compress_ratio=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    num_groups = num_rows // next_n
    effective_len = N - next_n + 1
    argmax_idx = logits[::next_n, :effective_len].argmax(dim=-1).int()
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j
    seq_lens = torch.full(
        (num_groups,), N * compress_ratio, dtype=torch.int32, device="cuda"
    )
    return logits, pre_idx, seq_lens


def _check_correct(
    indices,
    logits,
    seq_lens,
    top_k,
    next_n=1,
    compress_ratio=1,
    require_all_checked=False,
):
    """Every selected value must be >= the k-th largest in its row.

    With ``require_all_checked=True`` every row must be non-degenerate
    (``N_eff >= top_k``) and actually verified — this turns the otherwise-silent
    "skip degenerate row" branch into a hard failure, guarding against a
    mis-parametrized test that quietly checks nothing.
    """
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    n_checked = 0
    for row in range(indices.shape[0]):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            if require_all_checked:
                raise AssertionError(
                    f"row={row}: N_eff={N_eff} < top_k={top_k} — degenerate row "
                    f"not allowed under require_all_checked"
                )
            continue
        row_logits = logits_f32[row, :N_eff]
        kth_value = torch.topk(row_logits, k=top_k).values[-1].item()
        sel = [int(i) for i in indices[row].cpu().tolist() if i >= 0]
        assert len(sel) == top_k, f"row={row}: got {len(sel)} indices, want {top_k}"
        assert len(set(sel)) == len(sel), f"row={row}: duplicate indices"
        assert all(i < N_eff for i in sel), f"row={row}: out-of-range index"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        assert (sel_vals < kth_value).sum() == 0, (
            f"row={row}: some selected values below kth-rank ({kth_value:.6f})"
        )
        n_checked += 1
    if require_all_checked:
        assert n_checked == indices.shape[0], (
            f"only {n_checked}/{indices.shape[0]} rows were verified"
        )


def _make_varlen_inputs(seq_len_list, N, dtype, seed):
    """Ragged batch: per-row seq_lens vary; no pre_idx (radix backends).

    Returns ``(logits[batch, N], seq_lens[batch] int32)`` where
    ``seq_lens[i] = seq_len_list[i]``.
    """
    batch_size = len(seq_len_list)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
    return logits, seq_lens


def _radix_ctas(N, dtype, batch_size):
    """ctas_per_group the radix (CuTe DSL) backend will use for this shape."""
    from flashinfer.topk_varlen.topk_varlen import _radix_get_chunk_config
    from flashinfer.utils import get_device_sm_count, get_shared_bytes_per_block_optin

    device = torch.device("cuda")
    num_sms = get_device_sm_count(device)
    smem_capacity = get_shared_bytes_per_block_optin(device)
    ctas, _chunk = _radix_get_chunk_config(N, dtype, batch_size, num_sms, smem_capacity)
    return ctas


# ---------------------------------------------------------------------------
# test_basic_decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [4096, 32768])
@pytest.mark.parametrize("batch_size", [1, 32])
def test_basic_decode(dtype, top_k, N, batch_size):
    """top_k_varlen with pre_idx: works on Blackwell (GVR) and any GPU (radix)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if top_k > N:
        pytest.skip("N < top_k")

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=42)
    pre_idx_arg = pre_idx if _IS_BLACKWELL else None

    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    # Correctness is verifiable on any GPU: Blackwell runs GVR (pre_idx), other
    # hardware runs the masked radix_cutlass fallback — both produce a valid top-K.
    _check_correct(indices, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_return_values
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_return_values(dtype, top_k):
    """Returned values must equal logits[row, indices]."""
    N, batch_size = 8192, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)

    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, return_values=True
    )
    torch.cuda.synchronize()

    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype  # auto-allocated values keep the logits dtype
    logits_f32 = logits.float()
    for row in range(batch_size):
        expected = logits_f32[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


# ---------------------------------------------------------------------------
# test_next_n
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("batch_size", [2, 16])
def test_next_n(dtype, top_k, batch_size):
    """next_n=2: two rows share one pre_idx / seq_len entry."""
    next_n, N = 2, 8192
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=7, next_n=next_n
    )

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, next_n=next_n
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


# ---------------------------------------------------------------------------
# test_compress_ratio
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_compress_ratio(dtype, top_k):
    """compress_ratio=4: seq_lens in uncompressed-token space."""
    compress_ratio, N, batch_size = 4, 4096, 8
    logits, pre_idx, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=55, compress_ratio=compress_ratio
    )

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, compress_ratio=compress_ratio
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=compress_ratio)


# ---------------------------------------------------------------------------
# test_preallocated_outputs
# ---------------------------------------------------------------------------


@requires_blackwell
def test_preallocated_outputs():
    """out_indices and out_values passed by caller are written in-place."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")

    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        out_indices=out_i,
        return_values=True,
        out_values=out_v,
    )
    torch.cuda.synchronize()

    assert ret_i is out_i
    assert ret_v is out_v
    _check_correct(out_i, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_large_batch
# ---------------------------------------------------------------------------


@requires_blackwell
def test_large_batch():
    """128 rows × 65536 cols stress test."""
    dtype, top_k, N, batch_size = torch.bfloat16, 1024, 65536, 128
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=9)

    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx)
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_repeated_calls
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_repeated_calls():
    """Repeated identical calls each return a valid top-K (no state corruption).

    Results need not be bit-identical: the radix_cutlass fallback runs with
    deterministic=False, so BF16 values that tie at the K-th boundary let two
    correct calls select different (equally valid) tied indices. Assert each
    call is a correct top-K rather than requiring identical index sets.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=3)
    pre_idx_arg = pre_idx if _IS_BLACKWELL else None

    idx1, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()
    idx2, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()

    _check_correct(idx1, logits, seq_lens, top_k, require_all_checked=True)
    _check_correct(idx2, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# test_no_pre_idx_selects_radix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_no_pre_idx_selects_radix():
    """pre_idx=None resolves auto to a radix backend (never GVR) and is correct.

    On Blackwell auto picks ``radix`` (CuTe DSL); on other hardware it picks
    ``radix_cutlass`` (masked CUTLASS). GVR requires pre_idx, so it is never
    selected here.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=77)

    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=None)
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    # auto without pre_idx must resolve to a radix backend, never gvr.
    assert flashinfer.top_k_varlen.suitable_auto_backends[0] in (
        "radix",
        "radix_cutlass",
    )
    _check_correct(indices, logits, seq_lens, top_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_skip_check_auto_backend():
    """skip_check=True with backend="auto" must not raise TypeError.

    When skip_check=True the decorator still calls heuristic_func with positional
    args (*args from the caller).  The heuristic's old **kwargs signature caused
    TypeError because the positional logits/seq_lens/top_k arguments overflowed
    the single 'suitable_backends' slot.  Spelling out the full signature fixes it.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=77)
    # Must not raise TypeError regardless of hardware.
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, backend="auto", skip_check=True
    )
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)
    _check_correct(indices, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# Radix-backend tests (run on any GPU, backend="radix_cutlass" forced explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_cutlass_return_values(dtype, top_k):
    """radix_cutlass backend: returned values must equal logits[row, indices]."""
    N, batch_size = 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)

    indices, values = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        return_values=True,
        backend="radix_cutlass",
    )
    torch.cuda.synchronize()

    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype  # auto-allocated values keep the logits dtype
    logits_f32 = logits.float()
    for row in range(batch_size):
        expected = logits_f32[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("batch_size", [2, 16])
def test_radix_cutlass_next_n(dtype, top_k, batch_size):
    """radix_cutlass backend: next_n=2 — two rows share one seq_len entry."""
    next_n, N = 2, 8192
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(num_rows, N, top_k, dtype, seed=7, next_n=next_n)

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, next_n=next_n, backend="radix_cutlass"
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_cutlass_compress_ratio(dtype, top_k):
    """radix_cutlass backend: compress_ratio=4 — seq_lens in uncompressed-token space."""
    compress_ratio, N, batch_size = 4, 4096, 8
    logits, _, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=55, compress_ratio=compress_ratio
    )

    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        compress_ratio=compress_ratio,
        backend="radix_cutlass",
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=compress_ratio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_radix_cutlass_preallocated_outputs():
    """radix_cutlass backend: out_indices and out_values are written in-place."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")

    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        out_indices=out_i,
        return_values=True,
        out_values=out_v,
        backend="radix_cutlass",
    )
    torch.cuda.synchronize()

    assert ret_i is out_i
    assert ret_v is out_v
    _check_correct(out_i, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_lb_config_validation
# ---------------------------------------------------------------------------


def test_lb_config_validation():
    """GvrTopKLBConfig raises ValueError on invalid arguments."""
    with pytest.raises(ValueError, match="power of 2"):
        GvrTopKLBConfig(max_batch_size=100)
    with pytest.raises(ValueError, match="power of 2"):
        GvrTopKLBConfig(max_batch_size=32)
    with pytest.raises(ValueError, match="power of 2"):
        GvrTopKLBConfig(max_batch_size=2048)
    with pytest.raises(ValueError, match="cluster_size"):
        GvrTopKLBConfig(cluster_size=0)
    with pytest.raises(ValueError, match="num_threads"):
        GvrTopKLBConfig(num_threads=256)


# ---------------------------------------------------------------------------
# test_load_balance_modes — True / False correct
# ---------------------------------------------------------------------------


def _make_ragged_gvr_inputs(top_k, dtype=torch.bfloat16, next_n=1):
    """4 long requests (> 64K threshold) + 12 short requests: a ragged batch.

    With ``next_n > 1``, each request contributes ``next_n`` logit rows, so
    ``logits`` has shape ``(batch_size * next_n, N)``.
    """
    N = 128 * 1024
    seq_len_list = [N] * 4 + [2048] * 12
    batch_size = len(seq_len_list)
    num_rows = batch_size * next_n
    torch.manual_seed(7)
    logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
    logits_f32 = logits.to(torch.float32)
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    for r in range(batch_size):
        # Primary row (nn=0): effective range is [0, seq_len - next_n + 1).
        effective_len = seq_len_list[r] - next_n + 1
        pre_idx[r, 0] = int(logits_f32[r * next_n, :effective_len].argmax().item())
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    return logits, seq_lens, pre_idx


@requires_blackwell
@pytest.mark.parametrize(
    "load_balance,next_n", [(True, 1), (False, 1), (True, 2), (False, 2)]
)
def test_load_balance_modes(load_balance, next_n):
    """load_balance=True/False, next_n=1/2 all produce correct GVR top-K on a ragged batch.

    The next_n=2, load_balance=False combination specifically exercises _run_gvr
    (the single-CTA path) whose order_row was previously constructed with a
    [::next_n] slice bug that made it too short when next_n > 1.
    """
    top_k = 512
    logits, seq_lens, pre_idx = _make_ragged_gvr_inputs(top_k, next_n=next_n)
    num_rows = logits.shape[0]
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        next_n=next_n,
        backend="gvr",
        load_balance=load_balance,
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@requires_blackwell
def test_gvr_lb_workspace_reuse():
    """Caller-provided workspace buffers are reused across GVR LB calls.

    Verifies that passing a pre-allocated workspace dict produces the same
    correct results as the default (locally-allocated) path, and that the
    same buffers can be safely reused across multiple calls.
    """
    top_k, batch_size = 512, 8
    logits, seq_lens, pre_idx = _make_ragged_gvr_inputs(top_k)
    batch_size = seq_lens.shape[0]

    # Compute max_batch_size: smallest power-of-2 in [64, 1024] >= batch_size.
    max_batch_size = next(m for m in (64, 128, 256, 512, 1024) if m >= batch_size)
    workspace = {
        "gvr_order_row": torch.empty(max_batch_size, dtype=torch.int32, device="cuda"),
        "gvr_counters": torch.empty(2, dtype=torch.int32, device="cuda"),
    }

    # First call — workspace gets populated.
    indices0, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", workspace=workspace
    )
    torch.cuda.synchronize()
    _check_correct(indices0, logits, seq_lens, top_k)

    # Second call with same workspace — must give identical correct results.
    indices1, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", workspace=workspace
    )
    torch.cuda.synchronize()
    _check_correct(indices1, logits, seq_lens, top_k)
    assert torch.equal(indices0, indices1), "workspace reuse changed the result"


@requires_blackwell
def test_gvr_no_lb_next_n():
    """load_balance=False with next_n=2: order_row must be request-level, not row-level.

    Regression test for the [::next_n] slice bug: seq_lens already has shape
    (num_requests,), so slicing it with [::next_n] produced an order_row that was
    next_n times too short, causing out-of-bounds kernel accesses when next_n > 1.
    """
    top_k, next_n, N = 512, 2, 8192
    batch_size = 8  # requests
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows, N, top_k, torch.bfloat16, seed=11, next_n=next_n
    )
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        next_n=next_n,
        backend="gvr",
        load_balance=False,
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


# ---------------------------------------------------------------------------
# test_gvr_row_width_alignment — GVR N must be vec-aligned; radix is unconstrained
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,align", [(torch.bfloat16, 8), (torch.float16, 8), (torch.float32, 4)]
)
def test_gvr_row_width_alignment(dtype, align):
    """GVR rejects misaligned N for explicit backend="gvr" and auto-routes past it.

    GVR uses 128-bit vectorized loads, so each row must be 16-byte aligned.
    The suitability check catches this and:
      - raises ValueError for explicit backend="gvr"
      - falls back to radix_cutlass for backend="auto" (pre_idx provided)
    """
    top_k, batch_size = 512, 4
    N_bad = 4096 + 1  # not a multiple of 4 or 8 for any supported dtype
    logits = torch.randn(batch_size, N_bad, dtype=dtype, device="cuda")
    seq_lens = torch.full((batch_size,), N_bad, dtype=torch.int32, device="cuda")
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")

    # Explicit backend="gvr" must fail (alignment check fires in the suitability
    # function; the decorator raises the generic problem-size error).
    with pytest.raises(ValueError, match="not supported"):
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", load_balance=False
        )

    # backend="auto" must succeed by routing to radix_cutlass (no alignment constraint).
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="auto"
    )
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_radix_cutlass_row_width_no_alignment_constraint():
    """radix_cutlass backend accepts any N (no vectorized-load alignment requirement)."""
    top_k, batch_size, N_bad = 512, 4, 4097
    logits = torch.randn(batch_size, N_bad, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.full((batch_size,), N_bad, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_cutlass"
    )
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)


# ---------------------------------------------------------------------------
# Shape-aware launch config (GvrTopKConfig.auto) + 256-bit N-alignment gate
# ---------------------------------------------------------------------------


def test_auto_gvr_knobs_256bit_alignment_gate():
    """_auto_gvr_knobs force-disables 256-bit loads unless N is 32-byte aligned.

    256-bit loads assume 32B-aligned rows (N*itemsize % 32); the up-front N check
    only guarantees 16B. The gate keeps a 256-bit kernel from being selected for a
    16B-but-not-32B-aligned N (which would fault). No GPU needed beyond dtype size.
    """
    from flashinfer.topk_varlen.topk_varlen import _n_is_256bit_aligned

    # bf16 itemsize 2 -> 256-bit needs N % 16 == 0.
    assert _n_is_256bit_aligned(torch.bfloat16, 4096)
    assert not _n_is_256bit_aligned(torch.bfloat16, 4104)  # %16 == 8
    # fp32 itemsize 4 -> 256-bit needs N % 8 == 0.
    assert _n_is_256bit_aligned(torch.float32, 8192)
    assert not _n_is_256bit_aligned(torch.float32, 8196)  # %8 == 4


@requires_blackwell
def test_lb_256bit_misaligned_no_crash():
    """LB on N=4104 bf16 (16B-aligned, NOT 32B) runs correctly, not fault.

    Regression for a latent bug: the LB kernel defaulted to 256-bit loads (32B
    alignment) for all dtypes, faulting on 16B-but-not-32B-aligned N. auto() now
    gates 256-bit off for such N and the 128-bit path runs correctly.
    """
    top_k, N, batch_size = 512, 4104, 16
    assert N % 8 == 0 and (N * 2) % 32 != 0  # 128-bit OK, 256-bit would fault
    torch.manual_seed(31)
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2).to(
        torch.bfloat16
    )
    seq_lens = torch.tensor(
        [N] * 4 + [2048] * (batch_size - 4), dtype=torch.int32, device="cuda"
    )
    lf = logits.float()
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    for r in range(batch_size):
        pre_idx[r, 0] = int(lf[r, : int(seq_lens[r])].argmax().item())
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", load_balance=True
    )
    torch.cuda.synchronize()  # would surface a misaligned-address fault
    _check_correct(indices, logits, seq_lens, top_k)


@requires_blackwell
def test_auto_gvr_knobs_shape_aware():
    """auto() picks a shape-appropriate config: large-N fp32 small-batch -> 1024
    threads + 256-bit + low min_blocks (vs the frozen 512/mb3 old default)."""
    from flashinfer.topk_varlen.topk_varlen import _auto_gvr_knobs

    logits = torch.randn(8, 131072, dtype=torch.float32, device="cuda")
    num_threads, knobs = _auto_gvr_knobs(logits, is_lb=False)
    assert num_threads == 1024
    assert knobs["use_256bit_load"] is True  # fp32, N>=16384, 32B-aligned
    assert knobs["min_blocks_per_mp"] <= 1


# ---------------------------------------------------------------------------
# radix (CuTe DSL) backend — Blackwell only
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_radix_basic(dtype, top_k, batch_size):
    """radix (CuTe DSL) single-CTA correctness across dtype/K/batch."""
    N = 8192  # < max_chunk for all dtypes -> single-CTA
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=42)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix")
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        # SMEM-forced split — the N=131072 shared-memory-overflow regression.
        (torch.bfloat16, 1024, 131072, 64),
        # Small-batch fan-out: one row split across many CTAs to fill the machine.
        (torch.bfloat16, 1024, 65536, 1),
        # fp32 has a smaller max_chunk (57536), so N=65536 forces a split too.
        (torch.float32, 2048, 65536, 32),
        (torch.float32, 2048, 131072, 32),
    ],
)
def test_radix_multi_cta_regime(dtype, top_k, N, batch_size):
    """radix multi-CTA path (ctas_per_group > 1): SMEM split + small-batch fan-out.

    This is the coverage the perf work most needs: the single-CTA-only path
    faulted on rows too large for shared memory (N=131072), and the multi-CTA
    split + global-histogram merge had no committed correctness test.
    """
    ctas = _radix_ctas(N, dtype, batch_size)
    assert ctas > 1, (
        f"expected multi-CTA, got ctas_per_group={ctas} for N={N} batch={batch_size}"
    )
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=101)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix")
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_next_n(dtype, top_k):
    """radix backend: next_n=2 (two rows share one seq_len entry)."""
    next_n, N, batch_size = 2, 8192, 8
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(num_rows, N, top_k, dtype, seed=7, next_n=next_n)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, next_n=next_n, backend="radix"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@requires_blackwell
@pytest.mark.parametrize("top_k,next_n", [(512, 1), (1024, 1), (512, 2), (1024, 2)])
def test_radix_compress_ratio(top_k, next_n):
    """radix backend: compress_ratio=4, varied top_k and next_n.

    Tests both axes independently covered by test_radix_compress_ratio and
    test_radix_next_n, plus their interaction (next_n > 1 with compress_ratio > 1)
    which is where _run_radix's kernel length formula must apply compress_ratio
    after the next_n adjustment, not before.
    """
    dtype, compress_ratio, N, batch_size = torch.bfloat16, 4, 4096, 8
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=55, next_n=next_n, compress_ratio=compress_ratio
    )
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        next_n=next_n,
        compress_ratio=compress_ratio,
        backend="radix",
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, compress_ratio=compress_ratio
    )


@requires_blackwell
def test_radix_next_n_compress_ratio():
    """radix backend: next_n=2 combined with compress_ratio=4.

    Regression test: the next_n per-row adjustment (in token units) must happen
    before dividing by compress_ratio, not after. Pre-dividing seq_lens and then
    subtracting next_n in compressed-index units gives the wrong column bound
    (off by up to compress_ratio-1 columns per row).
    """
    dtype, top_k, next_n, compress_ratio = torch.bfloat16, 512, 2, 4
    N, batch_size = 4096, 8
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=17, next_n=next_n, compress_ratio=compress_ratio
    )
    # _make_inputs fills seq_lens with N*compress_ratio (16384), which is divisible
    # by compress_ratio: there adjust-before-divide and the buggy divide-before-
    # adjust agree (both give 4095), so the regression would slip through. Override
    # to N*compress_ratio + 1 (16385), where the two orders diverge:
    # (16385-2+0+1)//4 = 4096 (correct, all N columns) vs
    # (16385//4)-2+0+1 = 4095 (buggy, one column short).
    seq_lens = torch.full_like(seq_lens, N * compress_ratio + 1)
    # Make the last column a guaranteed top-1 value so the off-by-one is
    # observable. Correct adjust-before-divide gives N_eff = N for every row, so
    # column N-1 is in range and must be selected. The buggy divide-before-adjust
    # gives N_eff = N-1 for the ofs=0 rows, dropping column N-1 from their search
    # window — so its absence from the selected indices flags the regression
    # deterministically (a k-th-value check misses it here: at seed=17 no
    # divergence row otherwise places column N-1 in the top-k).
    logits[:, N - 1] = logits.float().abs().max().item() + 1.0
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        next_n=next_n,
        compress_ratio=compress_ratio,
        backend="radix",
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, compress_ratio=compress_ratio
    )
    # Every row's correct N_eff == N, so the boosted last column must appear in
    # every row's selected indices; a divide-before-adjust bound would drop it
    # from the ofs=0 rows.
    assert (indices == (N - 1)).any(dim=1).all(), (
        "column N-1 (a guaranteed top-1 value) is missing from some row's "
        "top-k — the next_n adjustment was applied after the compress_ratio "
        "divide (divide-before-adjust regression)"
    )


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_return_values(dtype):
    """radix backend: returned values equal logits[row, indices]."""
    top_k, N, batch_size = 512, 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, return_values=True, backend="radix"
    )
    torch.cuda.synchronize()
    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype  # auto-allocated values keep the logits dtype
    lf = logits.float()
    for row in range(batch_size):
        expected = lf[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


@requires_blackwell
@pytest.mark.parametrize("return_values", [True, False])
def test_radix_preallocated_outputs(return_values):
    """radix backend: out_indices written in-place; out_values honoured iff return_values=True.

    Covers the return_values=False + out_values-supplied case: _run_radix must pass
    None to the kernel (compiled with return_output_values=False) even when the caller
    has pre-allocated a values buffer.  Before the fix, the real tensor was forwarded
    unconditionally into a kernel compiled to expect None.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")
    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        out_indices=out_i,
        return_values=return_values,
        out_values=out_v,
        backend="radix",
    )
    torch.cuda.synchronize()
    assert ret_i is out_i
    if return_values:
        assert ret_v is out_v
    else:
        assert ret_v is None
    _check_correct(out_i, logits, seq_lens, top_k)


@requires_blackwell
@pytest.mark.parametrize(
    "backend,load_balance",
    [
        ("radix", None),
        ("radix_primitives", None),
        ("radix_cutlass", None),
        ("gvr", True),
        ("gvr", False),
        ("gvr_2", None),
    ],
)
def test_out_values_ignored_when_return_values_false(backend, load_balance):
    """out_values supplied but return_values=False must not corrupt the kernel call.

    _compile_radix specialises the kernel on return_output_values: when False the
    compiled signature has None for the values slot.  Passing a real tensor there
    (without the 'out_values if return_output_values else None' guard) causes a
    type mismatch.  Covers all backends plus both GVR load-balance paths.
    """
    # gvr_2 is fp32-only; the other backends keep the original bf16 coverage.
    dtype = torch.float32 if backend == "gvr_2" else torch.bfloat16
    top_k, N, batch_size = 512, 8192, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=99)
    # Pre-allocate a values buffer but deliberately do NOT set return_values=True.
    out_v = torch.full((batch_size, top_k), float("nan"), dtype=dtype, device="cuda")

    kwargs = dict(return_values=False, out_values=out_v, backend=backend)
    if backend == "gvr":
        kwargs["pre_idx"] = pre_idx
        kwargs["load_balance"] = load_balance
    elif backend == "gvr_2":
        kwargs["pre_idx"] = pre_idx

    ret_i, ret_v = flashinfer.top_k_varlen(logits, seq_lens, top_k, **kwargs)
    torch.cuda.synchronize()
    # return_values=False → second element must be None regardless of out_values.
    assert ret_v is None
    # Indices must still be correct.
    _check_correct(ret_i, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# Variable-length (true varlen) + degenerate seq_len coverage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", ["radix", "radix_primitives", "radix_cutlass"])
def test_varlen_ragged(backend):
    """Distinct per-row seq_lens: every row is masked to its own length.

    ``_make_inputs`` uses a uniform length, so this is the primary test of the
    varlen masking that ``top_k_varlen`` exists for. All rows are >= top_k so
    ``require_all_checked`` verifies every one.
    """
    if backend in ("radix", "radix_primitives") and not _IS_BLACKWELL:
        pytest.skip("CuTe DSL backends require Blackwell")
    dtype, top_k, N = torch.bfloat16, 512, 8192
    seq_len_list = [top_k, top_k + 1, 1024, 2048, 4096, 6000, 8000, N]
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=88)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend=backend)
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", ["radix", "radix_primitives", "radix_cutlass"])
def test_seq_len_equals_top_k(backend):
    """Degenerate seq_len == top_k: the top-K is exactly all valid indices [0, top_k)."""
    if backend in ("radix", "radix_primitives") and not _IS_BLACKWELL:
        pytest.skip("CuTe DSL backends require Blackwell")
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, seq_lens = _make_varlen_inputs([top_k] * batch_size, N, dtype, seed=64)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend=backend)
    torch.cuda.synchronize()
    for row in range(batch_size):
        sel = set(int(i) for i in indices[row].cpu().tolist() if i >= 0)
        assert sel == set(range(top_k)), (
            f"row={row}: seq_len==top_k must select all [0,{top_k}); got {len(sel)} unique"
        )


# ---------------------------------------------------------------------------
# CUDA graph capture / replay
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_graph_radix_multi_cta():
    """radix multi-CTA under CUDA graph capture/replay.

    Specifically exercises the row_states zero-init + kernel self-reset
    guardrail: a second replay with *fresh* input data must stay correct, i.e.
    the inter-CTA arrival counter must not carry stale state across replays.
    """
    if not _IS_BLACKWELL:
        pytest.skip("radix (CuTe DSL) requires Blackwell")
    dtype, top_k, N, batch_size = torch.bfloat16, 1024, 131072, 8
    assert _radix_ctas(N, dtype, batch_size) > 1  # ensure the multi-CTA path
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2).to(
        dtype
    )
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix", out_indices=out_i
        )

    # Warmup on a side stream (JIT compile + row_states alloc) before capture,
    # so capture itself performs no allocation.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)

    # Overwrite the captured input buffer with fresh data and replay again.
    # Zeroing out_i first means a no-op replay (or stale row_states) would leave
    # zeros and fail the check — so passing proves the kernel truly re-executes.
    fresh = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 3).to(
        dtype
    )
    logits.copy_(fresh)
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("load_balance", [False, True])
def test_cuda_graph_gvr(load_balance):
    """GVR under CUDA graph capture/replay — both single-CTA and LB paths.

    ``load_balance=True`` is the documented default whose docstring promises
    CUDA-graph safety; it runs the two-kernel prepare+main path with device-side
    counters/order_row. A ragged batch (long + short rows) exercises both LB
    branches. Zeroing ``out_i`` before the second replay proves the kernel
    re-executes rather than passing on stale warmup output.
    """
    top_k = 512
    logits, seq_lens, pre_idx = _make_ragged_gvr_inputs(top_k)
    batch_size = seq_lens.shape[0]
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            pre_idx=pre_idx,
            backend="gvr",
            load_balance=load_balance,
            out_indices=out_i,
        )

    # Warmup on a side stream so the first LB allocation (order_row / counters)
    # happens outside capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)

    # Zero the output and replay again on the same inputs: a no-op replay (or a
    # counter/order_row that carried stale state) would leave zeros and fail.
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# Auto-selection, cross-backend consistency, and input validation
# ---------------------------------------------------------------------------


def test_backend_heuristic_priority():
    """Auto-selection is shape/dtype-aware and tracks the measured winners.

    Hardware-independent: exercises the heuristic directly with meta tensors
    (only .dtype/.shape are read) so a regression in the decision rules is
    caught even off-GPU. The boundaries are grounded in the B200 sweep
    documented on the heuristic itself.
    """
    from flashinfer.topk_varlen.topk_varlen import _top_k_varlen_heuristic

    def order(suitable, dtype, batch, n_cols):
        logits = torch.empty(batch, n_cols, dtype=dtype, device="meta")
        seq_lens = torch.empty(batch, dtype=torch.int32, device="meta")
        return _top_k_varlen_heuristic(suitable, logits, seq_lens, 1024)

    all4 = ["radix_cutlass", "radix", "gvr_2", "gvr"]  # unordered on purpose

    # fp32 + hint: gvr_2 always first; small problems rank radix over gvr.
    assert order(all4, torch.float32, 1, 8192) == [
        "gvr_2",
        "radix",
        "gvr",
        "radix_cutlass",
    ]
    # fp32 large batch x long rows: gvr ahead of radix; the fp32 big corner
    # (N >= 64K and B*N >= 2^23) ranks radix_cutlass over radix.
    assert order(all4, torch.float32, 256, 131072) == [
        "gvr_2",
        "gvr",
        "radix_cutlass",
        "radix",
    ]
    # B*N = 2^23 but N < 64K: gvr first, radix over radix_cutlass.
    assert order(all4, torch.float32, 256, 32768) == [
        "gvr_2",
        "gvr",
        "radix",
        "radix_cutlass",
    ]
    # bf16 (gvr_2 never suitable): radix wins everywhere below B*N = 2^23...
    assert order(["gvr", "radix", "radix_cutlass"], torch.bfloat16, 64, 65536) == [
        "radix",
        "gvr",
        "radix_cutlass",
    ]
    # ...and gvr only above it; radix_cutlass never leads in half precision.
    assert order(["gvr", "radix", "radix_cutlass"], torch.bfloat16, 256, 131072) == [
        "gvr",
        "radix",
        "radix_cutlass",
    ]
    # no-hint fallbacks
    assert order(["radix", "radix_cutlass"], torch.bfloat16, 256, 131072) == [
        "radix",
        "radix_cutlass",
    ]
    assert order(["radix", "radix_cutlass"], torch.float32, 256, 131072) == [
        "radix_cutlass",
        "radix",
    ]
    assert order(["radix_cutlass"], torch.float32, 1, 4096) == ["radix_cutlass"]

    # radix_filter admission (auto-vs-oracle study, PR #4811): hint-free fp32
    # from 32K columns up, except the single-row case at >= 512K, and at every
    # N once B >= 256 (ahead of the fp32 radix_cutlass corner).
    hf = ["radix_cutlass", "radix", "radix_filter"]
    assert order(hf, torch.float32, 16, 32768) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.float32, 16, 8192) == ["radix", "radix_cutlass"]
    assert order(hf, torch.float32, 1, 65536) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.float32, 1, 524288) == ["radix", "radix_cutlass"]
    assert order(hf, torch.float32, 256, 8192) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.float32, 256, 131072) == [
        "radix_filter",
        "radix_cutlass",
        "radix",
    ]
    # fp32 with a hint but gvr_2 unsuitable: radix_filter ranks ahead of gvr.
    assert order(hf + ["gvr"], torch.float32, 64, 131072)[:2] == ["radix_filter", "gvr"]
    # half precision: gvr only for B >= 256 with 32K-512K columns (the old
    # B*N >= 2^23 rule picked it at B=16 x 2M, a 7x loss to radix); radix_filter
    # in the mid band for small batches and from 128K up for B >= 64.
    hh = ["gvr", "radix", "radix_cutlass", "radix_filter"]
    assert order(hh, torch.bfloat16, 16, 2097152) == ["radix", "gvr", "radix_cutlass"]
    assert order(hh, torch.bfloat16, 256, 131072) == [
        "gvr",
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hh, torch.bfloat16, 256, 32768) == ["gvr", "radix", "radix_cutlass"]
    assert order(hh, torch.bfloat16, 64, 524288) == [
        "radix_filter",
        "radix",
        "gvr",
        "radix_cutlass",
    ]
    assert order(hf, torch.float16, 16, 65536) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.bfloat16, 64, 8192) == ["radix", "radix_cutlass"]
    assert order(hf, torch.bfloat16, 256, 8192) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    # None tensors (skip_check / doc examples): static fallback order.
    assert _top_k_varlen_heuristic(all4, None, None, None) == [
        "gvr_2",
        "gvr",
        "radix",
        "radix_cutlass",
    ]


@requires_blackwell
def test_cross_backend_value_consistency():
    """radix, radix_cutlass, gvr, and gvr_2 select the same top-K *value* multiset.

    Compares sorted selected values (not indices) so ties don't cause spurious
    failures. fp32 keeps ties rare; any real divergence between backends fails.
    """
    dtype, top_k, N, batch_size = torch.float32, 1024, 8192, 8
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=123)
    idx_r, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix")
    idx_p, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    idx_c, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix_cutlass")
    idx_g, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr"
    )
    idx_g2, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr_2"
    )
    torch.cuda.synchronize()
    lf = logits.float()
    for row in range(batch_size):
        vr = lf[row][idx_r[row].long()].sort(descending=True).values
        vp = lf[row][idx_p[row].long()].sort(descending=True).values
        assert torch.allclose(vr, vp, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs radix_primitives value multisets differ"
        )
        vc = lf[row][idx_c[row].long()].sort(descending=True).values
        vg = lf[row][idx_g[row].long()].sort(descending=True).values
        vg2 = lf[row][idx_g2[row].long()].sort(descending=True).values
        assert torch.allclose(vr, vc, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs radix_cutlass value multisets differ"
        )
        assert torch.allclose(vr, vg, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs gvr value multisets differ"
        )
        assert torch.allclose(vr, vg2, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs gvr_2 value multisets differ"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("N,batch", [(262144, 1), (262144, 8), (1048576, 64)])
@pytest.mark.parametrize("hint_kind", ["identity", "random"])
@pytest.mark.parametrize("use_hints", [False, True], ids=["hints_off", "hints_on"])
def test_walkfirst_garbage_hint_no_fallback(
    N, batch, hint_kind, use_hints, monkeypatch
):
    """A garbage pre_idx must never send a row to the exact fallback.  Hints are
    ignored by default (hints_off: the call is identical to hintless); with
    FLASHINFER_TOPK_USE_HINTS=1 the smallest hinted value may only TIGHTEN the
    sample-derived threshold, so a garbage hint changes nothing.  Pre-fix, the
    hint rung replaced the sample and identity hints on the multi-CTA forms
    overflowed staging (6-8x slower at 1M).  Checks exactness and that no row
    reports the fallback path (status block 0)."""
    from flashinfer.topk_varlen.topk_varlen import _experimental_prim_buffers

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("walkfirst_primitives requires Ampere+")
    if use_hints:
        monkeypatch.setenv("FLASHINFER_TOPK_USE_HINTS", "1")
    top_k = 512 if batch == 1 else 1024
    torch.manual_seed(N // 1024 + batch)
    logits = (torch.randn(batch, N, device="cuda") * 2.0).contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    if hint_kind == "identity":
        pre_idx = (
            torch.arange(top_k, dtype=torch.int32, device="cuda")
            .expand(batch, top_k)
            .contiguous()
        )
    else:
        pre_idx = torch.randint(0, N, (batch, top_k), dtype=torch.int32, device="cuda")
    out = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        out_indices=out,
        pre_idx=pre_idx,
        backend="walkfirst_primitives",
    )
    torch.cuda.synchronize()
    ref = torch.topk(logits, top_k, dim=1).values
    got = torch.sort(logits.gather(1, out.long()), dim=1, descending=True).values
    assert torch.equal(got, ref), "garbage hint changed the selected value multiset"
    _, status = _experimental_prim_buffers(batch, logits.device)
    fallback_rows = int((status[:batch] != 0).sum())
    assert fallback_rows == 0, f"{fallback_rows}/{batch} rows took the exact fallback"


def _gvr2_check_complete_exact(out, logits, kv, top_k, ref_vals, sentinel):
    """Every output slot written and the selected value multiset equals the
    reference (rows masked to their own length)."""
    torch.cuda.synchronize()
    assert int((out == sentinel).sum()) == 0, "unwritten output slots"
    for r in range(out.shape[0]):
        idx = out[r].long()
        assert bool((idx >= 0).all()) and bool((idx < int(kv[r])).all()), (
            f"row={r}: index outside the valid window"
        )
        assert idx.numel() == torch.unique(idx).numel(), f"row={r}: duplicate indices"
        got = logits[r][idx].sort(descending=True).values
        assert torch.equal(got, ref_vals[r]), f"row={r}: value multiset differs"


@requires_blackwell
@pytest.mark.parametrize("n_valid", [3072, 4096], ids=["n3072", "n4096"])
def test_gvr2_high_anchor_hint_completeness(n_valid):
    """Port of TensorRT-LLM PR #18501's regression: anchor-only hints whose
    gathered values all sit ABOVE the true k-th value (an argmax anchor over
    the all-zero cold-start buffer, with row[0] = second-max) bracket the
    sampling band so it holds fewer than top_k entries.  The register-family
    kernel must then escape to the key-space ranking instead of stopping at
    the histogram total (pre-fix: out[tot:k) left unwritten -- 130,304 of
    131,072 slots per cell here)."""
    top_k, bs = 512, 256
    gen = torch.Generator(device="cuda").manual_seed(top_k + n_valid)
    logits = torch.randn(
        (bs, n_valid), generator=gen, dtype=torch.float32, device="cuda"
    )
    logits[:, 0] = torch.topk(logits, 2, dim=1).values[:, 1]
    ref_vals = torch.topk(logits, top_k, dim=1).values
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = logits.argmax(dim=1).to(torch.int32)
    kv = torch.full((bs,), n_valid, dtype=torch.int32, device="cuda")
    out = torch.full((bs, top_k), -7, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits, kv, top_k, out_indices=out, pre_idx=pre_idx, backend="gvr_2"
    )
    _gvr2_check_complete_exact(out, logits, kv, top_k, ref_vals, -7)


@requires_blackwell
def test_gvr2_neginf_tail_completeness():
    """Port of TensorRT-LLM PR #18501's second regression: an in-window -inf in
    the row's tail column (n_valid % 4 == 1) drags the hint-free bracket to
    -inf, every classify product becomes NaN and the histogram total is zero
    (pre-fix: whole rows unwritten).  Odd rows keep fewer than top_k finite
    entries so the -inf tie class exercises the escape's fill-lane bound
    (pre-fix: duplicate indices)."""
    top_k, bs, npad, n_valid = 1024, 256, 4096, 4093
    gen = torch.Generator(device="cuda").manual_seed(top_k + n_valid)
    logits = torch.randn((bs, npad), generator=gen, dtype=torch.float32, device="cuda")
    logits[:, n_valid:] = 3e38  # poison past the window
    logits[:, n_valid - 1] = float("-inf")  # in-window -inf in the tail column
    logits[1::2, 500:n_valid] = float("-inf")  # odd rows: n_finite < top_k
    masked = logits.clone()
    masked[:, n_valid:] = float("-inf")
    ref_vals = torch.topk(masked, top_k, dim=1).values
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device="cuda")
    kv = torch.full((bs,), n_valid, dtype=torch.int32, device="cuda")
    out = torch.full((bs, top_k), -7, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits, kv, top_k, out_indices=out, pre_idx=pre_idx, backend="gvr_2"
    )
    _gvr2_check_complete_exact(out, logits, kv, top_k, ref_vals, -7)


@requires_blackwell
@pytest.mark.xfail(
    strict=True,
    reason="gvr_2 drops a +inf entry (classify transform maps +inf to NaN); "
    "DKG issue #58, listed as an adjacent defect in TensorRT-LLM PR #18501",
)
def test_gvr2_plus_inf_selected():
    """A single +inf must be in the top-k (finite + inf inputs are inside the
    kernel's exactness contract).  Strict xfail: flips to a failure -- i.e.
    remove the marker -- once the upstream fix lands."""
    top_k, n = 1024, 4096
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((1, n), generator=gen, dtype=torch.float32, device="cuda")
    logits[0, 17] = float("inf")
    kv = torch.full((1,), n, dtype=torch.int32, device="cuda")
    pre_idx = (
        torch.arange(top_k, dtype=torch.int32, device="cuda").view(1, -1).contiguous()
    )
    out, _ = flashinfer.top_k_varlen(
        logits, kv, top_k, pre_idx=pre_idx, backend="gvr_2"
    )
    torch.cuda.synchronize()
    assert bool((out == 17).any()), "+inf entry missing from the top-k"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_unknown_backend_rejected():
    """Unregistered backend names — including the pre-rename 'radix_cutedsl' — raise.

    Matches the specific rejection error (not a bare Exception) so an unrelated
    failure — OOM, a missing dependency, an input assertion — cannot satisfy it.
    """
    from flashinfer.utils import BackendSupportedError

    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=5)
    for bad in ("radix_cutedsl", "not_a_backend"):
        with pytest.raises((BackendSupportedError, ValueError), match=bad):
            flashinfer.top_k_varlen(logits, seq_lens, top_k, backend=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_input_validation():
    """1-D logits and non-int32 seq_lens are rejected with ValueErrors (real
    exceptions with a message, so the checks also hold under ``python -O``)."""
    top_k = 512
    logits = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.full((4,), 4096, dtype=torch.int32, device="cuda")
    # logits must be 2-D
    with pytest.raises(ValueError, match="2-D CUDA"):
        flashinfer.top_k_varlen(logits[0], seq_lens[:1], top_k)
    # seq_lens must be int32
    with pytest.raises(ValueError, match="int32"):
        flashinfer.top_k_varlen(logits, seq_lens.long(), top_k)


def _malformed_hints(batch, top_k):
    dev = "cuda"
    return {
        "transposed": torch.zeros(top_k, batch, dtype=torch.int32, device=dev).t(),
        "wrong_batch": torch.zeros(batch // 2, top_k, dtype=torch.int32, device=dev),
        "wrong_width": torch.zeros(batch, top_k // 2, dtype=torch.int32, device=dev),
        "int64": torch.zeros(batch, top_k, dtype=torch.int64, device=dev),
        "cpu": torch.zeros(batch, top_k, dtype=torch.int32),
        "misaligned": torch.zeros(batch * top_k + 4, dtype=torch.int32, device=dev)[
            1 : 1 + batch * top_k
        ].view(batch, top_k),
        "three_d": torch.zeros(batch, top_k, 1, dtype=torch.int32, device=dev),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "kind",
    [
        "transposed",
        "wrong_batch",
        "wrong_width",
        "int64",
        "cpu",
        "misaligned",
        "three_d",
    ],
)
def test_malformed_hint_is_discarded_with_warning(kind):
    """A malformed ``pre_idx`` (wrong shape, dtype, device, layout or
    alignment) is dropped with a RuntimeWarning and the call runs hint-free
    and exact, under ``auto`` and under an explicit hint-free backend. The
    hint-consuming backends refuse the call up front instead of failing
    inside the kernel (they cannot run without a hint)."""
    from flashinfer.utils import BackendSupportedError

    batch, n, top_k = 4, 8192, 1024
    torch.manual_seed(11)
    logits = torch.randn(batch, n, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), n, dtype=torch.int32, device="cuda")
    bad = _malformed_hints(batch, top_k)[kind]
    ref = torch.sort(torch.topk(logits, top_k, dim=1).values, dim=1).values
    with pytest.warns(RuntimeWarning, match="pre_idx"):
        indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=bad)
    assert bool(((indices >= 0) & (indices < n)).all()), "index out of range"
    assert torch.equal(torch.sort(logits.gather(1, indices.long()), dim=1).values, ref)
    # explicit hint-free backend (radix on Blackwell+, radix_cutlass elsewhere):
    # the hint is dropped with the same warning and the result stays exact
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    hint_free = (
        "radix"
        if flashinfer.top_k_varlen.is_backend_supported("radix", cc)
        else "radix_cutlass"
    )
    with pytest.warns(RuntimeWarning, match="pre_idx"):
        indices, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, pre_idx=bad, backend=hint_free
        )
    assert bool(((indices >= 0) & (indices < n)).all()), "index out of range"
    assert torch.equal(torch.sort(logits.gather(1, indices.long()), dim=1).values, ref)
    # explicit hint-consuming backends: refused by their checker up front
    # (the @backend_requirement decorator reports a failed explicit-backend
    # check as ValueError("Problem size is not supported ..."))
    for backend in ("gvr", "gvr_2"):
        if flashinfer.top_k_varlen.is_backend_supported(backend, major * 10 + minor):
            with pytest.raises(
                (BackendSupportedError, ValueError), match="not supported"
            ):
                flashinfer.top_k_varlen(
                    logits, seq_lens, top_k, pre_idx=bad, backend=backend
                )
            # skip_check=True bypasses the checkers: the body must still refuse
            # instead of handing the discarded hint (None) to the kernel host
            with (
                pytest.warns(RuntimeWarning, match="pre_idx"),
                pytest.raises(BackendSupportedError, match="well-formed"),
            ):
                flashinfer.top_k_varlen(
                    logits,
                    seq_lens,
                    top_k,
                    pre_idx=bad,
                    backend=backend,
                    skip_check=True,
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_output_buffer_contract():
    """Caller-provided ``out_indices`` / ``out_values`` must be contiguous, on
    the logits device, of the right dtype and exactly ``[num_rows, top_k]``;
    anything else is a ValueError. The gvr_2 host used to accept a wider
    buffer and pack the result into it at stride ``top_k``, so rows 0 and 1
    of the caller's buffer held two result rows each and the rest stayed
    untouched."""
    batch, n, top_k = 4, 8192, 1024
    logits = torch.randn(batch, n, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), n, dtype=torch.int32, device="cuda")
    bad = {
        "wider": torch.empty(batch, 2 * top_k, dtype=torch.int32, device="cuda"),
        "taller": torch.empty(2 * batch, top_k, dtype=torch.int32, device="cuda"),
        "short_flat": torch.empty(batch * top_k - 1, dtype=torch.int32, device="cuda"),
        # right element count, wrong 2-D shape: would be a silent re-layout
        "transposed_shape": torch.empty(top_k, batch, dtype=torch.int32, device="cuda"),
        "misaligned": torch.empty(batch * top_k + 4, dtype=torch.int32, device="cuda")[
            1 : 1 + batch * top_k
        ].view(batch, top_k),
        "int64": torch.empty(batch, top_k, dtype=torch.int64, device="cuda"),
        "non_contiguous": torch.empty(
            top_k, batch, dtype=torch.int32, device="cuda"
        ).t(),
        "cpu": torch.empty(batch, top_k, dtype=torch.int32),
    }
    for buf in bad.values():
        with pytest.raises(ValueError, match="out_indices"):
            flashinfer.top_k_varlen(logits, seq_lens, top_k, out_indices=buf)
    with pytest.raises(ValueError, match="out_values"):
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            return_values=True,
            out_values=torch.empty(batch, top_k, dtype=torch.bfloat16, device="cuda"),
        )
    # a flat buffer with exactly num_rows * top_k elements is viewed in place
    # (the contract the radix_filter in-place test relies on), for every backend
    flat_i = torch.full((batch * top_k,), -7, dtype=torch.int32, device="cuda")
    idx, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, out_indices=flat_i)
    assert idx.data_ptr() == flat_i.data_ptr() and tuple(idx.shape) == (batch, top_k)
    assert int((flat_i == -7).sum()) == 0
    good_i = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    good_v = torch.empty(batch, top_k, dtype=torch.float32, device="cuda")
    idx, vals = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        return_values=True,
        out_indices=good_i,
        out_values=good_v,
    )
    assert idx.data_ptr() == good_i.data_ptr() and vals.data_ptr() == good_v.data_ptr()


# ---------------------------------------------------------------------------
# Coverage hardening (from the critical review): multi-CTA values, LB caps,
# degenerate short rows, radix_cutlass under CUDA graph.
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        (torch.bfloat16, 1024, 131072, 64),  # SMEM-split multi-CTA
        (torch.float32, 2048, 65536, 32),  # fp32 multi-CTA
    ],
)
def test_radix_multi_cta_return_values(dtype, top_k, N, batch_size):
    """radix return_values on the multi-CTA path: the inter-CTA histogram-merge
    value-gather is otherwise unverified (single-CTA value tests don't cover it)."""
    assert _radix_ctas(N, dtype, batch_size) > 1
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=202)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix", return_values=True
    )
    torch.cuda.synchronize()
    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    lf = logits.float()
    for row in range(batch_size):
        expected = lf[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: multi-CTA values do not match logits[row, indices]"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", ["radix", "radix_primitives", "radix_cutlass"])
def test_seq_len_less_than_top_k(backend):
    """Rows with seq_len < top_k: every valid index [0, seq_len) is selected.

    The two backends pad the surplus slots differently — ``radix`` writes the
    ``-1`` sentinel, ``radix_cutlass`` leaves masked-region indices (>= seq_len)
    — so this asserts the backend-agnostic guarantee (all valid entries chosen,
    unique, in-range) rather than a specific padding representation.
    """
    if backend in ("radix", "radix_primitives") and not _IS_BLACKWELL:
        pytest.skip("CuTe DSL backends require Blackwell")
    dtype, top_k, N = torch.bfloat16, 512, 4096
    seq_len_list = [top_k - 1, top_k // 2, 17, 1]  # all strictly < top_k
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=71)
    # return_values=True exercises the value-gather path. With seq_len < top_k the
    # kernel writes the -1 sentinel into surplus slots, so radix_cutlass's gather
    # must clamp the index (a raw -1 trips a device-side bounds assert) and zero
    # those slots — this combination guards that regression.
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend=backend, return_values=True
    )
    torch.cuda.synchronize()
    lf = logits.float()
    for row, sl in enumerate(seq_len_list):
        idx = indices[row]
        in_range = sorted(i for i in idx.cpu().tolist() if 0 <= i < sl)
        assert in_range == list(range(sl)), (
            f"{backend} row={row} seq_len={sl}: expected all valid indices "
            f"[0,{sl}); got {len(in_range)} unique in-range"
        )
        # Values at valid slots must equal logits[row, idx].
        valid = (idx >= 0) & (idx < sl)
        if valid.any():
            got = values[row][valid].float()
            exp = lf[row][idx[valid].long()]
            assert torch.allclose(got, exp, rtol=1e-3, atol=1e-3), (
                f"{backend} row={row}: gathered values mismatch logits[row, idx]"
            )
        # radix_cutlass zeros the -1 sentinel slots; assert it did.
        if backend == "radix_cutlass":
            sentinel = idx < 0
            if sentinel.any():
                assert (values[row][sentinel] == 0).all(), (
                    f"{backend} row={row}: sentinel (-1) value slots not zeroed"
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("next_n", [1, 2])
def test_cuda_graph_radix_cutlass(next_n):
    """radix_cutlass (the non-Blackwell auto default) under CUDA graph replay.

    Also exercises next_n>1 (the repeat_interleave/arange masking branch) under
    capture. Fresh-data replay proves re-execution.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 8192, 8
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=44, next_n=next_n
    )
    out_i = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            next_n=next_n,
            backend="radix_cutlass",
            out_indices=out_i,
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(
        out_i, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )

    fresh = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 3).to(dtype)
    logits.copy_(fresh)
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(
        out_i, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )


def test_lb_max_batch_size_boundaries():
    """_lb_max_batch_size rounds up to the next power-of-2 cap in [64, 1024]."""
    from flashinfer.topk_varlen.topk_varlen import _lb_max_batch_size

    assert _lb_max_batch_size(1) == 64
    assert _lb_max_batch_size(64) == 64
    assert _lb_max_batch_size(65) == 128
    assert _lb_max_batch_size(256) == 256
    assert _lb_max_batch_size(512) == 512
    assert _lb_max_batch_size(1024) == 1024
    with pytest.raises(ValueError):
        _lb_max_batch_size(1025)


# ---------------------------------------------------------------------------
# radix_primitives (CuTe DSL primitives API, coarse-histogram) backend
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_radix_primitives_basic(dtype, top_k, batch_size):
    N = 8192
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=31)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        # Shapes that force the *radix* backend multi-CTA; radix_primitives
        # streams them on one CTA per row with no SMEM staging at all.
        (torch.bfloat16, 1024, 131072, 64),
        (torch.bfloat16, 1024, 65536, 1),
        (torch.float32, 2048, 65536, 32),
        (torch.float32, 2048, 131072, 32),
    ],
)
def test_radix_primitives_large_n(dtype, top_k, N, batch_size):
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=33)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k,next_n", [(512, 2), (1024, 2)])
def test_radix_primitives_next_n(dtype, top_k, next_n):
    N, num_rows = 8192, 8
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=35, next_n=next_n
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, next_n=next_n, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@requires_blackwell
def test_radix_primitives_compress_ratio():
    dtype, top_k, N, batch_size, cr = torch.bfloat16, 512, 8192, 4, 4
    logits, _, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=37, compress_ratio=cr
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, compress_ratio=cr, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=cr)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_return_values(dtype):
    top_k, N, batch_size = 1024, 8192, 8
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=39)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives", return_values=True
    )
    torch.cuda.synchronize()
    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    lf = logits.float()
    for row in range(batch_size):
        expected = lf[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


@requires_blackwell
def test_radix_primitives_unaligned_rows():
    """Row width not a multiple of the 16B vector: exercises the scalar
    prologue/tail split (row byte address changes alignment per row)."""
    dtype, top_k, N = torch.bfloat16, 512, 8190  # N*2 % 16 != 0
    seq_len_list = [N, 7000, 4096, 513]
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=41)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_heavy_ties(dtype):
    """Coarsely quantized logits: the threshold bin holds many exact
    duplicates, exercising the in-smem exact tie select (eq_count > remaining
    but <= TIE_CAP)."""
    top_k, N, batch_size = 512, 8192, 4
    torch.manual_seed(43)
    logits = (
        torch.randint(0, 64, (batch_size, N), device="cuda").to(torch.float32) / 8.0
    ).to(dtype)
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_tie_overflow(dtype):
    """Near-constant rows: >TIE_CAP elements share the threshold bin, forcing
    the exact gmem refinement path.  Boosted columns must still be selected."""
    top_k, N = 512, 8192
    logits = torch.zeros(2, N, dtype=dtype, device="cuda")
    logits[0, 100] = 5.0
    logits[0, 7000] = 4.0
    seq_lens = torch.full((2,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    for row in range(2):
        sel = indices[row].long()
        assert sel.unique().numel() == top_k, f"row={row}: duplicate indices"
        assert (sel >= 0).all() and (sel < N).all(), f"row={row}: out-of-range"
    picked = set(indices[0].cpu().tolist())
    assert 100 in picked and 7000 in picked, "boosted columns must be selected"


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_radix_primitives_16bit_inbin_tie_overflow(dtype):
    """Regression: 16-bit tie-overflow refinement must OR the coarse-bin top
    bits into the final pivot key.

    All elements share ONE coarse bin (values are 8 consecutive ULPs of 1.0;
    a 13-bit bin leaves exactly 3 key bits free), so eq_count = N > TIE_CAP
    forces the exact refinement path.  The buggy pivot held only the refined
    low byte, so ``key > pivot`` admitted nearly every bin member and the row
    filled in atomic arrival order, dropping genuinely-larger values.
    """
    top_k, N = 1024, 8192
    ulp = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-10
    torch.manual_seed(45)
    logits = (
        1.0 + torch.randint(0, 8, (3, N), device="cuda").to(torch.float32) * ulp
    ).to(dtype)
    seq_lens = torch.tensor([N, N - 1, top_k + 9], dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_radix_primitives_multi_cta_overflow_cleanup(dtype):
    """Regression: multi-CTA FINAL_MC emission must be fenced from the rank-0
    row cleanup.

    Two-valued rows overflow the tie stage (half the row shares the threshold
    bin), sending every group through the refinement + FINAL_MC path where
    all ranks emit through the shared g_out/g_eqf atomics.  Without that
    barrier, rank 0's cleanup raced those emissions: counters restarted
    mid-row (duplicate indices) and leaked nonzero into the NEXT call, whose
    first output slots then kept stale garbage.  Repeated fresh-data calls on
    the shared group state make the race bite reliably.
    """
    top_k, N = 512, 131072
    seq_lens = torch.tensor([N, top_k + 33], dtype=torch.int32, device="cuda")
    for it in range(6):
        torch.manual_seed(47 + it)
        logits = torch.where(torch.rand(2, N, device="cuda") < 0.5, 1.0, -1.0).to(dtype)
        indices, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix_primitives"
        )
        torch.cuda.synchronize()
        _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
def test_radix_primitives_inf_flood_multirow():
    """Regression: > top_k in-range +inf values must classify as TIES.

    The fp32 float-boundary collect gave the above-top-bin boundary as +inf,
    so ``v >= hi_b`` classified every +inf as GREATER-THAN -- more gt hits
    than the histogram promised.  The scan-collect's positional stores had no
    top_k cap, so the excess spilled past the row's output slots into the
    NEXT row's indices (row 0 always looked fine; every later row picked
    ~half random values, some duplicated).  Multi-row is essential: a single
    row hides the bug (the spill lands out-of-tensor and the first top_k
    emissions happen to be infs, i.e. a correct answer).
    """
    top_k, N, batch = 2048, 32768, 16
    torch.manual_seed(101)
    logits = torch.randn(batch, N, device="cuda")
    logits = torch.where(
        torch.rand(batch, N, device="cuda") < 0.2,
        torch.full_like(logits, float("inf")),
        logits,
    ).contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize(
    "nnan,negative,N,batch",
    [
        (10, False, 32768, 4),  # few NaNs: used to be silently dropped
        (100, False, 32768, 4),  # > tie slack: used to UNDERFILL (-1 pads)
        (3000, False, 32768, 4),  # > top_k: used to return an EMPTY row
        (10, True, 32768, 4),  # negative-sign NaN patterns
        (1500, False, 65536, 2),  # multi-CTA group path
    ],
)
def test_radix_primitives_nan_inputs(nnan, negative, N, batch):
    """Regression: in-range NaNs must rank top (torch.topk semantics).

    The fp32 float-boundary collect classified with ordered compares, which
    every NaN fails, so NaNs counted by the (integer-bin) histogram were
    dropped by the collect: rows underfilled with -1 pads once the NaN count
    exceeded the threshold-bin slack, and came back empty with > top_k NaNs.
    The classify branch is now inverted around a strict-GT threshold
    (coarse_bin_gt_threshold_f32) so NaNs of either sign land in the gt arm.
    torch.equal is NaN-hostile: compare NaN counts + finite values.
    """
    top_k = 2048
    torch.manual_seed(303)
    logits = torch.randn(batch, N, device="cuda")
    for r in range(batch):
        idx = torch.randperm(N, device="cuda")[:nnan]
        logits[r, idx] = float("nan")
        if negative:
            lb = logits[r].view(torch.int32)
            lb[idx] = lb[idx] | torch.tensor(
                -0x80000000, device="cuda", dtype=torch.int32
            )
    logits = logits.contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    for r in range(batch):
        sel = indices[r]
        assert int((sel < 0).sum()) == 0, f"row{r}: -1 pads in a full row"
        sel = sel.long()
        assert int(sel.max()) < N and sel.unique().numel() == top_k, f"row{r}: dup/oor"
        got = logits[r][sel]
        ref = torch.topk(logits[r], top_k).values
        assert int(torch.isnan(got).sum()) == min(nnan, top_k), f"row{r}: NaN count"
        got_fin = got[~torch.isnan(got)].sort(descending=True).values
        ref_fin = ref[~torch.isnan(ref)]
        assert torch.equal(got_fin, ref_fin), f"row{r}: finite part mismatch"


@requires_blackwell
@pytest.mark.parametrize("edge", [2.0, -2.0, 8.0, 0.5])
def test_radix_primitives_binade_edge_tie_overflow(edge):
    """Regression: fp32 coarse bins that straddle a 2^24 ordered-key
    boundary (fp16 binade edges: values near +/-2^m, m odd) must run the
    high-byte refinement round.

    The overflow refinement skips the (24, 8) round when the coarse bin
    provably pins key bits [24, 32); 30 binade-edge bins violate that bound
    (found by adversarial review + exhaustive host enumeration,
    proto_wide_predicate.py).  With the round wrongly skipped, values on
    opposite sides of the boundary (e.g. 2.0 vs its fp32 predecessor, both
    in one coarse bin) get misordered by the masked low-bit compares and
    the strictly-larger values are dropped.
    """
    top_k, N = 2048, 16384
    torch.manual_seed(7)
    below = torch.nextafter(
        torch.tensor(edge, device="cuda"), torch.tensor(0.0, device="cuda")
    )
    logits = torch.full((2, N), -100.0 if edge > 0 else -1e9, device="cuda")
    logits[:, :3000] = below if edge > 0 else edge
    logits[:, 3000:3500] = edge if edge > 0 else below
    for r in range(2):
        logits[r] = logits[r][torch.randperm(N, device="cuda")]
    logits = logits.contiguous()
    seq_lens = torch.full((2,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("N,batch", [(8192, 4), (65536, 2), (32768, 16)])
def test_radix_primitives_approx_ties(N, batch):
    """approx_ties=True: sglang-compatible tie-truncation semantics.

    A row whose threshold coarse bin holds > TIE_CAP distinct candidates is
    filled with an arbitrary first-arrival subset of that bin instead of the
    exact smallest-key refinement.  Contract checked here: full row of
    unique in-range indices, every selected value from the tie bin or above
    (>= 1.0 in this construction), and bit-exact agreement with exact mode
    on rows without tie overflow.
    """
    top_k = 2048
    torch.manual_seed(11)
    # rows 0..: half in-bin (1.0 + eps), half fill at -5.0 -> tie overflow
    logits = torch.full((batch, N), -5.0, device="cuda")
    m = N // 2
    logits[:, :m] = 1.0 + torch.rand(batch, m, device="cuda") * 2**-12
    for r in range(batch):
        logits[r] = logits[r][torch.randperm(N, device="cuda")]
    logits = logits.contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives", approx_ties=True
    )
    torch.cuda.synchronize()
    for r in range(batch):
        sel = indices[r]
        assert int((sel < 0).sum()) == 0, f"row{r}: pads in a full row"
        sel = sel.long()
        assert int(sel.max()) < N and sel.unique().numel() == top_k, f"row{r} dup/oor"
        assert bool((logits[r][sel] >= 1.0).all()), f"row{r}: picked below the tie bin"

    # no-overflow rows: approx must be bit-identical to exact
    torch.manual_seed(12)
    xr = (torch.randn(batch, N, device="cuda") * 2.0).contiguous()
    ia, _ = flashinfer.top_k_varlen(
        xr, seq_lens, top_k, backend="radix_primitives", approx_ties=True
    )
    torch.cuda.synchronize()
    _check_correct(ia, xr, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize(
    "top_k,pattern,dtype",
    [
        (4096, "randn", torch.float32),
        (4096, "randn", torch.bfloat16),
        (3000, "randn", torch.float32),
        (4096, "constant", torch.float32),  # uniform ties, remaining > TIE_CAP
        (4096, "one_bin", torch.float32),  # distinct in-bin, EQFILL after rounds
        (4096, "constant", torch.bfloat16),
    ],
)
def test_radix_primitives_large_top_k(top_k, pattern, dtype):
    """top_k > TIE_CAP (2048): the staged tie machinery caps at TIE_CAP, so
    remaining > TIE_CAP rows resolve through the masked EQFILL arm (exact:
    survivors of all refinement rounds are provably key-identical).  Also
    covers the multi-CTA shape and short rows."""
    torch.manual_seed(21)
    N, batch = 16384, 3
    if pattern == "randn":
        logits = torch.randn(batch, N, device="cuda") * 2.0
    elif pattern == "constant":
        logits = torch.full((batch, N), 1.5, device="cuda")
    else:  # one_bin
        logits = 1.0 + torch.rand(batch, N, device="cuda") * 2**-12
    logits = logits.to(dtype).contiguous()
    # row 2 is short but non-degenerate (N_eff must stay >= top_k for the
    # strict checker); the degenerate short-row -1 fill is covered by
    # test_top_k_varlen's short-row cases at small k.
    seq_lens = torch.tensor([N, N - 3, top_k + 21], dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    # multi-CTA shape (same pattern, so constant/one_bin also exercise the
    # EQFILL_MC arm)
    if pattern == "randn":
        logits2 = torch.randn(1, 65536, device="cuda").to(dtype).contiguous()
    elif pattern == "constant":
        logits2 = torch.full((1, 65536), 1.5, device="cuda").to(dtype).contiguous()
    else:  # one_bin
        logits2 = (
            (1.0 + torch.rand(1, 65536, device="cuda") * 2**-12).to(dtype).contiguous()
        )
    seq2 = torch.full((1,), 65536, dtype=torch.int32, device="cuda")
    idx2, _ = flashinfer.top_k_varlen(logits2, seq2, top_k, backend="radix_primitives")
    torch.cuda.synchronize()
    _check_correct(idx2, logits2, seq2, top_k, require_all_checked=True)


@requires_blackwell
def test_radix_primitives_empty_batch():
    """Zero rows must early-return instead of launching a zero-block grid."""
    logits = torch.empty(0, 4096, dtype=torch.float32, device="cuda")
    seq_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, 512, backend="radix_primitives"
    )
    assert indices.shape == (0, 512)


@requires_blackwell
def test_radix_primitives_multi_cta_mixed_dtypes():
    """bf16 and fp32 multi-CTA groups back-to-back in one process.

    Regression: the two dtypes have different row_states layouts (histogram
    sizes), and the kernel's end-of-launch self-reset only zeroes its own
    layout's offsets.  A shared scratch buffer let bf16's (deliberately
    un-reset) stale tie buffer alias into fp32's group-1 histogram,
    corrupting every row handled by group >= 1.  Buffers are now keyed by
    layout; this test locks that in.  batch=2 ensures group 1 is exercised.
    """
    from flashinfer.topk_varlen.topk_varlen import _prim_get_group_config
    from flashinfer.utils import get_device_sm_count

    nsms = get_device_sm_count(torch.device("cuda"))
    for dtype, N, top_k in (
        (torch.bfloat16, 65536, 1024),
        (torch.float32, 65536, 2048),
        (torch.bfloat16, 131072, 1024),
        (torch.float32, 131072, 2048),
    ):
        cpg, _chunk = _prim_get_group_config(N, dtype, 2, nsms)
        assert cpg > 1, f"expected multi-CTA at batch=2 N={N}, got cpg={cpg}"
        logits, seq_lens = _make_varlen_inputs([N, N - 12345], N, dtype, seed=53)
        indices, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix_primitives"
        )
        torch.cuda.synchronize()
        _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_multi_cta_tie_overflow(dtype):
    """Multi-CTA group + tie overflow: the cooperative gmem refinement path
    (per-round global 256-bin merges) on a near-constant long row."""
    top_k, N = 1024, 65536
    logits = torch.zeros(1, N, dtype=dtype, device="cuda")
    logits[0, 123] = 5.0
    seq_lens = torch.full((1,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    sel = indices[0].long()
    assert sel.unique().numel() == top_k
    assert (sel >= 0).all() and (sel < N).all()
    assert 123 in indices[0].cpu().tolist(), "boosted column must be selected"


@requires_blackwell
@pytest.mark.parametrize("return_values", [True, False])
def test_radix_primitives_preallocated_outputs(return_values):
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 8192, 4
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=45)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")
    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        backend="radix_primitives",
        return_values=return_values,
        out_indices=out_i,
        out_values=out_v,
    )
    torch.cuda.synchronize()
    assert ret_i is out_i
    if return_values:
        assert ret_v is out_v
    else:
        assert ret_v is None
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
def test_cuda_graph_radix_primitives():
    """radix_primitives under CUDA graph capture/replay.

    The backend keeps no persistent device state (no row_states), so the
    guarantee under test is simply that capture/replay re-executes correctly,
    including with fresh input data."""
    dtype, top_k, N, batch_size = torch.bfloat16, 1024, 131072, 8
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2).to(
        dtype
    )
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix_primitives", out_indices=out_i
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)

    fresh = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 3).to(
        dtype
    )
    logits.copy_(fresh)
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# cutlass_primitives: the vendored library as one backend
# ---------------------------------------------------------------------------


def _cutlass_primitives_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16], ids=["f32", "bf16", "f16"]
)
@pytest.mark.parametrize(
    "N,batch,top_k",
    [
        (4096, 256, 512),
        (16384, 64, 1024),
        (16384, 8, 2048),
        (65536, 8, 512),
        (65536, 148, 1024),
        (65536, 256, 2048),
        (262144, 64, 1024),
        (1048576, 8, 1024),
    ],
)
def test_cutlass_primitives_exact(N, batch, top_k, dtype):
    """Value multiset equals torch.topk on every row, full and ragged lengths; every kernel
    the library's router can pick is covered by the (N, batch) grid."""
    torch.manual_seed(N // 1024 + batch)
    logits = (torch.randn(batch, N, device="cuda") * 2.0).to(dtype).contiguous()
    for seq_lens in (
        torch.full((batch,), N, dtype=torch.int32, device="cuda"),
        torch.randint(top_k + 1, N + 1, (batch,), dtype=torch.int32, device="cuda"),
    ):
        out = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, out_indices=out, backend="cutlass_primitives"
        )
        torch.cuda.synchronize()
        _check_correct(out, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("kind", ["constant", "two_values", "short_rows"])
def test_cutlass_primitives_degenerate_rows(kind):
    """Low-entropy rows take the exact fallback; rows shorter than k pad with -1."""
    batch, N, top_k = 16, 65536, 1024
    if kind == "constant":
        logits = torch.full((batch, N), 0.5, device="cuda")
    elif kind == "two_values":
        logits = torch.where(torch.rand(batch, N, device="cuda") < 0.001, 3.0, -1.0)
    else:
        logits = torch.randn(batch, N, device="cuda")
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    if kind == "short_rows":
        seq_lens = torch.tensor(
            [0, 1, 100, 1023, 1024, 1025, 4096, 16384] * 2,
            dtype=torch.int32,
            device="cuda",
        )
    out = torch.full((batch, top_k), -7, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits, seq_lens, top_k, out_indices=out, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    for r in range(batch):
        n_eff = int(seq_lens[r])
        valid = min(n_eff, top_k)
        assert (out[r, valid:] == -1).all(), f"row {r}: padding"
        idx = out[r, :valid].long()
        assert (
            idx.numel() == torch.unique(idx).numel()
            and bool((idx >= 0).all())
            and bool((idx < n_eff).all())
        )
        if valid == top_k:
            got = logits[r, idx].sort(descending=True).values
            ref = torch.topk(logits[r, :n_eff], top_k).values
            assert torch.equal(got, ref), f"row {r}: values differ"


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
def test_cuda_graph_cutlass_primitives():
    """Capture and replay: the library launches on the caller's stream through the TVM-FFI
    environment stream, so it must capture like any other backend."""
    batch, N, top_k = 64, 65536, 1024
    logits = (torch.randn(batch, N, device="cuda") * 2.0).contiguous()
    seq_lens = torch.randint(
        top_k + 1, N + 1, (batch,), dtype=torch.int32, device="cuda"
    )
    out = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, out_indices=out, backend="cutlass_primitives"
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out, logits, seq_lens, top_k, require_all_checked=True)
    logits.copy_((torch.randn(batch, N, device="cuda") * 3.0))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("next_n", [2, 3])
@pytest.mark.parametrize("N", [8192, 65536])
def test_cutlass_primitives_next_n(dtype, top_k, next_n, N):
    """next_n rows share one seq_len entry; row i of a group sees i % next_n more tokens."""
    num_rows = 8 * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=41, next_n=next_n
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, next_n=next_n, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("cr", [2, 4])
@pytest.mark.parametrize("N", [8192, 65536, 262144])
def test_cutlass_primitives_compress_ratio(cr, N):
    """compress_ratio divides the token length into compressed-block units."""
    dtype, top_k, batch_size = torch.bfloat16, 512, 6
    logits, _, _ = _make_inputs(batch_size, N, top_k, dtype, seed=43, compress_ratio=cr)
    # ragged token lengths, all long enough for a full top-k in block units
    g = torch.Generator(device="cuda").manual_seed(43)
    seq_lens = torch.randint(
        (top_k + 1) * cr, N * cr + 1, (batch_size,), device="cuda", generator=g
    ).to(torch.int32)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, compress_ratio=cr, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(
        indices, logits, seq_lens, top_k, compress_ratio=cr, require_all_checked=True
    )


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N, batch", [(8192, 8), (65536, 8), (262144, 16)])
def test_cutlass_primitives_return_values(dtype, N, batch):
    """values equal logits[row, indices] exactly; padding slots carry -inf values."""
    top_k = 1024
    logits, seq_lens = _make_varlen_inputs([N] * batch, N, dtype, seed=45)
    seq_lens[0] = top_k // 2  # a padded row
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    torch.cuda.synchronize()
    assert values.shape == (batch, top_k) and values.dtype == dtype
    _check_correct(indices, logits, seq_lens, top_k)
    for row in range(batch):
        valid = min(top_k, int(seq_lens[row]))
        expected = logits[row][indices[row, :valid].long()]
        assert torch.equal(expected, values[row, :valid]), f"row={row}: values differ"
        assert torch.isneginf(values[row, valid:]).all(), f"row={row}: padding values"
    # preallocated outputs are written in place
    out_i = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch, top_k, dtype=dtype, device="cuda")
    ri, rv = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        backend="cutlass_primitives",
        return_values=True,
        out_indices=out_i,
        out_values=out_v,
    )
    torch.cuda.synchronize()
    assert ri.data_ptr() == out_i.data_ptr() and rv.data_ptr() == out_v.data_ptr()
    # order within a row is unspecified: compare the rows as sorted multisets
    assert torch.equal(rv.float().sort(dim=1).values, values.float().sort(dim=1).values)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N, batch", [(8192, 8), (65536, 8), (262144, 16)])
def test_cutlass_primitives_paged_rows(dtype, N, batch):
    """Logits living in a wider arena (row stride > N) and a column-sliced view: same answers
    as the contiguous copy; a misaligned slice is refused."""
    top_k, pad = 1024, 64
    arena = (torch.randn(batch, N + pad, device="cuda") * 2.0).to(dtype)
    logits = arena[:, :N]
    assert not logits.is_contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    ref, _ = flashinfer.top_k_varlen(
        logits.contiguous(), seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(
        indices, logits.contiguous(), seq_lens, top_k, require_all_checked=True
    )
    lf = logits.float()
    for row in range(batch):
        got = lf[row][indices[row].long()].sort().values
        want = lf[row][ref[row].long()].sort().values
        assert torch.equal(got, want), f"row={row}: arena and contiguous answers differ"
        assert torch.equal(lf[row][indices[row].long()], values[row].float())
    shifted = arena[
        :, 16 : 16 + N
    ]  # a slice starting inside the arena, rows still aligned
    out, _ = flashinfer.top_k_varlen(
        shifted, seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(out, shifted.contiguous(), seq_lens, top_k, require_all_checked=True)
    odd = arena[
        :, 1 : 1 + N
    ]  # 4-byte offset: misaligned rows, copied into a padded arena
    out, _ = flashinfer.top_k_varlen(odd, seq_lens, top_k, backend="cutlass_primitives")
    torch.cuda.synchronize()
    _check_correct(out, odd.contiguous(), seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize(
    "N, dtype", [(4100, torch.float32), (16386, torch.bfloat16), (65541, torch.float32)]
)
def test_cutlass_primitives_unaligned_row_length(N, dtype):
    """Row lengths that are not whole 16-byte vectors: exact, ragged, with values."""
    top_k, batch = 512, 6
    logits, seq_lens = _make_varlen_inputs([N] * batch, N, dtype, seed=49)
    g = torch.Generator(device="cuda").manual_seed(49)
    seq_lens = torch.randint(top_k + 1, N + 1, (batch,), device="cuda", generator=g).to(
        torch.int32
    )
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    for row in range(batch):
        assert torch.equal(logits[row][indices[row].long()], values[row])


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("top_k", [5000, 8192])
@pytest.mark.parametrize("N", [16384, 65536, 262144])
def test_cutlass_primitives_large_k(top_k, N):
    """k above 4096 where the part's shared memory holds the 8K tie stage."""
    batch = 6
    logits, seq_lens = _make_varlen_inputs([N] * batch, N, torch.float32, seed=47)
    from flashinfer.topk_varlen.topk_varlen import (
        _cutlass_primitives_top_k_varlen_check,
    )

    if not _cutlass_primitives_top_k_varlen_check(logits, seq_lens, top_k):
        pytest.skip("the tie stage for this k does not fit this part's shared memory")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
