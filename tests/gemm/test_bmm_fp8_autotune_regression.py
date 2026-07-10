"""
Regression test for the CublasFp8GemmRunner cache-key bug (issue #3085).

Root cause: ``get_cache_key_extras`` included ``a.shape`` (which contains the
dynamic M dimension).  The autotuner builds the ``ProfilingCacheKey.extras``
field from ``get_cache_key_extras`` both when *storing* a profiling result
(using synthetic / bucketed tensors) and when *looking up* that result (using
the real runtime tensors).  When the real M does not equal the bucket M the
extras differ and the lookup misses, silently falling back to the un-tuned
heuristic (tactic=-1) and discarding every profiling result.

Fix (PR #3437): ``get_cache_key_extras`` returns only dtypes, which are
synthesis-invariant (the same for synthetic and real tensors).  A separate
``_algo_cache_key`` method still includes shapes for the shape-specific
internal cuBLASLt algo-buffer cache.

To reproduce the bug manually, revert ``get_cache_key_extras`` in
``flashinfer/gemm/gemm_base.py`` from::

    return (a.dtype, b.dtype, out.dtype)

back to::

    return (a.shape, b.shape, a.dtype, b.dtype, out.dtype)

then re-run this test — it will fail and print ``[BUG CONFIRMED]``.
"""
import pytest
import torch
import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.autotuner.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import map_to_hybrid_bucket_uncapped
from flashinfer.utils import get_compute_capability


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def test_autotune_cache_hit_across_bucket_boundary():
    """Verify that a profiling result for bucket M=M_PROF is reused at runtime
    for M=M_RUNTIME where M_RUNTIME != M_PROF but both map to the same bucket.

    The bug caused a cache miss here, so every bmm_fp8 call after autotune()
    silently fell back to tactic=-1 (heuristic) instead of using the tuned one.
    """
    if get_compute_capability(torch.device("cuda"))[0] < 10:
        pytest.skip("cublas FP8 GEMM requires SM100+")

    K, N = 2688, 5376

    # M_PROF is its own profiling bucket (synthetic M == real M during profiling).
    # The cache stores: nearest_profile=M_PROF, extras derived from M_PROF tensors.
    M_PROF = 4096
    assert map_to_hybrid_bucket_uncapped(M_PROF) == M_PROF

    # M_RUNTIME maps to the same bucket as M_PROF, but M_RUNTIME != M_PROF.
    # Before the fix, lookup extras used M_RUNTIME shape != stored M_PROF shape
    # -> cache miss.  After the fix, extras are dtypes only -> cache hit.
    M_RUNTIME = 3600
    assert map_to_hybrid_bucket_uncapped(M_RUNTIME) == M_PROF

    B_fp8, B_scale = to_float8(torch.randn(N, K, device="cuda", dtype=torch.bfloat16))
    B_fp8 = B_fp8.t()

    # Phase 1: profile with M=M_PROF so the cache has an entry for bucket M_PROF.
    A_prof, As_prof = to_float8(
        torch.randn(M_PROF, K, device="cuda", dtype=torch.bfloat16)
    )
    with torch.inference_mode(), autotune():
        flashinfer.bmm_fp8(
            A_prof.unsqueeze(0),
            B_fp8.unsqueeze(0),
            As_prof,
            B_scale,
            torch.bfloat16,
            out=None,
            backend="cublas",
        )

    # Confirm a cache entry was stored for bucket M_PROF.
    at = AutoTuner.get()
    prof_entries = [
        (k, v)
        for k, v in at.profiling_cache.items()
        if k.custom_op == "fp8_gemm" and k.nearest_profile[0][1] == M_PROF
    ]
    assert len(prof_entries) == 1, (
        f"Expected exactly one profiling cache entry for bucket M={M_PROF}, "
        f"got {len(prof_entries)}"
    )
    stored_key, stored_val = prof_entries[0]
    stored_extras = stored_key.extras

    # Phase 2: call with M=M_RUNTIME (no re-profiling) and observe the lookup.
    A_rt, As_rt = to_float8(
        torch.randn(M_RUNTIME, K, device="cuda", dtype=torch.bfloat16)
    )

    orig_search_cache = AutoTuner.search_cache
    search_log = []

    def patched_search_cache(self, custom_op, runners, input_shapes, tuning_config, inputs=None):
        result = orig_search_cache(
            self, custom_op, runners, input_shapes, tuning_config, inputs=inputs
        )
        if custom_op == "fp8_gemm":
            is_hit, runner_id, tactic, _ = result
            extras_used = (
                runners[0].get_cache_key_extras(inputs) if inputs is not None else ()
            )
            search_log.append(
                {"hit": is_hit, "tactic": tactic, "extras_used": extras_used}
            )
        return result

    AutoTuner.search_cache = patched_search_cache
    try:
        with torch.inference_mode():
            flashinfer.bmm_fp8(
                A_rt.unsqueeze(0),
                B_fp8.unsqueeze(0),
                As_rt,
                B_scale,
                torch.bfloat16,
                out=None,
                backend="cublas",
            )
    finally:
        AutoTuner.search_cache = orig_search_cache

    assert search_log, "search_cache was never called for fp8_gemm"
    final = search_log[-1]
    lookup_extras = final["extras_used"]

    # With the fix, extras are dtypes only -> identical for M_PROF and M_RUNTIME.
    # With the bug, extras include shapes -> stored (M_PROF) != lookup (M_RUNTIME).
    assert final["hit"], (
        "[BUG CONFIRMED] Cache miss at runtime for M_RUNTIME={} (bucket M={}):\n"
        "  Stored extras (profiled with M_PROF={}): {}\n"
        "  Lookup extras (runtime with M_RUNTIME={}): {}\n"
        "  extras differ -> ProfilingCacheKey mismatch -> tactic=-1 (heuristic)\n"
        "  Root cause: get_cache_key_extras included the dynamic M dimension.\n"
        "  Fix: return only (a.dtype, b.dtype, out.dtype) from get_cache_key_extras.".format(
            M_RUNTIME, M_PROF,
            M_PROF, stored_extras,
            M_RUNTIME, lookup_extras,
        )
    )
    assert final["tactic"] >= 0, (
        f"Expected a tuned tactic >= 0 but got tactic={final['tactic']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
