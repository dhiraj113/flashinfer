"""Thorough functional campaign for the radix_primitives top_k_varlen backend.

Strict per-row oracle (beyond tests/topk_varlen's _check_correct):
  * exact value-multiset equality vs torch.topk (not just >= kth)
  * -1 sentinel padding in ALL surplus slots (prim-documented contract)
  * no duplicate indices, all in [0, N_eff)
  * return_values gather is bitwise-exact vs logits[row, idx]
  * poisoned tails: logits[row, N_eff:] filled with +inf / NaN / 1e30 so ANY
    out-of-bounds element influencing selection is an instant failure
  * determinism: value-multiset must be identical across repeat runs (hard);
    index-set variation is counted and reported (warn only, ties are the
    only legitimate source)

Parts: A adversarial-patterns, B config-edges, C lifecycle/state,
       D randomized fuzz, Z hostile-input probes (run LAST: may kill context).
Usage: python thorough_functional.py <A|B|C|D|Z|abc|dz|all>
"""

import random
import sys
import time

import torch

import flashinfer

DEV = "cuda"
POISONS = [float("inf"), float("nan"), 1e30]
FAILURES = []
WARNINGS = []


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------


def gen_logits(pattern, num_rows, N, dtype, k, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if pattern == "randn":
        x = torch.randn(num_rows, N, device=DEV) * 2.0
    elif pattern == "constant":
        x = torch.full((num_rows, N), 1.2345, device=DEV)
    elif pattern == "two_values":
        x = torch.where(torch.rand(num_rows, N, device=DEV) < 0.5, 1.0, -1.0)
    elif pattern == "quantized8":
        x = torch.randint(0, 8, (num_rows, N), device=DEV).float() - 3.5
    elif pattern == "quantized2k":
        x = torch.randint(0, 2048, (num_rows, N), device=DEV).float() / 64.0 - 16.0
    elif pattern == "low_mantissa":
        # fp32: distinct values inside ONE fp16 coarse bin (refine path);
        # bf16/fp16: collapses to constant (tie-overflow path)
        x = 1.0 + torch.randint(0, 16, (num_rows, N), device=DEV).float() * 2**-20
    elif pattern == "low_mantissa_coarse":
        x = 1.0 + torch.randint(0, 64, (num_rows, N), device=DEV).float() * 2**-9
    elif pattern == "fp16_midpoint":
        # exact midpoints of adjacent fp16 values: the ties-to-even hazard in
        # fp32 coarse binning (the hole sglang has, prim's pass-2 must fix)
        base = (torch.randn(num_rows, N, device=DEV) * 2.0).half()
        up = (base.view(torch.int16) + 1).view(torch.float16)
        x = (base.float() + up.float()) * 0.5
    elif pattern == "extremes":
        x = torch.randn(num_rows, N, device=DEV) * 2.0
        m = torch.rand(num_rows, N, device=DEV)
        x = torch.where(m < 0.02, torch.full_like(x, float("inf")), x)
        x = torch.where((m >= 0.02) & (m < 0.04), torch.full_like(x, float("-inf")), x)
        x = torch.where((m >= 0.04) & (m < 0.08), x * 30000.0, x)  # fp16-inf range
        x = torch.where((m >= 0.08) & (m < 0.12), x * 1e-7, x)  # fp16-subnormal range
        x = torch.where((m >= 0.12) & (m < 0.14), torch.full_like(x, -0.0), x)
        x = torch.where((m >= 0.14) & (m < 0.16), torch.zeros_like(x), x)
    elif pattern == "ascending":
        x = torch.arange(N, device=DEV).float().unsqueeze(0).expand(num_rows, N) / max(
            N, 1
        ) + torch.arange(num_rows, device=DEV).float().unsqueeze(1)
        x = x.contiguous()
    elif pattern == "descending":
        x = (
            torch.arange(N - 1, -1, -1, device=DEV)
            .float()
            .unsqueeze(0)
            .expand(num_rows, N)
            / max(N, 1)
        ).contiguous()
    elif pattern == "one_bin":
        # everything inside one fp16 coarse bin: pass-1 histogram degenerate,
        # entire row lands in refine/tie machinery with count >> TIE_CAP
        x = 1.0 + torch.rand(num_rows, N, device=DEV) * 2**-12
    elif pattern == "kties_boundary":
        # k-1 clear winners, then a giant tie group exactly at the threshold
        ranks = torch.rand(num_rows, N, device=DEV).argsort(-1).argsort(-1)
        x = torch.randn(num_rows, N, device=DEV)
        x = torch.where(ranks < max(k - 1, 0), torch.full_like(x, 100.0), x)
        tie_hi = min(N, k - 1 + 4 * max(k, 1))
        x = torch.where(
            (ranks >= max(k - 1, 0)) & (ranks < tie_hi), torch.full_like(x, 50.0), x
        )
    else:
        raise ValueError(pattern)
    return x.to(dtype).contiguous()


PATTERNS = [
    "randn",
    "constant",
    "two_values",
    "quantized8",
    "quantized2k",
    "low_mantissa",
    "low_mantissa_coarse",
    "fp16_midpoint",
    "extremes",
    "ascending",
    "descending",
    "one_bin",
    "kties_boundary",
]


def sample_seq_lens(groups, N, compress, next_n, k, seed, style="ragged"):
    rng = random.Random(seed)
    full = N * compress
    if style == "full":
        return [full] * groups
    base = [
        full,
        max(1, full - 1),
        min(full, max(1, k * compress + next_n)),
        max(1, min(full, (k - 1) * compress)),
        1,
        2,
        max(1, min(full, 17)),
        max(1, next_n),
        min(full, max(1, k * compress // 2 + 1)),
    ]
    out = [base[i] if i < len(base) else rng.randint(1, full) for i in range(groups)]
    rng.shuffle(out)
    if full not in out:
        out[0] = full  # always keep one full row
    return out


def compute_neff(seq_list, next_n, compress, N, num_rows):
    neff = []
    for row in range(num_rows):
        g, ofs = row // next_n, row % next_n
        acl = seq_list[g] - next_n + ofs + 1
        ne = acl // compress if acl > 0 else 0
        neff.append(min(max(ne, 0), N))
    return neff


def poison_tails(logits, neff):
    N = logits.shape[1]
    for row, ne in enumerate(neff):
        if ne < N:
            logits[row, ne:] = POISONS[row % len(POISONS)]


# ---------------------------------------------------------------------------
# strict oracle
# ---------------------------------------------------------------------------


def strict_check(tag, indices, values, logits, neff, top_k):
    lf = logits.float()
    idx_cpu = indices.cpu()
    for row in range(logits.shape[0]):
        ne = neff[row]
        kexp = min(top_k, ne)
        row_idx = idx_cpu[row].tolist()
        valid = [i for i in row_idx if i != -1]
        n_sentinel = sum(1 for i in row_idx if i == -1)
        assert len(valid) == kexp, (
            f"{tag} row{row}: {len(valid)} valid indices, want {kexp} (N_eff={ne}) "
            f"first16={row_idx[:16]}"
        )
        assert n_sentinel == top_k - kexp, (
            f"{tag} row{row}: {n_sentinel} sentinels, want {top_k - kexp}"
        )
        assert len(set(valid)) == len(valid), f"{tag} row{row}: duplicate indices"
        bad = [i for i in valid if not (0 <= i < ne)]
        assert not bad, f"{tag} row{row}: out-of-range indices {bad[:8]} (N_eff={ne})"
        if kexp > 0:
            rl = lf[row, :ne]
            sel = rl[torch.tensor(valid, device=DEV, dtype=torch.long)]
            ref = torch.topk(rl, kexp).values
            sel_sorted = torch.sort(sel, descending=True).values
            if not torch.equal(sel_sorted, ref):
                d = (sel_sorted != ref).nonzero().flatten()
                j = int(d[0]) if d.numel() else -1
                raise AssertionError(
                    f"{tag} row{row}: value multiset mismatch at rank {j}: "
                    f"got {sel_sorted[j].item() if j >= 0 else '?'} "
                    f"want {ref[j].item() if j >= 0 else '?'} (N_eff={ne} kexp={kexp})"
                )
            if values is not None:
                m = (indices[row] >= 0) & (indices[row] < ne)
                got = values[row][m].float()
                exp = lf[row][indices[row][m].long()]
                assert torch.equal(got, exp), (
                    f"{tag} row{row}: return_values gather mismatch "
                    f"(max abs diff {(got - exp).abs().max().item()})"
                )


def run_case(
    tag,
    pattern,
    dtype,
    k,
    N,
    groups,
    next_n=1,
    compress=1,
    seed=0,
    rv=False,
    seq_style="ragged",
    runs=1,
    out_prefill=False,
):
    num_rows = groups * next_n
    logits = gen_logits(pattern, num_rows, N, dtype, k, seed)
    seq_list = sample_seq_lens(groups, N, compress, next_n, k, seed + 1, seq_style)
    neff = compute_neff(seq_list, next_n, compress, N, num_rows)
    poison_tails(logits, neff)
    seq_t = torch.tensor(seq_list, dtype=torch.int32, device=DEV)
    kw = {}
    if out_prefill:
        kw["out_indices"] = torch.full(
            (num_rows, k), 0x7F7F7F7F, dtype=torch.int32, device=DEV
        )
        if rv:
            kw["out_values"] = torch.full(
                (num_rows, k), float("nan"), dtype=dtype, device=DEV
            )
    prev_vals, prev_idx = None, None
    for r in range(runs):
        idx, vals = flashinfer.top_k_varlen(
            logits,
            seq_t,
            k,
            next_n=next_n,
            compress_ratio=compress,
            return_values=rv,
            backend="radix_primitives",
            **kw,
        )
        torch.cuda.synchronize()
        strict_check(tag, idx, vals if rv else None, logits, neff, k)
        if runs > 1:
            sidx = torch.sort(idx, dim=-1).values.cpu()
            svals = torch.sort(
                torch.where(
                    idx >= 0,
                    torch.gather(logits.float(), 1, idx.clamp(min=0).long()),
                    torch.tensor(float("-inf"), device=DEV),
                ),
                dim=-1,
            ).values.cpu()
            if prev_vals is not None:
                assert torch.equal(svals, prev_vals), (
                    f"{tag}: value-multiset NONDETERMINISM between runs {r - 1},{r}"
                )
                if not torch.equal(sidx, prev_idx):
                    WARNINGS.append(f"{tag}: index set varies across runs (tie order)")
            prev_vals, prev_idx = svals, sidx


def guard(fn, tag, *a, **kw):
    try:
        fn(tag, *a, **kw)
        print(f"  {tag}: PASS", flush=True)
        return True
    except AssertionError as e:
        FAILURES.append(str(e))
        print(f"  {tag}: FAIL -- {e}", flush=True)
        return True
    except RuntimeError as e:
        FAILURES.append(f"{tag}: RUNTIME ERROR {e}")
        print(f"  {tag}: RUNTIME ERROR -- {e}", flush=True)
        return False  # CUDA context likely dead


# ---------------------------------------------------------------------------
# Part A: adversarial data patterns x representative configs
# ---------------------------------------------------------------------------

A_CONFIGS = [
    # (dtype, k, N, groups)  -- single-CTA and multi-CTA regimes
    (torch.float32, 2048, 8192, 4),
    (torch.bfloat16, 1024, 8192, 4),
    (torch.float16, 1024, 8192, 4),
    (torch.float32, 2048, 131072, 2),
    (torch.bfloat16, 1024, 131072, 2),
    (torch.float16, 1024, 65536, 3),
]


def part_A():
    print("=== PART A: adversarial patterns ===", flush=True)
    n = 0
    for dtype, k, N, groups in A_CONFIGS:
        for pi, pat in enumerate(PATTERNS):
            if pat == "fp16_midpoint" and dtype != torch.float32:
                continue
            n += 1
            rv = pi % 3 == 0
            tag = f"A[{n}] {pat} {str(dtype)[6:]} k={k} N={N} b={groups} rv={rv}"
            ok = guard(
                run_case, tag, pat, dtype, k, N, groups, seed=100 + n, rv=rv, runs=3
            )
            if not ok:
                return


# ---------------------------------------------------------------------------
# Part B: config edges
# ---------------------------------------------------------------------------


def part_B():
    print("=== PART B: config edges ===", flush=True)
    # k edges
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for k in (1, 2, 32, 33, 512, 1000, 2047, 2048):
            tag = f"B.k {str(dtype)[6:]} k={k}"
            if not guard(run_case, tag, "quantized8", dtype, k, 4096, 4, seed=200 + k):
                return
    # N edges (incl. odd / prime / MC boundary); k may exceed N (degenerate rows)
    for dtype in (torch.float32, torch.bfloat16):
        for N in (64, 100, 2048, 4095, 4097, 8191, 8193, 65535, 65536, 65537, 131072):
            for groups in (1, 149):
                k = 1024
                pat = "quantized2k" if N % 2 else "randn"
                tag = f"B.N {str(dtype)[6:]} N={N} b={groups}"
                if not guard(run_case, tag, pat, dtype, k, N, groups, seed=300 + N):
                    return
    # k > N degenerate whole-batch
    for dtype in (torch.float32, torch.bfloat16):
        tag = f"B.kgtN {str(dtype)[6:]} k=512 N=64"
        if not guard(run_case, tag, "randn", dtype, 512, 64, 5, seed=333, rv=True):
            return
    # next_n x compress_ratio (with tiny seq_lens producing N_eff=0 rows)
    for dtype in (torch.float32, torch.bfloat16):
        for nn in (1, 2, 3, 4):
            for cr in (1, 2, 4):
                tag = f"B.nncr {str(dtype)[6:]} nn={nn} cr={cr}"
                if not guard(
                    run_case,
                    tag,
                    "quantized8",
                    dtype,
                    512,
                    8192,
                    5,
                    next_n=nn,
                    compress=cr,
                    seed=400 + nn * 10 + cr,
                    rv=(nn + cr) % 2 == 0,
                ):
                    return
    # MC + next_n/compress + return_values
    for dtype in (torch.float32, torch.bfloat16):
        tag = f"B.mc-nncr {str(dtype)[6:]} nn=2 cr=4 N=131072"
        if not guard(
            run_case,
            tag,
            "randn",
            dtype,
            2048 if dtype == torch.float32 else 1024,
            131072,
            2,
            next_n=2,
            compress=4,
            seed=444,
            rv=True,
        ):
            return
    # preallocated outputs with garbage prefill (catches unwritten slots)
    for rv in (False, True):
        tag = f"B.prealloc rv={rv}"
        if not guard(
            run_case,
            tag,
            "kties_boundary",
            torch.float32,
            2048,
            32768,
            3,
            seed=456,
            rv=rv,
            out_prefill=True,
        ):
            return


# ---------------------------------------------------------------------------
# Part C: lifecycle / state stress
# ---------------------------------------------------------------------------


def part_C():
    print("=== PART C: lifecycle & state ===", flush=True)

    def repeated_mc(tag):
        # 30 fresh-data MC calls on the worst-tie pattern: state self-reset
        for it in range(30):
            run_case(
                f"{tag} it{it}",
                "constant" if it % 3 == 0 else "two_values",
                torch.float32,
                2048,
                131072,
                2,
                seed=500 + it,
            )

    if not guard(lambda tag: repeated_mc(tag), "C.repeated-mc fp32 30x"):
        return

    def dtype_interleave(tag):
        # fp32 <-> bf16 MC alternation: row_states layout aliasing regression
        for it in range(8):
            run_case(
                f"{tag} f32 it{it}",
                "constant",
                torch.float32,
                2048,
                131072,
                2,
                seed=600 + it,
            )
            run_case(
                f"{tag} bf16 it{it}",
                "constant",
                torch.bfloat16,
                1024,
                131072,
                2,
                seed=650 + it,
            )

    if not guard(lambda tag: dtype_interleave(tag), "C.dtype-interleave-mc 8x"):
        return

    def single_mc_interleave(tag):
        for it in range(6):
            run_case(
                f"{tag} mc it{it}",
                "quantized8",
                torch.float32,
                2048,
                131072,
                1,
                seed=700 + it,
            )
            run_case(
                f"{tag} single it{it}",
                "quantized8",
                torch.float32,
                2048,
                8192,
                4,
                seed=750 + it,
            )

    if not guard(lambda tag: single_mc_interleave(tag), "C.single-mc-interleave 6x"):
        return

    def cuda_graph(tag, dtype, k, N, groups):
        num_rows = groups
        logits = gen_logits("randn", num_rows, N, dtype, k, seed=800)
        seq_list = sample_seq_lens(groups, N, 1, 1, k, 801)
        neff = compute_neff(seq_list, 1, 1, N, num_rows)
        poison_tails(logits, neff)
        seq_t = torch.tensor(seq_list, dtype=torch.int32, device=DEV)
        out_i = torch.empty(num_rows, k, dtype=torch.int32, device=DEV)

        def call():
            flashinfer.top_k_varlen(
                logits, seq_t, k, backend="radix_primitives", out_indices=out_i
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
        for rep in range(5):
            pat = PATTERNS[rep % len(PATTERNS)]
            if pat == "fp16_midpoint" and dtype != torch.float32:
                pat = "two_values"
            fresh = gen_logits(pat, num_rows, N, dtype, k, seed=810 + rep)
            new_seq = sample_seq_lens(groups, N, 1, 1, k, 820 + rep)
            new_neff = compute_neff(new_seq, 1, 1, N, num_rows)
            poison_tails(fresh, new_neff)
            logits.copy_(fresh)
            seq_t.copy_(torch.tensor(new_seq, dtype=torch.int32, device=DEV))
            g.replay()
            torch.cuda.synchronize()
            strict_check(f"{tag} rep{rep} pat={pat}", out_i, None, logits, new_neff, k)

    if not guard(
        lambda tag: cuda_graph(tag, torch.float32, 2048, 8192, 8), "C.graph-single fp32"
    ):
        return
    if not guard(
        lambda tag: cuda_graph(tag, torch.bfloat16, 1024, 131072, 2), "C.graph-mc bf16"
    ):
        return


# ---------------------------------------------------------------------------
# Part D: randomized fuzz
# ---------------------------------------------------------------------------

N_POOL = [
    64,
    128,
    257,
    1000,
    2048,
    4095,
    4096,
    8191,
    8192,
    8193,
    16384,
    32768,
    65535,
    65536,
    65537,
    98304,
    131072,
]
K_POOL = [1, 2, 3, 16, 32, 33, 100, 256, 511, 512, 513, 1000, 1024, 1536, 2047, 2048]
B_POOL = [1, 2, 3, 5, 8, 16, 17, 64, 148, 149, 256]


def part_D(iters=100, seed=20260827):
    print(f"=== PART D: randomized fuzz (seed={seed}) ===", flush=True)
    rng = random.Random(seed)
    for it in range(iters):
        dtype = rng.choice([torch.float32, torch.bfloat16, torch.float16])
        k = rng.choice(K_POOL)
        N = rng.choice(N_POOL)
        groups = rng.choice(B_POOL)
        nn = rng.choice([1, 1, 1, 2, 3, 4])
        cr = rng.choice([1, 1, 1, 2, 4])
        while groups * nn * N > 40_000_000:
            groups = max(1, groups // 2)
        pat = rng.choice(PATTERNS)
        if pat == "fp16_midpoint" and dtype != torch.float32:
            pat = "quantized2k"
        rv = rng.random() < 0.3
        tag = (
            f"D[{it}] {pat} {str(dtype)[6:]} k={k} N={N} b={groups} "
            f"nn={nn} cr={cr} rv={rv}"
        )
        if not guard(
            run_case,
            tag,
            pat,
            dtype,
            k,
            N,
            groups,
            next_n=nn,
            compress=cr,
            seed=1000 + it,
            rv=rv,
        ):
            return


# ---------------------------------------------------------------------------
# Part Z: hostile-input probes (LAST: may poison the CUDA context)
# ---------------------------------------------------------------------------


def part_Z():
    print("=== PART Z: hostile-input probes ===", flush=True)

    def probe(name, fn):
        try:
            r = fn()
            print(f"  Z.{name}: {r}", flush=True)
        except Exception as e:
            print(f"  Z.{name}: raised {type(e).__name__}: {str(e)[:200]}", flush=True)

    def seq_zero():
        logits = gen_logits("randn", 3, 4096, torch.float32, 512, 900)
        seq_t = torch.tensor([0, 4096, 0], dtype=torch.int32, device=DEV)
        idx, _ = flashinfer.top_k_varlen(logits, seq_t, 512, backend="radix_primitives")
        torch.cuda.synchronize()
        strict_check("Z.seq0", idx, None, logits, [0, 4096, 0], 512)
        return "PASS (all -1 for zero-length rows)"

    probe("seq_zero", seq_zero)

    def stream_concurrency():
        # two streams, same dtype => SHARED MC row_states buffer; probe whether
        # concurrent MC calls corrupt each other (report-only)
        k, N = 2048, 131072
        l1 = gen_logits("randn", 1, N, torch.float32, k, 901)
        l2 = gen_logits("two_values", 1, N, torch.float32, k, 902)
        seq_t = torch.full((1,), N, dtype=torch.int32, device=DEV)
        o1 = torch.empty(1, k, dtype=torch.int32, device=DEV)
        o2 = torch.empty(1, k, dtype=torch.int32, device=DEV)
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        # warmup (sequential, correct by construction)
        flashinfer.top_k_varlen(
            l1, seq_t, k, backend="radix_primitives", out_indices=o1
        )
        flashinfer.top_k_varlen(
            l2, seq_t, k, backend="radix_primitives", out_indices=o2
        )
        torch.cuda.synchronize()
        bad = 0
        for _ in range(20):
            with torch.cuda.stream(s1):
                flashinfer.top_k_varlen(
                    l1, seq_t, k, backend="radix_primitives", out_indices=o1
                )
            with torch.cuda.stream(s2):
                flashinfer.top_k_varlen(
                    l2, seq_t, k, backend="radix_primitives", out_indices=o2
                )
            torch.cuda.synchronize()
            try:
                strict_check("Z.stream l1", o1, None, l1, [N], k)
                strict_check("Z.stream l2", o2, None, l2, [N], k)
            except AssertionError:
                bad += 1
        return f"{'CLEAN' if bad == 0 else f'CORRUPT in {bad}/20 rounds'} (shared MC state across streams)"

    probe("stream_concurrency", stream_concurrency)

    def noncontig_logits():
        big = gen_logits("randn", 4, 8192, torch.float32, 512, 903)
        x = big[:, :4096]  # row stride 8192 != N: non-contiguous
        assert not x.is_contiguous()
        seq_t = torch.full((4,), 4096, dtype=torch.int32, device=DEV)
        idx, _ = flashinfer.top_k_varlen(x, seq_t, 512, backend="radix_primitives")
        torch.cuda.synchronize()
        strict_check("Z.noncontig", idx, None, x.contiguous(), [4096] * 4, 512)
        return "PASS (handled strided input correctly?!)"

    probe("noncontig_logits", noncontig_logits)

    def batch_zero():
        x = torch.empty(0, 4096, dtype=torch.float32, device=DEV)
        seq_t = torch.empty(0, dtype=torch.int32, device=DEV)
        idx, _ = flashinfer.top_k_varlen(x, seq_t, 512, backend="radix_primitives")
        torch.cuda.synchronize()
        return f"returned shape {tuple(idx.shape)}"

    probe("batch_zero", batch_zero)

    def overlong_seq():
        logits = gen_logits("randn", 2, 4096, torch.float32, 512, 904)
        seq_t = torch.tensor([4096 + 999, 4096], dtype=torch.int32, device=DEV)
        idx, _ = flashinfer.top_k_varlen(logits, seq_t, 512, backend="radix_primitives")
        torch.cuda.synchronize()
        strict_check("Z.overlong", idx, None, logits, [4096, 4096], 512)
        return "PASS (clamped to row width)"

    probe("overlong_seq", overlong_seq)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    fuzz_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260827
    t0 = time.time()
    torch.cuda.init()
    steps = {
        "A": part_A,
        "B": part_B,
        "C": part_C,
        "D": lambda: part_D(seed=fuzz_seed),
        "Z": part_Z,
    }
    order = {"abc": "ABC", "dz": "DZ", "all": "ABCDZ"}.get(part.lower(), part.upper())
    for p in order:
        tp = time.time()
        steps[p]()
        print(f"--- part {p} done in {time.time() - tp:.0f}s ---", flush=True)
    print(f"\n===== SUMMARY ({time.time() - t0:.0f}s) =====", flush=True)
    if WARNINGS:
        uniq = sorted(set(WARNINGS))
        print(f"warnings ({len(WARNINGS)} total, {len(uniq)} unique):", flush=True)
        for w in uniq[:20]:
            print(f"  WARN {w}", flush=True)
    if FAILURES:
        print(f"FAILURES: {len(FAILURES)}", flush=True)
        for f in FAILURES[:50]:
            print(f"  FAIL {f}", flush=True)
        sys.exit(1)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
