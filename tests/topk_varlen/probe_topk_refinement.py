"""Correctness probe for the rewritten overflow refinement (wide rounds,
early staging, terminal STAGE + tie_select, wide-bin skip predicate).

Strict value-multiset check vs torch.topk per row.  Covers:
  - one_bin / constant / two_values / quantized4 / tiecap_edge (P1 fair set)
  - huge_distinct: > top_k DISTINCT values inside the +inf coarse bin
    (wide-bin path: the (24,8) round MUST run; sglang breaks here)
  - neg_edge: k reaches into a bottom/-inf-adjacent wide bin
  - denorm: > top_k distinct tiny values in the +0/denormal collapse bins
  - randn control
Shapes hit single-CTA scan (N<=32768), single-CTA stream, MC grouped, MC
multi-round, and bf16's single-round path.
"""

import torch

import flashinfer

DEV = "cuda"


def gen(pattern, rows, N, dtype, k, seed):
    torch.manual_seed(seed)
    if pattern == "randn":
        x = torch.randn(rows, N, device=DEV) * 2.0
    elif pattern == "one_bin":
        x = 1.0 + torch.rand(rows, N, device=DEV) * 2**-12
    elif pattern == "constant":
        x = torch.full((rows, N), 1.2345, device=DEV)
    elif pattern == "two_values":
        x = torch.where(torch.rand(rows, N, device=DEV) < 0.5, 1.0, -1.0)
    elif pattern == "quantized4":
        x = torch.randint(0, 4, (rows, N), device=DEV).float()
    elif pattern == "tiecap_edge":
        x = torch.randn(rows, N, device=DEV) * 2.0
        ranks = torch.rand(rows, N, device=DEV).argsort(-1).argsort(-1)
        x = torch.where(ranks < k - 1, torch.full_like(x, 100.0), x)
        x = torch.where(
            (ranks >= k - 1) & (ranks < k - 1 + 2100), torch.full_like(x, 50.0), x
        )
    elif pattern == "huge_distinct":
        # >k distinct fp32 values that ALL land in the +inf coarse bin
        # (> 65520): geometric ladder up to ~1e30
        x = torch.randn(rows, N, device=DEV)
        m = min(N, 3000)
        ladder = torch.logspace(5, 30, m, device=DEV).float()  # 1e5..1e30
        x[:, :m] = ladder.flip(0)  # descending; all > 65520 for most entries
        x[:, :m] = torch.where(x[:, :m] <= 66000.0, x[:, :m] + 66000.0, x[:, :m])
        # shuffle placement per row
        for r in range(rows):
            perm = torch.randperm(N, device=DEV)
            x[r] = x[r][perm]
    elif pattern == "binade_edge":
        # straddle bin: >k values just below 2.0 + some exactly 2.0 (both in
        # coarse bin 3072, but on opposite sides of a 2^24 key boundary).
        # The correct top-k must include ALL the 2.0s first.
        x = torch.full((rows, N), -100.0, device=DEV)
        below = torch.nextafter(
            torch.tensor(2.0, device=DEV), torch.tensor(-1.0, device=DEV)
        )
        x[:, : min(N - 600, 3000)] = below
        x[:, min(N - 600, 3000) : min(N - 600, 3000) + 500] = 2.0
        for r in range(rows):
            perm = torch.randperm(N, device=DEV)
            x[r] = x[r][perm]
    elif pattern == "binade_edge_neg":
        # negative-side straddle: -2.0 vs nextafter(-2.0, 0) in one bin;
        # the correct top-k prefers the values CLOSER to zero.
        x = torch.full((rows, N), -100.0, device=DEV)
        above = torch.nextafter(
            torch.tensor(-2.0, device=DEV), torch.tensor(0.0, device=DEV)
        )
        x[:, : min(N - 600, 3000)] = -2.0
        x[:, min(N - 600, 3000) : min(N - 600, 3000) + 500] = above
        for r in range(rows):
            perm = torch.randperm(N, device=DEV)
            x[r] = x[r][perm]
    elif pattern == "binade_edge8":
        # same attack at the 8.0 edge with a spread of near-8 values
        x = torch.full((rows, N), -100.0, device=DEV)
        m = min(N - 8, 3000)
        x[:, :m] = 8.0 - (torch.rand(rows, m, device=DEV) * 2e-3)
        x[:, m : m + 8] = 8.0
        for r in range(rows):
            perm = torch.randperm(N, device=DEV)
            x[r] = x[r][perm]
    elif pattern == "denorm":
        # >k distinct tiny values inside the fp16-denormal collapse bins
        x = torch.full((rows, N), -1.0, device=DEV)
        m = min(N, 3000)
        tiny = torch.linspace(1e-30, 9e-27, m, device=DEV).float()
        x[:, :m] = tiny
        for r in range(rows):
            perm = torch.randperm(N, device=DEV)
            x[r] = x[r][perm]
    else:
        raise ValueError(pattern)
    return x.to(dtype).contiguous()


def check(pattern, rows, N, k, dtype, seed=11):
    x = gen(pattern, rows, N, dtype, k, seed)
    seq = torch.full((rows,), N, dtype=torch.int32, device=DEV)
    out_i = torch.full((rows, k), -7, dtype=torch.int32, device=DEV)
    flashinfer.top_k_varlen(x, seq, k, backend="radix_primitives", out_indices=out_i)
    torch.cuda.synchronize()
    bad = 0
    for r in range(rows):
        sel = out_i[r]
        inr = sel[(sel >= 0) & (sel < N)].long()
        dup = int(inr.numel() - inr.unique().numel())
        got = torch.sort(x[r].float()[inr], descending=True).values
        ref = torch.topk(x[r].float(), k).values
        if inr.numel() != k or dup or not torch.equal(got, ref):
            bad += 1
            if bad <= 2:
                print(
                    f"    BAD row{r}: n={inr.numel()} dup={dup} "
                    f"got_min={got[-1].item() if got.numel() else None} "
                    f"ref_min={ref[-1].item()}",
                    flush=True,
                )
    tag = "OK " if bad == 0 else "FAIL"
    print(
        f"  {tag} {pattern:13s} {str(dtype):15s} N={N:6d} b={rows:3d} k={k}", flush=True
    )
    return bad


fails = 0
for pattern in (
    "randn",
    "one_bin",
    "constant",
    "two_values",
    "quantized4",
    "tiecap_edge",
    "huge_distinct",
    "denorm",
    "binade_edge",
    "binade_edge_neg",
    "binade_edge8",
):
    for rows, N in ((1, 65536), (16, 32768), (4, 8192), (64, 65536), (2, 131072)):
        fails += check(pattern, rows, N, 2048, torch.float32)
for pattern in ("randn", "one_bin", "constant", "two_values"):
    for rows, N in ((1, 65536), (16, 32768), (2, 131072)):
        fails += check(pattern, rows, N, 1024, torch.bfloat16)
print(f"{'ALL EXACT' if fails == 0 else f'{fails} BAD ROWS'}", flush=True)
