import torch
import flashinfer
from flashinfer.autotuner import autotune

torch.manual_seed(42)
_ = torch.randn(16, 16, device="cuda") @ torch.randn(16, 16, device="cuda")
torch.cuda.synchronize()

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

M, K, N = 8192, 2688, 5376

A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
A_fp8, A_scale = to_float8(A_bf16)

B_raw = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
B_fp8, B_scale = to_float8(B_raw)
B_fp8 = B_fp8.t()  # [K, N] column-major

# Works
result = flashinfer.bmm_fp8(
    A_fp8.unsqueeze(0), B_fp8.unsqueeze(0),
    A_scale, B_scale, torch.bfloat16,
    out=None, backend="auto",
)
print(f"Without autotune: OK {result.shape}")

# Segfaults
with torch.inference_mode(), autotune():
    result = flashinfer.bmm_fp8(
        A_fp8.unsqueeze(0), B_fp8.unsqueeze(0),
        A_scale, B_scale, torch.bfloat16,
        out=None, backend="auto",
    )
    print(f"With autotune: OK {result.shape}")
