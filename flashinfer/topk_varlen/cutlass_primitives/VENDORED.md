# Vendored from cutlass-primitives

Do not edit these files in FlashInfer; change the library and re-vendor.

- upstream: ssh://git@gitlab-master.nvidia.com:12051/dnallapa/cutlass-primitives.git
- tag: v0.1.14
- commit: 4755198
- date: 2026-09-04
- command: python tools/vendor_into_flashinfer.py <flashinfer>

Layout: `device/`, `block/`, `dispatch/` are the shared layers; `topk/` holds the phases,
the kernels and the router.  FlashInfer-specific glue lives in
`flashinfer/topk_varlen/kernels/cutlass_primitives_backend.py`, not here.
