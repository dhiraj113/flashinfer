"""FlashInfer glue for the vendored ``cutlass_primitives`` top-k library: the one backend.

The library (``flashinfer/topk_varlen/cutlass_primitives/``, see its VENDORED.md) holds the
kernels and their router; this file owns everything FlashInfer-specific:

* the eligibility check in FlashInfer's style;
* compilation through ``build_and_load_cute_dsl_kernel`` so artifacts persist in FlashInfer's
  DSL cache, with dynamic row counts (``cute.sym_int``) and the TVM-FFI environment stream,
  instead of the library's in-process ``cute.compile`` cache and explicit torch stream;
* the small per-call buffers (status words, slab workspace) FlashInfer callers never see.

Routing is the library's: rows that fit a CTA's registers take the register-resident kernel,
rows up to eight register slices in batches that fit one wave of clusters take its clustered
form, and longer rows the streaming kernel with the cluster or slab merge chosen from the
device facts.  ``next_n``, ``compress_ratio`` and ``return_values`` are compile-time
specializations of the same kernels (the library's ``phases/varlen.py``).
Hints (``pre_idx``) are accepted and ignored; the library's design keeps hints off because
stale hints measured slower than none in every form tried.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from ..cutlass_primitives.dispatch.device import device_facts
from ..cutlass_primitives.topk.dispatch.router import choose
from ..cutlass_primitives.topk.kernels import register_cluster as RCL
from ..cutlass_primitives.topk.kernels import register_resident as REG
from ..cutlass_primitives.topk.kernels import streaming as STR

_CUTLASS_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
_DTYPE_TAGS = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
_VENDORED_DIR = Path(__file__).resolve().parent.parent / "cutlass_primitives"
_compiled: dict = {}
_status: dict = {}


def cutlass_primitives_supported(
    logits: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_values: bool,
) -> bool:
    """Eligibility: fp32/fp16/bf16 2-D contiguous rows whose stride is a multiple of 16 bytes,
    k up to 4096, any ``next_n`` dividing the row count and any ``compress_ratio``, values on
    request, SM80 or newer."""
    if (
        logits.dim() != 2
        or not logits.is_contiguous()
        or logits.dtype not in _CUTLASS_DTYPES
    ):
        return False
    if top_k > 4096 or next_n < 1 or compress_ratio < 1 or logits.shape[0] % next_n:
        return False
    if (logits.shape[1] * logits.element_size()) % 16:
        return False
    return torch.cuda.get_device_capability(logits.device)[0] >= 8


def _fake(dt, shape, align=None):
    order = tuple(range(len(shape) - 1, -1, -1))
    if align is None:
        return cute.runtime.make_fake_compact_tensor(dt, shape, stride_order=order)
    return cute.runtime.make_fake_compact_tensor(
        dt, shape, stride_order=order, assumed_align=align
    )


def _compiled_kernel(
    kind: str,
    dtype: torch.dtype,
    k: int,
    n: int,
    config,
    device: torch.device,
    next_n: int,
    compress_ratio: int,
    return_values: bool,
):
    """The FlashInfer-cached compiled launcher for one (kernel, dtype, k, N, configuration,
    next_n, compress_ratio, values) specialization."""
    facts = device_facts(device)
    key = (
        kind,
        dtype,
        k,
        n,
        config,
        facts.capability,
        next_n,
        compress_ratio,
        return_values,
    )
    if key in _compiled:
        return _compiled[key]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    cdt = _CUTLASS_DTYPES[dtype]
    rows = cute.sym_int()
    groups = cute.sym_int()  # rows // next_n length entries
    i32 = cutlass.Int32
    varlen = (next_n, compress_ratio, return_values)
    # the values output, or a static one-element placeholder when none is requested (TVM-FFI
    # ties every symbolic dimension of one name together, so an unused (rows, k) would not do)
    values_fake = _fake(cdt, (rows, k), 16) if return_values else _fake(cdt, (1, 1), 16)
    fakes: tuple
    kern: Any
    if kind == "register":
        assert isinstance(config, REG.RegisterConfig)
        kern = REG.RegisterTopK(cdt, k, config, facts.shared_memory_optin, *varlen)
        fakes = (
            _fake(cdt, (rows, n), 16),
            _fake(i32, (groups,)),
            _fake(i32, (rows, k), 16),
            values_fake,
            _fake(i32, (rows, REG.STATUS_WORDS), 16),
        )
    elif kind == "register_cluster":
        assert isinstance(config, RCL.RegisterClusterConfig)
        kern = RCL.RegisterClusterTopK(
            cdt, k, config, facts.shared_memory_optin, *varlen
        )
        fakes = (
            _fake(cdt, (rows, n), 16),
            _fake(i32, (groups,)),
            _fake(i32, (rows, k), 16),
            values_fake,
            _fake(i32, (rows, RCL.STATUS_WORDS), 16),
        )
    else:
        assert isinstance(config, STR.StreamingConfig)
        kern = STR.StreamingTopK(cdt, k, config, facts.shared_memory_optin, *varlen)
        if config.merge == "slab" and config.splits > 1:
            words = STR.slab_words_per_row(config.splits, config.stage)
            slab_fakes = (_fake(i32, (rows, words), 16), _fake(i32, (rows,)))
        else:  # unused by the kernel: static one-element placeholders (TVM-FFI ties every `rows` dim together)
            slab_fakes = (_fake(i32, (1, 1), 16), _fake(i32, (1,)))
        fakes = (
            _fake(cdt, (rows, n), 16),
            _fake(i32, (groups,)),
            _fake(i32, (rows, k), 16),
            values_fake,
            _fake(i32, (rows, STR.STATUS_WORDS), 16),
        ) + slab_fakes

    def compile_fn():
        return cute.compile(
            kern.launch,
            *fakes,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    # the configuration record's full name is the cache key's meaning; its hash keeps the
    # artifact file name within the filesystem's limit
    config_hash = hashlib.sha1(config.name().encode()).hexdigest()[:12]
    tag = f"{kind}_{_DTYPE_TAGS[dtype]}_k{k}_N{n}_sm{facts.capability[0]}{facts.capability[1]}_{config_hash}"
    if next_n != 1 or compress_ratio != 1:
        tag += f"_nn{next_n}_cr{compress_ratio}"
    if return_values:
        tag += "_vals"
    compiled = build_and_load_cute_dsl_kernel(
        "cutlass_primitives_topk",
        tag,
        compile_fn,
        extra_key_files=tuple(str(f) for f in sorted(_VENDORED_DIR.rglob("*.py")))
        + (__file__,),
    )
    _compiled[key] = compiled
    return compiled


def _status_buffer(rows: int, words: int, device: torch.device) -> torch.Tensor:
    key = (device, rows, words)
    if key not in _status:
        _status[key] = torch.empty(rows, words, dtype=torch.int32, device=device)
    return _status[key]


def run_cutlass_primitives(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_values: bool,
    out_indices: torch.Tensor,
    out_values: Optional[torch.Tensor],
    pre_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Top-k indices of each row of ``logits`` into ``out_indices`` (rows, top_k, int32), and
    the selected values into ``out_values`` when ``return_values``.  Row r sees
    ``(seq_lens[r // next_n] - next_n + r % next_n + 1) // compress_ratio`` elements (the
    ``radix`` backends' semantics); rows shorter than top_k are padded with -1 (values: -inf).
    """
    rows, n = logits.shape
    if return_values and out_values is None:
        out_values = torch.empty(rows, top_k, dtype=logits.dtype, device=logits.device)
    if rows == 0:
        return out_indices, (out_values if return_values else None)
    kind, config = choose(device_facts(logits.device), logits.dtype, top_k, n, rows)
    compiled = _compiled_kernel(
        kind,
        logits.dtype,
        top_k,
        n,
        config,
        logits.device,
        next_n,
        compress_ratio,
        return_values,
    )
    values = (
        out_values if return_values else _no_values_buffer(logits.device, logits.dtype)
    )
    if kind in ("register", "register_cluster"):
        words = REG.STATUS_WORDS if kind == "register" else RCL.STATUS_WORDS
        compiled(
            logits,
            seq_lens,
            out_indices,
            values,
            _status_buffer(rows, words, logits.device),
        )
    else:
        assert isinstance(config, STR.StreamingConfig)
        slab, counters = STR._slab_workspace(logits.device, rows, config)
        if config.merge == "slab" and config.splits > 1:
            slab = slab.view(rows, -1)
        else:
            slab = slab.view(1, 1)  # placeholders matching the static one-element fakes
        compiled(
            logits,
            seq_lens,
            out_indices,
            values,
            _status_buffer(rows, STR.STATUS_WORDS, logits.device),
            slab,
            counters,
        )
    return out_indices, (out_values if return_values else None)


_no_values: dict = {}


def _no_values_buffer(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The (1, 1) stand-in matching the static values placeholder the kernel was compiled with."""
    key = (device, dtype)
    if key not in _no_values:
        _no_values[key] = torch.empty(1, 1, dtype=dtype, device=device)
    return _no_values[key]
