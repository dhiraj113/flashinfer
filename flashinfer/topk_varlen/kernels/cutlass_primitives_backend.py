"""FlashInfer glue for the vendored ``cutlass_primitives`` top-k library: the one backend.

The library (``flashinfer/topk_varlen/cutlass_primitives/``, see its VENDORED.md) holds the
kernels and their router; this file owns everything FlashInfer-specific:

* the eligibility check in FlashInfer's style;
* compilation through ``build_and_load_cute_dsl_kernel`` so artifacts persist in FlashInfer's
  DSL cache, with dynamic row counts (``cute.sym_int``) and the TVM-FFI environment stream,
  instead of the library's in-process ``cute.compile`` cache and explicit torch stream;
* the small per-call buffers (status words, slab workspace): cached per (device, stream,
  shape) so concurrent streams never share one, or carved from the caller's
  ``workspace["cutlass_primitives_workspace"]`` (sized by ``cutlass_primitives_workspace_bytes``)
  when the caller wants to own every byte the backend touches.

Routing is the library's: rows that fit a CTA's registers take the register-resident kernel,
rows up to eight register slices in batches that fit one wave of clusters take its clustered
form, and longer rows the streaming kernel with the cluster or slab merge chosen from the
device facts.  ``next_n``, ``compress_ratio`` and ``return_values`` are compile-time
specializations of the same kernels (the library's ``phases/varlen.py``).
Hints (``pre_idx``) are accepted and ignored; the library's design keeps hints off because
stale hints measured slower than none in every form tried.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from ..cutlass_primitives.dispatch.device import device_facts
from ..cutlass_primitives.topk.dispatch.router import choose
from ..cutlass_primitives.topk.dispatch.workspace import carve, workspace_layout
from ..cutlass_primitives.topk.kernels import census_split as CS
from ..cutlass_primitives.topk.kernels import register_cluster as RCL
from ..cutlass_primitives.topk.kernels.layout import arena_bytes, arena_view
from ..cutlass_primitives.topk.kernels import register_resident as REG
from ..cutlass_primitives.topk.kernels import streaming as STR

WORKSPACE_KEY = "cutlass_primitives_workspace"


@functools.cache
def _cached_choose(device: torch.device, dtype: torch.dtype, k: int, n: int, rows: int):
    """The router's (kernel, configuration) for a problem, memoized: the eligibility check and
    the launch both ask, on every call, and the answer is a pure function of these arguments."""
    return choose(device_facts(device), dtype, k, n, rows)


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
    """Eligibility: fp32/fp16/bf16 2-D logits in any layout (aligned rows and paged arenas are
    read in place, anything else is copied once into a padded arena), any k (rows shorter than
    k are padded; k beyond the tie stages takes the radix refine, a wider split or the exact
    select), any ``next_n`` dividing the row count and any ``compress_ratio``, values on
    request, SM80 or newer."""
    if logits.dim() != 2 or logits.dtype not in _CUTLASS_DTYPES:
        return False
    if top_k < 1 or next_n < 1 or compress_ratio < 1 or logits.shape[0] % next_n:
        return False
    if torch.cuda.get_device_capability(logits.device)[0] < 8:
        return False
    if logits.shape[0] == 0:
        return True
    try:  # the router declines only what no configuration on this part can hold
        _cached_choose(
            logits.device, logits.dtype, top_k, logits.shape[1], logits.shape[0]
        )
    except ValueError:
        return False
    return True


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
    stride: int,
    col_offset: int,
    config,
    device: torch.device,
    next_n: int,
    compress_ratio: int,
    return_values: bool,
):
    """The FlashInfer-cached compiled launcher for one (kernel, dtype, k, N, row stride and
    column offset, configuration, next_n, compress_ratio, values) specialization.  The kernel
    sees the logits as a compact (rows, stride) view and reads N columns of each row from
    col_offset."""
    facts = device_facts(device)
    key = (
        kind,
        dtype,
        k,
        n,
        stride,
        col_offset,
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
    varlen = (next_n, compress_ratio, return_values, n, col_offset)
    # the values output, or a static one-element placeholder when none is requested (TVM-FFI
    # ties every symbolic dimension of one name together, so an unused (rows, k) would not do)
    values_fake = _fake(cdt, (rows, k), 16) if return_values else _fake(cdt, (1, 1), 16)
    fakes: tuple
    kern: Any
    if kind == "register":
        assert isinstance(config, REG.RegisterConfig)
        kern = REG.RegisterTopK(cdt, k, config, facts.shared_memory_optin, *varlen)
        fakes = (
            _fake(cdt, (rows, stride), 16),
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
            _fake(cdt, (rows, stride), 16),
            _fake(i32, (groups,)),
            _fake(i32, (rows, k), 16),
            values_fake,
            _fake(i32, (rows, RCL.STATUS_WORDS), 16),
        )
    elif kind == "census_split":
        assert isinstance(config, CS.CensusSplitConfig)
        kern = CS.CensusSplitTopK(cdt, k, config, facts.shared_memory_optin, *varlen)
        words = CS.slab_words_per_row(config.splits, config.tie_slab)
        fakes = (
            _fake(cdt, (rows, stride), 16),
            _fake(i32, (groups,)),
            _fake(i32, (rows, k), 16),
            values_fake,
            _fake(i32, (rows, CS.STATUS_WORDS), 16),
            _fake(i32, (rows, words), 16),
            _fake(i32, (rows, 2)),
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
            _fake(cdt, (rows, stride), 16),
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
    if stride != n or col_offset:
        tag += f"_stride{stride}_off{col_offset}"
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
    """The cached (rows, words) status buffer for the current stream.  Keyed by the stream so
    two streams running the same shape at once never write one buffer (launches on one stream
    are ordered; on two they are not)."""
    key = (device, torch.cuda.current_stream(device).cuda_stream, rows, words)
    if key not in _status:
        _status[key] = torch.empty(rows, words, dtype=torch.int32, device=device)
    return _status[key]


def cutlass_primitives_workspace_bytes(logits: torch.Tensor, top_k: int) -> int:
    """Bytes the ``cutlass_primitives`` backend needs in ``workspace["cutlass_primitives_workspace"]``
    for ``logits`` (its shape, dtype, device and layout all count) at ``top_k``.

    The size follows the library's kernel choice for the row count, so an engine that runs
    several batch sizes takes the maximum over them (each query is a cheap host computation).
    With a workspace the call allocates nothing: the status words, the slab merge's buffers
    and the padded copy of a misaligned input all come out of it.  Zero rows need none.
    """
    rows, n = logits.shape
    if rows == 0:
        return 0
    kind, config = _cached_choose(logits.device, logits.dtype, top_k, n, rows)
    return workspace_layout(kind, config, rows, arena_bytes(logits)).total_bytes


def _caller_buffers(
    workspace: torch.Tensor,
    logits: torch.Tensor,
    kind: str,
    config,
    words: int,
    rows: int,
):
    """(status (rows, words), slab (rows, slab words) or None, counters or None, arena bytes or
    None) carved from the caller's workspace; the counters are zeroed on the stream because the
    kernel needs zero arrivals at launch and the caller's memory may hold anything."""
    layout = workspace_layout(kind, config, rows, arena_bytes(logits))
    ws = carve(workspace, layout, logits.device)
    slab = counters = None
    if ws.slab is not None:
        assert ws.counters is not None
        slab, counters = ws.slab.view(rows, -1), ws.counters
        counters.zero_()
    return ws.status.view(rows, words), slab, counters, ws.arena


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
    workspace: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Top-k indices of each row of ``logits`` into ``out_indices`` (rows, top_k, int32), and
    the selected values into ``out_values`` when ``return_values``.  Row r sees
    ``(seq_lens[r // next_n] - next_n + r % next_n + 1) // compress_ratio`` elements (the
    ``radix`` backends' semantics); rows shorter than top_k are padded with -1 (values: -inf).
    ``workspace["cutlass_primitives_workspace"]`` (a contiguous CUDA tensor of at least
    ``cutlass_primitives_workspace_bytes(logits, top_k)`` bytes, 256-byte-aligned base, any
    dtype, any content) supplies every buffer the backend would otherwise cache or allocate.
    """
    rows, n = logits.shape
    if return_values and out_values is None:
        out_values = torch.empty(rows, top_k, dtype=logits.dtype, device=logits.device)
    if rows == 0:
        return out_indices, (out_values if return_values else None)
    kind, config = _cached_choose(logits.device, logits.dtype, top_k, n, rows)
    words = {
        "register": REG.STATUS_WORDS,
        "register_cluster": RCL.STATUS_WORDS,
        "census_split": CS.STATUS_WORDS,
        "streaming": STR.STATUS_WORDS,
    }[kind]
    caller_ws = workspace.get(WORKSPACE_KEY) if workspace else None
    arena_buf = slab = counters = None
    if caller_ws is not None:
        status, slab, counters, arena_buf = _caller_buffers(
            caller_ws, logits, kind, config, words, rows
        )
    else:
        status = _status_buffer(rows, words, logits.device)
    # a paged arena or sliced view: the kernel takes a compact (rows, stride) view over the
    # storage and reads n columns of each row from col_offset (the library's layout rules);
    # misaligned rows are copied into a padded arena (the caller's workspace when given)
    arena, col_offset = arena_view(logits, arena_buf)
    compiled = _compiled_kernel(
        kind,
        logits.dtype,
        top_k,
        n,
        arena.shape[1],
        col_offset,
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
        compiled(arena, seq_lens, out_indices, values, status)
    elif kind == "census_split":
        assert isinstance(config, CS.CensusSplitConfig)
        if slab is None:
            slab, counters = CS._slab_workspace(logits.device, rows, config)
            slab = slab.view(rows, -1)
        else:
            assert counters is not None
            counters = counters.view(rows, 2)
        compiled(arena, seq_lens, out_indices, values, status, slab, counters)
    else:
        assert isinstance(config, STR.StreamingConfig)
        if slab is None:
            # the library's per-(device, stream, shape) cache, or its shared one-element
            # placeholders for configurations that do not merge through the slab
            slab, counters = STR._slab_workspace(logits.device, rows, config)
            if config.merge == "slab" and config.splits > 1:
                slab = slab.view(rows, -1)
            else:
                slab = slab.view(1, 1)  # matching the static one-element fakes
        compiled(arena, seq_lens, out_indices, values, status, slab, counters)
    return out_indices, (out_values if return_values else None)


_no_values: dict = {}


def _no_values_buffer(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The (1, 1) stand-in matching the static values placeholder the kernel was compiled with."""
    key = (device, dtype)
    if key not in _no_values:
        _no_values[key] = torch.empty(1, 1, dtype=dtype, device=device)
    return _no_values[key]
