"""Kernel 3, streaming sample-filter top-k: the composition (see docs/kernels/streaming.md).

One CTA per row.  Phases in order: sample (three barriers) -> filter pass -> verdict and
repair -> resolution, with the exact radix select as the fallback for anything the sampled
path cannot certify.  Short rows go straight to the radix select.  This file names the
phases and owns the shared-memory plan and the configuration; the phases hold the logic.

With ``splits > 1`` the row is shared by a cluster of CTAs launched as ``cluster=(1, splits,
1)``: every CTA samples the whole row (identical threshold), filters its own slice, and the
verdict and resolution merge over distributed shared memory.  The slab merge for parts
without clusters is the next increment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

try:  # the allocator moved between DSL releases; both spellings are in use on our parts
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # pragma: no cover - older nvidia-cutlass-dsl
    from cutlass.utils.smem_allocator import SmemAllocator
try:
    from cuda.bindings import driver as cuda_driver
except ImportError:  # pragma: no cover - older cuda-python
    from cuda import cuda as cuda_driver

from ...device.cluster import cluster_sync
from ...device.launch import release_dependent_grid, wait_for_prior_grid
from ...device.timers import read_clock64

from ..phases.aim import aim_tight, aim_wide
from ..phases.elements import Elements
from ..phases.fallback import exact_select_row
from ..phases.filter_pass import filter_pass
from ..phases.repair import verdict_and_repair, verdict_and_repair_cluster
from ..phases.resolve import emit_and_select, emit_and_select_cluster
from ..phases.sample import Threshold, sample_probe, sample_threshold
from ..phases.slab import merge_slab, publish_and_arrive, slab_words_per_row
from ..phases.varlen import effective_length, gather_values
from .register_resident import _no_values
from ...device.cluster import cluster_rank

__all__ = ["StreamingConfig", "StreamingTopK", "topk_streaming"]

_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
STATUS_WORDS = 12  # per row: [0] sampled path failed, [1] arm (0 identity, 1 census, 2 sampled, 3 radix), [2] survivors, [3..7] phase clocks, [8..11] slab-merge sub-phase clocks


@dataclass(frozen=True)
class StreamingConfig:
    """Every performance knob of the streaming kernel.  All fields are compile-time constants
    and all appear in the artifact name."""

    # shape
    threads: int = 1024
    ctas_per_sm: int = 1
    unroll: int = 4
    # split
    splits: int = 1  # CTAs per row
    merge: str = "cluster"  # "cluster": DSMEM merge (SM90+, splits <= 8); "slab": global-memory last arriver
    # sample
    aim: str = "tight"  # "tight" | "wide"
    floor_multiple: int = 2
    span_ext: float = 1.5
    sample_vectors: int = 1  # adjacent 16-byte vectors per thread in the sample (1, 2 or 4): the survivor spread shrinks with the square root
    # stage
    stage: int = 8192
    tie_capacity: int = 2048
    # repair
    overflow_offset: int = 64
    # epilogue
    ballot_limit: int = 128
    # scheduling
    short_cutoff: int = 16384  # rows at or below take the radix select directly
    pdl: bool = False
    packed_compare: bool = False  # 16-bit rows: setp.le.{f16x2,bf16x2} classify
    # instrumentation
    telemetry: bool = False

    def validate(self, k: int, elems: Elements, shared_memory_limit: int) -> None:
        if self.threads not in (256, 512, 1024):
            raise ValueError(f"threads must be 256, 512 or 1024, got {self.threads}")
        if self.unroll * elems.per_vector > 32:
            raise ValueError(
                "unroll * elements per vector must fit the 32-bit dead mask"
            )
        if self.aim not in ("tight", "wide"):
            raise ValueError(f"unknown aim policy {self.aim!r}")
        if self.sample_vectors not in (1, 2, 4):
            raise ValueError("sample_vectors must be 1, 2 or 4")
        if self.stage < 2048 or self.stage % 256:
            raise ValueError(
                "stage must be a multiple of 256 and at least 2048 (the census histogram aliases both stage halves)"
            )
        if (
            self.merge == "slab"
            and self.splits > 1
            and self.stage < max(768, 256 * self.splits)
        ):
            raise ValueError(
                "the slab merge's scratch needs a stage of at least max(768, 256 x splits)"
            )
        if k >= 3 * self.stage * self.splits // 4:
            raise ValueError(
                f"stage {self.stage} x {self.splits} too small for k={k}: k must sit below 3/4 of it (the balanced aim needs room on both sides)"
            )
        if self.merge not in ("cluster", "slab"):
            raise ValueError(f"unknown merge {self.merge!r}")
        if self.merge == "cluster" and self.splits not in (1, 2, 3, 4, 6, 8):
            raise ValueError(
                f"cluster splits must be one of 1, 2, 3, 4, 6, 8 (portable cluster sizes), got {self.splits}"
            )
        if self.merge == "slab" and self.splits not in (1, 2, 4, 8, 16, 32):
            raise ValueError(
                f"slab splits must be one of 1, 2, 4, 8, 16, 32, got {self.splits}"
            )
        if self.tie_capacity < max(2 * self.threads, k):
            raise ValueError(
                f"tie_capacity must be at least max(2 * threads, k) = {max(2 * self.threads, k)}"
            )
        if self.tie_capacity > 4 * self.threads:
            raise ValueError(
                "tie_capacity above 4 * threads exceeds the radix select's per-thread slots"
            )
        if self.packed_compare and elems.is_f32:
            raise ValueError("packed_compare applies to 16-bit rows only")
        need = self.shared_memory_bytes()
        if need * self.ctas_per_sm > shared_memory_limit:
            raise ValueError(
                f"{need} B x {self.ctas_per_sm} CTAs exceeds the {shared_memory_limit} B shared-memory budget"
            )

    def shared_memory_bytes(self) -> int:
        stage = 2 * (self.stage + 4) * 4
        ties = 2 * self.tie_capacity * 4
        return (
            stage + ties + 2 * 256 * 4 + (3 * (self.threads // 32) + 64) * 4 + 1024
        )  # + alignment slack

    def name(self) -> str:
        return "_".join(f"{f.name}{getattr(self, f.name)}" for f in fields(self))


class StreamingTopK:
    """A compiled streaming top-k for one (dtype, k, config); call it on (rows, N) inputs."""

    def __init__(
        self,
        dtype,
        k: int,
        config: StreamingConfig,
        shared_memory_limit: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        return_values: bool = False,
    ):
        self.dtype = dtype
        self.elems = Elements.of(dtype)
        self.k = k
        self.config = config
        config.validate(k, self.elems, shared_memory_limit)
        self.aim_policy = aim_tight if config.aim == "tight" else aim_wide
        self.threads = config.threads
        self.warps = config.threads // 32
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        self.return_values = return_values

    @cute.jit
    def exact_select(
        self,
        row_ptr,
        length,
        out_row,
        s_scratch,
        s_tie_keys,
        s_tie_idx,
        s_slots,
        s_slots_u32,
        s_result,
        tidx,
        check_constant: cutlass.Constexpr,
    ):
        """The exact path for one whole row (census, then the radix select over the rank-k
        bin's key range if it overflowed the tie stage; with ``check_constant`` a range pass
        first).  ``s_scratch``: at least 4096 Int32 (the dead stage, both halves).  Returns the
        arm taken (1 census, 3 radix) for the status word."""
        cfg = self.config
        return exact_select_row(
            self.elems,
            row_ptr,
            length,
            self.k,
            out_row,
            s_scratch,
            s_tie_keys,
            s_tie_idx,
            cfg.tie_capacity,
            cfg.ballot_limit,
            s_slots,
            s_slots_u32,
            s_result,
            tidx,
            self.threads,
            check_constant,
            cfg.telemetry,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        status: cute.Tensor,
        slab: cute.Tensor,
        counters: cute.Tensor,
    ):
        cfg = self.config
        elems = self.elems
        threads = cutlass.const_expr(self.threads)
        k = cutlass.const_expr(self.k)
        n_cols = cutlass.const_expr(x.shape[1])
        splits = cutlass.const_expr(cfg.splits)
        clustered = cutlass.const_expr(splits > 1 and cfg.merge == "cluster")
        slabbed = cutlass.const_expr(splits > 1 and cfg.merge == "slab")
        tidx, _, _ = cute.arch.thread_idx()
        row, rank_y, _ = cute.arch.block_idx()
        rows, _, _ = cute.arch.grid_dim()
        rank = cutlass.Int32(0)
        if cutlass.const_expr(clustered):
            rank = cluster_rank()
        if cutlass.const_expr(slabbed):
            rank = rank_y
        telemetry = cutlass.const_expr(cfg.telemetry)

        smem = SmemAllocator()
        # one allocation for both stage halves: the exact fallback's 4096-bin census and the
        # slab merge's scratch alias the dead stage from s_keys and may run into s_idx
        s_stage = smem.allocate_array(
            cutlass.Int32, 2 * (cfg.stage + 4), byte_alignment=128
        )
        s_keys = s_stage
        s_idx = s_stage + (cfg.stage + 4)
        s_hist = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_merged = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tie_keys = smem.allocate_array(
            cutlass.Uint32, cfg.tie_capacity, byte_alignment=128
        )
        s_tie_idx = smem.allocate_array(
            cutlass.Int32, cfg.tie_capacity, byte_alignment=128
        )
        s_slots_u32 = smem.allocate_array(
            cutlass.Uint32, 2 * self.warps, byte_alignment=128
        )
        s_slots = smem.allocate_array(cutlass.Int32, self.warps, byte_alignment=128)
        s_result = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 8, byte_alignment=128)

        if cutlass.const_expr(cfg.pdl):
            wait_for_prior_grid()
        mark = cutlass.Int64(
            0
        )  # telemetry: after the PDL wait, so phases exclude the previous kernel's tail
        if telemetry:
            mark = read_clock64()

        row_ptr = x.iterator + cutlass.Int64(row) * n_cols
        out_row = out.iterator + cutlass.Int64(row) * k
        status_row = status.iterator + cutlass.Int64(row) * STATUS_WORDS
        probe = sample_probe(
            elems, row_ptr, n_cols, tidx, threads, cfg.sample_vectors
        )  # in flight while the length loads
        length = effective_length(
            lengths, row, n_cols, self.next_n, self.compress_ratio
        )
        emitter = cutlass.Int32(
            0
        )  # 1 on the CTA that wrote this row's indices (for the values gather)

        # the two direct arms are rank 0's alone; peers of a cluster idle through them (no
        # cluster barrier or DSMEM access happens on these arms)
        if cutlass.Int32(k) >= length:
            if rank == 0:
                emitter = cutlass.Int32(1)
                for i in range(tidx, k, threads):
                    v = cutlass.Int32(-1)
                    if i < length:
                        v = cutlass.Int32(i)
                    out_row[i] = v
                if tidx == 0:
                    status_row[0] = cutlass.Int32(0)
                    status_row[1] = cutlass.Int32(0)
        else:
            if length <= cutlass.Int32(cfg.short_cutoff):
                if rank == 0:
                    emitter = cutlass.Int32(1)
                    arm = self.exact_select(
                        row_ptr,
                        length,
                        out_row,
                        s_keys,
                        s_tie_keys,
                        s_tie_idx,
                        s_slots,
                        s_slots_u32,
                        s_result,
                        tidx,
                        False,
                    )
                    if tidx == 0:
                        status_row[0] = cutlass.Int32(0)
                        status_row[1] = arm
            else:
                # this CTA's slice: equal vector-aligned chunks, the last one shorter
                chunk = (
                    (length + cutlass.Int32(splits - 1)) // cutlass.Int32(splits)
                    + cutlass.Int32(elems.per_vector - 1)
                ) & ~cutlass.Int32(elems.per_vector - 1)
                start = rank * chunk
                count = length - start
                if count > chunk:
                    count = chunk
                if count < 0:
                    count = cutlass.Int32(0)
                samples = cutlass.const_expr(
                    threads * elems.per_vector * cfg.sample_vectors
                )
                aim = self.aim_policy(
                    k, length, rows, samples, 3 * cfg.stage * splits // 4
                )
                th = Threshold(
                    *sample_threshold(
                        elems,
                        row_ptr,
                        length,
                        n_cols,
                        probe,
                        aim,
                        cfg.floor_multiple,
                        cfg.span_ext,
                        s_hist,
                        s_slots_u32,
                        s_result,
                        tidx,
                        threads,
                        cfg.telemetry,
                        cfg.sample_vectors,
                    )
                )
                if telemetry:
                    if tidx == 0:
                        status_row[3] = (read_clock64() - mark).to(cutlass.Int32)
                        if cutlass.const_expr(not slabbed):
                            for w in cutlass.range_constexpr(3):
                                status_row[8 + w] = s_result[12 + w]
                    mark = read_clock64()
                bar = th.bar
                scale = th.scale
                if tidx == 0:
                    s_count[0] = cutlass.Int32(0)  # a degenerate sample stages nothing
                    s_result[6] = cutlass.Int32(
                        0
                    )  # winner cursor (rank 0's is the row's)
                    s_result[7] = cutlass.Int32(0)  # tie cursor
                if th.degenerate == 0:
                    filter_pass(
                        elems,
                        row_ptr,
                        start,
                        count,
                        bar,
                        scale,
                        cfg.packed_compare,
                        cfg.stage,
                        s_count,
                        s_hist,
                        s_keys,
                        s_idx,
                        tidx,
                        threads,
                        cfg.unroll,
                    )
                else:
                    cute.arch.barrier()
                if telemetry:
                    if tidx == 0:
                        status_row[4] = (read_clock64() - mark).to(cutlass.Int32)
                    mark = read_clock64()
                survivors = s_count[0]
                ok = cutlass.Int32(0)
                finisher = cutlass.Int32(
                    1
                )  # the CTA that resolves the row and writes its status
                if cutlass.const_expr(slabbed):
                    # the row's verdict happens on the last arriver, after the others have left
                    slab_row = slab.iterator + cutlass.Int64(row) * slab_words_per_row(
                        splits, cfg.stage
                    )
                    slab_keys = slab_row
                    slab_idx = slab_row + splits * cfg.stage
                    tables = slab_row + 2 * splits * cfg.stage
                    finisher = publish_and_arrive(
                        elems,
                        rank,
                        splits,
                        bar,
                        scale,
                        cfg.stage,
                        s_count,
                        s_keys,
                        s_idx,
                        s_hist,
                        s_merged,
                        s_result,
                        slab_keys,
                        slab_idx,
                        tables,
                        counters.iterator + row,
                        tidx,
                        threads,
                    )
                    if telemetry:
                        if tidx == 0:
                            status_row[5] = (read_clock64() - mark).to(cutlass.Int32)
                        mark = read_clock64()
                    if finisher == 1:
                        ok = merge_slab(
                            elems,
                            k,
                            splits,
                            cfg.stage,
                            out_row,
                            s_keys,
                            s_idx,
                            s_merged,
                            s_tie_keys,
                            s_tie_idx,
                            cfg.tie_capacity,
                            cfg.ballot_limit,
                            s_slots,
                            s_result,
                            slab_keys,
                            slab_idx,
                            tables,
                            counters.iterator + row,
                            tidx,
                            threads,
                            cfg.telemetry,
                        )
                        survivors = s_result[8]
                        if telemetry:
                            if tidx == 0:
                                for w in cutlass.range_constexpr(4):
                                    status_row[8 + w] = s_result[12 + w]
                else:
                    if cutlass.const_expr(clustered):
                        bar, scale, survivors, ok = verdict_and_repair_cluster(
                            elems,
                            row_ptr,
                            start,
                            count,
                            k,
                            bar,
                            scale,
                            th.floor_bar,
                            th.floor_scale,
                            th.degenerate,
                            cfg.packed_compare,
                            cfg.stage,
                            splits,
                            s_count,
                            s_hist,
                            s_keys,
                            s_idx,
                            s_result,
                            tidx,
                            threads,
                            cfg.unroll,
                        )
                    else:
                        bar, scale, survivors, ok = verdict_and_repair(
                            elems,
                            row_ptr,
                            start,
                            count,
                            k,
                            bar,
                            scale,
                            th.floor_bar,
                            th.floor_scale,
                            th.degenerate,
                            cfg.packed_compare,
                            cfg.stage,
                            cfg.overflow_offset,
                            s_count,
                            s_hist,
                            s_keys,
                            s_idx,
                            s_result,
                            tidx,
                            threads,
                            cfg.unroll,
                        )
                    if telemetry:
                        if tidx == 0:
                            status_row[5] = (read_clock64() - mark).to(cutlass.Int32)
                        mark = read_clock64()
                    if ok == 1:
                        if cutlass.const_expr(clustered):
                            ok = emit_and_select_cluster(
                                elems,
                                k,
                                rank,
                                splits,
                                bar,
                                scale,
                                out_row,
                                s_keys,
                                s_idx,
                                s_hist,
                                s_merged,
                                s_tie_keys,
                                s_tie_idx,
                                cfg.tie_capacity,
                                cfg.ballot_limit,
                                s_count,
                                s_slots,
                                s_result,
                                tidx,
                                threads,
                            )
                        else:
                            ok = emit_and_select(
                                elems,
                                k,
                                survivors,
                                bar,
                                scale,
                                out_row,
                                s_keys,
                                s_idx,
                                s_hist,
                                s_tie_keys,
                                s_tie_idx,
                                cfg.tie_capacity,
                                cfg.ballot_limit,
                                s_slots,
                                s_result,
                                tidx,
                                threads,
                            )
                    else:
                        if cutlass.const_expr(clustered):
                            cluster_sync()  # match the resolution's barrier so no CTA exits early
                    if rank != 0:
                        finisher = cutlass.Int32(0)
                if telemetry:
                    if tidx == 0:
                        status_row[6] = (read_clock64() - mark).to(cutlass.Int32)
                    mark = read_clock64()
                arm = cutlass.Int32(2)
                if finisher == 1:
                    emitter = cutlass.Int32(1)
                    if ok == 0:
                        arm = self.exact_select(
                            row_ptr,
                            length,
                            out_row,
                            s_keys,
                            s_tie_keys,
                            s_tie_idx,
                            s_slots,
                            s_slots_u32,
                            s_result,
                            tidx,
                            True,
                        )
                    if telemetry:
                        if tidx == 0:
                            status_row[7] = (read_clock64() - mark).to(cutlass.Int32)
                            if (
                                ok == 0
                            ):  # the exact select's sub-phases replace the sample's
                                for w in cutlass.range_constexpr(3):
                                    status_row[8 + w] = s_result[12 + w]
                    if tidx == 0:
                        status_row[0] = cutlass.Int32(1) - ok
                        status_row[1] = arm
                        status_row[2] = survivors
        if cutlass.const_expr(self.return_values):
            # the emitter wrote the ties and the fallback itself; in a cluster the peers' winners
            # preceded the cluster barrier, in a slab the last arriver copied everything: after
            # this block barrier every index of the row is visible to the emitter
            if emitter == 1:
                cute.arch.barrier()
                gather_values(
                    self.dtype,
                    row_ptr,
                    out_row,
                    values.iterator + cutlass.Int64(row) * k,
                    k,
                    tidx,
                    threads,
                )
        if cutlass.const_expr(cfg.pdl):
            release_dependent_grid()

    @cute.jit
    def launch(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        status: cute.Tensor,
        slab: cute.Tensor,
        counters: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        """Launch on the caller's stream (torch's current stream, so CUDA-graph capture and
        stream ordering with the surrounding torch work both hold)."""
        splits = self.config.splits
        if cutlass.const_expr(splits > 1 and self.config.merge == "cluster"):
            self.kernel(x, lengths, out, values, status, slab, counters).launch(
                grid=(x.shape[0], splits, 1),
                block=(self.threads, 1, 1),
                cluster=(1, splits, 1),
                min_blocks_per_mp=self.config.ctas_per_sm,
                use_pdl=self.config.pdl,
                stream=stream,
            )
        else:
            self.kernel(x, lengths, out, values, status, slab, counters).launch(
                grid=(x.shape[0], splits, 1),
                block=(self.threads, 1, 1),
                min_blocks_per_mp=self.config.ctas_per_sm,
                use_pdl=self.config.pdl,
                stream=stream,
            )


_slabs: dict = {}


def _slab_workspace(device, rows: int, config: StreamingConfig):
    """Per-(device, rows, splits, stage) slab and zeroed arrival counters, kept alive so the
    counters' self-reset carries from one launch to the next."""
    key: tuple
    if config.merge != "slab" or config.splits == 1:
        key = (device, "none")
        if key not in _slabs:
            _slabs[key] = (
                torch.zeros(1, device=device, dtype=torch.int32),
                torch.zeros(1, device=device, dtype=torch.int32),
            )
        return _slabs[key]
    key = (device, rows, config.splits, config.stage)
    if key not in _slabs:
        words = rows * slab_words_per_row(config.splits, config.stage)
        _slabs[key] = (
            torch.empty(words, device=device, dtype=torch.int32),
            torch.zeros(rows, device=device, dtype=torch.int32),
        )
    return _slabs[key]


_compiled: dict = {}


def topk_streaming(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    config: StreamingConfig | None = None,
    out: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Indices of the k largest elements of each row of ``x`` (rows, N), as an int32 (rows, k)
    tensor.  ``lengths`` (rows // next_n, int32) limits each row to its first elements: row r
    sees ``(lengths[r // next_n] - next_n + r % next_n + 1) // compress_ratio`` of them (the
    speculative-decode stride and the KV-index compression; both default to 1, when it is
    ``lengths[r]``).  Rows shorter than k are padded with -1.  NaN ranks above +inf, as in
    torch.  Order within a row is unspecified.  ``out``, ``status`` (rows * STATUS_WORDS,
    int32) and ``values`` (rows, k, x's dtype: the selected elements, -inf in padding) are
    written in place when given, so a caller in a hot loop allocates nothing; values are only
    produced when ``values`` is passed.  ``config`` defaults to the dispatcher's choice for the
    device and problem.
    """
    from ...dispatch.device import device_facts

    from ..dispatch.streaming_policy import streaming_config_for

    assert x.dim() == 2 and x.is_cuda and x.is_contiguous() and x.dtype in _DTYPES
    rows, n = x.shape
    if (n * x.element_size()) % 16:
        raise ValueError(
            f"row stride must be a multiple of 16 bytes for the vector loads (N={n}, {x.dtype})"
        )
    facts = device_facts(x.device)
    if config is None:
        config = streaming_config_for(facts, x.dtype, k, n, rows)
    if lengths is None:
        lengths = torch.full(
            (rows // next_n,), n * compress_ratio, device=x.device, dtype=torch.int32
        )
    if out is None:
        out = torch.empty(rows, k, device=x.device, dtype=torch.int32)
    if status is None:
        status = torch.empty(rows * STATUS_WORDS, device=x.device, dtype=torch.int32)
    vals = values if values is not None else _no_values(x.device, x.dtype)
    key = (
        x.dtype,
        k,
        rows,
        n,
        config,
        facts.capability,
        next_n,
        compress_ratio,
        values is not None,
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    slab, counters = _slab_workspace(x.device, rows, config)
    args = (
        from_dlpack(x),
        from_dlpack(lengths),
        from_dlpack(out),
        from_dlpack(vals),
        from_dlpack(status),
        from_dlpack(slab),
        from_dlpack(counters),
        stream,
    )
    if key not in _compiled:
        kern = StreamingTopK(
            _DTYPES[x.dtype],
            k,
            config,
            facts.shared_memory_optin,
            next_n,
            compress_ratio,
            values is not None,
        )
        _compiled[key] = cute.compile(kern.launch, *args)
    _compiled[key](*args)
    return out
