"""The top-k router: one entry point, the kernel chosen from the device facts and the problem.

This is what a consumer sees as a single backend.  The rule is the plan's dispatch order:
rows that fit a CTA's registers take the register-resident kernel; everything longer takes the
streaming kernel, whose own policy picks the split and merge.  Both kernels share the output
contract, so the choice is invisible to the caller.
"""

from __future__ import annotations

import torch

from ...dispatch.device import device_facts

from ..kernels.register_resident import (
    RegisterConfig,
    register_config_for,
    topk_register,
)
from ..kernels.streaming import StreamingConfig, topk_streaming
from .streaming_policy import streaming_config_for

__all__ = ["REGISTER_MAX_ROW", "choose", "topk"]

REGISTER_MAX_ROW = (
    1024 * 16
)  # words per thread x threads; 16-bit rows hold twice as many


def choose(
    facts, dtype: torch.dtype, k: int, n: int, rows: int
) -> tuple[str, RegisterConfig | StreamingConfig]:
    """(kernel name, configuration) for a batch of ``rows`` rows of ``n`` elements."""
    per_word = 1 if dtype == torch.float32 else 2
    if n <= REGISTER_MAX_ROW * per_word and k <= 4096:
        return "register", register_config_for(facts, dtype, k, n, rows)
    return "streaming", streaming_config_for(facts, dtype, k, n, rows)


def topk(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest elements of each row of ``x`` (rows, N), int32 (rows, k).

    ``lengths`` (rows, int32) limits each row to its first elements; rows shorter than k are
    padded with -1.  NaN ranks above +inf.  Order within a row is unspecified.
    """
    rows, n = x.shape
    kernel, config = choose(device_facts(x.device), x.dtype, k, n, rows)
    if isinstance(config, RegisterConfig):
        return topk_register(x, k, lengths=lengths, config=config, out=out)
    return topk_streaming(x, k, lengths=lengths, config=config, out=out)
