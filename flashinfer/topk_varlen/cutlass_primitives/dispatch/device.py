"""Device facts, queried once per device.  The only place that looks at the hardware.

Everything above this module receives these facts as values; nothing else calls into torch or
the driver for capabilities.  Co-resident cluster counts come from the driver's occupancy
query (per-GPC placement makes ``SMs // size`` wrong: B200 fits 45 clusters of 3, not 49).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

__all__ = ["DeviceFacts", "device_facts", "max_active_clusters"]


@dataclass(frozen=True)
class DeviceFacts:
    name: str
    index: int
    capability: tuple
    sm_count: int
    shared_memory_optin: int  # bytes per block with the opt-in attribute
    supports_clusters: bool  # thread-block clusters and DSMEM (SM90+)
    supports_pdl: bool  # programmatic dependent launch (SM90+)
    packed_bf16_compare: bool  # setp.le.bf16x2 (SM90+); fp16x2 is available everywhere


@functools.cache
def _facts(index: int) -> DeviceFacts:
    props = torch.cuda.get_device_properties(index)
    cc = (props.major, props.minor)
    return DeviceFacts(
        name=props.name,
        index=index,
        capability=cc,
        sm_count=props.multi_processor_count,
        shared_memory_optin=getattr(props, "shared_memory_per_block_optin", 48 * 1024),
        supports_clusters=cc[0] >= 9,
        supports_pdl=cc[0] >= 9,
        packed_bf16_compare=cc[0] >= 9,
    )


def device_facts(device=None) -> DeviceFacts:
    """Facts for a torch device (default: the current one)."""
    index = torch.device(device or torch.cuda.current_device()).index
    if index is None:
        index = torch.cuda.current_device()
    return _facts(index)


@functools.cache
def max_active_clusters(index: int, cluster_size: int) -> int:
    """Clusters of ``cluster_size`` CTAs the device can run at once (driver occupancy query).

    Falls back to seven eighths of ``SMs // size`` if the query is unavailable.
    """
    facts = _facts(index)
    try:
        from cutlass.utils.hardware_info import HardwareInfo

        return int(HardwareInfo(index).get_max_active_clusters(cluster_size))
    except Exception:  # noqa: BLE001 - the query is an optimization, never a requirement
        return (facts.sm_count // cluster_size) * 7 // 8
