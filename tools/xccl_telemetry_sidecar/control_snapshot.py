from __future__ import annotations

import struct
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mmap

XCCL_CONTROL_SNAPSHOT_MAGIC = 0x5843434C  # 'XCCL'
XCCL_CONTROL_SNAPSHOT_LAYOUT_VERSION = 1
XCCL_CONTROL_SNAPSHOT_SIZE = 58

XCCL_MODE_NORMAL = 0
XCCL_MODE_HOST = 1
XCCL_MODE_NET = 2
XCCL_MODE_MIXED = 3
XCCL_MODE_SEVERE = 4

_FMT = "<IHHQQQffBBHHIBB6s"


def monotonic_ns() -> int:
    return time.monotonic_ns()


def mode_to_u8(mode: str) -> int:
    m = mode.upper()
    if m == "HOST":
        return XCCL_MODE_HOST
    if m == "NET":
        return XCCL_MODE_NET
    if m == "MIXED":
        return XCCL_MODE_MIXED
    if m == "SEVERE":
        return XCCL_MODE_SEVERE
    return XCCL_MODE_NORMAL


def publish_control_frame(
    mm: "mmap.mmap",
    comm_key: int,
    host_score: float,
    net_score: float,
    mode: str,
    severity: int,
    target_channels: int,
    target_window: int,
    pacing_ns: int,
    cooldown_level: int,
    stable_epochs: int,
) -> None:
    host_score = max(0.0, min(1.0, float(host_score)))
    net_score = max(0.0, min(1.0, float(net_score)))
    severity = max(0, min(3, int(severity)))
    target_channels = max(0, min(0xFFFF, int(target_channels)))
    target_window = max(1, min(0xFFFF, int(target_window)))
    pacing_ns = max(0, min(0xFFFFFFFF, int(pacing_ns)))
    cooldown_level = max(0, min(0xFF, int(cooldown_level)))
    stable_epochs = max(0, min(0xFF, int(stable_epochs)))

    v0 = struct.unpack_from("<Q", mm, 16)[0]
    odd = v0 + 1
    even = v0 + 2
    if even == 0:
        odd, even = 1, 2

    struct.pack_into("<Q", mm, 16, odd)
    payload = struct.pack(
        _FMT,
        XCCL_CONTROL_SNAPSHOT_MAGIC,
        XCCL_CONTROL_SNAPSHOT_LAYOUT_VERSION,
        XCCL_CONTROL_SNAPSHOT_SIZE,
        int(comm_key),
        int(odd),
        monotonic_ns(),
        host_score,
        net_score,
        mode_to_u8(mode),
        severity,
        target_channels,
        target_window,
        pacing_ns,
        cooldown_level,
        stable_epochs,
        b"\x00" * 6,
    )
    mm[:XCCL_CONTROL_SNAPSHOT_SIZE] = payload
    struct.pack_into("<Q", mm, 16, even)

