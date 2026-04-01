"""
与 xccl_telemetry_hint.h 中 TelemetryHintSnapshot 一致的 36 字节布局 + §5.3 发布协议。
"""
from __future__ import annotations

import os
import struct
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mmap

SIZE = 36
FMT_TAIL = "<QffffI"  # ts_ns + 4 floats + flags

XCCL_HINT_F_SEVERE = 1 << 0
XCCL_HINT_F_CAUTION = 1 << 1
XCCL_HINT_F_STALE_WRITER = 1 << 2


def monotonic_ns() -> int:
    return time.monotonic_ns()


def publish_frame(
    mm: "mmap.mmap",
    cnp: float,
    ce: float,
    pcie: float,
    rnic: float,
    flags: int,
) -> None:
    """奇数 → payload → 偶数；字段 clamp 到合理范围由调用方负责。"""
    cnp = max(0.0, min(1.0, float(cnp)))
    ce = max(0.0, min(1.0, float(ce)))
    pcie = max(0.0, min(1.0, float(pcie)))
    rnic = max(0.0, min(1.0, float(rnic)))
    flags = int(flags) & 0xFFFFFFFF

    v0 = struct.unpack_from("<Q", mm, 0)[0]
    odd = v0 + 1
    even = v0 + 2
    if even == 0:
        odd, even = 1, 2
    struct.pack_into("<Q", mm, 0, odd)
    tail = struct.pack(FMT_TAIL, monotonic_ns(), cnp, ce, pcie, rnic, flags)
    mm[8 : 8 + 28] = tail
    struct.pack_into("<Q", mm, 0, even)


def open_mmap_region(path: str, length: int = 4096) -> "mmap.mmap":
    """打开或创建文件；若已存在且小于 length 则扩展，避免无谓截断已映射文件。"""
    import mmap as mmap_mod

    if os.path.exists(path):
        sz = os.path.getsize(path)
        if sz < length:
            with open(path, "r+b") as f:
                f.truncate(length)
    else:
        with open(path, "w+b") as f:
            f.truncate(length)
    f = open(path, "r+b")
    mm = mmap_mod.mmap(f.fileno(), length)
    f.close()
    return mm


def init_zero(mm: "mmap.mmap", length: int) -> None:
    mm[:length] = b"\x00" * length
