from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Optional

XCCL_HOST_SIGNAL_MAGIC = 0x58484353
XCCL_HOST_SIGNAL_LAYOUT_VERSION = 1
XCCL_HOST_SIGNAL_SIZE = 60

_FMT = "<IHHQQQIIIIIff"


@dataclass
class HostSignalFrame:
    comm_key: int
    version: int
    ts_ns: int
    cq_posted: int
    cq_completed: int
    cq_backlog: int
    rtt_baseline_us: int
    rtt_ewma_us: int
    completion_stretch: float
    cpu_poll_delay_norm: float


class HostSignalReader:
    def __init__(self, path: str):
        self._path = path
        self._mm: Optional[mmap.mmap] = None

    def open_if_needed(self) -> bool:
        if self._mm is not None:
            return True
        if not self._path or not os.path.isfile(self._path):
            return False
        f = open(self._path, "rb")
        try:
            self._mm = mmap.mmap(f.fileno(), 4096, access=mmap.ACCESS_READ)
        finally:
            f.close()
        return True

    def read_frame(self) -> Optional[HostSignalFrame]:
        if not self.open_if_needed():
            return None
        assert self._mm is not None

        for _ in range(4):
            v1 = struct.unpack_from("<Q", self._mm, 16)[0]
            if v1 == 0 or (v1 & 1) != 0:
                continue
            raw = self._mm[:XCCL_HOST_SIGNAL_SIZE]
            vals = struct.unpack(_FMT, raw)
            v2 = struct.unpack_from("<Q", self._mm, 16)[0]
            if v1 != v2 or (v2 & 1) != 0:
                continue

            magic, layout_version, struct_size = vals[0], vals[1], vals[2]
            if (
                magic != XCCL_HOST_SIGNAL_MAGIC
                or layout_version != XCCL_HOST_SIGNAL_LAYOUT_VERSION
                or struct_size != XCCL_HOST_SIGNAL_SIZE
                or vals[4] != v1
            ):
                continue

            return HostSignalFrame(
                comm_key=vals[3],
                version=vals[4],
                ts_ns=vals[5],
                cq_posted=vals[6],
                cq_completed=vals[7],
                cq_backlog=vals[8],
                rtt_baseline_us=vals[9],
                rtt_ewma_us=vals[10],
                completion_stretch=float(vals[11]),
                cpu_poll_delay_norm=float(vals[12]),
            )
        return None

