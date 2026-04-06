from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Optional

XCCL_HOST_SIGNAL_MAGIC = 0x58484353
XCCL_HOST_SIGNAL_LAYOUT_VERSION = 2
XCCL_HOST_SIGNAL_SIZE = 76
XCCL_HOST_SIGNAL_LAYOUT_VERSION_V1 = 1
XCCL_HOST_SIGNAL_SIZE_V1 = 60

_FMT_V2 = "<IHHQQQIIIIIIfffff"
_FMT_V1 = "<IHHQQQIIIIIff"


@dataclass
class HostSignalFrame:
    comm_key: int
    version: int
    ts_ns: int
    cq_posted: int
    cq_completed: int
    cq_backlog: int
    cq_backlog_max: int
    rtt_baseline_us: int
    rtt_ewma_us: int
    completion_stretch: float
    cpu_poll_delay_norm: float
    cq_backlog_ewma: float
    completion_drain_rate: float
    poll_gap_norm: float


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
            header = self._mm[:8]
            if len(header) < 8:
                continue
            magic, layout_version, struct_size = struct.unpack("<IHH", header)
            if magic != XCCL_HOST_SIGNAL_MAGIC:
                continue
            if layout_version == XCCL_HOST_SIGNAL_LAYOUT_VERSION and struct_size == XCCL_HOST_SIGNAL_SIZE:
                raw = self._mm[:XCCL_HOST_SIGNAL_SIZE]
                vals = struct.unpack(_FMT_V2, raw)
                mode = "v2"
            elif layout_version == XCCL_HOST_SIGNAL_LAYOUT_VERSION_V1 and struct_size == XCCL_HOST_SIGNAL_SIZE_V1:
                raw = self._mm[:XCCL_HOST_SIGNAL_SIZE_V1]
                vals = struct.unpack(_FMT_V1, raw)
                mode = "v1"
            else:
                continue
            v2 = struct.unpack_from("<Q", self._mm, 16)[0]
            if v1 != v2 or (v2 & 1) != 0:
                continue

            if vals[4] != v1:
                continue

            if mode == "v2":
                cq_backlog_max = vals[9]
                rtt_baseline_us = vals[10]
                rtt_ewma_us = vals[11]
                completion_stretch = float(vals[12])
                cpu_poll_delay_norm = float(vals[13])
                cq_backlog_ewma = float(vals[14])
                completion_drain_rate = float(vals[15])
                poll_gap_norm = float(vals[16])
            else:
                cq_backlog_max = vals[8]
                rtt_baseline_us = vals[9]
                rtt_ewma_us = vals[10]
                completion_stretch = float(vals[11])
                cpu_poll_delay_norm = float(vals[12])
                cq_backlog_ewma = float(vals[8])
                completion_drain_rate = 1.0
                poll_gap_norm = 0.0

            return HostSignalFrame(
                comm_key=vals[3],
                version=vals[4],
                ts_ns=vals[5],
                cq_posted=vals[6],
                cq_completed=vals[7],
                cq_backlog=vals[8],
                cq_backlog_max=cq_backlog_max,
                rtt_baseline_us=rtt_baseline_us,
                rtt_ewma_us=rtt_ewma_us,
                completion_stretch=completion_stretch,
                cpu_poll_delay_norm=cpu_poll_delay_norm,
                cq_backlog_ewma=cq_backlog_ewma,
                completion_drain_rate=completion_drain_rate,
                poll_gap_norm=poll_gap_norm,
            )
        return None

