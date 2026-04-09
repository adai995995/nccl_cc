#!/usr/bin/env python3
"""
向 TelemetryHint 共享区写入合法快照（design_docs §5.3：奇数写入中 → 写 payload → 偶数发布）。

推荐与文件 mmap 联用：
  python3 write_telemetry_hint.py --path /tmp/xccl_hint.bin --cnp 0.3

另开终端：
  export NCCL_CC_HINT_ENABLE=1 NCCL_CC_HINT_MMAP_PATH=/tmp/xccl_hint.bin
  export NCCL_CC_HINT_TTL_NS=10000000000
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET ./allreduce_test

期望日志含：XCCL hint: telemetry mmap active
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
import time

SIZE = 36  # TelemetryHintSnapshot
FMT_TAIL = "<QffffI"  # ts_ns + 4 floats + flags = 8+4*4+4 = 28

XCCL_HINT_F_SEVERE = 1 << 0
XCCL_HINT_F_CAUTION = 1 << 1
XCCL_HINT_F_STALE_WRITER = 1 << 2


def monotonic_ns() -> int:
    return time.monotonic_ns()


def publish_frame(mm: mmap.mmap, cnp: float, ce: float, pcie: float, rnic: float, flags: int) -> None:
    v0 = struct.unpack_from("<Q", mm, 0)[0]
    odd = v0 + 1
    even = v0 + 2
    if even == 0:
        odd, even = 1, 2
    struct.pack_into("<Q", mm, 0, odd)
    tail = struct.pack(FMT_TAIL, monotonic_ns(), cnp, ce, pcie, rnic, flags)
    mm[8 : 8 + 28] = tail
    struct.pack_into("<Q", mm, 0, even)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write XCCL TelemetryHintSnapshot to mmap file or shm")
    ap.add_argument(
        "--path",
        default="",
        help="文件路径，或目录（将写入 目录/xccl_telemetry_hint.bin）；NCCL_CC_HINT_MMAP_PATH 需与此文件一致",
    )
    ap.add_argument("--shm", action="store_true", help="使用 /dev/shm/xccl_telemetry_hint（需与 NCCL 默认名一致时可 ln -s）")
    ap.add_argument("--cnp", type=float, default=0.25)
    ap.add_argument("--ce", type=float, default=0.0)
    ap.add_argument("--pcie", type=float, default=0.0)
    ap.add_argument("--rnic", type=float, default=0.0)
    ap.add_argument("--flags", type=lambda x: int(x, 0), default=0)
    ap.add_argument("--interval", type=float, default=0.5, help="循环发布间隔（秒）；0 只写一次")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    once = args.once or args.interval <= 0
    length = max(SIZE, 4096)

    if args.shm:
        shm_path = "/dev/shm/xccl_telemetry_hint"
        f = open(shm_path, "w+b")
        f.truncate(length)
        mm = mmap.mmap(f.fileno(), length)
        f.close()
        target = shm_path
    elif args.path:
        file_path = args.path
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "xccl_telemetry_hint.bin")
            print(
                f"注意: --path 是目录，已自动使用文件:\n  {file_path}\n"
                f"请设置: export NCCL_CC_HINT_MMAP_PATH={file_path}",
                file=sys.stderr,
            )
        f = open(file_path, "w+b")
        f.truncate(length)
        mm = mmap.mmap(f.fileno(), length)
        f.close()
        target = file_path
    else:
        ap.error("请指定 --path FILE 或 --shm")

    mm[:length] = b"\x00" * length

    if once:
        publish_frame(mm, args.cnp, args.ce, args.pcie, args.rnic, args.flags)
        mm.flush()
        mm.close()
        print(f"OK: published one frame ({SIZE} bytes layout) to {target}")
        return

    print(f"Loop publishing to {target}; Ctrl+C to stop")
    try:
        while True:
            publish_frame(mm, args.cnp, args.ce, args.pcie, args.rnic, args.flags)
            mm.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        mm.close()
        print("stopped")


if __name__ == "__main__":
    main()
