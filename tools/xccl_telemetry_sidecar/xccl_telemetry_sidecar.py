#!/usr/bin/env python3
"""
XCCL Host Telemetry Sidecar（MVP）
见 design_docs/xccl_host_telemetry_sidecar_design.md

周期采集 netdev / ethtool / PSI，归一化后写入 TelemetryHintSnapshot（与 NCCL xccl_telemetry_hint.h 一致）。

用法：
  export XCCL_TELEMETRY_FILE=/path/to/xccl_telemetry_hint.bin
  # 与 NCCL 一致：
  export NCCL_CC_HINT_MMAP_PATH=$XCCL_TELEMETRY_FILE
  python3 xccl_telemetry_sidecar.py --ifaces eth0

  # 多口取 max 利用率（默认）
  python3 xccl_telemetry_sidecar.py --ifaces eth0,eth1 --interval-ms 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from collectors import (
    mlx5_rdma_bytes_total,
    parse_ethtool_S,
    path_for_iface,
    read_iface_speed_mbits,
    read_netdev_statistics,
    read_psi_memory_some_avg10,
)
from snapshot import (
    XCCL_HINT_F_CAUTION,
    XCCL_HINT_F_SEVERE,
    init_zero,
    open_mmap_region,
    publish_frame,
)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    ap = argparse.ArgumentParser(description="XCCL telemetry sidecar → mmap snapshot")
    ap.add_argument(
        "--file",
        default=os.environ.get("XCCL_TELEMETRY_FILE")
        or os.environ.get("NCCL_CC_HINT_MMAP_PATH", ""),
        help="mmap 文件路径（与 NCCL_CC_HINT_MMAP_PATH 一致）",
    )
    ap.add_argument(
        "--ifaces",
        default=os.environ.get("XCCL_TELEMETRY_IFACES", "eth0"),
        help="逗号分隔 netdev 名，见附录 ibdev2netdev",
    )
    ap.add_argument("--interval-ms", type=int, default=int(os.environ.get("XCCL_TELEMETRY_INTERVAL_MS", "200")))
    ap.add_argument("--ecn-thresh", type=float, default=float(os.environ.get("XCCL_ECN_DELTA_THRESH", "1000.0")), help="周期内 rx_ecn_mark 增量超过此值开始打满 cnp（线性饱和）")
    ap.add_argument("--ce-thresh", type=float, default=float(os.environ.get("XCCL_CE_ERR_DELTA_THRESH", "100.0")), help="周期内错误计数总增量阈值")
    ap.add_argument("--severe", type=float, default=0.85, help="任 level >= 此值则 SEVERE")
    ap.add_argument("--caution", type=float, default=0.55, help="未 SEVERE 且 level >= 此值则 CAUTION")
    ap.add_argument("--dry-run", action="store_true", help="只打印不算写文件")
    args = ap.parse_args()

    if not args.file and not args.dry_run:
        print("ERROR: 请设置 --file 或 XCCL_TELEMETRY_FILE / NCCL_CC_HINT_MMAP_PATH", file=sys.stderr)
        sys.exit(2)

    ifaces = [x.strip() for x in args.ifaces.split(",") if x.strip()]
    if not ifaces:
        print("ERROR: empty --ifaces", file=sys.stderr)
        sys.exit(2)

    interval_s = max(0.05, args.interval_ms / 1000.0)

    mm = None
    length = 4096
    if not args.dry_run:
        mm = open_mmap_region(args.file, length)
        init_zero(mm, length)

    prev_net: dict[str, dict[str, int]] = {}
    prev_ecn: dict[str, int] = {}
    prev_rdma_bytes: dict[str, int] = {}
    t_prev: float | None = None
    debug = os.environ.get("XCCL_TELEMETRY_DEBUG", "").strip() in ("1", "true", "yes")

    print(f"sidecar: ifaces={ifaces} interval={args.interval_ms}ms file={args.file!r} dry_run={args.dry_run}", flush=True)
    print(
        "note: RoCE/GDR 流量常不计入 netdev rx_bytes/tx_bytes；rnic 已优先用 ethtool RDMA 字节计数",
        flush=True,
    )

    while True:
        t_now = time.monotonic()
        dt = interval_s if t_prev is None else max(1e-3, t_now - t_prev)
        t_prev = t_now

        rnic_max = 0.0
        ce_sum_delta = 0.0
        cnp_max = 0.0

        for iface in ifaces:
            p = path_for_iface(iface)
            if not os.path.isdir(p):
                print(f"WARN: {p} missing", flush=True)
                continue

            speed_m = read_iface_speed_mbits(p)
            stats = read_netdev_statistics(p)
            eth = parse_ethtool_S(iface)
            rdma_total = mlx5_rdma_bytes_total(eth)

            prev = prev_net.get(iface)
            if prev is not None:
                d_rx = max(0, stats["rx_bytes"] - prev["rx_bytes"])
                d_tx = max(0, stats["tx_bytes"] - prev["tx_bytes"])
                d_netdev = d_rx + d_tx
                d_rdma = max(0, rdma_total - prev_rdma_bytes.get(iface, 0))
                # RoCE：RDMA 与 netdev 计数通常不同源；取较大增量避免漏计
                bytes_delta = max(d_rdma, d_netdev)

                total_bps = bytes_delta * 8.0 / dt
                if speed_m > 0:
                    util = total_bps / (speed_m * 1e6)
                    rnic_max = max(rnic_max, clamp01(util))
                if debug and (d_rdma > 0 or d_netdev > 0):
                    print(
                        f"debug {iface}: d_rdma_bytes={d_rdma} d_netdev_bytes={d_netdev} speed_mbits={speed_m}",
                        flush=True,
                    )
                err = (
                    max(0, stats["rx_errors"] - prev["rx_errors"])
                    + max(0, stats["tx_errors"] - prev["tx_errors"])
                    + max(0, stats["rx_dropped"] - prev["rx_dropped"])
                    + max(0, stats["tx_dropped"] - prev["tx_dropped"])
                )
                ce_sum_delta += err
            prev_net[iface] = stats.copy()
            prev_rdma_bytes[iface] = rdma_total

            ecn = int(eth.get("rx_ecn_mark", 0))
            p_e = prev_ecn.get(iface)
            if p_e is not None:
                d_ecn = max(0, ecn - p_e)
                cnp_max = max(cnp_max, clamp01(d_ecn / max(1e-6, args.ecn_thresh)))
            prev_ecn[iface] = ecn

        ce_level = clamp01(ce_sum_delta / max(1e-6, args.ce_thresh))
        pcie = read_psi_memory_some_avg10()
        if pcie is None:
            pcie = 0.0
        else:
            pcie = clamp01(pcie)

        flags = 0
        levels = [cnp_max, ce_level, pcie, rnic_max]
        mx = max(levels)
        if mx >= args.severe:
            flags |= XCCL_HINT_F_SEVERE
        elif mx >= args.caution:
            flags |= XCCL_HINT_F_CAUTION

        if args.dry_run:
            # 100G 链路上「几千字节/200ms」对应利用率约 1e-6 量级，.4f 会全 0；附 ppm（占线速百万分比）便于人读
            rnic_ppm = rnic_max * 1e6
            print(
                f"cnp={cnp_max:.4f} ce={ce_level:.4f} pcie={pcie:.4f} "
                f"rnic={rnic_max:.6f} ({rnic_ppm:.3f} ppm) flags={flags}",
                flush=True,
            )
        else:
            assert mm is not None
            publish_frame(mm, cnp_max, ce_level, pcie, rnic_max, flags)
            mm.flush()

        elapsed = time.monotonic() - t_now
        sleep_s = max(0.0, interval_s - elapsed)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
