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
    mlx5_drop_retry_total,
    mlx5_rdma_bytes_total,
    parse_ethtool_S,
    path_for_iface,
    read_host_psi_mix_avg10,
    read_iface_speed_mbits,
    read_netdev_statistics,
)
from control_snapshot import publish_control_frame
from host_signal_snapshot import HostSignalReader
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
    ap.add_argument("--drop-retry-thresh", type=float, default=float(os.environ.get("XCCL_DROP_RETRY_DELTA_THRESH", "100.0")), help="周期内 ethtool drop/retry 总增量阈值")
    ap.add_argument("--mode-margin", type=float, default=float(os.environ.get("XCCL_MODE_MARGIN", "0.10")), help="host/net 判因滞回边界")
    ap.add_argument("--severe", type=float, default=0.85, help="任 level >= 此值则 SEVERE")
    ap.add_argument("--caution", type=float, default=0.55, help="未 SEVERE 且 level >= 此值则 CAUTION")
    ap.add_argument("--dry-run", action="store_true", help="只打印不算写文件")
    ap.add_argument(
        "--host-signal-file",
        default=os.environ.get("XCCL_HOST_SIGNAL_FILE")
        or os.environ.get("NCCL_CC_HOST_SIGNAL_MMAP_PATH", "/dev/shm/xccl_host_signal_snapshot"),
        help="NCCL host signal mmap 文件路径（S1: 只读融合）",
    )
    ap.add_argument(
        "--cq-backlog-thresh",
        type=float,
        default=float(os.environ.get("XCCL_CQ_BACKLOG_THRESH", "128.0")),
        help="cq_backlog 归一化阈值",
    )
    ap.add_argument(
        "--completion-stretch-thresh",
        type=float,
        default=float(os.environ.get("XCCL_COMPLETION_STRETCH_THRESH", "1.5")),
        help="completion stretch 归一化阈值（stretch-1 的分母）",
    )
    ap.add_argument(
        "--control-file",
        default=os.environ.get("XCCL_CONTROL_SNAPSHOT_FILE")
        or os.environ.get("NCCL_CC_CONTROL_SNAPSHOT_MMAP_PATH", "/dev/shm/xccl_control_snapshot"),
        help="XCCL control snapshot 输出路径（S3）",
    )
    ap.add_argument("--control-comm-key", type=int, default=int(os.environ.get("XCCL_CONTROL_COMM_KEY", "0")), help="control snapshot comm_key，0=wildcard")
    ap.add_argument("--control-base-window", type=int, default=int(os.environ.get("XCCL_CONTROL_BASE_WINDOW", "256")))
    ap.add_argument("--control-min-window", type=int, default=int(os.environ.get("XCCL_CONTROL_MIN_WINDOW", "16")))
    ap.add_argument("--control-max-window", type=int, default=int(os.environ.get("XCCL_CONTROL_MAX_WINDOW", "512")))
    ap.add_argument("--control-host-max-channels", type=int, default=int(os.environ.get("XCCL_CONTROL_HOST_MAX_CHANNELS", "4")))
    ap.add_argument("--control-host-min-channels", type=int, default=int(os.environ.get("XCCL_CONTROL_HOST_MIN_CHANNELS", "1")))
    ap.add_argument("--control-max-pacing-ns", type=int, default=int(os.environ.get("XCCL_CONTROL_MAX_PACING_NS", "50000")))
    ap.add_argument("--control-switch-stable-epochs", type=int, default=int(os.environ.get("XCCL_CONTROL_SWITCH_STABLE_EPOCHS", "3")))
    ap.add_argument("--control-cooldown-epochs", type=int, default=int(os.environ.get("XCCL_CONTROL_COOLDOWN_EPOCHS", "5")))
    ap.add_argument(
        "--control-window-max-step",
        type=int,
        default=int(os.environ.get("XCCL_CONTROL_WINDOW_MAX_STEP", "16")),
        help="每个周期 target_window 最大变化步长",
    )
    ap.add_argument(
        "--control-channel-update-period",
        type=int,
        default=int(os.environ.get("XCCL_CONTROL_CHANNEL_UPDATE_PERIOD", "3")),
        help="target_channels 允许更新的最小周期数",
    )
    ap.add_argument(
        "--control-pacing-smooth-alpha",
        type=float,
        default=float(os.environ.get("XCCL_CONTROL_PACING_SMOOTH_ALPHA", "0.35")),
        help="pacing_ns 一阶平滑系数，越小越平滑",
    )
    ap.add_argument(
        "--host-psi-weight",
        type=float,
        default=float(os.environ.get("XCCL_HOST_PSI_WEIGHT", "0.05")),
        help="host_psi_mix 在 host 归因中的弱提示权重",
    )
    ap.add_argument(
        "--host-psi-gate-backlog",
        type=float,
        default=float(os.environ.get("XCCL_HOST_PSI_GATE_BACKLOG", "0.10")),
        help="backlog 低于该值时不放大 PSI",
    )
    ap.add_argument(
        "--host-psi-gate-stretch",
        type=float,
        default=float(os.environ.get("XCCL_HOST_PSI_GATE_STRETCH", "0.10")),
        help="stretch 低于该值时不放大 PSI",
    )
    ap.add_argument(
        "--net-strong-thresh",
        type=float,
        default=float(os.environ.get("XCCL_NET_STRONG_THRESH", "0.60")),
        help="host 信号不可用时，NET 归因所需强证据阈值",
    )
    ap.add_argument("--control-host-enter", type=float, default=float(os.environ.get("XCCL_CONTROL_HOST_ENTER", "0.55")))
    ap.add_argument("--control-net-enter", type=float, default=float(os.environ.get("XCCL_CONTROL_NET_ENTER", "0.55")))
    ap.add_argument("--control-recovery-enter", type=float, default=float(os.environ.get("XCCL_CONTROL_RECOVERY_ENTER", "0.75")))
    ap.add_argument("--control-host-exit", type=float, default=float(os.environ.get("XCCL_CONTROL_HOST_EXIT", "0.35")))
    ap.add_argument("--control-net-exit", type=float, default=float(os.environ.get("XCCL_CONTROL_NET_EXIT", "0.35")))
    ap.add_argument("--control-normal-exit", type=float, default=float(os.environ.get("XCCL_CONTROL_NORMAL_EXIT", "0.30")))
    ap.add_argument("--control-enter-epochs", type=int, default=int(os.environ.get("XCCL_CONTROL_ENTER_EPOCHS", "3")))
    ap.add_argument("--control-exit-epochs", type=int, default=int(os.environ.get("XCCL_CONTROL_EXIT_EPOCHS", "5")))
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
    ctrl_mm = None
    if not args.dry_run and args.control_file:
        ctrl_mm = open_mmap_region(args.control_file, length)
        init_zero(ctrl_mm, length)

    prev_net: dict[str, dict[str, int]] = {}
    prev_ecn: dict[str, int] = {}
    prev_rdma_bytes: dict[str, int] = {}
    prev_drop_retry: dict[str, int] = {}
    t_prev: float | None = None
    debug = os.environ.get("XCCL_TELEMETRY_DEBUG", "").strip() in ("1", "true", "yes")
    host_reader = HostSignalReader(args.host_signal_file)
    active_mode = "MIXED"
    mode_candidate = ""
    mode_candidate_epochs = 0
    mode_stable_epochs = 0
    cooldown_left = 0
    backlog_ewma_norm = 0.0
    backlog_spike_norm = 0.0
    applied_target_window = int(round(args.control_base_window))
    applied_target_channels = 0
    applied_pacing_ns = 0
    channel_hold_epochs = 0
    should_control = False
    control_state = "NORMAL"
    enter_epochs = 0
    exit_epochs = 0

    print(f"sidecar: ifaces={ifaces} interval={args.interval_ms}ms file={args.file!r} dry_run={args.dry_run}", flush=True)
    print(f"sidecar: host_signal_file={args.host_signal_file!r}", flush=True)
    print(f"sidecar: control_file={args.control_file!r}", flush=True)
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
        drop_retry_sum_delta = 0.0

        for iface in ifaces:
            p = path_for_iface(iface)
            if not os.path.isdir(p):
                print(f"WARN: {p} missing", flush=True)
                continue

            speed_m = read_iface_speed_mbits(p)
            stats = read_netdev_statistics(p)
            eth = parse_ethtool_S(iface)
            rdma_total = mlx5_rdma_bytes_total(eth)
            drop_retry_total = mlx5_drop_retry_total(eth)

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
            p_dr = prev_drop_retry.get(iface)
            if p_dr is not None:
                drop_retry_sum_delta += max(0, drop_retry_total - p_dr)
            prev_drop_retry[iface] = drop_retry_total

            ecn = int(eth.get("rx_ecn_mark", 0))
            p_e = prev_ecn.get(iface)
            if p_e is not None:
                d_ecn = max(0, ecn - p_e)
                cnp_max = max(cnp_max, clamp01(d_ecn / max(1e-6, args.ecn_thresh)))
            prev_ecn[iface] = ecn

        ce_level = clamp01(ce_sum_delta / max(1e-6, args.ce_thresh))
        drop_retry_norm = clamp01(drop_retry_sum_delta / max(1e-6, args.drop_retry_thresh))
        psi = read_host_psi_mix_avg10()
        psi_cpu = clamp01(psi["cpu"])
        psi_mem = clamp01(psi["memory"])
        psi_io = clamp01(psi["io"])
        host_psi_mix = clamp01(psi["mix"])

        host_frame = host_reader.read_frame()
        cq_backlog_norm = 0.0
        cq_backlog_max_norm = 0.0
        completion_stretch_norm = 0.0
        cpu_poll_delay_norm = 0.0
        drain_pressure_norm = 0.0
        poll_gap_norm = 0.0
        host_snapshot_ok = host_frame is not None
        host_cause_score = 0.0
        if host_frame is not None:
            backlog_for_cause = host_frame.cq_backlog_ewma if host_frame.cq_backlog_ewma > 0.0 else float(host_frame.cq_backlog)
            cq_backlog_norm = clamp01(backlog_for_cause / max(1e-6, args.cq_backlog_thresh))
            cq_backlog_max_norm = clamp01(float(host_frame.cq_backlog_max) / max(1e-6, args.cq_backlog_thresh))
            completion_stretch_norm = clamp01(
                (max(1.0, host_frame.completion_stretch) - 1.0)
                / max(1e-6, args.completion_stretch_thresh)
            )
            poll_gap_norm = clamp01(host_frame.poll_gap_norm)
            cpu_poll_delay_norm = clamp01(max(host_frame.cpu_poll_delay_norm, poll_gap_norm))
            drain_ratio = clamp01(host_frame.completion_drain_rate)
            drain_pressure_norm = clamp01(1.0 - drain_ratio)
            # backlog 分离“持续压力(ewma)”与“尖峰(spike)”；优先使用 NCCL 导出的 ewma/max
            backlog_ewma_norm = cq_backlog_norm
            backlog_spike_norm = max(backlog_spike_norm * 0.8, cq_backlog_max_norm)
            psi_gate = 1.0 if (
                backlog_ewma_norm >= args.host_psi_gate_backlog
                or completion_stretch_norm >= args.host_psi_gate_stretch
            ) else 0.0
            host_cause_score = clamp01(
                0.60 * backlog_ewma_norm
                + 0.20 * drain_pressure_norm
                + 0.15 * cpu_poll_delay_norm
                + max(0.0, min(0.20, args.host_psi_weight)) * host_psi_mix * psi_gate
            )
        else:
            # host 根因信号缺失时，不以 PSI 直接代替 host 归因
            backlog_ewma_norm *= 0.9
            backlog_spike_norm *= 0.9

        symptom_score = completion_stretch_norm
        net_cause_score = clamp01(
            0.65 * cnp_max + 0.30 * drop_retry_norm + 0.05 * rnic_max
        )
        if host_snapshot_ok and host_cause_score > net_cause_score + args.mode_margin:
            raw_mode = "HOST"
        elif net_cause_score > host_cause_score + args.mode_margin:
            raw_mode = "NET"
        elif (not host_snapshot_ok) and net_cause_score >= args.net_strong_thresh:
            # host 缺失时，仅在 net 强证据成立下进入 NET；否则保持保守
            raw_mode = "NET"
        else:
            raw_mode = "MIXED"

        if raw_mode == active_mode:
            mode_stable_epochs += 1
            mode_candidate = ""
            mode_candidate_epochs = 0
        else:
            if raw_mode == mode_candidate:
                mode_candidate_epochs += 1
            else:
                mode_candidate = raw_mode
                mode_candidate_epochs = 1
            if cooldown_left == 0 and mode_candidate_epochs >= max(1, args.control_switch_stable_epochs):
                active_mode = raw_mode
                mode_stable_epochs = 1
                mode_candidate = ""
                mode_candidate_epochs = 0
                cooldown_left = max(0, args.control_cooldown_epochs)
            else:
                mode_stable_epochs += 1

        # 先做控制门控：避免把 AIMD/外控当作常态兜底
        enter_cond = (host_cause_score >= args.control_host_enter) or (net_cause_score >= args.control_net_enter)
        exit_cond = (
            host_cause_score <= args.control_host_exit
            and net_cause_score <= args.control_net_exit
            and symptom_score <= args.control_normal_exit
        )
        if not should_control:
            enter_epochs = enter_epochs + 1 if enter_cond else 0
            exit_epochs = 0
            if enter_epochs >= max(1, args.control_enter_epochs):
                should_control = True
                enter_epochs = 0
        else:
            exit_epochs = exit_epochs + 1 if exit_cond else 0
            enter_epochs = 0
            if exit_epochs >= max(1, args.control_exit_epochs):
                should_control = False
                exit_epochs = 0

        if not should_control:
            control_state = "NORMAL"
        else:
            sev = max(host_cause_score, net_cause_score, symptom_score)
            control_state = "RECOVERY" if sev >= args.control_recovery_enter else "GUARDED"

        applied_mode = active_mode if should_control else "NORMAL"

        max_score = max(host_cause_score, net_cause_score, symptom_score)
        target_window = int(round(args.control_base_window))
        target_channels = 0
        pacing_ns = 0

        if control_state == "NORMAL":
            # 显式 no-control：尽量贴近原生 NCCL 行为
            target_window = int(round(args.control_base_window))
            target_channels = 0
            pacing_ns = 0
        elif control_state == "GUARDED" and applied_mode == "HOST":
            host_factor = clamp01(host_cause_score)
            target_window = int(round(args.control_base_window * (1.0 - 0.15 * host_factor)))
            c_hi = max(args.control_host_min_channels, args.control_host_max_channels)
            c_lo = max(1, min(args.control_host_min_channels, args.control_host_max_channels))
            target_channels = int(round(c_hi - 0.5 * (c_hi - c_lo) * host_factor))
        elif control_state == "GUARDED" and applied_mode == "NET":
            net_factor = clamp01(net_cause_score)
            target_window = int(round(args.control_base_window * (1.0 - 0.20 * net_factor)))
            pacing_ns = int(round(args.control_max_pacing_ns * 0.40 * net_factor))
        elif control_state == "GUARDED" and applied_mode == "MIXED":
            mix_factor = clamp01(max_score)
            target_window = int(round(args.control_base_window * (1.0 - 0.12 * mix_factor)))
            target_channels = int(round(max(1, args.control_host_max_channels) * (1.0 - 0.30 * host_cause_score)))
            pacing_ns = int(round(args.control_max_pacing_ns * 0.25 * net_cause_score))
        elif applied_mode == "HOST":
            host_factor = clamp01(host_cause_score)
            target_window = int(round(args.control_base_window * (1.0 - 0.35 * host_factor)))
            c_hi = max(args.control_host_min_channels, args.control_host_max_channels)
            c_lo = max(1, min(args.control_host_min_channels, args.control_host_max_channels))
            if c_hi == c_lo:
                target_channels = c_lo
            else:
                target_channels = int(round(c_hi - (c_hi - c_lo) * host_factor))
        elif applied_mode == "NET":
            net_factor = clamp01(net_cause_score)
            target_window = int(round(args.control_base_window * (1.0 - 0.45 * net_factor)))
            pacing_ns = int(round(args.control_max_pacing_ns * net_factor))
        elif applied_mode == "MIXED":
            mix_factor = clamp01(max_score)
            target_window = int(round(args.control_base_window * (1.0 - 0.30 * mix_factor)))
            target_channels = int(round(max(1, args.control_host_max_channels) * (1.0 - 0.5 * host_cause_score)))
            pacing_ns = int(round(args.control_max_pacing_ns * 0.5 * net_cause_score))

        target_window = max(args.control_min_window, min(args.control_max_window, target_window))
        if target_channels > 0:
            target_channels = max(1, min(0xFFFF, target_channels))
        pacing_ns = max(0, min(0xFFFFFFFF, pacing_ns))

        # Actuator 防抖：模式稳定不等于动作可大跳
        if control_state == "NORMAL":
            applied_target_window = int(round(args.control_base_window))
            applied_target_channels = 0
            applied_pacing_ns = 0
            channel_hold_epochs = 0
            target_window = applied_target_window
            target_channels = applied_target_channels
            pacing_ns = applied_pacing_ns
        else:
            wstep = max(1, args.control_window_max_step)
            target_window = max(applied_target_window - wstep, min(applied_target_window + wstep, target_window))
            applied_target_window = target_window

            channel_hold_epochs += 1
            channel_period = max(1, args.control_channel_update_period)
            if channel_hold_epochs >= channel_period:
                applied_target_channels = target_channels
                channel_hold_epochs = 0
            target_channels = applied_target_channels

            p_alpha = clamp01(args.control_pacing_smooth_alpha)
            applied_pacing_ns = int(round((1.0 - p_alpha) * applied_pacing_ns + p_alpha * pacing_ns))
            pacing_ns = max(0, min(0xFFFFFFFF, applied_pacing_ns))

        flags = 0
        levels = [host_cause_score, net_cause_score, symptom_score]
        mx = max(levels)
        if mx >= args.severe:
            flags |= XCCL_HINT_F_SEVERE
        elif mx >= args.caution:
            flags |= XCCL_HINT_F_CAUTION

        if args.dry_run:
            # 100G 链路上「几千字节/200ms」对应利用率约 1e-6 量级，.4f 会全 0；附 ppm（占线速百万分比）便于人读
            rnic_ppm = rnic_max * 1e6
            print(
                f"cnp={cnp_max:.4f} ce={ce_level:.4f} "
                f"drop_retry={drop_retry_norm:.4f} "
                f"host_cause={host_cause_score:.4f}(psi_mix={host_psi_mix:.4f},psi_cpu={psi_cpu:.4f},"
                f"psi_mem={psi_mem:.4f},psi_io={psi_io:.4f},cq={cq_backlog_norm:.4f},"
                f"cq_max={cq_backlog_max_norm:.4f},cq_ewma={backlog_ewma_norm:.4f},cq_spike={backlog_spike_norm:.4f},"
                f"drain={drain_pressure_norm:.4f},poll_gap={poll_gap_norm:.4f},cpu={cpu_poll_delay_norm:.4f},"
                f"host_ok={1 if host_snapshot_ok else 0}) "
                f"net_cause={net_cause_score:.4f}(cnp={cnp_max:.4f},"
                f"drop_retry={drop_retry_norm:.4f},rnic={rnic_max:.4f}) "
                f"symptom={symptom_score:.4f}(stretch={completion_stretch_norm:.4f}) "
                f"mode={applied_mode} should_ctl={1 if should_control else 0} state={control_state} "
                f"tw={target_window} tc={target_channels} pace_ns={pacing_ns} "
                f"enter={enter_epochs} exit={exit_epochs} stable={mode_stable_epochs} cooldown={cooldown_left} "
                f"rnic={rnic_max:.6f} ({rnic_ppm:.3f} ppm) flags={flags}",
                flush=True,
            )
        else:
            assert mm is not None
            ce_level_ext = max(ce_level, drop_retry_norm)
            publish_frame(mm, cnp_max, ce_level_ext, host_cause_score, rnic_max, flags)
            mm.flush()
            if ctrl_mm is not None:
                severity = 3 if (flags & XCCL_HINT_F_SEVERE) else (2 if (flags & XCCL_HINT_F_CAUTION) else 0)
                publish_control_frame(
                    ctrl_mm,
                    args.control_comm_key,
                    host_cause_score,
                    net_cause_score,
                    applied_mode,
                    severity,
                    target_channels,
                    target_window,
                    pacing_ns,
                    cooldown_left,
                    mode_stable_epochs,
                )
                ctrl_mm.flush()

        if cooldown_left > 0:
            cooldown_left -= 1

        elapsed = time.monotonic() - t_now
        sleep_s = max(0.0, interval_s - elapsed)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
