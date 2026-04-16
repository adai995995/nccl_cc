#!/usr/bin/env python3
"""
Phase V3 时间线分析脚本

将 NCCL v2-minimal 控制日志 + benchmark latency CSV + stress 事件
对齐到统一时间轴，绘制多面板时间线图。

用法:
  python3 analyze_timeline.py <nccl_timeline.csv> <bench_latency.csv> <events.csv> \
                               [baseline_latency.csv] [baseline_events.csv]

输出:
  - timeline_plot.png : 多面板时间线图
  - timeline_stats.txt: 三阶段统计对比
"""
import sys
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Will output stats only.")


def read_csv(path):
    """Simple CSV reader returning dict of column_name -> list of values."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return {}
    header = lines[0].split(",")
    data = {h: [] for h in header}
    for line in lines[1:]:
        parts = line.split(",")
        for i, h in enumerate(header):
            if i < len(parts):
                try:
                    data[h].append(float(parts[i]))
                except ValueError:
                    data[h].append(parts[i])
    return data


def read_events(path):
    """Read events CSV -> list of (ts_us, event_name)."""
    events = []
    if not os.path.exists(path):
        return events
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ts_us"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                events.append((float(parts[0]), parts[1]))
    return events


def phase_stats(latencies, label):
    """Compute statistics for a list of latencies."""
    if not latencies:
        return f"  {label}: (no data)"
    a = np.array(latencies)
    return (f"  {label}: n={len(a)}  mean={a.mean():.1f}  p50={np.median(a):.1f}  "
            f"p99={np.percentile(a,99):.1f}  max={a.max():.1f}  stddev={a.std():.1f}")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    nccl_csv_path = sys.argv[1]
    bench_csv_path = sys.argv[2]
    events_path = sys.argv[3]
    baseline_csv_path = sys.argv[4] if len(sys.argv) > 4 else None
    baseline_events_path = sys.argv[5] if len(sys.argv) > 5 else None

    nccl = read_csv(nccl_csv_path)
    bench = read_csv(bench_csv_path)
    events = read_events(events_path)

    baseline = read_csv(baseline_csv_path) if baseline_csv_path else None
    baseline_events = read_events(baseline_events_path) if baseline_events_path else []

    if not nccl.get("ts_us"):
        print("ERROR: NCCL timeline CSV is empty or missing ts_us column")
        sys.exit(1)
    if not bench.get("ts_us"):
        print("ERROR: Bench CSV missing ts_us column")
        sys.exit(1)

    nccl_ts = np.array(nccl["ts_us"])
    nccl_pressure = np.array(nccl["pressure"])
    nccl_backlog = np.array(nccl["backlog_norm"])
    nccl_pollgap = np.array(nccl.get("deviation", nccl.get("backlog_raw", nccl.get("loop_delay_norm", nccl.get("poll_gap_norm", [])))))
    nccl_window = np.array(nccl["window"])
    nccl_states = nccl["state"]

    bench_ts = np.array(bench["ts_us"])
    bench_wall = np.array(bench["wall_us"])
    bench_skew = np.array(bench["skew_us"]) if "skew_us" in bench else None

    t0 = min(nccl_ts[0], bench_ts[0])
    nccl_t_ms = (nccl_ts - t0) / 1000.0
    bench_t_ms = (bench_ts - t0) / 1000.0

    if baseline and "ts_us" in baseline:
        base_ts = np.array(baseline["ts_us"])
        base_wall = np.array(baseline["wall_us"])
        base_t0 = base_ts[0]
        base_t_ms = (base_ts - base_t0) / 1000.0
    else:
        base_t_ms = None
        base_wall = None

    stress_intervals = []
    ev_start = None
    for ts, ev in events:
        t_ms = (ts - t0) / 1000.0
        if ev == "stress_start":
            ev_start = t_ms
        elif ev == "stress_stop" and ev_start is not None:
            stress_intervals.append((ev_start, t_ms))
            ev_start = None

    base_stress_intervals = []
    if baseline_events:
        base_ev_start = None
        for ts, ev in baseline_events:
            if base_ts is not None and len(base_ts) > 0:
                t_ms = (ts - base_t0) / 1000.0
            else:
                t_ms = ts / 1000.0
            if ev == "stress_start":
                base_ev_start = t_ms
            elif ev == "stress_stop" and base_ev_start is not None:
                base_stress_intervals.append((base_ev_start, t_ms))
                base_ev_start = None

    out_dir = os.path.dirname(nccl_csv_path) or "."
    stats_path = os.path.join(out_dir, "timeline_stats.txt")

    with open(stats_path, "w") as sf:
        def p(msg):
            print(msg)
            sf.write(msg + "\n")

        p("=" * 60)
        p("Phase V3 Timeline Statistics")
        p("=" * 60)
        p(f"NCCL timeline records: {len(nccl_ts)}")
        p(f"Bench iterations:      {len(bench_ts)}")
        p(f"Duration:              {(nccl_ts[-1]-nccl_ts[0])/1e6:.1f}s")
        p("")

        if stress_intervals:
            s_start, s_end = stress_intervals[0]
            clean1_mask = bench_t_ms < s_start
            stress_mask = (bench_t_ms >= s_start) & (bench_t_ms <= s_end)
            clean2_mask = bench_t_ms > s_end

            p("v2-minimal latency by phase (us):")
            p(phase_stats(bench_wall[clean1_mask].tolist(), "CLEAN-1"))
            p(phase_stats(bench_wall[stress_mask].tolist(), "STRESS "))
            p(phase_stats(bench_wall[clean2_mask].tolist(), "CLEAN-2"))
            p("")

            if base_t_ms is not None and base_wall is not None and base_stress_intervals:
                bs0, bs1 = base_stress_intervals[0]
                bc1 = base_t_ms < bs0
                bst = (base_t_ms >= bs0) & (base_t_ms <= bs1)
                bc2 = base_t_ms > bs1
                p("Baseline (no CC) latency by phase (us):")
                p(phase_stats(base_wall[bc1].tolist(), "CLEAN-1"))
                p(phase_stats(base_wall[bst].tolist(), "STRESS "))
                p(phase_stats(base_wall[bc2].tolist(), "CLEAN-2"))
                p("")

            nccl_clean1 = nccl_t_ms < s_start
            nccl_stress = (nccl_t_ms >= s_start) & (nccl_t_ms <= s_end)
            nccl_clean2 = nccl_t_ms > s_end

            p("NCCL controller by phase:")
            for label, mask in [("CLEAN-1", nccl_clean1), ("STRESS", nccl_stress), ("CLEAN-2", nccl_clean2)]:
                if mask.sum() == 0:
                    p(f"  {label}: (no data)")
                    continue
                p_vals = nccl_pressure[mask]
                w_vals = nccl_window[mask]
                states_in = [nccl_states[i] for i in range(len(mask)) if mask[i]]
                shrink_pct = sum(1 for s in states_in if s == "SHRINK") / len(states_in) * 100
                p(f"  {label}: pressure mean={p_vals.mean():.3f} max={p_vals.max():.3f}  "
                  f"window mean={w_vals.mean():.0f} min={w_vals.min():.0f}  "
                  f"SHRINK={shrink_pct:.1f}%")

            control_react_ms = None
            if s_start is not None:
                first_shrink_idx = None
                for i in range(len(nccl_t_ms)):
                    if nccl_t_ms[i] >= s_start and nccl_states[i] == "SHRINK":
                        first_shrink_idx = i
                        break
                if first_shrink_idx is not None:
                    control_react_ms = nccl_t_ms[first_shrink_idx] - s_start
                    p(f"\nControl reaction latency: {control_react_ms:.1f}ms "
                      f"(from stress_start to first SHRINK)")
            p("")

            if s_end is not None:
                first_recover_idx = None
                for i in range(len(nccl_t_ms)):
                    if nccl_t_ms[i] >= s_end and nccl_states[i] == "RECOVER":
                        first_recover_idx = i
                        break
                if first_recover_idx is not None:
                    recover_ms = nccl_t_ms[first_recover_idx] - s_end
                    p(f"Recovery latency: {recover_ms:.1f}ms "
                      f"(from stress_stop to first RECOVER)")

                window_full_idx = None
                max_w = nccl_window.max()
                for i in range(len(nccl_t_ms)):
                    if nccl_t_ms[i] >= s_end and nccl_window[i] >= max_w * 0.95:
                        window_full_idx = i
                        break
                if window_full_idx is not None:
                    restore_ms = nccl_t_ms[window_full_idx] - s_end
                    p(f"Window restore latency: {restore_ms:.1f}ms "
                      f"(from stress_stop to window >= 95% max)")

        p("\n" + "=" * 60)

    print(f"\nStats written to: {stats_path}")

    if not HAS_MPL:
        print("Skipping plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    for ax in axes:
        for s_start, s_end in stress_intervals:
            ax.axvspan(s_start, s_end, alpha=0.15, color="red", zorder=0)

    # Panel 1: Latency (v2-minimal + baseline)
    ax = axes[0]
    ax.plot(bench_t_ms, bench_wall / 1000.0, "b-", linewidth=0.5, alpha=0.7, label="v2-minimal")
    if base_t_ms is not None and base_wall is not None:
        ax.plot(base_t_ms, base_wall / 1000.0, "gray", linewidth=0.5, alpha=0.5, label="baseline")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Phase V3: Timeline Analysis", fontsize=12, fontweight="bold")

    # Panel 2: Pressure + components
    ax = axes[1]
    ax.plot(nccl_t_ms, nccl_pressure, "r-", linewidth=0.6, label="pressure")
    ax.plot(nccl_t_ms, nccl_backlog, "orange", linewidth=0.4, alpha=0.7, label="backlog_norm")
    ax.plot(nccl_t_ms, nccl_pollgap, "purple", linewidth=0.4, alpha=0.7, label="deviation")
    ax.axhline(y=float(nccl.get("pressure", [0.4])[0]) if False else 0.40,
               color="red", linestyle="--", alpha=0.5, label="pressure_thresh")
    ax.axhline(y=0.15, color="green", linestyle="--", alpha=0.5, label="exit_thresh")
    ax.set_ylabel("Pressure")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    # Panel 3: Window
    ax = axes[2]
    ax.plot(nccl_t_ms, nccl_window, "g-", linewidth=0.8)
    ax.set_ylabel("Window")
    max_w = nccl_window.max()
    floor_vals = nccl.get("floor", None)
    if floor_vals:
        ax.axhline(y=float(floor_vals[0]), color="orange", linestyle="--",
                   alpha=0.5, label=f"floor={int(float(floor_vals[0]))}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, max_w * 1.1)

    # Panel 4: Controller state
    ax = axes[3]
    state_map = {"HOLD": 0, "SHRINK": -1, "RECOVER": 1}
    state_nums = np.array([state_map.get(s, 0) for s in nccl_states])
    colors = np.array(["green" if s == "RECOVER" else "red" if s == "SHRINK" else "gray"
                        for s in nccl_states])
    ax.scatter(nccl_t_ms, state_nums, c=colors, s=1, alpha=0.6)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["SHRINK", "HOLD", "RECOVER"])
    ax.set_ylabel("State")

    # Panel 5: Skew (if available)
    ax = axes[4]
    if bench_skew is not None:
        ax.plot(bench_t_ms, bench_skew / 1000.0, "m-", linewidth=0.5, alpha=0.7, label="v2-minimal")
        ax.set_ylabel("Rank Skew (ms)")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.set_visible(False)

    axes[-1].set_xlabel("Time (ms from start)")

    for s_start, s_end in stress_intervals:
        axes[0].annotate("STRESS START", xy=(s_start, axes[0].get_ylim()[1]),
                         fontsize=7, color="red", ha="left", va="top")
        axes[0].annotate("STRESS STOP", xy=(s_end, axes[0].get_ylim()[1]),
                         fontsize=7, color="green", ha="left", va="top")

    plot_path = os.path.join(out_dir, "timeline_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
