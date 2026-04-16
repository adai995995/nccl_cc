#!/usr/bin/env python3
"""
从多个 OUTDIR 的 timeline_stats.txt 抽取阶段 1/2 关心的指标，汇总为 CSV（stdout）。

解析字段（来自 analyze_timeline.py 输出）：
  - v2 NCCL：CLEAN-1 / STRESS / CLEAN-2 的 SHRINK%
  - Recovery / Window restore / Control reaction（ms）
  - v2-minimal 与 Baseline（若有）各阶段 latency：mean、p99、max

用法:
  python3 summarize_phase12_runs.py <outdir1> [outdir2 ...] > summary.csv
  python3 summarize_phase12_runs.py --json summary.json <outdir1> ...

若某目录缺少 timeline_stats.txt，跳过并打印警告到 stderr。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


def _find_timeline_stats(outdir: str) -> Optional[str]:
    p = os.path.join(outdir, "timeline_stats.txt")
    if os.path.isfile(p):
        return p
    return None


def _parse_run_meta(outdir: str) -> Dict[str, str]:
    meta_path = os.path.join(outdir, "run_meta.txt")
    out: Dict[str, str] = {}
    if not os.path.isfile(meta_path):
        return out
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("git_rev="):
                out["git_rev"] = line.split("=", 1)[1].strip()
            elif line.startswith("ITERS="):
                out["ITERS"] = line.split("=", 1)[1].strip()
            elif line.startswith("PHASE1_END="):
                out["PHASE1_END"] = line.split("=", 1)[1].strip()
            elif line.startswith("PHASE2_END="):
                out["PHASE2_END"] = line.split("=", 1)[1].strip()
    return out


def _parse_phase_latency_line(line: str) -> Optional[Dict[str, float]]:
    # "  CLEAN-1: n=347  mean=16779.8  p50=18497.0  p99=39661.3  max=40035.8  stddev=8083.7"
    m = re.search(
        r"n=(\d+)\s+mean=([\d.]+)\s+p50=([\d.]+)\s+p99=([\d.]+)\s+max=([\d.]+)",
        line,
    )
    if not m:
        return None
    return {
        "n": int(m.group(1)),
        "mean": float(m.group(2)),
        "p50": float(m.group(3)),
        "p99": float(m.group(4)),
        "max": float(m.group(5)),
    }


def _parse_nccl_shrink_line(line: str) -> Optional[float]:
    m = re.search(r"SHRINK=([\d.]+)%", line)
    if not m:
        return None
    return float(m.group(1))


def parse_timeline_stats_text(content: str) -> Dict[str, Any]:
    """Parse analyze_timeline.py stats output into flat dict."""
    d: Dict[str, Any] = {}
    lines = content.splitlines()
    section: Optional[str] = None

    for line in lines:
        if not line.strip():
            if section in ("v2_lat", "base_lat"):
                section = None
            continue

        s = line.strip()
        if s.startswith("v2-minimal latency by phase"):
            section = "v2_lat"
            continue
        if s.startswith("Baseline (no CC) latency by phase"):
            section = "base_lat"
            continue
        if s.startswith("NCCL controller by phase"):
            section = "nccl"
            continue
        if s.startswith("Control reaction latency"):
            m = re.search(r"Control reaction latency:\s*([\d.]+)ms", line)
            if m:
                d["reaction_ms"] = float(m.group(1))
            continue
        if s.startswith("Recovery latency"):
            m = re.search(r"Recovery latency:\s*([\d.]+)ms", line)
            if m:
                d["recovery_ms"] = float(m.group(1))
            continue
        if s.startswith("Window restore latency"):
            m = re.search(r"Window restore latency:\s*([\d.]+)ms", line)
            if m:
                d["window_restore_ms"] = float(m.group(1))
            continue

        if section == "v2_lat" and "mean=" in line and "CLEAN-1" in line:
            pl = _parse_phase_latency_line(line)
            if pl:
                for k, v in pl.items():
                    d[f"v2_lat_clean1_{k}"] = v
        elif section == "v2_lat" and "mean=" in line and "STRESS" in line:
            pl = _parse_phase_latency_line(line)
            if pl:
                for k, v in pl.items():
                    d[f"v2_lat_stress_{k}"] = v
        elif section == "v2_lat" and "mean=" in line and "CLEAN-2" in line:
            pl = _parse_phase_latency_line(line)
            if pl:
                for k, v in pl.items():
                    d[f"v2_lat_clean2_{k}"] = v
        elif section == "base_lat" and "mean=" in line and "CLEAN-1" in line:
            pl = _parse_phase_latency_line(line)
            if pl:
                for k, v in pl.items():
                    d[f"base_lat_clean1_{k}"] = v
        elif section == "base_lat" and "mean=" in line and "STRESS" in line:
            pl = _parse_phase_latency_line(line)
            if pl:
                for k, v in pl.items():
                    d[f"base_lat_stress_{k}"] = v
        elif section == "base_lat" and "mean=" in line and "CLEAN-2" in line:
            pl = _parse_phase_latency_line(line)
            if pl:
                for k, v in pl.items():
                    d[f"base_lat_clean2_{k}"] = v
        elif section == "nccl" and "SHRINK=" in line and "CLEAN-1" in line:
            sh = _parse_nccl_shrink_line(line)
            if sh is not None:
                d["v2_shrink_clean1_pct"] = sh
        elif section == "nccl" and "SHRINK=" in line and "STRESS" in line:
            sh = _parse_nccl_shrink_line(line)
            if sh is not None:
                d["v2_shrink_stress_pct"] = sh
        elif section == "nccl" and "SHRINK=" in line and "CLEAN-2" in line:
            sh = _parse_nccl_shrink_line(line)
            if sh is not None:
                d["v2_shrink_clean2_pct"] = sh

    return d


def parse_one_outdir(outdir: str) -> Dict[str, Any]:
    outdir = os.path.abspath(outdir)
    row: Dict[str, Any] = {
        "outdir": outdir,
        "outdir_basename": os.path.basename(outdir),
    }
    row.update(_parse_run_meta(outdir))
    stats_path = _find_timeline_stats(outdir)
    if not stats_path:
        row["error"] = "missing timeline_stats.txt"
        return row
    with open(stats_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    parsed = parse_timeline_stats_text(content)
    row.update(parsed)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdirs", nargs="*", help="包含 timeline_stats.txt 的目录（可多个）")
    ap.add_argument(
        "--json",
        metavar="FILE",
        help="同时写入 JSON 数组到 FILE",
    )
    args = ap.parse_args()
    if not args.outdirs:
        ap.print_help()
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for od in args.outdirs:
        if not os.path.isdir(od):
            print(f"WARN: not a directory, skip: {od}", file=sys.stderr)
            continue
        r = parse_one_outdir(od)
        if r.get("error"):
            print(f"WARN: {r['error']}: {od}", file=sys.stderr)
        rows.append(r)

    if not rows:
        print("ERROR: no rows to output", file=sys.stderr)
        sys.exit(1)

    # Union of all keys for CSV header
    all_keys: List[str] = []
    seen = set()
    priority = [
        "outdir_basename",
        "outdir",
        "git_rev",
        "ITERS",
        "PHASE1_END",
        "PHASE2_END",
        "v2_shrink_clean1_pct",
        "v2_shrink_stress_pct",
        "v2_shrink_clean2_pct",
        "reaction_ms",
        "recovery_ms",
        "window_restore_ms",
    ]
    for k in priority:
        if any(k in row for row in rows):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    for row in rows:
        for k in sorted(row.keys()):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    w = csv.DictWriter(sys.stdout, fieldnames=all_keys, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in all_keys})

    if args.json:
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(rows, jf, indent=2)


if __name__ == "__main__":
    main()
