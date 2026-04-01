"""
从 netdev sysfs、ethtool -S、/proc/pressure/memory 采集原始计数（周期增量在调用方计算）。
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional


def _read_int(path: str) -> int:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def read_iface_speed_mbits(path_iface: str) -> int:
    """/sys/class/net/<iface>/speed，单位 Mb/s；失败返回 0。"""
    return _read_int(os.path.join(path_iface, "speed"))


def read_netdev_statistics(path_iface: str) -> Dict[str, int]:
    """读取 statistics/ 下常用计数。"""
    stat = os.path.join(path_iface, "statistics")
    keys = (
        "rx_bytes",
        "tx_bytes",
        "rx_packets",
        "tx_packets",
        "rx_errors",
        "tx_errors",
        "rx_dropped",
        "tx_dropped",
    )
    out: Dict[str, int] = {}
    for k in keys:
        out[k] = _read_int(os.path.join(stat, k))
    return out


def parse_ethtool_S(iface: str) -> Dict[str, int]:
    """
    解析 `ethtool -S <iface>` 输出为 {name: value}。
    仅解析形如 '     rx_ecn_mark: 123' 的行。
    """
    try:
        p = subprocess.run(
            ["ethtool", "-S", iface],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if p.returncode != 0:
        return {}
    out: Dict[str, int] = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        # NIC statistics: 或无前缀
        m = re.match(r"^([a-zA-Z0-9_]+):\s*(\d+)\s*$", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


@dataclass
class PsiMemory:
    some_avg10: Optional[float]  # 0~1 若可解析


def read_psi_memory_some_avg10() -> Optional[float]:
    """
    解析 /proc/pressure/memory 中 some avg10=xx.xx。
    返回 0~1（百分比/100）；文件不存在返回 None。
    """
    path = "/proc/pressure/memory"
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            line = f.readline()
    except OSError:
        return None
    # some avg10=0.00 avg60=...
    m = re.search(r"some\s+avg10=([\d.]+)", line)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    # avg10 为百分比，如 0.05 表示 0.05%
    return min(1.0, v / 100.0) if v <= 100.0 else min(1.0, v)


def path_for_iface(name: str) -> str:
    return os.path.join("/sys/class/net", name)


# mlx5 RoCE：优先用 *_rdma_* 列；部分驱动/场景下流量记在 vport unicast（*_rdma_* 恒为 0）
_MLX5_RDMA_ONLY_KEYS = (
    "rx_vport_rdma_ucast_bytes",
    "tx_vport_rdma_ucast_bytes",
    "rx_vport_rdma_unicast_bytes",
    "tx_vport_rdma_unicast_bytes",
    "rx_vport_rdma_mcast_bytes",
    "tx_vport_rdma_mcast_bytes",
    "rx_vport_rdma_multicast_bytes",
    "tx_vport_rdma_multicast_bytes",
)


def mlx5_rdma_bytes_total(eth: Dict[str, int]) -> int:
    """从 ethtool -S 汇总「可用于带宽」的字节计数（单调递增，用于周期增量）。"""
    total = sum(int(eth.get(k, 0)) for k in _MLX5_RDMA_ONLY_KEYS)
    if total > 0:
        return total
    # 回退：与 sysfs rx_bytes/tx_bytes 通常同量级，RoCE 也常体现在此
    return int(eth.get("rx_vport_unicast_bytes", 0)) + int(eth.get("tx_vport_unicast_bytes", 0))
