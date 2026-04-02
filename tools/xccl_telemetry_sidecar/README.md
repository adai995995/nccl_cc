# XCCL Host Telemetry Sidecar（MVP）

对应设计：`design_docs/xccl_host_telemetry_sidecar_design.md`

## 依赖

- Python 3.8+
- `ethtool`（解析 `rx_ecn_mark` 等，可选）
- 读 `/sys/class/net/<iface>/`、`/proc/pressure/memory` 的权限（一般非 root 即可）

## 快速开始

```bash
cd /export/xxl/nccl_cc/tools/xccl_telemetry_sidecar

# 先建映射文件（与 NCCL 一致）
export XCCL_TELEMETRY_FILE=/export/xxl/nccl_cc/test/th_data/xccl_telemetry_hint.bin
export NCCL_CC_HINT_ENABLE=1
export NCCL_CC_HINT_MMAP_PATH=$XCCL_TELEMETRY_FILE
export NCCL_AIMD_ENABLE=1

# 只观察数值，不写文件
python3 xccl_telemetry_sidecar.py --ifaces eth0 --dry-run

# 后台写快照（按 ibdev2netdev 选一个实际用于训练的口）
python3 xccl_telemetry_sidecar.py --ifaces eth0 --file "$XCCL_TELEMETRY_FILE" &
```

另开终端跑 NCCL 测试程序。

## 环境变量

| 变量 | 含义 |
|------|------|
| `XCCL_TELEMETRY_FILE` | mmap 文件路径（与 `NCCL_CC_HINT_MMAP_PATH` 一致） |
| `XCCL_TELEMETRY_IFACES` | 逗号分隔 netdev，默认 `eth0` |
| `XCCL_TELEMETRY_INTERVAL_MS` | 周期，默认 `200` |
| `XCCL_ECN_DELTA_THRESH` | `rx_ecn_mark` 周期增量归一化分母，默认 `1000` |
| `XCCL_CE_ERR_DELTA_THRESH` | 错误类增量归一化分母，默认 `100` |
| `XCCL_DROP_RETRY_DELTA_THRESH` | ethtool drop/retry 增量归一化分母，默认 `100` |
| `XCCL_MODE_MARGIN` | `HOST/NET` 判因滞回边界，默认 `0.10` |
| `XCCL_HOST_SIGNAL_FILE` | NCCL host signal 快照路径（默认 `/dev/shm/xccl_host_signal_snapshot`） |
| `XCCL_CQ_BACKLOG_THRESH` | `cq_backlog` 归一化分母，默认 `128` |
| `XCCL_COMPLETION_STRETCH_THRESH` | `(stretch-1)` 归一化分母，默认 `1.5` |
| `XCCL_CONTROL_SNAPSHOT_FILE` | 控制快照输出路径（默认 `/dev/shm/xccl_control_snapshot`） |
| `XCCL_CONTROL_COMM_KEY` | 控制快照 `comm_key`，默认 `0`（wildcard） |
| `XCCL_CONTROL_BASE_WINDOW` | 控制策略基础窗口，默认 `256` |
| `XCCL_CONTROL_MIN_WINDOW` | 控制策略最小窗口，默认 `16` |
| `XCCL_CONTROL_MAX_WINDOW` | 控制策略最大窗口，默认 `512` |
| `XCCL_CONTROL_HOST_MAX_CHANNELS` | HOST 模式最大 channel cap，默认 `4` |
| `XCCL_CONTROL_HOST_MIN_CHANNELS` | HOST 模式最小 channel cap，默认 `1` |
| `XCCL_CONTROL_MAX_PACING_NS` | NET 模式最大 pacing，默认 `50000` |
| `XCCL_CONTROL_SWITCH_STABLE_EPOCHS` | 切换模式所需连续稳定周期，默认 `3` |
| `XCCL_CONTROL_COOLDOWN_EPOCHS` | 模式切换后冷却周期，默认 `5` |

## 当前实现（S0+S1 子集）

- **`rnic_pressure`**：`Σ(rx_bytes+tx_bytes)` 增量速率 / 口速（多口取 max）
- **`cnp_level`**：`rx_ecn_mark` 增量 / `XCCL_ECN_DELTA_THRESH`（多口取 max）
- **`ce_level`**：errors/dropped 增量之和 / `XCCL_CE_ERR_DELTA_THRESH`
- **`drop_retry_norm`（S2）**：ethtool `drop/retry/retrans/timeout` 候选计数增量归一化
- **`host_psi_mix`（S2）**：`/proc/pressure/{cpu,memory,io}` 的 `some avg10` 平均
- **`host_score`（S2）**：融合 `cq_backlog/completion_stretch/cpu_poll_delay/host_psi_mix`
- **`net_score`（S2）**：融合 `cnp/rtt_stretch/drop_retry/rnic`
- **`mode`（S2）**：按 `host_score/net_score` + `margin` 判为 `HOST/NET/MIXED`
- **`target_window/target_channels/pacing_ns`（S3）**：按 mode 映射并写入 `XcclControlSnapshot`
- **`cooldown_level/stable_epochs`（S3）**：模式切换防抖状态
- **`flags`**：`SEVERE` / `CAUTION`（见 `--severe` / `--caution`）

## 与测试脚本关系

- `test/write_telemetry_hint.py`：无硬件依赖，用于 CI/手测写帧协议。
- 本目录：**真实 counter → 归一化 → 同一写帧协议**。

## 注意

- 首周期无上一采样点，部分 level 可能为 0；属正常现象。
- 多口训练时请将 `--ifaces` 与 **`NCCL_IB_HCA` 对应的 netdev** 对齐（见设计附录 A）。
