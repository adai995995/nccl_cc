# 阶段 0：工程可重复性与数据契约

## 1. 目标

在扩展任何科学结论之前，保证**每一次实验 run 的输入、输出、进度同步方式**可审计、可复现。特别解决多机 MPI 场景下「等不到 CSV 行数、events 不完整」等工程问题，避免后续阶段建立在损坏数据上。

## 2. 前置条件

- 单机可成功跑通 `run_timeline_experiment.sh`，`events.csv` 含 `stress_start` 与 `stress_stop`。
- `oracle_bench` / `oracle_bench_mpi` 与 NCCL 构建路径一致，`LD_LIBRARY_PATH` 正确。
- 多机时：各节点 SSH 互通、`stress-ng` 安装在 `STRESS_NODE`、`HOSTS` 与 `NCCL_SOCKET_IFNAME` 与实际网络一致。

## 3. 数据契约（一次 run 的必交物）

每次实验应在独立子目录（建议带时间戳）中至少包含：

| 产物 | 说明 |
|------|------|
| `run_meta.txt` | 时间、`HOSTS`/`STRESS_NODE`（若 MPI）、主要 `NCCL_*`、`ITERS`/`WARMUP`/`COUNT`、`PHASE1_END`/`PHASE2_END`、stress 完整命令行、NCCL 构建路径或 git commit |
| `*_latency_*.csv` | bench 逐 iter（或约定格式）输出 |
| `*_events_*.csv` | 至少两列 `ts_us,event`，且含成对 `stress_start` / `stress_stop`（及未来 net 等事件） |
| `*.log` | `tee` 保存的完整 stdout/stderr |
| v2-minimal 时 | `nccl_timeline*.csv`（单机一份；MPI 可为每 rank 一份，**分析时注明选用的 rank**） |

**CSV 行数约定**：与脚本一致——header 1 行 + 每个 post-warmup iter 一行；`wait_for_csv_lines` 的目标行号与 `PHASE1_END`/`PHASE2_END` 对齐（见 `run_timeline_experiment.sh` 注释）。

## 4. 自变量 / 因变量 / 控制变量

- **本阶段自变量**：部署方式（仅本机 vs MPI）、轮询 CSV 的实现（本机路径 vs 远程 rank0 路径）。
- **因变量**：run 是否完整结束、events 是否成对、日志是否无致命错误。
- **控制变量**：同一组 `ITERS`、`PHASE*`、`COUNT`、同一 NCCL 二进制。

## 5. 操作步骤

1. 定义 `OUTDIR` 命名规范，例如 `timeline_results_<tag>_<YYYYMMDD_HHMMSS>`。
2. 单机执行阶段 0 验收：连续 **3** 次完整 run，检查产物清单与 `analyze_timeline.py` 可执行性。
3. MPI：确认 bench CSV 实际写入位置；**进度等待**必须与「能观察到文件增长」的路径一致（例如对 rank0 节点 `ssh` 执行 `wc -l`，或 scp/rsync 到本机再轮询——与团队约定一种并写进 `run_meta.txt`）。
4. 每次 run 后运行自检脚本或人工检查：`grep stress events.csv`、`wc -l` latency CSV 与 `ITERS` 关系。

## 6. 退出标准（阶段完成判据）

- 单机与 MPI（若使用）各 **≥3** 次连续成功，events 均含 start/stop。
- `run_meta.txt` 模板固定，字段无遗漏。
- 文档中记录「MPI 进度轮询」的最终实现方式，供阶段 1–2 引用。

## 7. 风险与注意事项

- 远端路径在本机 `wc -l` 会导致永久阻塞或过早注入压力，数据不可信。
- `stress-ng --timeout` 应长于 bench 在 stress 段可能停留的时间，或脚本在 `stress_stop` 主动 kill，避免双轨竞态。

## 8. 仓库对应关系

- `../run_timeline_experiment.sh`：单机轮询与事件写入参考实现。
- `../run_timeline_experiment_mpi.sh`：多机部署与 collect；**阶段 0 应校验其等待逻辑与 CSV 位置一致**。
