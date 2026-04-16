# 阶段 2：Detector 检测评估（与执行解耦）

## 1. 目标

在**不争论「控制是否有收益」**的前提下，量化评估：**内部检测状态**与**外部可知的真值压力阶段**是否一致。输出误触率、反应延迟、恢复延迟、震荡等指标，为改进 detector 提供可优化目标函数。

## 2. 前置条件

- 阶段 0–1 已完成；`nccl_timeline_*.csv` 列定义稳定（至少包含可用于判断「是否进入收缩/高压状态」的列，如 state / pressure / window 等，以实际 CSV 表头为准）。
- 三阶段边界 `PHASE1_END`/`PHASE2_END` 与 `events.csv` 中 `stress_start`/`stress_stop` 时间戳可用。

## 3. 真值（Ground Truth）定义

| 真值标签 | 定义 |
|-----------|------|
| `clean` | 不在脚本声明的压力窗口内（含 CLEAN1 与 CLEAN2） |
| `host_stress` | `stress_start` 与 `stress_stop` 之间，且压力类型为 host（stress-ng 等） |
| `net_stress` | （预留）net 注入 start/stop 之间 |

真值由 **实验脚本写入的 events** + **iter 与时间的映射** 联合得到；若 timeline 为控制周期采样而非 per-iter，需定义对齐规则（例如：每个周期映射到最近 bench iter 或最近 wall 时间）。

## 4. 检测输出定义

从 `nccl_timeline_*.csv` 提取每个采样点的：

- 用于判定「是否认为存在持续压力」的量（如综合 score、某 pressure 阈值、state 枚举）。
- 可选：是否发生 window 收缩、mode 切换。

**检测判决规则**需在实验前写死，例如：

- `predict_stress = 1` 当且仅当 `state == SHRINK`（示例，以实际实现替换）。
- 或使用 `pressure > T_high` 持续 `K` 个连续采样点视为阳性。

## 5. 因变量（核心指标）

| 指标 | 含义 | 建议计算方式 |
|------|------|----------------|
| 误触率（FPR 类） | clean 真值下被判为压力 | clean 段采样中 `predict_stress=1` 的比例或时间占比 |
| 漏检率（FNR 类） | host_stress 真值下长期未判压力 | stress 段内 `predict_stress=0` 的时长占比 |
| 反应延迟 | 压力已开始但检测尚未稳定阳性 | `stress_start` 到「连续 K 点 predict=1」的首时刻之差 |
| 恢复延迟 | 压力已结束但检测仍维持高压态 | `stress_stop` 到「连续 K 点 predict=0」的首时刻之差 |
| 震荡 | 压力段内检测翻转 | stress 段内 `predict` 0→1 与 1→0 切换次数，或 window 翻转次数 |

`K` 为超参数（如 3～10 个控制周期），在 `run_meta.txt` 中记录。

## 6. 自变量与控制变量

- **自变量**：detector 版本（不同 commit/分支/阈值）、`K`、`T_high`。
- **控制变量**：与阶段 1 相同的 `ITERS`、`PHASE*`、压力注入方式、算子与规模。

## 7. 操作步骤

1. 固定一套 host 三阶段实验（阶段 1 的 B 配置）。
2. 为每个 detector 变体运行 **N≥5** 次（不同时间戳目录），每次保留完整 timeline + events + bench。
3. 用脚本（建议小 Python）解析 timeline 与 events，按上表输出每个 run 的五个量。
4. 汇总：均值、标准差，或箱线图；保留 1～2 个典型 run 的 `timeline_plot.png` 作为附录。

## 8. 产物清单

- 每 run：阶段 1 所列文件 + `detector_metrics.json`（或 `.txt`），含上述指标与参数 `K`、阈值。
- 汇总：`detector_summary.csv`（每行一个 run）。

## 9. 退出标准

- 至少一个「基线 detector」在 clean 段误触率有数值且可复现；改进版本有前后对比表。
- 文档中固定「真值—预测」对齐规则，避免口头争议。

## 10. 与阶段 3 的边界

本阶段**不**以吞吐或 tail 改善为成功标准；若需评估「固定 oracle 检测下 actuator 收益」，属于阶段 3 的子实验，应单独标注。

## 11. 仓库对应关系

- 三阶段实验与时间线：`../run_timeline_experiment.sh`、`../analyze_timeline.py`（**五参数**：含 `baseline_latency` 与 `baseline_events` 时，会在 `timeline_stats.txt` 中追加 **Baseline 分阶段延迟**，便于与 v2-minimal 对照）。
- **重复 N 次 + 汇总表**：`../run_phase12_sweep.sh`（`N`、`PREFIX`、`SUMMARY_OUT` 等环境变量见脚本头注释）；仅对已有目录汇总：`ONLY_SUMMARIZE=1 PREFIX=... ./run_phase12_sweep.sh`。
- **从多目录抽取指标列**：`../summarize_phase12_runs.py`，输出 CSV（含 `v2_shrink_clean1_pct`、`v2_shrink_clean2_pct`、`recovery_ms`、各阶段 mean/p99/max 等）。
