# 阶段 1：V3 时间线与三阶段统计

## 1. 目标

在**固定三阶段时序**（CLEAN1 → STRESS → CLEAN2）下，将 **NCCL 控制平面时间线**与 **bench 延迟曲线**、**压力注入事件**对齐，形成可发表的「现象—反应—恢复」证据链。本阶段不改 detector 算法时，也可作为版本回归基线。

## 2. 前置条件

- 阶段 0 退出标准已满足。
- 已安装 `analyze_timeline.py` 所需依赖（至少 numpy；作图需 matplotlib）。

## 3. 实验设计

### 3.1 三阶段定义

| 阶段 | iter 范围（约定） | 说明 |
|------|-------------------|------|
| CLEAN1 | `1 … PHASE1_END` | 无压力或仅环境基线 |
| STRESS | `PHASE1_END+1 … PHASE2_END` | 注入 host（或后续 net）压力 |
| CLEAN2 | `PHASE2_END+1 … ITERS` | 撤压后的恢复段 |

`PHASE1_END`、`PHASE2_END` 在整组对比实验中保持不变。

### 3.2 对照组（最小集）

| 标签 | 说明 |
|------|------|
| A Baseline | `NCCL_AIMD_ENABLE=0`，无 v2-minimal，无 timeline 文件（或按现有脚本） |
| B v2-minimal | `NCCL_AIMD_ENABLE=1`，`NCCL_CC_V2_MINIMAL=1`，开启 timeline 输出 |

单机用 `run_timeline_experiment.sh`；多机用 `run_timeline_experiment_mpi.sh`，并固定分析 rank（建议 rank0，与 bench CSV 来源一致）。

### 3.3 指标（因变量）

优先报告**分阶段**统计（由 `analyze_timeline.py` 的 `timeline_stats.txt` 与图支撑）：

- p99、max、mean、std（若脚本对 skew 有列则一并记录）
- 可视化：backlog / window / mode / latency 等与时间的对齐关系

**主指标**：STRESS 段相对 CLEAN1 的 tail 恶化；CLEAN2 相对 STRESS 的恢复程度。

## 4. 操作步骤

1. 设置 `OUTDIR`、`ITERS`、`WARMUP`、`COUNT`、`PHASE1_END`、`PHASE2_END`，写入 `run_meta.txt`。
2. 执行 A，再执行 B（或脚本内顺序执行，与现有一致）。
3. 对 B 的 timeline + bench + events 运行：  
   `python3 analyze_timeline.py <nccl_timeline.csv> <bench_latency.csv> <events.csv> [baseline_latency.csv] [baseline_events.csv]`  
   MPI 多文件时先选定单个 `nccl_timeline_*_rank0.csv`（或协议约定 rank）。
4. 将生成的 `timeline_plot.png`、`timeline_stats.txt` 归档到同一 run 目录。

## 5. 产物清单

- Baseline：`baseline_latency_*.csv`、`baseline_*.log`、`baseline_events_*.csv`
- v2-minimal：`bench_latency_*.csv`、`bench_*.log`、`events_*.csv`、`nccl_timeline_*.csv`
- 分析输出：`timeline_plot.png`、`timeline_stats.txt`

## 6. 退出标准

- 同一配置下 A/B 至少各 1 次完整 run；若用于论文主图，建议 B **≥3** 次。
- 图中可清晰区分三阶段；stats 中三段均有有效样本数。
- 文档化所用 `analyze_timeline.py` 命令行与 rank 选择规则。

## 7. 风险与注意事项

- 时间戳基准：bench 与 NCCL timeline 须同源时钟语义（均为 monotonic 或均为 wall，与实现一致）；若混用需在 meta 中说明。
- CLEAN 段若出现持续 SHRINK，应记录为阶段 2（detector）问题而非在本阶段强行解释收益。

## 8. 仓库对应关系

- `../run_timeline_experiment.sh`、`../run_timeline_experiment_mpi.sh`
- `../analyze_timeline.py`
