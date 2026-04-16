# 阶段 5：Workload 与拓扑泛化

## 1. 目标

证明阶段 1–4 的结论**不局限于**单一 micro-bench（如固定 allreduce + 单机 8GPU）或单一压力节点。在**控制变量尽量一致**的前提下，扩展算子、参与 rank 偏斜、多机拓扑与多 job 干扰。

## 2. 前置条件

- 阶段 1–3 在默认场景下已有可引用结论；阶段 4 可选（若已确定 window-only 为主，本阶段仍以 window 为主角扩展）。

## 3. 泛化维度

| 维度 | 示例 | 说明 |
|------|------|------|
| 集体算子 | allreduce → alltoall、reduce_scatter 等 | bench 或 harness 需支持稳定计时与 CSV 输出 |
| Rank 偏斜 | 压力仅打在 rank `r` 所在节点 | MPI 下与 `STRESS_NODE`、rank–host 映射文档化 |
| 拓扑 | 2×8、4×8、非对称 GPU 数 | 记录 `HOSTS` 与实际 NVLink/PCIe 拓扑（若与结论相关） |
| 多 job | 同机第二组进程占 CPU/网卡 | 难控，建议最后做；每次记录干扰进程命令 |

## 4. 实验设计原则

- **每次只改一个泛化维度**，其余与「默认成功配置」对齐。
- 每个新场景仍走 **L1（现象）→ L2（Oracle，若适用）→ L3（v2-minimal）** 的缩短流程；若 L1 无 tail 放大，则记录「该场景无控制收益空间」而非强行调参。

## 5. 自变量 / 因变量 / 控制变量

- **自变量**：算子类型、压力节点/rank、拓扑、是否存在第二 job。
- **因变量**：与阶段 1 相同的主指标（分阶段 p99/max/skew）；可增加「受影响 rank 比例」等叙述性指标（若 bench 提供 per-rank）。
- **控制变量**：消息规模、`ITERS`、压力强度分级（建议低/中/高三档以免单次扫参过大）。

## 6. 操作步骤

1. 建立「场景卡片」：`scenario_id`、算子、拓扑、压力位置、bench 命令行。
2. 每场景至少：baseline + v2-minimal 各 **≥3** run；关键场景补 Oracle。
3. 所有结果路径写入 `scenario_registry.csv`（场景 ID → OUTDIR 列表）。

## 7. 产物清单

- 每场景每 run：阶段 0 契约中的全套文件 + `scenario_id` 在 `run_meta.txt` 首行。
- 可选：与阶段 2 相同的 detector 指标，用于说明「泛化后 detector 是否退化」。

## 8. 退出标准

- 至少 **2** 个非默认算子或 **1** 种偏斜 rank 配置完成并有汇总表。
- 明确写出：**哪些场景收益与默认一致、哪些场景无现象、哪些场景负收益**及初步原因（日志与 timeline）。

## 9. 风险与注意事项

- alltoall 等对启动顺序敏感，MPI 参数错误会导致「假负收益」。
- 多 job 实验难以复现，宜报告区间或多次 seed，避免单点 hero number。

## 10. 仓库对应关系

- `../run_timeline_experiment_mpi.sh`：`HOSTS`、`STRESS_NODE` 与 rank 映射。
- `oracle_bench` / `oracle_bench_mpi`：若需新算子，在源码与 Makefile 中扩展并在此文档「场景卡片」中链接 commit。
