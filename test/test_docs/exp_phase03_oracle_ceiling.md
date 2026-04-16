# 阶段 3：Oracle 上限与因果对照

## 1. 目标

对每个拟研究的 **failure mode**，先测定**理论上界**：在「已知压力类型与时段」的前提下，**理想干预**能否显著降低 collective tail（p99/max/skew）而不无谓损伤 clean 段。若 Oracle 无收益，则不应再投入 full controller 调参。

本阶段涵盖文档中的 **Oracle A（host）** 与 **Oracle B（net）** 思想；实现可分时推进。

## 2. 前置条件

- 阶段 0–1 满足；阶段 2 可选（Oracle 实验可与简单 detector 并行，但报告时需说明是否人工给定干预时段）。

## 3. Failure mode 与 Oracle 定义

### 3.1 Host 持续压力（已有主线）

- **注入**：与现脚本一致，`taskset` + `stress-ng` 与 bench 争用 CPU（单机或 `STRESS_NODE` 远程）。
- **Oracle 行为示例**：在 STRESS 段将 `target_window` 固定为满窗的 1/2 或 1/3（或等价 env/补丁），**仅在该段**启用；CLEAN 段与原生一致。
- **对照组**：原生 NCCL；Oracle；v2-minimal（闭环）。

### 3.2 Net 持续压力（Oracle B）

- **注入**（择一或可组合，需可持续数十秒以上）：  
  - `tc netem` 延迟/抖动/限速；或  
  - 跨节点背景流量（如 `iperf`）占满瓶颈链路；或  
  - 可控 switch 队列（若环境具备）。
- **Oracle**：在已知为 net 拥塞的时段，优先尝试 **window 收缩**；若假设为 in-flight 过深，可尝试 **pacing**（与阶段 4 消融配合时需标注）。
- **注意**：尖峰仅持续单个 collective 的场景与本阶段目标不符（控制链来不及），应在 `run_meta.txt` 说明注入时间尺度。

## 4. 自变量 / 因变量 / 控制变量

- **自变量**：failure mode（host / net）、Oracle 强度（如 1/2 窗、1/3 窗）、是否仅 STRESS 段启用。
- **因变量**：分阶段 p99、max、skew、带宽；可选「相对 baseline stress 段的改善百分比」。
- **控制变量**：`ITERS`、`PHASE*`、`COUNT`、拓扑与 `NCCL_SOCKET_IFNAME`；net 实验时记录 `tc` 或背景流命令。

## 5. 操作步骤（每个 failure mode）

1. **L1 现象**：仅 baseline，确认 STRESS 段相对 CLEAN1 tail 显著恶化。
2. **L2 Oracle**：同一时序下启用 Oracle，记录 tail 是否改善。
3. **L3 闭环**：同配置跑 v2-minimal（或当前最优闭环），与 Oracle 对比差距。
4. 全部结果写入同一 `OUTDIR`，附 `run_meta.txt`。

## 6. 产物清单

- 与阶段 1 相同类型的 latency、events、log、timeline（若 Oracle 路径仍写 timeline）。
- Oracle 专用说明：`oracle_spec.txt`（人工规则：哪段 iter、哪类压力、动作参数）。

## 7. 退出标准

- Host：Oracle 在 STRESS 段相对 baseline 的 **p99 或 max 至少一项**有明确改善趋势（建议预先定义阈值，如 ≥10% 或统计显著，与团队一致即可）。
- Net：至少一种可持续注入下完成「baseline vs Oracle」各 ≥3 次；若 Oracle 始终无收益，**记录负结果**并考虑更换瓶颈场景或指标。
- 论文级叙述：**Oracle 与 v2-minimal 的差距**可解释为「检测/执行延迟」或「动作粒度不足」。

## 8. 风险与注意事项

- Oracle 与 v2-minimal 混用同一 env 时，务必避免并行写同一 timeline 文件。
- Net 注入可能影响 SSH；建议带外管理或本地控制台，或将 phase 时长与注入脚本容错设计好。

## 9. 仓库对应关系

- Host 三阶段：`../run_timeline_experiment.sh`、`../run_timeline_experiment_mpi.sh`（需扩展 Oracle 分支时可复制为 `run_oracle_experiment*.sh` 或通过 env 切换）。
- 分析：`../analyze_timeline.py`；汇总可用自写脚本对比多 `timeline_stats.txt`。
