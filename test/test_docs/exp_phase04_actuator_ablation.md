# 阶段 4：Actuator 消融（执行器因果）

## 1. 目标

在 **同一 failure mode 与同一压力时序** 下，分离 **window / pacing / channel（或其它执行器）** 对 tail 与吞吐的边际贡献，避免「三把锤子」叠加导致不可解释的负收益。

## 2. 前置条件

- 阶段 3 在至少一种场景下已证明 **Oracle（window 类）有正收益**，或已明确仅某类执行器有效。
- 代码或构建支持**单独启用**各类 actuator（若仅编译期开关，则每个变体单独构建并在 `run_meta.txt` 记录 artifact）。

## 3. 实验矩阵（最小）

| Run ID | window | pacing | channel | 压力类型 |
|--------|--------|--------|---------|----------|
| W | 仅开 | 关 | 关 | host |
| P | 关 | 仅开 | 关 | host |
| C | 关 | 关 | 仅开 | host |

对 **net** 场景（若阶段 3 已建立）重复上述矩阵或按假设缩减为「net 下仅 P + W」等，避免无意义组合。

## 4. 自变量 / 因变量 / 控制变量

- **自变量**：启用的 actuator 集合、（可选）detector 版本——**建议本阶段固定 detector**，只改 actuator，便于归因。
- **因变量**：分阶段 p99、max、skew、mean、带宽；可选 pipeline 相关指标（若 bench 输出）。
- **控制变量**：`ITERS`、`PHASE*`、注入命令、NCCL 拓扑相关 env。

## 5. 操作步骤

1. 为每个 Run ID 建立独立 `OUTDIR` 与 `run_meta.txt`，明确二进制或 `NCCL_*` 组合。
2. 使用与阶段 1 相同的三阶段脚本骨架，仅替换「控制策略」相关 env/二进制。
3. 每个单元格至少 **3** 次重复；记录失败 run（如 hang）原因。
4. 汇总表：行 = Run ID，列 = 各阶段 p99/max/skew 均值±方差。

## 6. 产物清单

- 每 run：latency、events、log；若该变体输出 timeline 则一并保存。
- `ablation_summary.csv`：聚合指标。

## 7. 退出标准

- 完成 host 矩阵至少 W/P/C 三行均有数据。
- 能回答：**谁带来 tail 收益、谁主要带来负收益或震荡**；结论写入一页摘要，供架构裁剪（如 window-only 优先）使用。

## 8. 风险与注意事项

- channel 类改动易破坏 NCCL 原生 pipeline，负收益为常见结果，需在文档中如实记录。
- pacing 过粗可能导致吞吐坍塌，应与阶段 6 的「渐进引入」一致，本阶段只做**单因子**边界。

## 9. 与阶段 2、3 的边界

- 不在本阶段同时大改 detector；若需「Oracle 触发 pacing-only」，归入阶段 3 并标注为 hybrid oracle。

## 10. 仓库对应关系

- 以 `run_timeline_experiment.sh` 为模板，通过不同 `make` 目标或 `NCCL_BUILD` 指向不同构建输出；或增加 `run_ablation_matrix.sh` 调用子脚本。
