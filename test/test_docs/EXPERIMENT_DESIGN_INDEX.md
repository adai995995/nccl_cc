# NCCL CC 分阶段实验设计文档索引

本目录除 `test1.md` / `test2.md` / `test3.md` 讨论材料外，新增**可执行级**阶段设计说明。建议按编号顺序推进；后一阶段以前一阶段的「退出标准」为前提。

| 阶段 | 文档 | 一句话目标 |
|------|------|------------|
| 0 | [exp_phase00_engineering_contract.md](./exp_phase00_engineering_contract.md) | 固定一次 run 的数据契约与可重复性（含 MPI） |
| 1 | [exp_phase01_timeline_v3.md](./exp_phase01_timeline_v3.md) | 三阶段压力 + 时间线对齐 + 统计与作图 |
| 2 | [exp_phase02_detector_eval.md](./exp_phase02_detector_eval.md) | 检测与真值对齐：误触、反应/恢复、震荡 |
| 3 | [exp_phase03_oracle_ceiling.md](./exp_phase03_oracle_ceiling.md) | Oracle 上限：host / net 场景下干预是否有效 |
| 4 | [exp_phase04_actuator_ablation.md](./exp_phase04_actuator_ablation.md) | window / pacing / channel 等执行器消融 |
| 5 | [exp_phase05_workload_topology.md](./exp_phase05_workload_topology.md) | 算子、偏斜 rank、多机等泛化 |
| 6 | [exp_phase06_multidim_control.md](./exp_phase06_multidim_control.md) | 在 window 站稳后引入多维组合策略 |

**相关脚本与工具（仓库内）**

- 单机三阶段：`../run_timeline_experiment.sh`
- MPI 三阶段：`../run_timeline_experiment_mpi.sh`
- 时间线分析：`../analyze_timeline.py`
- 阶段 1+2 批量重复与汇总：`../run_phase12_sweep.sh`、`../summarize_phase12_runs.py`（从多目录 `timeline_stats.txt` 抽 SHRINK%、Recovery 等列成 CSV）

---

文档版本：与仓库目录 `test_docs` 同步维护；修改实验契约时请同步更新阶段 0 文档。
