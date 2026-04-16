当前实验结果与综合结论（可直接用于汇报/论文“实验结果”部分）
1) 实验体系与场景（你已经把“可控 failure mode”稳定复现出来了）
你构造并稳定复现了一个“持续型局部 host pressure”场景：

单机 8GPU
强制走网络（禁用 P2P/SHM，避免 NVLink/IPC 掩盖问题）
taskset 绑核 + stress-ng 同核竞争，制造 CQ draining / host 侧进度受阻 的持续压力
用 oracle_bench 逐 iter 统计：wall latency、max GPU latency、rank skew、带宽
Phase V3 进一步把 NCCL 控制平面状态（pressure/window/state）与 bench latency 做时间戳对齐并可视化
这一步本身是重要成果：你现在有一套可重复、可量化、可对齐时间线的实验方法论。

2) Phase V1（Oracle 上限）：证明“缩窗能治 tail 放大”
在上述 host pressure 场景下，数据清晰显示：

压力注入会显著恶化 tail 与 skew：stress-baseline 相比 no-stress，mean/p99/max 上升、rank skew 大幅膨胀、带宽下降。
人工 oracle 缩窗能显著改善 tail：把 window 固定砍到 1/2 或 1/3，p99/max/skew 明显下降，带宽也能回升。
典型证据（test/oracle_results/summary_20260406_155742.txt）：

stress-baseline：mean 37.7ms，p99 74.4ms，bw 0.19 GB/s
oracle-third：p99 56.0ms，max 70.3ms，bw 0.35 GB/s
结论 A（强证据）：在持续 host pressure 下，原生 NCCL 的持续注入确实存在“全局 tail 放大”；并且 window containment 是能直接打到放大机制的“最小有效干预”。

3) Phase V2（v2-minimal 闭环）：证明“不用 sidecar，仅 window-only 也能出显著正收益”
你实现的 v2-minimal 满足“收缩式设计”目标：NCCL 内部自闭环、只动 window、不动 channels/pacing、无 sidecar，并且你修复了“多控制器抢写 window”的关键 bug（v2-minimal 独占 window 控制权）。

证据（test/oracle_results/summary_20260406_182843.txt）：

stress-baseline：mean 31.4ms，p99 63.9ms，bw 0.23 GB/s
v2-minimal+stress：mean 13.9ms，p99 51.8ms，bw 0.53 GB/s
结论 B（强证据）：在该 host pressure failure mode 下，“把闭环放进 NCCL 内部 + window-only containment”不仅可行，而且能取得非常强的正收益（接近/部分超过 oracle 的表现），证明你的核心机制路线成立。

4) Phase V3（时间线分析）：证明“问题主要在 detector，而不是执行链路”
你做了 Phase V3 的完整工具链：

NCCL 每控制周期输出 timeline CSV（pressure/state/window/计数器等）
bench 每 iter 输出带 CLOCK_MONOTONIC 时间戳的 latency CSV
stress_start/stop 事件写入 events CSV
自动产出 timeline_plot.png + timeline_stats.txt（reaction / recovery / restore latency）
并通过多轮迭代定位到：

早期版本的误触主要来自 信号选择与采样/线程问题（包括曾经的“多控制线程同时写 timeline”与 sleep 周期异常）。
修复后控制平面确实能稳定以 ~1ms 运行，窗口执行链路可控、可观测。
不同版本里出现的“CLEAN 段 SHRINK、恢复慢、窗口震荡”等，根因集中在 压力 detector 的鲁棒性（基线漂移、冷启动、门控与冻结策略）。
结论 C（已证明、但属于工程结论）：你已经把不确定性从“控制系统整体太复杂”收缩到了一个清晰可迭代的问题：detector 的稳定性（误触率、恢复时延、震荡），并且具备足够的时间线证据来驱动下一步改进。

综合结论（最重要的 3 句话）
你已经证明了一个真实且可控的 failure mode：持续 host pressure 会导致原生 NCCL 的全局 tail/skew 放大。
你已经证明了最小有效干预的因果有效性：window containment 能显著缓解 tail 放大（oracle 上限成立），并且 v2-minimal（NCCL 内闭环、window-only）能在该场景获得显著正收益。
你当前的主要未完成项不是“再加 signals/再加 sidecar/再加三杆”，而是把 detector 做到跨 run 稳定（低误触、少震荡、恢复快）；Phase V3 已提供可观测与量化手段来完成这一步。
