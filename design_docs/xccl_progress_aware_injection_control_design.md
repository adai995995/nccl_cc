# XCCL：Progress-Aware Library-Level 注入控制（实现导向总设计）

> 目标读者：在 NCCL `net_ib` 路径上做实现/调试的人。  
> 目标：把“设计意图”落到“可实现、可验证、可逐步迭代”的工程规范（结构体、状态机、不变量、挂载点、回滚对称性、并发约束、调试指标）。

---

## 1. 背景与问题定义

原生 NCCL 在 RDMA 路径上主要依赖以下机制维持通信推进：

- RC / RNR 等硬件可靠传输机制
- 有限的 CQ 轮询，防止 SQ 被塞满
- collective 算法与 pipeline 自带的隐式流控
- 协议/算法选择带来的吞吐-时延折中

这些机制能够保证“能跑”和“基础性能”，但不提供一个**显式的 library-level 注入控制器**来根据运行时拥塞症状动态调节发送强度。真实 AI 集群中，性能恶化可能来自网络拥塞、PCIe/host 背压、RNIC 压力、多 QP completion 拖尾、pipeline 推进效率下降、在途工作集过大等综合因素。

因此，XCCL 的工程目标是：在 NCCL library 层增加一个**comm-local、progress-aware**的注入整形机制，显式控制 in-flight chunk 数，并基于 completion/progress/queue pressure（+ 可选 telemetry hint）动态调节注入窗口。

---

## 2. 设计目标与原则

### 2.1 总体目标

实现一个**comm-local、progress-aware、可选接入外部 telemetry hints** 的 NCCL library-level 注入控制框架，具备：

1. 显式控制每个“发送侧上下文（send-side communication context）”的 in-flight chunk 数（library-level injection window）
2. 将多个 WR completion 聚合为 chunk 级症状信号（avg/max/span/service_time）
3. 基于 progress efficiency、queue pressure、completion 指标进行 epoch 决策
4. 支持在 collective 调用前接收 telemetry hint（低频、非 fast path）
5. 支持长 collective 运行中低频 epoch 更新
6. 与原生 NCCL request/completion 语义兼容（`wc->wr_id` 需恢复）
7. 正确处理多 QP、多 CQE、slot 复用、rollback、stale CQE 等边界

### 2.2 设计原则

- **comm-local 主闭环**：主控制器必须是 `CollectiveCC`，不依赖其它 comm 的细粒度进度
- **低侵入**：主要修改 `net_ib` 发送/完成路径 + 辅助模块
- **correctness 优先**：先保证 inflight/finalize/shadow slot/rollback 正确，再升级控制器
- **fast path 轻量化**：fast path 不做重型采样；控制逻辑按 epoch 低频更新
- **telemetry 是 hint**：外部 telemetry 仅做 gating/bias，不替代本地症状闭环
- **弱 DeviceCC（可选）**：仅 coarse cap/emergency brake，不做跨-comm 强调度

---

## 3. 当前实现基线与挂载点（对齐代码）

> 本文档的设计将尽量对齐现有实现（便于增量演进），关键挂载点如下。

### 3.1 初始化/清理

- `nccl/src/transport/net_ib/init.cc`
  - `ncclIbInit()` 调用 `ncclIbAIMDInit()`
  - `ncclIbFinalize()` 调用 `ncclIbAIMDFinalize()`

### 3.2 发送路径

- `nccl/src/transport/net_ib/p2p.cc`
  - `ncclIbIsend()` 内部最终调用 `ncclIbMultiSend()`
  - 在 `ncclIbMultiSend()` 中已存在：
    - `CollectiveCC` 获取/创建（按“当前实现的 key：`comm`/`ncclIbSendComm*`”区分）
    - 但文档与接口层面建议将 CollectiveCC 的硬粒度进一步收敛为“单发送侧上下文”
    - 窗口检查 `inflight_chunks` vs `lib_window`
    - `chunk_id_slot` 分配（`next_chunk_id % max_chunks`）
    - `ncclIbRecordChunkSend(...)` 初始化 chunk 状态与 `chunk_cqes_pending`
    - per-QP shadow slot 分配 + `lastWr->wr_id` 编码（每 QP 一个 signaled CQE）
    - post 失败时释放 shadow、修正 `chunk_cqes_pending` / `inflight`（局部逻辑）

### 3.3 Completion 路径

- `nccl/src/transport/net_ib/p2p.cc`
  - `ncclIbCompletionEventProcess()` 入口处先调用 `ncclIbOnCompletionWithCC(commBase, wc, devIndex)`
  - `ncclIbOnCompletionWithCC()` 位于 `nccl/src/transport/net_ib/aimd_cc.cc`
    - 识别/解码编码 `wr_id`
    - 查 shadow slot，恢复 `wc->wr_id`（保持 NCCL 原语义）
    - 更新 chunk RTT/统计，`chunk_cqes_pending--`
    - `prev==1` 触发 `FinalizeChunkRTT()` 并在 finalize 内 `inflight_chunks--`

---

## 4. 术语与粒度

- **CollectiveCC**：comm-local 拥塞控制对象，系统主控制器
- **CollectiveCC 硬粒度（强定义）**：当前版本的 engineering key 绑定粒度应为“单发送侧上下文（send-side communication context）”，建议等价于 `ncclIbSendComm*`（或等价的单 peer/单发送连接上下文）。不建议把它绑定到整个 communicator 级别的所有发送活动。

> 说明：当前代码可能用 `comm` 或 `ncclIbSendComm*` 作为 key 做简化；但为了后续 correctness 与 telemetry 归属清晰，本设计文档规定 CollectiveCC 的归属对象必须是“发送侧上下文”，并且所有 inflight/finalize/统计窗口都以该归属为边界。
- **Chunk**：注入控制与统计的基本粒度（由若干 WR 组成；多 QP 时每 QP 有一个 signaled CQE）
- **ChunkTracker**：按 chunk 聚合 completion 的观测层（pending CQE、统计、finalize）
- **Shadow Pool**：wr_id 兼容层，保存原始 `wr_id` 与 XCCL 元数据并在 completion 时恢复
- **Completion latency**：`post_send -> CQE` 的完成时延（包含 host/RNIC/队列效应，不等同于纯 wire RTT）
- **Progress efficiency**：单位时间内 finalized chunks 的推进速度
- **Queue pressure**：`inflight_chunks / effective_window`
- **Telemetry hint**：sidecar/低频通道提供的粗粒度 hint（CNP/CE/PCIe stall/RNIC proxy）
- **DeviceCC（可选）**：device-level coarse guardrail，只输出 cap

---

## 5. 必须满足的不变量（Correctness Contract）

### 5.1 inflight 不变量（chunk 粒度）

- **I1**：每个成功 post（会产生至少一个 CQE）的 chunk，最终必须 **exactly once finalize**，且 finalize 必须执行 `inflight_chunks--`。
- **I2**：每个“窗口预留成功但最终没有任何 QP 成功 post”的 chunk，必须 rollback，并执行 `inflight_chunks--`（因为不会有 CQE 回来触发 finalize）。
- **I3**：`inflight_chunks` 不能因为 early return、`original_wr_id==0`、shadow 竞态等原因泄漏（泄漏会导致窗口永久满、系统卡死）。
- **I4（强身份）**：chunk 的真实身份 = `(cc_idx, chunk_slot, chunk_generation)`。completion 处理必须校验 generation 不一致则视为 stale CQE/延迟 CQE，禁止触发 `pending_cqes--` 与 finalize。

### 5.2 chunk tracker 不变量

- **C1**：`pending_cqes` 初值为 0；每个 signaled WR 所属 QP 的 `post` 成功后递增一次（即 `pending_cqes == posted_qps`）。
- **C2**：`pending_cqes` 只能单调递减到 0；出现负数意味着 stale CQE/重复减、或初始化/补偿不对。
- **C3**：finalize 必须 CAS 保证 **exactly once**。

### 5.3 shadow slot 不变量

- **S1**：slot 复用必须具备 **generation/version** 机制；否则 stale/late CQE 可能命中新分配的 slot，破坏 `pending_cqes`/统计甚至错误还原 `wr_id`。
- **S2**：发送失败 rollback 必须释放所有已分配 slot（防泄漏）。
- **S3**：completion 必须在恢复 `wc->wr_id` 后才交还给 NCCL 原生逻辑。

---

## 6. 数据结构（实现建议）

> 本节给出推荐的“实现版本结构体”。可以在不破坏现有函数入口的前提下，逐步扩展 `aimd_cc.h` 中现有结构体字段。

### 6.1 WR_ID 编码（沿用现状）

当前实现（`aimd_cc.h`）使用 64 位 `wr_id` 编码，包含：

- `[63:48]` Magic（识别受控 WR）
- `[47:40]` QP index（0..255）
- `[39:30]` CC index（0..1023）
- `[29:14]` Shadow slot（0..65535）
- `[13:0]` Chunk tag（建议至少包含 `chunk_generation tag`，用于 stale CQE 校验；当前实现可能先把它当作 `chunk_slot` 简化）

> **正确性硬要求**：仅依赖 `chunk_slot`（不携带 generation tag）会在 `chunk_slot` 复用、CQE 延迟/乱序时破坏 stale 判定。工程上需要确保 `wr_id` 的低位 tag 能使 completion 判断“该 CQE 属于哪个 chunk_generation”。
>
> **编码语义补充（必须）**：
> - `chunk_generation_tag` 可以是完整 generation，也可以是截断低位（例如 `chunk_generation & ((1<<k)-1)`），具体取决于位宽预算。
> - `chunk_generation_tag` 在 `wr_id` 中**仅用于 fast-path 预过滤**，不能作为最终强校验依据。
> - 即使 `chunk_generation_tag` 因截断发生碰撞，最终 stale 判定仍必须依赖：
>   1) `ShadowSlot.chunk_generation` 与 tag 的一致性检查（快速）
>   2) `ChunkSlot.chunk_generation` 与 `ShadowSlot.chunk_generation` 的强一致性检查（最终）
> - 只有两层都通过，才允许执行 `pending_cqes--`。

### 6.2 Shadow Pool（强建议 Phase 1 落地）

```c
struct ShadowSlot {
  std::atomic<uint32_t> active;  // 0/1
  uint32_t generation;           // 每次分配递增

  uint64_t original_wr_id;
  uint32_t cc_idx;
  uint32_t chunk_slot;          // chunk_slot index（复用粒度）
  uint32_t chunk_generation;    // 与 ChunkSlot.chunk_generation 强校验（防 stale 命中复用槽）
  uint8_t  qp_idx;

  uint64_t send_ts_ns;
};
```

实现要点：

- 分配仅对会产生 CQE 的 WR（通常是 signaled `lastWr`）执行。
- 分配时在锁内/或 CAS 找到 inactive slot，`generation++`，写字段，最后 `active=1`（release）。
- completion 时先 `active.load(acquire)`，inactive 则 stale 直接返回；active 才读取字段并释放 slot。
- stale CQE 的安全策略：**不做 `pending_cqes--`、不 finalize、可计数/限频日志**。

### 6.3 ChunkTracker（建议：slot + generation 模型）

```c
enum ChunkStatus : uint32_t {
  CHUNK_UNUSED = 0,
  CHUNK_POSTED = 1,
  CHUNK_FINALIZED = 2,
  CHUNK_FAILED = 3,
};

struct ChunkSlot {
  uint32_t chunk_generation;   // chunk_generation：每次复用 chunk_slot 时递增
  std::atomic<uint32_t> status;

  // 工程协议：pending_cqes 只在“某个 signaled WR 所属 QP 的 post 成功”后递增
  // 不建议采用“先设为 nqps 再补偿”的模式，避免部分成功/提前 return 路径导致统计漂移
  std::atomic<uint32_t> pending_cqes;  // 当前该 chunk 已成功 post 的 signaled CQE 数
  std::atomic<uint32_t> finalized_flag; // 0->1 exactly once

  uint64_t send_ts_ns;
  std::atomic<uint64_t> first_cqe_ts_ns;
  std::atomic<uint64_t> last_cqe_ts_ns;

  std::atomic<uint64_t> sum_completion_ns;
  std::atomic<uint64_t> max_completion_ns;
  std::atomic<uint64_t> min_completion_ns;
  std::atomic<uint32_t> samples;

  std::atomic<uint64_t> representative_original_wr_id;
};

struct ChunkTracker {
  int max_chunks;
  std::atomic<uint64_t> next_chunk_seq;   // 单调递增，用于分配
  ChunkSlot* slots;
};
```

finalize 产出（用于控制器输入）：

- `avg_completion_ns = sum/samples`
- `max_completion_ns`
- `span_ns = last_cqe_ts - first_cqe_ts`
- `service_time_ns = last_cqe_ts - send_ts_ns`

字段语义说明：

- `representative_original_wr_id` 仅用于调试/日志关联，不参与 correctness 判定或控制决策。

### 6.4 CollectiveCC（progress-aware + epoch）

```c
enum CCState : uint8_t { CC_NORMAL=0, CC_CAUTION=1, CC_SEVERE=2 };

struct TelemetryHintSnapshot {
  uint64_t version;
  uint64_t ts_ns;
  float cnp_level, ce_level, pcie_stall_level, rnic_pressure;
  uint32_t flags;
};

struct CollectiveCC {
  uintptr_t comm_key;
  uint64_t collective_id; // 日志用途
  int enabled;

  // window
  std::atomic<uint32_t> inflight_chunks;
  uint32_t local_window;
  uint32_t effective_window;
  uint32_t min_window, max_window;
  uint32_t device_cap;   // 可选，默认=max_window
  CCState state;

  // progress counters
  std::atomic<uint64_t> posted_chunks_total;
  std::atomic<uint64_t> finalized_chunks_total;

  // epoch
  uint64_t epoch_interval_ns;  // 1ms~5ms
  uint64_t epoch_last_ts_ns;
  uint64_t epoch_last_finalized;
  uint32_t stable_epochs;
  uint32_t cooldown_epochs;
  // epoch due：由 completion/finalize 轻量置位；真正的 ccEpochUpdate 在 send/epoch 边界执行
  std::atomic<uint32_t> epoch_due;
  uint64_t epoch_due_ts_ns;

  // metrics (EWMA)
  double ewma_avg_completion_ns;
  double ewma_tail_completion_ns; // first version 可用 ewma(max) 近似
  double ewma_span_ns;
  double progress_efficiency;     // finalized/sec
  double queue_pressure;          // inflight/effective_window

  // parameters
  double alpha_slow;
  double beta_fast;

  // telemetry (optional)
  TelemetryHintSnapshot hint;
  uint64_t hint_ttl_ns;        // hint 有效期：now - hint.ts_ns <= hint_ttl_ns 才算有效

  // tracker
  ChunkTracker tracker;

  pthread_mutex_t mutex; // 低频 epoch 更新用
};
```

---

## 7. 发送路径（实现细化）

### 7.1 推荐的 send path 分解（便于验证与 rollback 对称）

在 `net_ib/p2p.cc:ncclIbMultiSend()` 中，建议将 XCCL 逻辑抽象为如下步骤（可逐步重构为函数）：

1. `cc = GetOrCreateCollectiveCC(comm, collective_id)`
2. `effective_window = min(local_window, device_cap)`（由 epoch 更新维护，send fast path 仅读）
3. **窗口预留**：若 `inflight >= effective_window`，返回 retry（例如 `ncclInProgress`）
4. **分配 chunk_slot**：`chunk_slot = next_chunk_seq % max_chunks`
5. **初始化 ChunkSlot**：
   - `status=POSTED, pending_cqes=0, finalized_flag=0`
   - `send_ts_ns=now`, 清空统计
   - 设置 `chunk_generation`（例如：`chunk_generation = next_chunk_seq / max_chunks` 或独立递增计数）
   - 定义 `chunk_generation_tag = chunk_generation`（用于编码/解码 stale CQE 校验）
6. 对每个 QP：
   - 为该 QP 的 signaled `lastWr` 分配 shadow slot（保存 `original_wr_id`、`chunk_slot`、`chunk_generation`、`send_ts_ns`）
   - 编码 `lastWr->wr_id = ENCODE(qp_idx, cc_idx, chunk_generation_tag, shadow_slot)`（低位 tag 用于 completion stale 校验）
   - `ibv_post_send`（对该 QP 的 WR 链）
   - 若该 QP 的 `wrap_ibv_post_send` 返回成功，则：
     - `pending_cqes++`（该 QP 的 signaled CQE 将来会触发 pending 递减）
     - `posted_qps++`
7. post 失败：
   - 若某 QP 的 post 失败：释放该 QP 已分配的 shadow slot；**不要**对 pending_cqes 做“减法补偿”（避免统计漂移）
   - chunk 级回滚条件：如果所有 QP post 后 `posted_qps==0`，则该 chunk 不会产生受控 CQE，必须回滚 `inflight--` 并标记 chunk failed/unset

inflight 所有权模型（强定义）：

- `inflight_chunks++` 发生在“chunk 通过窗口检查并完成窗口预留”时（credit 预占），而不是每个 QP post 成功时。
- 若最终 `posted_qps==0`，必须 rollback `inflight_chunks--`（因为无受控 CQE 返回）。
- 若 `posted_qps>0`，该 credit 必须由 finalize 路径归还（`inflight_chunks--`）。
- 该模型与 library-level injection window 语义一致：窗口约束的是“被放行的 chunk credit”。

### 7.2 关键：post 部分成功的语义

多 QP 场景下，可能出现：

- QP0 post 成功（未来会有 1 个 CQE）
- QP1..QP(n-1) post 失败（不会有 CQE）

此时必须保证：

- `pending_cqes` 最终等于“实际会回来的 CQE 数”
- finalize 在最后一个 CQE 到来时仍能触发 exactly once，并执行 `inflight--`

工程建议：

- chunk 使用 `planned_qps`/`posted_qps` 两个概念：
  - `planned_qps = nqps`（该 chunk 计划覆盖的 QP 个数）
  - `posted_qps` 仅在每个 QP 的 signaled WR post 成功后递增
  - `pending_cqes == posted_qps`（强工程协议），只有当 `pending_cqes` 递减到 0 才触发 finalize

### 7.3 chunk_slot 复用前置条件（必须）

分配 `chunk_slot` 时，目标 slot 必须满足：

- `status` 属于 `CHUNK_UNUSED` 或 `CHUNK_FINALIZED` 或 `CHUNK_FAILED`

若目标 slot 仍是活跃态（例如 `CHUNK_POSTED`）：

- 视为 tracker 资源耗尽或逻辑异常
- 不允许覆盖复用
- 返回 retry（推荐）或计数并在 debug 模式 assert

说明：

- `generation` 机制用于识别 stale/延迟 CQE；
- 但 `generation` 不能替代“禁止覆盖活跃 slot”这个前置约束。

---

## 8. Completion 路径（实现细化）

### 8.1 completion 处理流程（仅针对受控 WR）

对每个 CQE：

1. `IS_MY_WR(wc->wr_id)` 判断是否受控
2. 解码得到 `qp_idx, cc_idx, chunk_generation_tag, shadow_slot`
3. 查 ShadowSlot：
   - inactive：判定 stale，直接 return（不更新 pending、不 finalize）
   - active：读取 `original_wr_id, send_ts_ns, chunk_slot, chunk_generation`（可交叉校验），释放 slot
4. chunk generation 校验（强制）：
   - 若 `shadow.chunk_generation != chunk_generation_tag` 则判定 stale，禁止 `pending_cqes--` 与 finalize（仅计数/限频日志）
   - 定位 `ChunkSlot(cc_idx, chunk_slot)`
   - 若 `ChunkSlot.chunk_generation != shadow.chunk_generation` 则判定内部状态不一致/乱序，按 stale 丢弃（仅计数/限频日志）
5. `wc->wr_id = original_wr_id`（恢复 NCCL 原生语义）
6. 更新 ChunkSlot 统计：
   - `completion_ns = now - send_ts_ns`
   - 更新 `first_cqe_ts_ns / last_cqe_ts_ns`
   - `sum/max/min/samples`
7. `prev = pending_cqes.fetch_sub(1)`
   - `prev <= 0`：计数异常（重复减/未 stale），只计数/日志，禁止 finalize
   - `prev == 1`：触发 finalize
8. finalize exactly once：
   - `if (finalized_flag.exchange(1)==0)` 执行 finalize
   - 生成 `avg/max/span/service_time`
   - 更新 `CollectiveCC` 的轻量统计样本（用于后续 epoch 更新）
   - `inflight_chunks--`（系统推进点，必须执行）
   - `finalized_chunks_total++`
9. 返回给 NCCL 原生 completion 逻辑继续处理（请求 events 递减等）

### 8.2 Epoch 触发（低频）

v1 推荐模型：completion/finalize fast path 只做轻量判定，不执行重的 `ccEpochUpdate()`。

completion/finalize 中建议仅做：

- 置 `cc->epoch_due=1`（或写 `cc->epoch_due_ts_ns=now`），例如满足：
  - `now - epoch_last_ts_ns >= epoch_interval_ns` 或 `finalized_delta >= N`
- 记录 epoch bucket 的原始采样到轻量字段（不加锁或只做轻量写，避免 CQE 热点）

真正的 `ccEpochUpdate()` 由如下路径执行（避免 CQ fast path 负担）：

- send path / collective 边界：
  - 在 `ncclIbMultiSend()` 开头做 `ccEpochUpdateIfDue(cc, now)`（或在 `sendProxyProgress` 入口做一次）
  - 由同一个 proxy/progress 线程执行（与 CQ 消化路径一致，减少并发锁争用）

并发去重语义（必须）：

- `epoch_due` 是“需要更新”的脏位，不是计数器。
- `ccEpochUpdateIfDue()` 推荐使用 `if (epoch_due.exchange(0) == 1)` 抢占唯一执行权。
- 只有抢到执行权的一方执行 `ccEpochUpdate()`；其它并发调用直接返回。
- 执行完成后更新 `epoch_last_ts_ns / epoch_last_finalized`，作为下一轮基线。

> 这样可以保证：CQE path 只负责“观测与标记”，控制器真正决策发生在 send/epoch 边界的低频路径。

---

## 9. 控制器（progress-aware，epoch 更新）

### 9.1 输入信号（第一版最小集合）

- completion：`ewma_avg_completion_ns`、`ewma_tail_completion_ns`、`ewma_span_ns`
- progress：`progress_efficiency = Δfinalized / Δt`
- pressure：`queue_pressure = inflight / effective_window`
- telemetry hint（可选）：`cnp/ce/pcie_stall/rnic_pressure`

### 9.2 输出

- `local_window`
- `effective_window = min(local_window, device_cap)`

### 9.3 控制策略建议（工程友好、易 debug）

- **减窗快、恢复慢**
- 按状态机 `NORMAL/CAUTION/SEVERE` 工作

示例策略：

- **快速减窗**：当 `queue_pressure` 高且 `progress_efficiency` 下降，且 tail/span 恶化（或 hint severe）
  - `local_window = max(min_window, local_window * beta_fast)`
  - 设置 `cooldown_epochs = K`
- **慢速恢复**：连续 `stable_epochs >= STABLE_EPOCHS` 且无 severe hint
  - `local_window = min(max_window, local_window + alpha_slow)`

telemetry 仅做 gating/bias：

- severe hint：抑制增窗、允许更激进减窗
- 不允许 telemetry 单独驱动增窗（避免外部噪声直接破坏闭环）

---

## 10. Telemetry hint（可选增强，非 fast path）

### 10.1 形态

建议统一为规整化 hint：

```c
struct XCCLHints {
  uint64_t version;
  uint64_t ts_ns;
  float cnp_level;
  float ce_level;
  float pcie_stall_level;
  float rnic_pressure;
  uint32_t flags;
};
```

### 10.2 读取时机

- collective 调用前（一次 snapshot）
- epoch 边界或 epochdue 被触发时低频刷新

### 10.3 接入 API（推荐，便于代码挂载）

推荐将 telemetry hint 接入拆成两个明确的低频 API（不在 CQE fast path 调用）：

1. `ccOnCollectiveBegin(cc, now_ns)`
   - 读取最新 hint snapshot
   - 写入 `cc->hint`，并记录 `cc->hint.ts_ns`
   - （可选）初始化本轮控制 baseline / epoch 观察起点

2. `ccRefreshHintIfNeeded(cc, now_ns)`
   - 仅当 `now_ns` 距离上次刷新超过 `epoch_interval`，或 `now_ns - hint.ts_ns > hint_ttl_ns` 时触发
   - 重新读取并更新 `cc->hint`

### 10.4 TTL 语义（必须）

- 引入 `hint_ttl_ns`（配置项，建议从 `NCCL_CC_HINT_TTL_NS` 读取或默认若干毫秒）
- hint 有效性判定：
  - 若 `now_ns - hint.ts_ns > hint_ttl_ns`，则视为无效
  - 无效时仅使用本地症状闭环（local EWMA/progress/queue pressure），telemetry 不参与严重决策或强减窗

### 10.5 传递方式

- 共享内存（sidecar 低频写，XCCL 低频读）
- fast path 不触碰共享内存

共享内存读取一致性（必须）：

- 读取 hint snapshot 时必须避免“半更新态”。
- 推荐协议（version double-check）：
  1. reader 读取 `version1`
  2. reader 读取完整 hint 结构
  3. reader 读取 `version2`
  4. 若 `version1 != version2`，或 `version1` 为写入中的奇数版本，则重试
- 可选替代：sidecar 双缓冲写，XCCL 只读稳定槽。

---

## 11.5 并发模型与锁/原子边界（实现约束）

### 原子字段（lock-free 热路径）

- `inflight_chunks`
- `ChunkSlot.pending_cqes`
- `ChunkSlot.finalized_flag`
- `CollectiveCC.epoch_due`

### 低频锁保护字段（epoch/control path）

- `local_window`
- `effective_window`
- `ewma_*` 指标
- `state`
- `stable_epochs / cooldown_epochs`

### 路径并发约束

- completion path 可并发进入同一个 `CollectiveCC`（多 CQE 并发）。
- send path 在实现上可能由同一 proxy/progress 线程主导，但文档不强依赖“单线程假设”；并发安全必须靠原子/锁边界保证。
- 对 `ChunkSlot`：
  - 生命周期初始化（slot 分配/清零）由 send path 执行；
  - completion 仅更新本生命周期内允许并发更新的原子字段（pending/samples/max/sum 等）。

---

## 11. DeviceCC（可选，弱化 guardrail）

DeviceCC 不参与细粒度调度，只输出 coarse cap：

- `device_cap = clamp(f(hints), cap_floor, cap_ceiling)`
- `effective_window = min(local_window, device_cap)`

启用条件：

- 默认关闭
- 仅在需要 coarse guardrail 或 emergency brake 时启用

---

## 12. 调试与可观测性（建议输出/统计）

### 12.1 Debug 开关建议

- `NCCL_CC_DEBUG`
- `NCCL_CC_TRACE_SEND`
- `NCCL_CC_TRACE_CQE`
- `NCCL_CC_TRACE_CONTROL`

### 12.2 建议导出的统计量（每 comm/每 CC）

- `posted_chunks_total / finalized_chunks_total`
- `inflight_chunks`
- `local_window / effective_window`
- `progress_efficiency`
- `queue_pressure`
- `ewma_avg/tail/span`
- decrease/increase 次数、state 切换次数
- stale CQE 次数（shadow inactive 或 generation 不匹配）
- rollback 次数、post partial 次数
- `pending_cqes` 异常次数（`prev<=0`）

---

## 13. 分阶段实施计划（工程落地顺序）

### Phase 1：Correctness hardening（优先）

- ShadowSlot generation（+ stale CQE 检测）
- finalize exactly once（`finalized_flag`）
- rollback 对称（窗口/slot/chunk 状态一致）
- `pending_cqes` 异常计数与保护

### Phase 2：progress-aware 观测层

- chunk finalized 产出 `avg/max/span/service_time`
- epoch bucket + EWMA

### Phase 3：调用前 hint 注入

- hints snapshot API（共享内存或低频通道）

### Phase 4：运行中 epoch 控制更新

- `epoch_interval` 控制逻辑稳定化，fast path 只读 `effective_window`

### Phase 5：弱 DeviceCC

- coarse cap + emergency brake

---

## 14. 附：与现有 AIMD/RTT-only 实现的兼容说明

在现有实现上逐步引入本设计时，保持以下兼容性：

- `ncclIbOnCompletionWithCC()` 必须继续在 completion 入口先执行，且必须恢复 `wc->wr_id`，让原生 `CompletionEventProcess` 正常处理 request events。
- 窗口控制仍以 `inflight_chunks` 为主计数，但其更新必须严格绑定“post 成功/rollback/finalize”三条路径。
- 第一阶段可继续用 RTT-only 的控制器输出 `local_window`，但观测层与 correctness 机制必须先硬化；控制器升级只在此基础上进行。
---
## 14.5 失败处理策略（运行时行为约定）
### stale CQE
- 行为：计数 + 限频日志
- 不触发 `pending_cqes--`
- 不触发 finalize
- 不 crash
### `pending_cqes.fetch_sub()` 返回 `prev<=0`
- 视为严重异常（重复减/状态错乱）
- 行为：计数 + 限频日志
- 禁止 finalize
- debug 模式可 assert
### chunk_slot 分配发现目标 slot 仍活跃
- 视为资源耗尽或生命周期异常
- 行为：返回 retry（推荐）或错误码
- debug 模式可 assert
### hint 过期
- 自动退化为 purely local controller
- telemetry 不参与强减窗/状态切换决策
---
## 15. 资源生命周期表（便于实现核对状态机）
| 资源 | 分配/初始化 | 正常推进 | stale/失败处理 | 释放/回收点 |
|---|---|---|---|---|
| `CollectiveCC` | `ncclIbGetOrCreateCollectiveCC`（按 send-side 上下文 key） | epochdue 标记、窗口更新、统计 EWMA | `finalize` 后仍可能存在延迟 CQE（由 ChunkSlot/ShadowSlot generation 防护） | `ncclIbAIMDFinalize`（或 CC 池回收策略） |
| `ShadowSlot` | per-QP signaled WR post 之前分配（generation++，写元数据） | completion 恢复 `wc->wr_id`，并做 chunk_generation 校验后释放（inactive=0） | inactive/stale：直接 return 不触发 `pending_cqes--` | completion fast path 释放 slot（active->0） |
| `ChunkSlot`（chunk_generation） | chunk 放行时分配 `chunk_slot`，并设置 `chunk_generation` | 每个 signaled QP post 成功则 `pending_cqes++`；pending 递减到 0 时 finalize | 若 `posted_qps==0`：立即 rollback（不等 completion） | finalize 中触发一次性状态迁移；chunk_generation 复用由下一次分配隔离 |
| `epoch bucket` | finalize 时轻量累积样本 | send path 入口消费并调用 `ccEpochUpdate()` | 若未触发 epochdue，则保持累积等待 | `ccEpochUpdate()` 后清零/更新基线 |

