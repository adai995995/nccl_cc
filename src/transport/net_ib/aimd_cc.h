/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_AIMD_CC_H_
#define NET_IB_AIMD_CC_H_

#include "common.h"
#include "xccl_telemetry_hint.h"
#include <stdint.h>

// ============================================================================
// 环境变量
// ============================================================================
#define NCCL_AIMD_ENABLE_ENV "NCCL_AIMD_ENABLE"
// Phase 2 观测层：NCCL_CC_METRICS=1 启用 chunk 级 sum/min/first/last/span 与 epoch bucket
#define NCCL_CC_METRICS_ENV "NCCL_CC_METRICS"
#define NCCL_CC_METRICS_LOG_MS_ENV "NCCL_CC_METRICS_LOG_MS"
// Phase 4：epoch 控制器（见 design_docs/xccl_phase4_epoch_controller_implementation_design.md）
#define NCCL_CC_EPOCH_ENABLE_ENV "NCCL_CC_EPOCH_ENABLE"
#define NCCL_CC_EPOCH_INTERVAL_NS_ENV "NCCL_CC_EPOCH_INTERVAL_NS"
// 以下为可调参数（在 aimd_cc.cc 的 GetOrCreateCollectiveCC 中解析），未设置时用默认值：
//   NCCL_AIMD_MIN_WINDOW       默认 16    [1, 1024]
//   NCCL_AIMD_MAX_WINDOW       默认 512   [1, 4096]
//   NCCL_AIMD_LIB_WINDOW       默认 256   [min_window, max_window]
//   NCCL_AIMD_ALPHA            默认 1.0   (0, 100]
//   NCCL_AIMD_BETA             默认 0.5   (0, 1]
//   NCCL_AIMD_UPDATE_INTERVAL_US  默认 1000  [100, 100000] (us)

// ============================================================================
// Shadow Pool 配置
// ============================================================================
#define NCCL_CC_MAX_QP 256
#define NCCL_CC_MAX_SHADOW_SLOTS 65536  // 每个QP最多65536个slot（Window 512×8 slice 可能达 4096+，预留余量）
#define NCCL_CC_MAX_COLLECTIVES 1024   // CollectiveCC 池大小

// ============================================================================
// WR_ID 编码配置
// ============================================================================
#define NCCL_CC_WR_MAGIC 0xCC01  // 标识受控WR的Magic Number

// chunk_slot 需要覆盖 cc->chunk_tracker.max_chunks<=1024，因此仅用 10bit。
#define NCCL_CC_CHUNK_SLOT_BITS 10
#define NCCL_CC_CHUNK_SLOT_MASK ((1U << NCCL_CC_CHUNK_SLOT_BITS) - 1U)

// 剩余 4bit 用于 generation tag（仅用于 fast-path stale 预过滤）。
#define NCCL_CC_CHUNK_GENERATION_TAG_BITS 4
#define NCCL_CC_CHUNK_GENERATION_TAG_MASK ((1U << NCCL_CC_CHUNK_GENERATION_TAG_BITS) - 1U)

// 64位 wr_id 布局（chunk_slot 需覆盖 max_chunks=1024）：
// [63:48] Magic (16 bit)
// [47:40] QP Index (8 bit)   - 支持 256 个 QP
// [39:30] CC Index (10 bit)  - 支持 1024 个 CC，足够
// [29:14] Shadow Slot (16 bit) - 支持 65536
// [13:10] Chunk Generation Tag (4 bit)
// [9:0]   Chunk Slot (10 bit, max_chunks<=1024)

#define NCCL_CC_ENCODE_WR_ID(qp_idx, cc_idx, chunk_slot, chunk_generation_tag, shadow_slot) \
    (((uint64_t)NCCL_CC_WR_MAGIC << 48) | \
     ((uint64_t)((qp_idx) & 0xFF) << 40) | \
     ((uint64_t)((cc_idx) & 0x3FF) << 30) | \
     ((uint64_t)((shadow_slot) & 0xFFFF) << 14) | \
     ((uint64_t)((chunk_generation_tag) & NCCL_CC_CHUNK_GENERATION_TAG_MASK) << NCCL_CC_CHUNK_SLOT_BITS) | \
     ((uint64_t)((chunk_slot) & NCCL_CC_CHUNK_SLOT_MASK)))

#define NCCL_CC_DECODE_WR_ID(wr_id, qp_idx_out, cc_idx_out, chunk_slot_out, chunk_generation_tag_out, slot_out) \
    do { \
        uint16_t magic = (wr_id >> 48) & 0xFFFF; \
        if (magic == NCCL_CC_WR_MAGIC) { \
            *(qp_idx_out)   = (int)((wr_id >> 40) & 0xFF); \
            *(cc_idx_out)   = (int)((wr_id >> 30) & 0x3FF); \
            *(slot_out)     = (int)((wr_id >> 14) & 0xFFFF); \
            *(chunk_slot_out) = (int)(wr_id & NCCL_CC_CHUNK_SLOT_MASK); \
            *(chunk_generation_tag_out) = (int)((wr_id >> NCCL_CC_CHUNK_SLOT_BITS) & NCCL_CC_CHUNK_GENERATION_TAG_MASK); \
        } else { \
            *(qp_idx_out) = *(cc_idx_out) = *(chunk_slot_out) = *(chunk_generation_tag_out) = *(slot_out) = -1; \
        } \
    } while(0)

// 检查是否是受控WR
#define NCCL_CC_IS_MY_WR(wr_id) \
    (((wr_id >> 48) & 0xFFFF) == NCCL_CC_WR_MAGIC)

// ============================================================================
// Shadow Request 结构
// ============================================================================
struct ShadowRequest {
    uint64_t original_wr_id;  // 完整的64位原生ID（指针或大索引）
    int cc_index;              // CC在池中的索引
    int active;                // 标记是否占用（0=空闲，1=占用）

    uint32_t shadow_generation; // 每次分配递增（用于 debug/观测）
    uint32_t chunk_slot;        // chunk tracker 的 slot index
    uint64_t chunk_generation;  // chunk_generation：用于强 stale 判定

    uint64_t send_timestamp;   // 发送时间戳（用于RTT计算）
};

// ============================================================================
// CollectiveCC 结构
// ============================================================================
struct ChunkTracker {
    uint64_t* chunk_send_times;     // 每个chunk的发送时间
    uint64_t* chunk_rtts;           // 每个chunk的最大WR RTT（与 Phase2 max_completion 同源，CAS max）
    int* chunk_wr_counts;           // 每个chunk包含的WR数量
    volatile int* chunk_completed_wrs; // 每个chunk已完成的WR数量
    volatile int* chunk_status;      // 0: idle, 1: in-flight, 2: completed(可复用)
    volatile int* chunk_cqes_pending;  // 每个chunk“已成功 post”的 signaled CQE 数（初值 0）
    volatile int* chunk_finalized_flag; // exactly-once finalize：0/1（CAS）

    uint64_t* chunk_generation;         // chunk_generation（与 ShadowSlot 强一致校验）

    uint64_t* chunk_original_wr_id; // p2p 时 original_wr_id 可能为 0：首 CQE 写入，后续 CQE 读取
    // Phase 2：chunk 级观测（仅当 NCCL_CC_METRICS=1 时由 completion 更新；与 chunk_rtts 并行）
    uint64_t* chunk_first_cqe_us;   // 0 表示尚无 CQE
    uint64_t* chunk_last_cqe_us;    // max(now)；RecordChunkSend 时置 0
    uint64_t* chunk_lat_sum_us;     // 各 CQE 的 wr_rtt 之和
    uint64_t* chunk_lat_min_us;     // 初值 UINT64_MAX
    uint32_t* chunk_lat_samples;    // 有效 CQE 数，应等于 posted QPs
    int max_chunks;                  // 最大可追踪的 chunk 数量（<=1024）
    uint64_t next_chunk_seq;        // 单调递增分配：slot = seq%max_chunks；gen=seq/max_chunks
    pthread_mutex_t mutex;           // 保护ChunkTracker
};

struct CollectiveCC {
    uintptr_t comm_key;              // 按 comm 区分，避免多路共享一 cc 占满 window 导致死锁
    uint64_t collective_id;          // 用于日志
    
    // RTT统计
    volatile uint64_t rtt_max;        // Max RTT observed in current window
    volatile uint64_t rtt_min;        // Min RTT observed (for baseline)
    uint64_t rtt_baseline;           // Baseline RTT (e.g., empty network RTT)
    double rtt_ewma;                 // EWMA(max RTT) - 对tail敏感但不抖
    
    // 窗口管理
    volatile int inflight_chunks;    // 当前in-flight的chunk数量
    volatile int lib_window;         // LIW: Library-level Injection Window
    int min_window;                  // 最小窗口（e.g., 1）
    int max_window;                  // 最大窗口（e.g., 512）
    
    // 控制参数
    uint64_t rtt_target_low;         // 空闲阈值 = 空载RTT × 1.5
    uint64_t rtt_target_high;        // 拥塞阈值 = 空载RTT × 3
    double alpha;                    // 加性增参数（e.g., 1）
    double beta;                     // 乘性减参数（e.g., 0.5）
    
    // 控制周期
    uint64_t update_interval_us;     // 窗口更新间隔（e.g., 1000us = 1ms）
    uint64_t last_update_time;       // 上次窗口更新时间
    
    // 统计
    int congestion_events;           // 拥塞事件计数
    int increase_events;              // 增加事件计数
    int total_updates;                // 总更新次数
    
    // 状态
    int enabled;                     // 是否启用
    
    // Chunk RTT 追踪器
    struct ChunkTracker chunk_tracker;
    
    // 互斥锁（用于低频操作，如窗口更新）
    pthread_mutex_t mutex;

    // Phase 2：epoch 聚合桶（finalize 写入与 UpdateLIW 读/打日志均在 cc->mutex 下，避免撕裂）
    uint64_t metrics_bucket_sum_avg_us;   // 各 chunk avg 之和（整数 avg = sum_lat/samples）
    uint64_t metrics_bucket_sum_max_us;   // 各 chunk max_completion 之和
    uint64_t metrics_bucket_sum_span_us;
    uint64_t metrics_bucket_sum_service_us;
    uint32_t metrics_bucket_chunk_count;
    uint64_t metrics_bucket_last_finalize_us;
    uint64_t metrics_o1_violations;       // samples != chunk_wr_counts 等
    uint64_t metrics_last_log_us;         // 限频日志

    // Phase 3：Telemetry hint（节点级快照缓存，见 design_docs/xccl_phase3_telemetry_hint_implementation_design.md）
    TelemetryHintSnapshot hint;
    uint64_t hint_ttl_ns;              // NCCL_CC_HINT_TTL_NS
    uint64_t hint_last_refresh_ns;     // 单调时钟 ns，上次成功读入提交
    uint64_t hint_read_failures;
    int hint_read_ok;
    int hint_valid;

    // Phase 4：epoch + effective_window（send 路径准入上界）
    uint32_t epoch_due;              /* 原子：finalize 置 1，IfDue exchange 清 0 */
    int device_cap;                  /* Phase 4 恒为 max_window；Phase 5 可动态 */
    volatile int effective_window;   /* min(lib_window, device_cap, hint_cap) 再 clamp max_chunks；0=未发布则回退 lib_window */
    uint64_t epoch_last_ts_ns;       /* 上次成功 ccEpochUpdate 单调时间 */
    uint64_t epoch_interval_ns;      /* 默认 1ms，可与 AIMD update_interval 分离 */
    uint64_t finalized_chunks_total; /* finalize 次数（观测，原子递增） */
};

// Phase 2：finalize 一次性样本（微秒）
struct NcclCcChunkSample {
  uint64_t chunk_seq;
  uint64_t chunk_generation;
  int      chunk_slot;
  uint64_t avg_completion_us;
  uint64_t max_completion_us;
  uint64_t min_completion_us;
  uint64_t span_us;
  uint64_t service_time_us;
  uint32_t samples;
};

// ============================================================================
// 函数声明
// ============================================================================

// 初始化/清理
ncclResult_t ncclIbAIMDInit(void);
ncclResult_t ncclIbAIMDFinalize(void);
int ncclIbIsAIMDEnabled(void);
int ncclIbCcMetricsEnabled(void);
int ncclIbCcHintEnabled(void);
int ncclIbCcEpochEnabled(void);
uint64_t ncclIbGetNanos(void);
/* Phase 4：send/MultiSend 与 PostSendWithCC 准入上界（epoch 关闭时等价 lib_window） */
int ncclCcGetEffectiveWindowForSend(const struct CollectiveCC* cc);

/* Phase 3：发送路径低频调用；幂等、内部节流 */
void ncclCcOnCollectiveBegin(struct CollectiveCC* cc, uint64_t now_ns);
void ncclCcRefreshHintIfNeeded(struct CollectiveCC* cc, uint64_t now_ns);
int ncclCcHintIsValid(const struct CollectiveCC* cc, uint64_t now_ns);

// Shadow Pool管理
ncclResult_t ncclIbInitShadowPool(void);
int ncclIbAllocateShadowSlot(int qp_index, uint64_t original_wr_id,
                              int cc_index, int chunk_slot, uint64_t chunk_generation);
void ncclIbReleaseShadowSlot(int qp_index, int shadow_slot);
uint64_t ncclIbGetMicros(void);

// CollectiveCC管理
ncclResult_t ncclIbGetOrCreateCollectiveCC(void* comm, uint64_t collective_id, int estimated_chunks, 
                                          struct CollectiveCC** cc_out);
ncclResult_t ncclIbDestroyCollectiveCC(uint64_t collective_id);

// RTT监测
void ncclIbRecordChunkSend(struct CollectiveCC* cc, int chunk_slot, uint64_t chunk_generation, int num_wrs);
void ncclIbUpdateChunkWRRTT(struct CollectiveCC* cc, int chunk_slot, uint64_t wr_rtt);
void ncclIbUpdateChunkRTTOnly(struct CollectiveCC* cc, int chunk_slot, uint64_t wr_rtt);  // 仅更新 RTT(max)，不增加 completed_wrs
int ncclIbIsChunkComplete(struct CollectiveCC* cc, int chunk_slot);
void ncclIbFinalizeChunkRTT(struct CollectiveCC* cc, int chunk_slot);
void ncclIbUpdateCollectiveRTT(struct CollectiveCC* cc, uint64_t chunk_rtt);

// AIMD控制
void ncclIbUpdateLIW(struct CollectiveCC* cc);

// 发送路径集成
ncclResult_t ncclIbPostSendWithCC(struct ncclIbSendComm* comm, 
                                  struct CollectiveCC* cc, 
                                  int chunk_id,
                                  int num_wrs,
                                  struct ibv_send_wr* wrs,
                                  int nreqs,
                                  ncclIbQp* qp);

// Completion路径集成
ncclResult_t ncclIbOnCompletionWithCC(struct ncclIbNetCommBase* commBase, 
                                       struct ibv_wc* wc, 
                                       int devIndex);

#endif // NET_IB_AIMD_CC_H_
