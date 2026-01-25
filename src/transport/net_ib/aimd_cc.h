/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_AIMD_CC_H_
#define NET_IB_AIMD_CC_H_

#include "common.h"
#include <stdint.h>

// ============================================================================
// 环境变量
// ============================================================================
#define NCCL_AIMD_ENABLE_ENV "NCCL_AIMD_ENABLE"
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

// 64位 wr_id 布局（Chunk ID 需 14 bit 覆盖 max_chunks=1024，原 8 bit 导致 256+ 截断→窗口死锁）：
// [63:48] Magic (16 bit)
// [47:40] QP Index (8 bit)   - 支持 256 个 QP
// [39:30] CC Index (10 bit)  - 支持 1024 个 CC，足够
// [29:14] Shadow Slot (16 bit) - 支持 65536
// [13:0]  Chunk ID (14 bit)  - 支持 16384，覆盖 max_chunks 1024

#define NCCL_CC_ENCODE_WR_ID(qp_idx, cc_idx, chunk_id, slot) \
    (((uint64_t)NCCL_CC_WR_MAGIC << 48) | \
     ((uint64_t)((qp_idx) & 0xFF) << 40) | \
     ((uint64_t)((cc_idx) & 0x3FF) << 30) | \
     ((uint64_t)((slot) & 0xFFFF) << 14) | \
     ((uint64_t)((chunk_id) & 0x3FFF)))

#define NCCL_CC_DECODE_WR_ID(wr_id, qp_idx_out, cc_idx_out, chunk_id_out, slot_out) \
    do { \
        uint16_t magic = (wr_id >> 48) & 0xFFFF; \
        if (magic == NCCL_CC_WR_MAGIC) { \
            *(qp_idx_out)   = (int)((wr_id >> 40) & 0xFF); \
            *(cc_idx_out)   = (int)((wr_id >> 30) & 0x3FF); \
            *(slot_out)     = (int)((wr_id >> 14) & 0xFFFF); \
            *(chunk_id_out) = (int)(wr_id & 0x3FFF); \
        } else { \
            *(qp_idx_out) = *(cc_idx_out) = *(chunk_id_out) = *(slot_out) = -1; \
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
    int chunk_id;              // Chunk ID（或NCCL Step）
    int cc_index;              // CC在池中的索引
    int active;                // 标记是否占用（0=空闲，1=占用）
    uint64_t send_timestamp;   // 发送时间戳（用于RTT计算）
};

// ============================================================================
// CollectiveCC 结构
// ============================================================================
struct ChunkTracker {
    uint64_t* chunk_send_times;     // 每个chunk的发送时间
    uint64_t* chunk_rtts;           // 每个chunk的最大WR RTT
    int* chunk_wr_counts;           // 每个chunk包含的WR数量
    volatile int* chunk_completed_wrs; // 每个chunk已完成的WR数量
    volatile int* chunk_status;      // 0: idle, 1: in-flight, 2: completed(可复用)
    volatile int* chunk_cqes_pending;  // 每个chunk待完成的CQE数（Multi-QP时=nqps）
    uint64_t* chunk_original_wr_id; // p2p 时 1 Slot 多 CQE：首 CQE 写入，后续 CQE 读取
    int max_chunks;                  // 最大可追踪的chunk数量
    int next_chunk_id;               // 单调递增分配，取模得 chunk_id_slot，避免多 comm 同 collective_id 时 slot 碰撞
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
};

// ============================================================================
// 函数声明
// ============================================================================

// 初始化/清理
ncclResult_t ncclIbAIMDInit(void);
ncclResult_t ncclIbAIMDFinalize(void);
int ncclIbIsAIMDEnabled(void);

// Shadow Pool管理
ncclResult_t ncclIbInitShadowPool(void);
int ncclIbAllocateShadowSlot(int qp_index, uint64_t original_wr_id, 
                              int cc_index, int chunk_id);
void ncclIbReleaseShadowSlot(int qp_index, int shadow_slot);
uint64_t ncclIbGetMicros(void);

// CollectiveCC管理
ncclResult_t ncclIbGetOrCreateCollectiveCC(void* comm, uint64_t collective_id, int estimated_chunks, 
                                          struct CollectiveCC** cc_out);
ncclResult_t ncclIbDestroyCollectiveCC(uint64_t collective_id);

// RTT监测
void ncclIbRecordChunkSend(struct CollectiveCC* cc, int chunk_id, int num_wrs, int nqps);
void ncclIbUpdateChunkWRRTT(struct CollectiveCC* cc, int chunk_id, uint64_t wr_rtt);
void ncclIbUpdateChunkRTTOnly(struct CollectiveCC* cc, int chunk_id, uint64_t wr_rtt);  // 仅更新 RTT(max)，不增加 completed_wrs
int ncclIbIsChunkComplete(struct CollectiveCC* cc, int chunk_id);
void ncclIbFinalizeChunkRTT(struct CollectiveCC* cc, int chunk_id);
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
