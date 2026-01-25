/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "aimd_cc.h"
#include "common.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// 全局变量
// ============================================================================

// Shadow Pool
struct ShadowRequest g_shadow_pool[NCCL_CC_MAX_QP][NCCL_CC_MAX_SHADOW_SLOTS];
int g_shadow_pool_next[NCCL_CC_MAX_QP];
pthread_mutex_t g_shadow_pool_mutex[NCCL_CC_MAX_QP];
int g_shadow_pool_initialized = 0;

// CollectiveCC Pool
struct CollectiveCC* g_cc_pool[NCCL_CC_MAX_COLLECTIVES];
pthread_mutex_t g_cc_pool_mutex;
int g_cc_pool_initialized = 0;

// 全局状态
int g_aimd_enabled = -1;  // -1: 未检查, 0: 禁用, 1: 启用

// ============================================================================
// 工具函数
// ============================================================================

uint64_t ncclIbGetMicros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

int ncclIbIsAIMDEnabled(void) {
    if (g_aimd_enabled == -1) {
        const char* env = getenv(NCCL_AIMD_ENABLE_ENV);
        if (env && atoi(env) > 0) {
            g_aimd_enabled = 1;
        } else {
            g_aimd_enabled = 0;
        }
    }
    return g_aimd_enabled;
}

// ============================================================================
// Shadow Pool 管理
// ============================================================================

ncclResult_t ncclIbInitShadowPool(void) {
    if (g_shadow_pool_initialized) {
        return ncclSuccess;
    }
    
    for (int qp = 0; qp < NCCL_CC_MAX_QP; qp++) {
        memset(g_shadow_pool[qp], 0, sizeof(g_shadow_pool[qp]));
        g_shadow_pool_next[qp] = 0;
        pthread_mutex_init(&g_shadow_pool_mutex[qp], NULL);
    }
    
    g_shadow_pool_initialized = 1;
    return ncclSuccess;
}

int ncclIbAllocateShadowSlot(int qp_index, uint64_t original_wr_id, 
                              int cc_index, int chunk_id) {
    if (qp_index < 0 || qp_index >= NCCL_CC_MAX_QP) {
        return -1;
    }
    
    pthread_mutex_lock(&g_shadow_pool_mutex[qp_index]);
    
    // 找一个空的slot（从next位置开始循环查找）
    int slot = g_shadow_pool_next[qp_index];
    int start_slot = slot;
    
    do {
        if (g_shadow_pool[qp_index][slot].active == 0) {
            // 找到空slot
            g_shadow_pool[qp_index][slot].original_wr_id = original_wr_id;
            g_shadow_pool[qp_index][slot].cc_index = cc_index;
            g_shadow_pool[qp_index][slot].chunk_id = chunk_id;
            g_shadow_pool[qp_index][slot].active = 1;
            g_shadow_pool[qp_index][slot].send_timestamp = ncclIbGetMicros();
            
            // 更新next指针（下次从下一个位置开始查找）
            g_shadow_pool_next[qp_index] = (slot + 1) % NCCL_CC_MAX_SHADOW_SLOTS;
            
            pthread_mutex_unlock(&g_shadow_pool_mutex[qp_index]);
            return slot;
        }
        slot = (slot + 1) % NCCL_CC_MAX_SHADOW_SLOTS;
    } while (slot != start_slot);
    
    // 没有空slot（理论上不应该发生，因为窗口控制了inflight数量）
    pthread_mutex_unlock(&g_shadow_pool_mutex[qp_index]);
    WARN("NET/IB: Shadow Pool full for QP %d", qp_index);
    return -1;
}

void ncclIbReleaseShadowSlot(int qp_index, int shadow_slot) {
    if (qp_index < 0 || qp_index >= NCCL_CC_MAX_QP ||
        shadow_slot < 0 || shadow_slot >= NCCL_CC_MAX_SHADOW_SLOTS) {
        return;
    }
    
    pthread_mutex_lock(&g_shadow_pool_mutex[qp_index]);
    g_shadow_pool[qp_index][shadow_slot].active = 0;
    pthread_mutex_unlock(&g_shadow_pool_mutex[qp_index]);
}

// ============================================================================
// CollectiveCC 管理
// ============================================================================

ncclResult_t ncclIbInitCCPool(void) {
    if (g_cc_pool_initialized) {
        return ncclSuccess;
    }
    
    memset(g_cc_pool, 0, sizeof(g_cc_pool));
    pthread_mutex_init(&g_cc_pool_mutex, NULL);
    g_cc_pool_initialized = 1;
    return ncclSuccess;
}

ncclResult_t ncclIbGetOrCreateCollectiveCC(void* comm, uint64_t collective_id, int estimated_chunks, 
                                          struct CollectiveCC** cc_out) {
    if (!g_cc_pool_initialized) {
        NCCLCHECK(ncclIbInitCCPool());
    }
    uintptr_t key = (uintptr_t)comm;
    
    pthread_mutex_lock(&g_cc_pool_mutex);
    
    // 按 comm 查找：每个连接独立 window，避免多 comm 共享一 cc 时 8 路占满 block 第 9 路
    for (int i = 0; i < NCCL_CC_MAX_COLLECTIVES; i++) {
        if (g_cc_pool[i] && g_cc_pool[i]->comm_key == key) {
            g_cc_pool[i]->collective_id = collective_id;  // 更新以便日志
            *cc_out = g_cc_pool[i];
            pthread_mutex_unlock(&g_cc_pool_mutex);
            return ncclSuccess;
        }
    }
    
    // 创建新的CollectiveCC
    int free_slot = -1;
    for (int i = 0; i < NCCL_CC_MAX_COLLECTIVES; i++) {
        if (g_cc_pool[i] == NULL) {
            free_slot = i;
            break;
        }
    }
    
    if (free_slot < 0) {
        pthread_mutex_unlock(&g_cc_pool_mutex);
        WARN("NET/IB: CC Pool full");
        return ncclInternalError;
    }
    
    struct CollectiveCC* cc = (struct CollectiveCC*)malloc(sizeof(struct CollectiveCC));
    if (!cc) {
        pthread_mutex_unlock(&g_cc_pool_mutex);
        return ncclInternalError;
    }
    
    memset(cc, 0, sizeof(struct CollectiveCC));
    cc->comm_key = key;
    cc->collective_id = collective_id;
    cc->enabled = 1;
    
    // 初始化窗口与 AIMD 参数（可由环境变量覆盖，便于 A/B 测试）
    int min_w = 16, max_w = 512, lib_w = 256;
    double alpha_val = 1.0, beta_val = 0.5;
    uint64_t uiu = 1000;
    int from_env = 0;
    const char *e;

    e = getenv("NCCL_AIMD_MIN_WINDOW"); if (e) { int v = atoi(e); if (v >= 1 && v <= 1024) { min_w = v; from_env = 1; } }
    e = getenv("NCCL_AIMD_MAX_WINDOW"); if (e) { int v = atoi(e); if (v >= 1 && v <= 4096) { max_w = v; from_env = 1; } }
    if (min_w > max_w) max_w = min_w;

    e = getenv("NCCL_AIMD_LIB_WINDOW");
    if (e) { int v = atoi(e); if (v >= min_w && v <= max_w) { lib_w = v; from_env = 1; } }
    else { if (256 < min_w) lib_w = min_w; else if (256 > max_w) lib_w = max_w; }

    e = getenv("NCCL_AIMD_ALPHA"); if (e) { double v = atof(e); if (v > 0 && v <= 100) { alpha_val = v; from_env = 1; } }
    e = getenv("NCCL_AIMD_BETA");  if (e) { double v = atof(e); if (v > 0 && v <= 1) { beta_val = v; from_env = 1; } }
    e = getenv("NCCL_AIMD_UPDATE_INTERVAL_US"); if (e) { int v = atoi(e); if (v >= 100 && v <= 100000) { uiu = (uint64_t)v; from_env = 1; } }

    cc->min_window = min_w;
    cc->max_window = max_w;
    cc->lib_window = lib_w;
    cc->inflight_chunks = 0;
    cc->alpha = alpha_val;
    cc->beta = beta_val;
    cc->update_interval_us = uiu;
    cc->last_update_time = 0;

    { static int _aimd_env_logged = 0; if (from_env && !_aimd_env_logged) { _aimd_env_logged = 1; INFO(NCCL_ENV, "AIMD params (from env): min_window=%d max_window=%d lib_window=%d alpha=%.2f beta=%.2f update_interval_us=%lu", min_w, max_w, lib_w, alpha_val, beta_val, (unsigned long)uiu); } }
    
    // 初始化RTT参数
    cc->rtt_max = 0;
    cc->rtt_min = 0;
    cc->rtt_baseline = 0;
    cc->rtt_ewma = 0.0;
    cc->rtt_target_low = 0;
    cc->rtt_target_high = 0;
    
    // 初始化ChunkTracker
    int max_chunks = estimated_chunks > 0 ? estimated_chunks : 1024;
    cc->chunk_tracker.max_chunks = max_chunks;
    cc->chunk_tracker.chunk_send_times = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_rtts = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_wr_counts = (int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_completed_wrs = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_status = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_cqes_pending = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_original_wr_id = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    
    if (!cc->chunk_tracker.chunk_send_times || !cc->chunk_tracker.chunk_rtts ||
        !cc->chunk_tracker.chunk_wr_counts || !cc->chunk_tracker.chunk_completed_wrs ||
        !cc->chunk_tracker.chunk_status || !cc->chunk_tracker.chunk_cqes_pending ||
        !cc->chunk_tracker.chunk_original_wr_id) {
        free(cc->chunk_tracker.chunk_send_times);
        free(cc->chunk_tracker.chunk_rtts);
        free(cc->chunk_tracker.chunk_wr_counts);
        free((void*)cc->chunk_tracker.chunk_completed_wrs);
        free((void*)cc->chunk_tracker.chunk_status);
        free((void*)cc->chunk_tracker.chunk_cqes_pending);
        free(cc->chunk_tracker.chunk_original_wr_id);
        free(cc);
        pthread_mutex_unlock(&g_cc_pool_mutex);
        return ncclInternalError;
    }
    
    memset(cc->chunk_tracker.chunk_send_times, 0, max_chunks * sizeof(uint64_t));
    memset(cc->chunk_tracker.chunk_rtts, 0, max_chunks * sizeof(uint64_t));
    memset(cc->chunk_tracker.chunk_wr_counts, 0, max_chunks * sizeof(int));
    memset((void*)cc->chunk_tracker.chunk_completed_wrs, 0, max_chunks * sizeof(int));
    memset((void*)cc->chunk_tracker.chunk_status, 0, max_chunks * sizeof(int));
    memset((void*)cc->chunk_tracker.chunk_cqes_pending, 0, max_chunks * sizeof(int));
    memset(cc->chunk_tracker.chunk_original_wr_id, 0, max_chunks * sizeof(uint64_t));
    
    pthread_mutex_init(&cc->chunk_tracker.mutex, NULL);
    pthread_mutex_init(&cc->mutex, NULL);
    
    g_cc_pool[free_slot] = cc;
    *cc_out = cc;
    
    pthread_mutex_unlock(&g_cc_pool_mutex);
    return ncclSuccess;
}

// ============================================================================
// RTT 监测
// ============================================================================

void ncclIbRecordChunkSend(struct CollectiveCC* cc, int chunk_id, int num_wrs, int nqps) {
    if (!cc || chunk_id < 0 || chunk_id >= cc->chunk_tracker.max_chunks) return;
    
    // CAS：仅抢到 0->1 或 2->1 的调用做初始化，防止多 QP/重入重复写 chunk_cqes_pending 等
    int* status_ptr = (int*)&cc->chunk_tracker.chunk_status[chunk_id];
    int expected = 0;
    if (__atomic_compare_exchange_n(status_ptr, &expected, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        // 从 idle 首次初始化
    } else {
        expected = 2;  // 复用：从 completed 再入
        if (!__atomic_compare_exchange_n(status_ptr, &expected, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED))
            return;  // 已是 1 (in-flight)，其他线程已初始化，直接返回
    }
    cc->chunk_tracker.chunk_rtts[chunk_id] = 0;
    cc->chunk_tracker.chunk_wr_counts[chunk_id] = num_wrs;
    cc->chunk_tracker.chunk_completed_wrs[chunk_id] = 0;
    cc->chunk_tracker.chunk_cqes_pending[chunk_id] = nqps > 0 ? nqps : 1;
    cc->chunk_tracker.chunk_send_times[chunk_id] = ncclIbGetMicros();
#ifdef AIMD_DEBUG
    /* 短期放宽 rate-limit 至 50ms，便于确认 cid 是否 0,1,2… 轮转；确认后可改回 500000(500ms) */
    { static uint64_t last=0; uint64_t n=ncclIbGetMicros(); if (n-last>50000) { printf("[AIMD] RecordChunkSend: cc=%p cid=%d nqps=%d cqes_pending=%d\n", (void*)cc, chunk_id, nqps>0?nqps:1, cc->chunk_tracker.chunk_cqes_pending[chunk_id]); last=n; } }
#endif
}

void ncclIbUpdateChunkWRRTT(struct CollectiveCC* cc, int chunk_id, uint64_t wr_rtt) {
    if (!cc || chunk_id < 0 || chunk_id >= cc->chunk_tracker.max_chunks) return;
    
    ncclIbUpdateChunkRTTOnly(cc, chunk_id, wr_rtt);
    // 原子增加完成计数（仅当每个 WR 对应一个 CQE 时使用；Multi-QP 下改用 chunk_cqes_pending）
    __sync_fetch_and_add(&cc->chunk_tracker.chunk_completed_wrs[chunk_id], 1);
}

void ncclIbUpdateChunkRTTOnly(struct CollectiveCC* cc, int chunk_id, uint64_t wr_rtt) {
    if (!cc || chunk_id < 0 || chunk_id >= cc->chunk_tracker.max_chunks) return;
    
    // 使用CAS更新chunk RTT（取max），无锁
    uint64_t* chunk_rtt_ptr = &cc->chunk_tracker.chunk_rtts[chunk_id];
    uint64_t old_rtt, new_rtt;
    do {
        old_rtt = __atomic_load_n(chunk_rtt_ptr, __ATOMIC_RELAXED);
        if (wr_rtt <= old_rtt) break;
        new_rtt = wr_rtt;
    } while (!__atomic_compare_exchange_n(chunk_rtt_ptr, &old_rtt, new_rtt, 
                                           0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

int ncclIbIsChunkComplete(struct CollectiveCC* cc, int chunk_id) {
    if (!cc || chunk_id < 0 || chunk_id >= cc->chunk_tracker.max_chunks) return 0;
    
    pthread_mutex_lock(&cc->chunk_tracker.mutex);
    int complete = (cc->chunk_tracker.chunk_status[chunk_id] == 1 &&
                    cc->chunk_tracker.chunk_completed_wrs[chunk_id] >= 
                    cc->chunk_tracker.chunk_wr_counts[chunk_id]);
    pthread_mutex_unlock(&cc->chunk_tracker.mutex);
    
    return complete;
}

void ncclIbFinalizeChunkRTT(struct CollectiveCC* cc, int chunk_id) {
    if (!cc || chunk_id < 0 || chunk_id >= cc->chunk_tracker.max_chunks) return;
    
    uint64_t chunk_rtt = 0;
    int did_finalize = 0;
    
    pthread_mutex_lock(&cc->chunk_tracker.mutex);
    if (cc->chunk_tracker.chunk_status[chunk_id] == 1) {
        chunk_rtt = cc->chunk_tracker.chunk_rtts[chunk_id];
        cc->chunk_tracker.chunk_status[chunk_id] = 0;  // 置 0 以便 chunk_id 复用
        did_finalize = 1;
    }
    pthread_mutex_unlock(&cc->chunk_tracker.mutex);
    
    // 必须始终扣减 inflight，否则 chunk_rtt==0（如 send_time 未设或 RTT 为 0）时 inflight 永不降→窗口死锁、卡死
    if (did_finalize) {
        int prev_inflight = __sync_fetch_and_sub(&cc->inflight_chunks, 1);
        // 诊断：rate-limit 打印，确认 Finalize 路径有被调用（若 blocked 时从无此打印，则 CQE 未进 OnCompletion 或 chunk_cqes_pending 未到 1）
        {
            static uint64_t last_fin_print = 0;
            uint64_t now_us = ncclIbGetMicros();
            if (now_us - last_fin_print > 1000000) {
                printf("[AIMD] FinalizeChunkRTT: cc=%p chunk=%d inflight_before=%d\n", (void*)cc, chunk_id, prev_inflight);
                fflush(stdout);
                last_fin_print = now_us;
            }
        }
    }
    if (chunk_rtt > 0)
        ncclIbUpdateCollectiveRTT(cc, chunk_rtt);
}

void ncclIbUpdateCollectiveRTT(struct CollectiveCC* cc, uint64_t chunk_rtt) {
    if (!cc) return;
    
    // 使用CAS更新Max RTT，无锁
    uint64_t old_max, new_max;
    do {
        old_max = __atomic_load_n(&cc->rtt_max, __ATOMIC_RELAXED);
        if (chunk_rtt <= old_max) break;
        new_max = chunk_rtt;
    } while (!__atomic_compare_exchange_n(&cc->rtt_max, &old_max, new_max, 
                                           0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    
    // 使用CAS更新Min RTT，无锁
    uint64_t old_min, new_min;
    do {
        old_min = __atomic_load_n(&cc->rtt_min, __ATOMIC_RELAXED);
        if (chunk_rtt >= old_min && old_min != 0) break;
        new_min = chunk_rtt;
    } while (!__atomic_compare_exchange_n(&cc->rtt_min, &old_min, new_min, 
                                           0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    
    // Baseline保护：设置硬下限，防止网络一直堵导致基线漂移
    const uint64_t RTT_HARD_MIN = 1;  // 1us硬下限
    uint64_t current_min = __atomic_load_n(&cc->rtt_min, __ATOMIC_RELAXED);
    if (current_min < RTT_HARD_MIN) {
        __atomic_store_n(&cc->rtt_min, RTT_HARD_MIN, __ATOMIC_RELAXED);
        current_min = RTT_HARD_MIN;
    }
    
    // 更新基线（如果min RTT显著降低）
    if (current_min < cc->rtt_baseline * 0.8 || cc->rtt_baseline == 0) {
        // 这个更新需要加锁（低频操作）
        pthread_mutex_lock(&cc->mutex);
        cc->rtt_baseline = current_min;
        cc->rtt_target_low = cc->rtt_baseline * 1.5;
        cc->rtt_target_high = cc->rtt_baseline * 3;
        pthread_mutex_unlock(&cc->mutex);
    }
    
    // EWMA更新（在窗口更新时进行，低频操作）
    // 触发窗口更新（如果到时间）
    ncclIbUpdateLIW(cc);
}

// ============================================================================
// AIMD 控制
// ============================================================================

void ncclIbUpdateLIW(struct CollectiveCC* cc) {
    if (!cc || !cc->enabled) return;
    
    uint64_t now = ncclIbGetMicros();
    uint64_t time_since_update = (cc->last_update_time == 0) ?
        cc->update_interval_us + 1 : (now - cc->last_update_time);
    
    // 控制周期检查
    if (time_since_update < cc->update_interval_us) {
        return;
    }
    
    // 这个函数1ms才调用一次，加锁是可以接受的
    pthread_mutex_lock(&cc->mutex);
    
    // 读取当前Max RTT（原子读取）
    uint64_t current_max = __atomic_load_n(&cc->rtt_max, __ATOMIC_RELAXED);
    
    // EWMA更新（对tail敏感但不抖）
    double alpha = 0.1;  // 平滑因子
    if (cc->rtt_ewma == 0) {
        cc->rtt_ewma = (double)current_max;
    } else {
        cc->rtt_ewma = alpha * (double)current_max + (1.0 - alpha) * cc->rtt_ewma;
    }
    
    double old_window = (double)cc->lib_window;
    double rtt_collective = cc->rtt_ewma;
    
    // AIMD算法
    if (cc->rtt_target_high > 0 && rtt_collective > (double)cc->rtt_target_high) {
        // 拥塞：乘性减
        double new_window = old_window * cc->beta;
        if (new_window < cc->min_window) {
            new_window = cc->min_window;
        }
        cc->lib_window = (int)new_window;
        cc->congestion_events++;
        
        if (cc->congestion_events % 100 == 0) {
            INFO(NCCL_NET, "AIMD[%lu]: Congestion (RTT=%.1f > %.1f), LIW %.1f -> %.1f",
                 cc->collective_id, rtt_collective, (double)cc->rtt_target_high, 
                 old_window, new_window);
        }
    } 
    else if (cc->rtt_target_low > 0 && rtt_collective < (double)cc->rtt_target_low) {
        // 空闲：加性增
        double new_window = old_window + cc->alpha;
        if (new_window > cc->max_window) {
            new_window = cc->max_window;
        }
        cc->lib_window = (int)new_window;
        cc->increase_events++;
    }
    // 中间区域：保持不变（稳定性）
    
    cc->last_update_time = now;
    cc->total_updates++;
    
    pthread_mutex_unlock(&cc->mutex);
}

// ============================================================================
// 初始化/清理
// ============================================================================

ncclResult_t ncclIbAIMDInit(void) {
#ifdef AIMD_DEBUG
    printf("[AIMD] AIMD_DEBUG build loaded\n"); fflush(stdout);
#else
    printf("[AIMD] AIMD build (release, no -DAIMD_DEBUG)\n"); fflush(stdout);
#endif
    if (!ncclIbIsAIMDEnabled()) {
        return ncclSuccess;
    }
    
    NCCLCHECK(ncclIbInitShadowPool());
    NCCLCHECK(ncclIbInitCCPool());
    
    return ncclSuccess;
}

// ============================================================================
// 发送路径集成
// ============================================================================

ncclResult_t ncclIbPostSendWithCC(struct ncclIbSendComm* comm, 
                                  struct CollectiveCC* cc, 
                                  int chunk_id,
                                  int num_wrs,
                                  struct ibv_send_wr* wrs,
                                  int nreqs,
                                  ncclIbQp* qp) {
    if (!cc || !cc->enabled) {
        return ncclSuccess;  // 未启用CC，直接返回
    }
    
    // 🔴 关键：使用原子操作检查并预留窗口空间（避免竞争）
    int old_inflight, new_inflight;
    do {
        old_inflight = __sync_fetch_and_add(&cc->inflight_chunks, 0);  // 读取当前值
        if (old_inflight >= cc->lib_window) {
            // 窗口已满，返回Success但让上层下一轮再试；限频日志确认是否因 Completion 未减 inflight 导致死锁
            static uint64_t last_print = 0;
            uint64_t now_us = ncclIbGetMicros();
            if (now_us - last_print > 1000000) {
                printf("[AIMD] PostSend blocked: inflight=%d window=%d chunk_id=%d\n",
                       old_inflight, cc->lib_window, chunk_id);
                fflush(stdout);
                last_print = now_us;
            }
            return ncclSuccess;  // 返回成功，但实际未发送（上层会重试）
        }
        new_inflight = old_inflight + 1;  // 预留1个chunk的空间
    } while (!__sync_bool_compare_and_swap(&cc->inflight_chunks, old_inflight, new_inflight));
    // 🔴 按 cc 单调分配 chunk_id_slot，与 p2p 一致，避免多路共 slot 导致 Finalize 少调、inflight 不降
    int next = __sync_fetch_and_add(&cc->chunk_tracker.next_chunk_id, 1);
    int chunk_id_slot = (int)((unsigned)next % (unsigned)cc->chunk_tracker.max_chunks);
    // 获取QP索引（简化：使用qp_num的低位作为索引）
    int qp_index = qp->qp->qp_num % NCCL_CC_MAX_QP;
    
    // 获取CC在池中的索引
    int cc_index = -1;
    pthread_mutex_lock(&g_cc_pool_mutex);
    for (int i = 0; i < NCCL_CC_MAX_COLLECTIVES; i++) {
        if (g_cc_pool[i] == cc) {
            cc_index = i;
            break;
        }
    }
    pthread_mutex_unlock(&g_cc_pool_mutex);
    
    if (cc_index < 0) {
        // 回滚窗口预留
        __sync_fetch_and_sub(&cc->inflight_chunks, 1);
        return ncclInternalError;
    }
    
    int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
    ncclIbRecordChunkSend(cc, chunk_id_slot, num_wrs, nqps > 0 ? nqps : 1);
    
    // 为每个WR分配Shadow Slot并编码wr_id
    for (int r = 0; r < nreqs; r++) {
        // 🔴 关键：先保存NCCL原生的wr_id（完整的64位，可能是指针）
        uint64_t original_wr_id = wrs[r].wr_id;
        
        // 🔴 关键：分配Shadow Slot并存储原生ID（用 chunk_id_slot 保证 shadow->chunk_id 不越界）
        int shadow_slot = ncclIbAllocateShadowSlot(qp_index, original_wr_id, cc_index, chunk_id_slot);
        if (shadow_slot < 0) {
            for (int j = 0; j < r; j++) {
                if (NCCL_CC_IS_MY_WR(wrs[j].wr_id)) {
                    int qp_i, cc_i, cid, sl;
                    NCCL_CC_DECODE_WR_ID(wrs[j].wr_id, &qp_i, &cc_i, &cid, &sl);
                    if (sl >= 0 && qp_i >= 0)
                        ncclIbReleaseShadowSlot(qp_i, sl);
                }
            }
            __sync_fetch_and_sub(&cc->inflight_chunks, 1);
            return ncclSystemError;
        }
        wrs[r].wr_id = NCCL_CC_ENCODE_WR_ID(qp_index, cc_index, chunk_id_slot, shadow_slot);
    }
    
    // ✅ 关键：窗口空间已在上面预留，这里不需要再增加
    // 如果发送失败，调用者需要调用ncclIbPostSendWithCCRollback来释放slot和窗口预留
    
    return ncclSuccess;
}

// 回滚函数（如果ibv_post_send失败时调用）
void ncclIbPostSendWithCCRollback(struct ncclIbSendComm* comm,
                                   struct CollectiveCC* cc,
                                   struct ibv_send_wr* wrs,
                                   int nreqs,
                                   ncclIbQp* qp) {
    if (!cc || !cc->enabled) return;
    (void)qp;
    for (int r = 0; r < nreqs; r++) {
        if (NCCL_CC_IS_MY_WR(wrs[r].wr_id)) {
            int qp_i, cc_i, cid, sl;
            NCCL_CC_DECODE_WR_ID(wrs[r].wr_id, &qp_i, &cc_i, &cid, &sl);
            if (sl >= 0 && qp_i >= 0)
                ncclIbReleaseShadowSlot(qp_i, sl);
        }
    }
    
    // 减少inflight计数
    __sync_fetch_and_sub(&cc->inflight_chunks, 1);
}

// ============================================================================
// Completion路径集成
// ============================================================================

ncclResult_t ncclIbOnCompletionWithCC(struct ncclIbNetCommBase* commBase, 
                                       struct ibv_wc* wc, 
                                       int devIndex) {
    if (!NCCL_CC_IS_MY_WR(wc->wr_id)) {
#ifdef AIMD_DEBUG
        { static uint64_t last=0; uint64_t n=ncclIbGetMicros(); if (n-last>500000) { printf("[AIMD] OnComp skip: !IS_MY_WR wr_id=0x%llx\n", (unsigned long long)wc->wr_id); last=n; } }
#endif
        return ncclSuccess;
    }
    
    int qp_idx, cc_idx, chunk_id, slot;
    NCCL_CC_DECODE_WR_ID(wc->wr_id, &qp_idx, &cc_idx, &chunk_id, &slot);
    
    if (qp_idx < 0 || cc_idx < 0 || chunk_id < 0 || slot < 0 ||
        qp_idx >= NCCL_CC_MAX_QP || cc_idx >= NCCL_CC_MAX_COLLECTIVES ||
        slot >= NCCL_CC_MAX_SHADOW_SLOTS)
        return ncclSuccess;
    
    struct CollectiveCC* cc = g_cc_pool[cc_idx];
    
    uint64_t original_wr_id = 0;
    uint64_t send_time = 0;
    
    pthread_mutex_lock(&g_shadow_pool_mutex[qp_idx]);
    struct ShadowRequest* shadow = &g_shadow_pool[qp_idx][slot];
    if (shadow->active) {
        original_wr_id = shadow->original_wr_id;
        chunk_id = shadow->chunk_id;
        send_time = shadow->send_timestamp;
        if (cc && cc->chunk_tracker.chunk_original_wr_id && chunk_id >= 0 && chunk_id < cc->chunk_tracker.max_chunks)
            cc->chunk_tracker.chunk_original_wr_id[chunk_id] = original_wr_id;
        shadow->active = 0;  // 立即释放 Shadow，与 WR 绑定，不等 Chunk 完成
    }
    pthread_mutex_unlock(&g_shadow_pool_mutex[qp_idx]);
    
    if (original_wr_id == 0 && cc && cc->chunk_tracker.chunk_original_wr_id &&
        chunk_id >= 0 && chunk_id < cc->chunk_tracker.max_chunks)
        original_wr_id = cc->chunk_tracker.chunk_original_wr_id[chunk_id];
    
    wc->wr_id = original_wr_id;
    // NCCL 的 wr_id 来自 req 数组偏移 (reqs[r]-base.reqs)<<(r*8)，第 0 个偏移为 0，故 original_wr_id==0 合法；
    // 若此处用 original_wr_id==0 提前 return，会跳过 chunk_cqes_pending-- 与 FinalizeChunkRTT，导致 inflight 泄漏。
    if (chunk_id < 0 || !cc || !cc->enabled)
        return ncclSuccess;
    // 🔴 越界保护：chunk_id 必须 < max_chunks，否则 chunk_cqes_pending 等会踩堆
    if (chunk_id >= cc->chunk_tracker.max_chunks)
        return ncclSuccess;
    
    uint64_t now = ncclIbGetMicros();
    uint64_t wr_rtt = (send_time > 0 && now >= send_time) ? (now - send_time) : 0;
    ncclIbUpdateChunkRTTOnly(cc, chunk_id, wr_rtt);
    
    int prev = __sync_fetch_and_sub(&cc->chunk_tracker.chunk_cqes_pending[chunk_id], 1);
#ifdef AIMD_DEBUG
    if (prev <= 0)
        printf("[AIMD] BAD prev=%d cc=%p cid=%d\n", prev, (void*)cc, chunk_id);
    { static uint64_t last_p1=0; uint64_t n=ncclIbGetMicros(); if (prev==1 && n-last_p1>500000) { printf("[AIMD] prev==1 -> Finalize cc=%p cid=%d\n", (void*)cc, chunk_id); last_p1=n; } }
#endif
    if (prev == 1) {
        __sync_fetch_and_add(&cc->chunk_tracker.chunk_completed_wrs[chunk_id],
                            cc->chunk_tracker.chunk_wr_counts[chunk_id]);
        ncclIbFinalizeChunkRTT(cc, chunk_id);
    }
    return ncclSuccess;
}

// ============================================================================
// 初始化/清理
// ============================================================================

ncclResult_t ncclIbAIMDFinalize(void) {
    // 清理CC Pool
    if (g_cc_pool_initialized) {
        pthread_mutex_lock(&g_cc_pool_mutex);
        for (int i = 0; i < NCCL_CC_MAX_COLLECTIVES; i++) {
            if (g_cc_pool[i]) {
                struct CollectiveCC* cc = g_cc_pool[i];
                // 清理chunk_tracker内存
                if (cc->chunk_tracker.chunk_send_times) free(cc->chunk_tracker.chunk_send_times);
                if (cc->chunk_tracker.chunk_rtts) free(cc->chunk_tracker.chunk_rtts);
                if (cc->chunk_tracker.chunk_wr_counts) free(cc->chunk_tracker.chunk_wr_counts);
                if (cc->chunk_tracker.chunk_completed_wrs) free((void*)cc->chunk_tracker.chunk_completed_wrs);
                if (cc->chunk_tracker.chunk_status) free((void*)cc->chunk_tracker.chunk_status);
                if (cc->chunk_tracker.chunk_cqes_pending) free((void*)cc->chunk_tracker.chunk_cqes_pending);
                if (cc->chunk_tracker.chunk_original_wr_id) free(cc->chunk_tracker.chunk_original_wr_id);
                pthread_mutex_destroy(&cc->chunk_tracker.mutex);
                pthread_mutex_destroy(&cc->mutex);
                free(cc);
                g_cc_pool[i] = NULL;
            }
        }
        pthread_mutex_unlock(&g_cc_pool_mutex);
    }
    
    return ncclSuccess;
}
