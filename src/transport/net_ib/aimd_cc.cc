/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "aimd_cc.h"
#include "common.h"
#include "xccl_host_signal_snapshot.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>

// ============================================================================
// 全局变量
// ============================================================================

// Shadow Pool
struct ShadowRequest g_shadow_pool[NCCL_CC_MAX_QP][NCCL_CC_MAX_SHADOW_SLOTS];
int g_shadow_pool_next[NCCL_CC_MAX_QP];
uint32_t g_shadow_pool_shadow_gen_next[NCCL_CC_MAX_QP];
pthread_mutex_t g_shadow_pool_mutex[NCCL_CC_MAX_QP];
int g_shadow_pool_initialized = 0;

// CollectiveCC Pool
struct CollectiveCC* g_cc_pool[NCCL_CC_MAX_COLLECTIVES];
pthread_mutex_t g_cc_pool_mutex;
int g_cc_pool_initialized = 0;

// 全局状态
int g_aimd_enabled = -1;  // -1: 未检查, 0: 禁用, 1: 启用
int g_cc_metrics_enabled = -1;  // Phase 2 观测 NCCL_CC_METRICS
static int g_cc_epoch_enabled = -1;  // Phase 4 NCCL_CC_EPOCH_ENABLE
static pthread_t g_cc_control_thread;
static int g_cc_control_thread_started = 0;
static volatile int g_cc_control_thread_running = 0;
static uint64_t g_cc_control_plane_interval_us = 1000ULL; // 默认 1ms
static uint64_t g_cc_external_snapshot_ttl_ns = 20000000ULL; // 默认 20ms
static int g_aimd_recovery_only = -1; // 1=仅恢复态允许本地 AIMD 回退
static uint64_t g_cc_local_fallback_hold_ns = 50000000ULL; // 默认 50ms

// v2-minimal 全局参数（detector v2：降低 clean 段误触、加快恢复；仍可用环境变量覆盖）
static FILE* g_v2_timeline_fp = NULL;
static int   g_cc_v2_minimal = -1;
static float g_v2_pressure_thresh = 0.58f;  // 进入高压带：默认略提高，减少无 stress 时 SHRINK
static float g_v2_exit_thresh     = 0.30f;  // 低压带：略放宽，便于累积 low_cnt、出现 RECOVER
static int   g_v2_enter_epochs    = 14;     // 需更持续高压才进入 SHRINK
static int   g_v2_exit_epochs     = 14;       // 低压持续更少周期即可恢复（原 20 偏慢）
static float g_v2_beta            = 0.7f;
static int   g_v2_alpha           = 1;
static float g_v2_backlog_thresh  = 32.0f;
static float g_v2_window_floor    = 0.4f;
static int   g_v2_middle_high_decay = 1;      // 中间带每周期衰减 high_cnt，避免尖峰快速锁死 SHRINK

// 消融：oracle 固定 pacing / channels（控制面每周期写入；与 v2-minimal 互斥）
static int     g_oracle_fixed_actuator_inited = 0;
static int     g_oracle_fixed_actuator = 0;
static uint32_t g_oracle_fixed_pacing_ns = 0;
static int     g_oracle_fixed_channels = 0;

static void ncclIbOracleFixedActuatorInitOnce(void) {
    if (g_oracle_fixed_actuator_inited) return;
    g_oracle_fixed_actuator_inited = 1;
    const char* pe = getenv(NCCL_CC_ORACLE_PACING_NS_ENV);
    const char* ce = getenv(NCCL_CC_ORACLE_CHANNELS_ENV);
    if (pe && pe[0]) {
        uint64_t v = strtoull(pe, NULL, 10);
        if (v > 0ULL && v <= 1000000000ULL) g_oracle_fixed_pacing_ns = (uint32_t)v;
    }
    if (ce && ce[0]) {
        int n = atoi(ce);
        if (n > 0 && n <= 256) g_oracle_fixed_channels = n;
    }
    g_oracle_fixed_actuator = (g_oracle_fixed_pacing_ns > 0U || g_oracle_fixed_channels > 0) ? 1 : 0;
    if (g_oracle_fixed_actuator) {
        INFO(NCCL_ENV, "Oracle fixed actuator: pacing_ns=%u channels_cap=%d (0=use all QPs)",
             g_oracle_fixed_pacing_ns, g_oracle_fixed_channels);
    }
}

static inline int ncclCcLoadInt(const volatile int* p) {
    return __atomic_load_n(p, __ATOMIC_RELAXED);
}

static inline void ncclCcStoreInt(volatile int* p, int v) {
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
}

static inline uint32_t ncclCcLoadU32(const volatile uint32_t* p) {
    return __atomic_load_n(p, __ATOMIC_RELAXED);
}

static inline void ncclCcStoreU32(volatile uint32_t* p, uint32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
}

static inline uint64_t ncclCcLoadU64(const volatile uint64_t* p) {
    return __atomic_load_n(p, __ATOMIC_RELAXED);
}

static inline void ncclCcStoreU64(volatile uint64_t* p, uint64_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
}

static inline float ncclCcClamp01f(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

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

int ncclIbCcMetricsEnabled(void) {
    if (g_cc_metrics_enabled == -1) {
        const char* env = getenv(NCCL_CC_METRICS_ENV);
        g_cc_metrics_enabled = (env && atoi(env) > 0) ? 1 : 0;
    }
    return g_cc_metrics_enabled;
}

int ncclIbCcEpochEnabled(void) {
    if (g_cc_epoch_enabled == -1) {
        const char* env = getenv(NCCL_CC_EPOCH_ENABLE_ENV);
        g_cc_epoch_enabled = (env && atoi(env) > 0 && ncclIbIsAIMDEnabled()) ? 1 : 0;
    }
    return g_cc_epoch_enabled;
}

int ncclIbCcV2MinimalEnabled(void) {
    if (g_cc_v2_minimal == -1) {
        const char* e = getenv(NCCL_CC_V2_MINIMAL_ENV);
        g_cc_v2_minimal = (e && atoi(e) > 0 && ncclIbIsAIMDEnabled()) ? 1 : 0;
        if (g_cc_v2_minimal) {
            const char* v;
            v = getenv("NCCL_CC_V2_PRESSURE_THRESH"); if (v) { float f = (float)atof(v); if (f > 0.f && f <= 1.f) g_v2_pressure_thresh = f; }
            v = getenv("NCCL_CC_V2_EXIT_THRESH");     if (v) { float f = (float)atof(v); if (f > 0.f && f <= 1.f) g_v2_exit_thresh = f; }
            v = getenv("NCCL_CC_V2_ENTER_EPOCHS");    if (v) { int n = atoi(v); if (n >= 1 && n <= 1000) g_v2_enter_epochs = n; }
            v = getenv("NCCL_CC_V2_EXIT_EPOCHS");     if (v) { int n = atoi(v); if (n >= 1 && n <= 1000) g_v2_exit_epochs = n; }
            v = getenv("NCCL_CC_V2_BETA");             if (v) { float f = (float)atof(v); if (f > 0.f && f < 1.f) g_v2_beta = f; }
            v = getenv("NCCL_CC_V2_ALPHA");            if (v) { int n = atoi(v); if (n >= 1 && n <= 100) g_v2_alpha = n; }
            v = getenv("NCCL_CC_V2_BACKLOG_THRESH");   if (v) { float f = (float)atof(v); if (f > 0.f) g_v2_backlog_thresh = f; }
            v = getenv("NCCL_CC_V2_WINDOW_FLOOR");     if (v) { float f = (float)atof(v); if (f > 0.f && f <= 1.f) g_v2_window_floor = f; }
            v = getenv("NCCL_CC_V2_MIDDLE_HIGH_DECAY"); if (v) { g_v2_middle_high_decay = (atoi(v) > 0) ? 1 : 0; }
            if (g_v2_exit_thresh >= g_v2_pressure_thresh) {
                WARN("V2-minimal: exit_thresh >= pressure_thresh; clamping exit to pressure - 0.05");
                g_v2_exit_thresh = g_v2_pressure_thresh - 0.05f;
                if (g_v2_exit_thresh < 0.0f) g_v2_exit_thresh = 0.0f;
            }
            INFO(NCCL_ENV, "V2-minimal enabled: pressure_thresh=%.2f exit_thresh=%.2f enter=%d exit=%d beta=%.2f alpha=%d backlog_thresh=%.0f window_floor=%.2f middle_high_decay=%d",
                 g_v2_pressure_thresh, g_v2_exit_thresh, g_v2_enter_epochs, g_v2_exit_epochs,
                 g_v2_beta, g_v2_alpha, g_v2_backlog_thresh, g_v2_window_floor, g_v2_middle_high_decay);

            const char* tl = getenv("NCCL_CC_V2_TIMELINE_FILE");
            if (tl && tl[0]) {
                g_v2_timeline_fp = fopen(tl, "w");
                if (g_v2_timeline_fp) {
                    fprintf(g_v2_timeline_fp,
                            "ts_us,pressure,backlog_norm,deviation,state,window,floor,high_cnt,low_cnt,fast_ewma,slow_ewma\n");
                    fflush(g_v2_timeline_fp);
                    INFO(NCCL_ENV, "V2-minimal timeline logging to: %s", tl);
                }
            }
        }
    }
    return g_cc_v2_minimal;
}

static int ncclIbAimdRecoveryOnly(void) {
    if (g_aimd_recovery_only == -1) {
        const char* e = getenv(NCCL_AIMD_RECOVERY_ONLY_ENV);
        g_aimd_recovery_only = (e && atoi(e) > 0) ? 1 : 0;
    }
    return g_aimd_recovery_only;
}

static float ncclCcOracleFactor(void) {
    static float f = -1.0f;
    if (f < 0.0f) {
        const char* e = getenv("NCCL_CC_ORACLE_FACTOR");
        f = (e && e[0]) ? (float)atof(e) : 1.0f;
        if (f < 0.1f) f = 0.1f;
        if (f > 1.0f) f = 1.0f;
        if (f < 1.0f) {
            INFO(NCCL_ENV, "NCCL_CC_ORACLE_FACTOR=%.2f (oracle mode: window scaled)", f);
        }
    }
    return f;
}

int ncclCcGetEffectiveWindowForSend(const struct CollectiveCC* cc) {
    if (!cc) return 0;
    int ew = ncclCcLoadInt(&cc->effective_window);
    int ext = ncclCcLoadInt(&cc->external_control_active);
    int base;
    if (ew > 0 && (ext || ncclIbCcEpochEnabled())) {
        base = ew;
    } else {
        base = ncclCcLoadInt(&cc->lib_window);
    }
    float factor = ncclCcOracleFactor();
    if (factor < 1.0f) {
        int adjusted = (int)(base * factor);
        if (adjusted < cc->min_window) adjusted = cc->min_window;
        return adjusted;
    }
    return base;
}

int ncclCcGetEffectiveChannelsForSend(const struct CollectiveCC* cc, int total_qps) {
    if (!cc || total_qps <= 0) return total_qps;
    int ec = ncclCcLoadInt(&cc->effective_channels);
    if (ec <= 0) return total_qps; // 0=未发布，回退全 QP
    if (ec > total_qps) ec = total_qps;
    if (ec < 1) ec = 1;
    return ec;
}

uint32_t ncclCcGetEffectivePacingNsForSend(const struct CollectiveCC* cc) {
    if (!cc) return 0;
    return ncclCcLoadU32(&cc->effective_pacing_ns);
}

uint64_t ncclIbGetNanos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void ncclCcControlPlaneSleepUs(uint64_t us) {
    struct timespec ts, rem;
    ts.tv_sec = (time_t)(us / 1000000ULL);
    ts.tv_nsec = (long)((us % 1000000ULL) * 1000ULL);
    while (nanosleep(&ts, &rem) == -1 && errno == EINTR) {
        ts = rem;
    }
}

void ncclCcApplyPacingForSend(struct CollectiveCC* cc) {
    if (!cc) return;
    uint32_t pace_ns = ncclCcGetEffectivePacingNsForSend(cc);
    if (pace_ns == 0) return;

    for (;;) {
        uint64_t now = ncclIbGetNanos();
        uint64_t last = ncclCcLoadU64(&cc->last_inject_ns);

        if (last == 0) {
            if (__atomic_compare_exchange_n(&cc->last_inject_ns, &last, now, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                return;
            continue;
        }

        uint64_t target = last + (uint64_t)pace_ns;
        if (now >= target) {
            if (__atomic_compare_exchange_n(&cc->last_inject_ns, &last, now, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                return;
            continue;
        }

        uint64_t wait_ns = target - now;
        if (wait_ns > 100000ULL) wait_ns = 100000ULL; // 最长 100us，避免长阻塞
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = (long)wait_ns;
        nanosleep(&ts, NULL);
    }
}

static void ncclCcControlPlaneIntervalInitOnce(void) {
    static int done;
    if (done) return;
    done = 1;
    const char* e = getenv("NCCL_CC_CONTROL_PLANE_INTERVAL_US");
    if (!e || !e[0]) return;
    uint64_t v = (uint64_t)strtoull(e, NULL, 10);
    if (v >= 100ULL && v <= 1000000ULL) g_cc_control_plane_interval_us = v;
}

static void ncclCcExternalSnapshotTtlInitOnce(void) {
    static int done;
    if (done) return;
    done = 1;
    const char* e = getenv(NCCL_CC_CONTROL_SNAPSHOT_TTL_NS_ENV);
    if (!e || !e[0]) return;
    uint64_t v = (uint64_t)strtoull(e, NULL, 10);
    if (v >= 1000000ULL && v <= 10000000000ULL) g_cc_external_snapshot_ttl_ns = v;
}

static void ncclCcLocalFallbackHoldInitOnce(void) {
    static int done;
    if (done) return;
    done = 1;
    const char* e = getenv(NCCL_CC_LOCAL_FALLBACK_HOLD_NS_ENV);
    if (!e || !e[0]) return;
    uint64_t v = (uint64_t)strtoull(e, NULL, 10);
    if (v >= 1000000ULL && v <= 10000000000ULL) g_cc_local_fallback_hold_ns = v;
}

static uint64_t g_hint_refresh_min_ns = 10000000ULL; /* 10ms 默认 */
static void ncclCcHintEnsureRefreshMinOnce(void) {
    static int done;
    if (done) return;
    done = 1;
    const char* e = getenv(NCCL_CC_HINT_REFRESH_MIN_NS_ENV);
    if (e && e[0]) {
        uint64_t v = (uint64_t)strtoull(e, NULL, 10);
        if (v < 1000000ULL) g_hint_refresh_min_ns = v * 1000000ULL;
        else g_hint_refresh_min_ns = v;
    }
}

int ncclIbCcHintEnabled(void) {
    static int g = -1;
    if (g == -1) {
        const char* e = getenv(NCCL_CC_HINT_ENABLE_ENV);
        g = (e && atoi(e) > 0) ? 1 : 0;
    }
    return g && xcclTelemetryHintTransportIsMapped();
}

static void ncclCcRecomputeHintValid(struct CollectiveCC* cc, uint64_t now_ns) {
    if (!cc->hint_read_ok || cc->hint.version == 0) {
        cc->hint_valid = 0;
        return;
    }
    if (now_ns - cc->hint.ts_ns > cc->hint_ttl_ns) {
        cc->hint_valid = 0;
        return;
    }
    cc->hint_valid = 1;
}

void ncclCcOnCollectiveBegin(struct CollectiveCC* cc, uint64_t now_ns) {
    if (!cc || !cc->enabled) return;
    if (!ncclIbCcHintEnabled()) return;
    ncclCcHintEnsureRefreshMinOnce();

    pthread_mutex_lock(&cc->mutex);
    ncclCcRecomputeHintValid(cc, now_ns);
    if (cc->hint_last_refresh_ns != 0 && (now_ns - cc->hint_last_refresh_ns) < g_hint_refresh_min_ns) {
        pthread_mutex_unlock(&cc->mutex);
        return;
    }
    pthread_mutex_unlock(&cc->mutex);

    TelemetryHintSnapshot snap;
    ncclResult_t r = xcclTelemetryHintReadSnapshot(&snap);

    pthread_mutex_lock(&cc->mutex);
    if (r == ncclSuccess) {
        memcpy(&cc->hint, &snap, sizeof(snap));
        cc->hint_read_ok = 1;
        cc->hint_last_refresh_ns = now_ns;
    } else {
        cc->hint_read_failures++;
        cc->hint_read_ok = 0;
    }
    ncclCcRecomputeHintValid(cc, now_ns);
    pthread_mutex_unlock(&cc->mutex);
}

void ncclCcRefreshHintIfNeeded(struct CollectiveCC* cc, uint64_t now_ns) {
    ncclCcOnCollectiveBegin(cc, now_ns);
}

int ncclCcHintIsValid(const struct CollectiveCC* cc, uint64_t now_ns) {
    if (!cc) return 0;
    struct CollectiveCC* c = (struct CollectiveCC*)cc;
    pthread_mutex_lock(&c->mutex);
    ncclCcRecomputeHintValid(c, now_ns);
    int v = c->hint_valid;
    pthread_mutex_unlock(&c->mutex);
    return v;
}

static void ncclIbUpdateLIWLocal(struct CollectiveCC* cc);

static int ncclIbExternalControlActive(void) {
    return xcclControlSnapshotEnabled() && xcclControlSnapshotTransportIsMapped();
}

static void ncclCcApplyExternalSnapshot(struct CollectiveCC* cc, const XcclControlSnapshot* snap, int snap_ok, uint64_t now_ns) {
    if (!cc) return;
    pthread_mutex_lock(&cc->mutex);

    int applied = 0;
    if (snap_ok && snap) {
        uint64_t snap_comm = snap->comm_key;
        uint64_t cc_comm = (uint64_t)cc->comm_key;
        uint64_t age_ns = (now_ns >= snap->ts_ns) ? (now_ns - snap->ts_ns) : 0;
        if ((snap_comm == 0ULL || snap_comm == cc_comm) &&
            age_ns <= g_cc_external_snapshot_ttl_ns && snap->target_window > 0) {
            int tw = (int)snap->target_window;
            if (tw < cc->min_window) tw = cc->min_window;
            if (tw > cc->max_window) tw = cc->max_window;
            if (tw > cc->device_cap) tw = cc->device_cap;
            if (tw > cc->chunk_tracker.max_chunks) tw = cc->chunk_tracker.max_chunks;

            ncclCcStoreInt(&cc->lib_window, tw);
            ncclCcStoreInt(&cc->effective_window, tw);
            ncclCcStoreInt(&cc->effective_channels, (int)snap->target_channels);
            ncclCcStoreU32(&cc->effective_pacing_ns, snap->pacing_ns);
            ncclCcStoreInt(&cc->external_control_active, 1);
            cc->external_snapshot_ts_ns = snap->ts_ns;
            cc->external_target_window = snap->target_window;
            cc->external_target_channels = snap->target_channels;
            cc->external_pacing_ns = snap->pacing_ns;
            applied = 1;
        }
    }

    if (!applied) {
        ncclCcStoreInt(&cc->external_control_active, 0);
        ncclCcStoreInt(&cc->effective_window, ncclCcLoadInt(&cc->lib_window));
        ncclCcStoreInt(&cc->effective_channels, 0);
        ncclCcStoreU32(&cc->effective_pacing_ns, 0);
    }
    pthread_mutex_unlock(&cc->mutex);
}

static void* ncclCcControlPlaneThreadMain(void* arg) {
    (void)arg;
    int recovery_only = ncclIbAimdRecoveryOnly();
    ncclCcLocalFallbackHoldInitOnce();
    float agg_backlog_ewma = 0.0f;
    uint64_t prev_agg_posted = 0ULL;
    uint64_t prev_agg_completed = 0ULL;
    uint64_t last_progress_ts_ns = 0ULL;
    while (g_cc_control_thread_running) {
        struct CollectiveCC* local_ccs[NCCL_CC_MAX_COLLECTIVES];
        int ncc = 0;

        pthread_mutex_lock(&g_cc_pool_mutex);
        for (int i = 0; i < NCCL_CC_MAX_COLLECTIVES; i++) {
            if (g_cc_pool[i] && g_cc_pool[i]->enabled) {
                local_ccs[ncc++] = g_cc_pool[i];
            }
        }
        pthread_mutex_unlock(&g_cc_pool_mutex);

        int external_active = ncclIbExternalControlActive();
        XcclControlSnapshot snap;
        int snap_ok = 0;
        uint64_t now_ns = ncclIbGetNanos();
        if (external_active && xcclControlSnapshotRead(&snap) == ncclSuccess) snap_ok = 1;

        uint64_t agg_posted = 0;
        uint64_t agg_completed = 0;
        uint64_t agg_backlog_sum = 0ULL;
        uint32_t agg_backlog_max = 0;
        uint32_t agg_rtt_baseline = 0;
        uint32_t agg_rtt_ewma = 0;
        float agg_stretch = 1.0f;

        int v2_active = ncclIbCcV2MinimalEnabled();
        for (int i = 0; i < ncc; i++) {
            struct CollectiveCC* cc = local_ccs[i];
            if (v2_active) {
                ncclCcStoreInt(&cc->external_control_active, 0);
            } else if (external_active) {
                ncclCcApplyExternalSnapshot(cc, &snap, snap_ok, now_ns);
                int ext_applied = ncclCcLoadInt(&cc->external_control_active);
                if (!ext_applied) {
                    int allow_local = 1;
                    if (recovery_only) {
                        allow_local = 0;
                    } else {
                        uint64_t last_ext = cc->external_snapshot_ts_ns;
                        if (last_ext > 0 && now_ns > last_ext &&
                            now_ns - last_ext < g_cc_local_fallback_hold_ns) {
                            allow_local = 0;
                        }
                    }
                    if (allow_local) {
                        ncclIbUpdateLIWLocal(cc);
                    }
                }
            } else {
                ncclCcStoreInt(&cc->external_control_active, 0);
                if (!recovery_only) ncclIbUpdateLIWLocal(cc);
            }

            uint64_t posted = ncclCcLoadU64(&cc->cq_posted_total);
            uint64_t completed = ncclCcLoadU64(&cc->cq_completed_total);
            agg_posted += posted;
            agg_completed += completed;

            uint64_t backlog64 = (posted > completed) ? (posted - completed) : 0ULL;
            agg_backlog_sum += backlog64;
            if (backlog64 > (uint64_t)agg_backlog_max) {
                agg_backlog_max = (backlog64 > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)backlog64;
            }

            pthread_mutex_lock(&cc->mutex);
            uint32_t base = (cc->rtt_baseline > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)cc->rtt_baseline;
            uint32_t ewma = (cc->rtt_ewma < 0.0) ? 0U :
                (cc->rtt_ewma > (double)0xFFFFFFFFU ? 0xFFFFFFFFU : (uint32_t)cc->rtt_ewma);
            float stretch = 1.0f;
            if (base > 0) {
                stretch = (float)((double)ewma / (double)base);
                if (stretch < 0.0f) stretch = 0.0f;
            }
            pthread_mutex_unlock(&cc->mutex);

            if (stretch >= agg_stretch) {
                agg_stretch = stretch;
                agg_rtt_baseline = base;
                agg_rtt_ewma = ewma;
            }
        }

        // 消融实验：固定 pacing / channel 上限（不缩窗；与 v2-minimal 互斥）
        ncclIbOracleFixedActuatorInitOnce();
        if (g_oracle_fixed_actuator && !ncclIbCcV2MinimalEnabled() && ncc > 0) {
            for (int i = 0; i < ncc; i++) {
                struct CollectiveCC* cc = local_ccs[i];
                if (g_oracle_fixed_pacing_ns > 0U) {
                    ncclCcStoreU32(&cc->effective_pacing_ns, g_oracle_fixed_pacing_ns);
                } else {
                    ncclCcStoreU32(&cc->effective_pacing_ns, 0);
                }
                if (g_oracle_fixed_channels > 0) {
                    ncclCcStoreInt(&cc->effective_channels, g_oracle_fixed_channels);
                } else {
                    ncclCcStoreInt(&cc->effective_channels, 0);
                }
            }
        }

        uint32_t agg_backlog = (agg_backlog_sum > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)agg_backlog_sum;
        float backlog_now = (float)agg_backlog;
        if (agg_backlog_ewma == 0.0f) {
            agg_backlog_ewma = backlog_now;
        } else {
            agg_backlog_ewma = 0.2f * backlog_now + 0.8f * agg_backlog_ewma;
        }

        uint64_t delta_posted = (agg_posted >= prev_agg_posted) ? (agg_posted - prev_agg_posted) : 0ULL;
        uint64_t delta_completed = (agg_completed >= prev_agg_completed) ? (agg_completed - prev_agg_completed) : 0ULL;
        float completion_drain_rate = 1.0f;
        if (delta_posted > 0ULL) {
            completion_drain_rate = (float)delta_completed / (float)delta_posted;
        } else if (delta_completed == 0ULL) {
            completion_drain_rate = 0.0f;
        } else {
            completion_drain_rate = 1.0f;
        }
        if (completion_drain_rate < 0.0f) completion_drain_rate = 0.0f;
        if (completion_drain_rate > 4.0f) completion_drain_rate = 4.0f;

        if (delta_completed > 0ULL) last_progress_ts_ns = now_ns;
        if (last_progress_ts_ns == 0ULL) last_progress_ts_ns = now_ns;
        uint64_t no_progress_ns = (now_ns >= last_progress_ts_ns) ? (now_ns - last_progress_ts_ns) : 0ULL;
        uint64_t base_gap_ns = g_cc_control_plane_interval_us * 1000ULL * 5ULL; // 5 个控制周期
        if (base_gap_ns == 0ULL) base_gap_ns = 1ULL;
        float poll_gap_norm = ncclCcClamp01f((float)no_progress_ns / (float)base_gap_ns);

        prev_agg_posted = agg_posted;
        prev_agg_completed = agg_completed;

        // ================================================================
        // v2-minimal 分支：纯 NCCL 内部自闭环窗口控制
        // ================================================================
        if (ncclIbCcV2MinimalEnabled()) {
            float backlog_norm = ncclCcClamp01f(agg_backlog_ewma / g_v2_backlog_thresh);

            // 快慢双 EWMA 自适应基线检测
            // fast: tau ≈ 50 epochs (50ms)，快速跟踪短期变化
            // slow: tau ≈ 2000 epochs (2s)，代表长期基线水平
            static float v2_fast_ewma = -1.0f;
            static float v2_slow_ewma = -1.0f;
            static int v2_warmup_count = 0;

            if (v2_fast_ewma < 0.0f) {
                v2_fast_ewma = backlog_norm;
                v2_slow_ewma = backlog_norm;
            }
            v2_fast_ewma = 0.02f * backlog_norm + 0.98f * v2_fast_ewma;

            if (v2_warmup_count < 500) { v2_warmup_count++; }

            float deviation = v2_fast_ewma - v2_slow_ewma;
            // warmup 期间抑制信号（前 500ms baseline 尚未稳定）
            float v2_pressure = (v2_warmup_count >= 500) ?
                ncclCcClamp01f(deviation / 0.04f) : 0.0f;

            static int v2_high_count = 0;
            static int v2_low_count = 0;

            if (v2_pressure >= g_v2_pressure_thresh) {
                v2_high_count++;
                v2_low_count = 0;
            } else if (v2_pressure <= g_v2_exit_thresh) {
                v2_low_count++;
                v2_high_count = 0;
            } else {
                // 中间区域 (exit_thresh, pressure_thresh)：默认每周期衰减 high_cnt，避免短时偏差反复累积成 SHRINK
                if (g_v2_middle_high_decay && v2_high_count > 0) {
                    v2_high_count--;
                }
            }

            int v2_should_shrink  = (v2_high_count >= g_v2_enter_epochs);
            int v2_should_recover = (v2_low_count  >= g_v2_exit_epochs);

            // 仅在非 SHRINK 态更新 slow baseline，避免基线被异常期 backlog 拉高
            if (!v2_should_shrink) {
                v2_slow_ewma = 0.0005f * backlog_norm + 0.9995f * v2_slow_ewma;
            }

            for (int i = 0; i < ncc; i++) {
                struct CollectiveCC* cc = local_ccs[i];
                int cur_w = ncclCcLoadInt(&cc->lib_window);
                int new_w = cur_w;
                int floor_w = (int)((float)cc->max_window * g_v2_window_floor);
                if (floor_w < cc->min_window) floor_w = cc->min_window;

                if (v2_should_shrink) {
                    new_w = (int)((float)cur_w * g_v2_beta);
                } else if (v2_should_recover) {
                    int gap = cc->max_window - cur_w;
                    int step = g_v2_alpha + gap / 16;
                    new_w = cur_w + step;
                    if (new_w > cc->max_window) new_w = cc->max_window;
                }

                if (new_w < floor_w) new_w = floor_w;

                ncclCcStoreInt(&cc->lib_window, new_w);
                ncclCcStoreInt(&cc->effective_window, new_w);
                ncclCcStoreInt(&cc->effective_channels, 0);
                ncclCcStoreU32(&cc->effective_pacing_ns, 0);
            }

            {
                int w0 = (ncc > 0) ? ncclCcLoadInt(&local_ccs[0]->effective_window) : -1;
                int fl0 = (ncc > 0) ? (int)((float)local_ccs[0]->max_window * g_v2_window_floor) : 0;
                const char* state = v2_should_shrink ? "SHRINK" : (v2_should_recover ? "RECOVER" : "HOLD");

                if (g_v2_timeline_fp) {
                    uint64_t ts_us = now_ns / 1000ULL;
                    fprintf(g_v2_timeline_fp,
                            "%lu,%.4f,%.4f,%.5f,%s,%d,%d,%d,%d,%.4f,%.4f\n",
                            (unsigned long)ts_us, v2_pressure, backlog_norm,
                            deviation, state, w0, fl0, v2_high_count, v2_low_count,
                            v2_fast_ewma, v2_slow_ewma);
                }

                static uint64_t v2_last_log_us = 0;
                uint64_t v2_now_us = ncclIbGetMicros();
                if (v2_last_log_us == 0 || v2_now_us - v2_last_log_us >= 5000000ULL) {
                    INFO(NCCL_NET, "V2MIN: pressure=%.3f (fast=%.3f slow=%.3f dev=%.4f) state=%s window=%d floor=%d high=%d low=%d ncc=%d",
                         v2_pressure, v2_fast_ewma, v2_slow_ewma, deviation, state, w0, fl0,
                         v2_high_count, v2_low_count, ncc);
                    v2_last_log_us = v2_now_us;
                }
            }

            // 发布 HostSignalSnapshot（供外部观测，不做控制输入）
            if (xcclHostSignalSnapshotEnabled() && xcclHostSignalSnapshotIsMapped()) {
                XcclHostSignalSnapshot hs;
                memset(&hs, 0, sizeof(hs));
                hs.magic = XCCL_HOST_SIGNAL_MAGIC;
                hs.layout_version = XCCL_HOST_SIGNAL_LAYOUT_VERSION;
                hs.struct_size = (uint16_t)sizeof(XcclHostSignalSnapshot);
                hs.comm_key = 0ULL;
                hs.ts_ns = now_ns;
                hs.cq_posted = (agg_posted > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)agg_posted;
                hs.cq_completed = (agg_completed > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)agg_completed;
                hs.cq_backlog = agg_backlog;
                hs.cq_backlog_max = agg_backlog_max;
                hs.rtt_baseline_us = agg_rtt_baseline;
                hs.rtt_ewma_us = agg_rtt_ewma;
                hs.completion_stretch = agg_stretch;
                hs.cpu_poll_delay_norm = 0.0f;
                hs.cq_backlog_ewma = agg_backlog_ewma;
                hs.completion_drain_rate = completion_drain_rate;
                hs.poll_gap_norm = poll_gap_norm;
                (void)xcclHostSignalSnapshotPublish(&hs);
            }

            ncclCcControlPlaneSleepUs(g_cc_control_plane_interval_us);
            continue; // 跳过后续 full-system 路径
        }
        // ================================================================
        // 以下为现有 full-system 路径（v2-minimal 不走）
        // ================================================================
        if (xcclHostSignalSnapshotEnabled() && xcclHostSignalSnapshotIsMapped()) {
            XcclHostSignalSnapshot hs;
            memset(&hs, 0, sizeof(hs));
            hs.magic = XCCL_HOST_SIGNAL_MAGIC;
            hs.layout_version = XCCL_HOST_SIGNAL_LAYOUT_VERSION;
            hs.struct_size = (uint16_t)sizeof(XcclHostSignalSnapshot);
            hs.comm_key = 0ULL;
            hs.ts_ns = now_ns;
            hs.cq_posted = (agg_posted > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)agg_posted;
            hs.cq_completed = (agg_completed > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)agg_completed;
            hs.cq_backlog = agg_backlog;
            hs.cq_backlog_max = agg_backlog_max;
            hs.rtt_baseline_us = agg_rtt_baseline;
            hs.rtt_ewma_us = agg_rtt_ewma;
            hs.completion_stretch = agg_stretch;
            hs.cpu_poll_delay_norm = 0.0f;
            hs.cq_backlog_ewma = agg_backlog_ewma;
            hs.completion_drain_rate = completion_drain_rate;
            hs.poll_gap_norm = poll_gap_norm;
            (void)xcclHostSignalSnapshotPublish(&hs);
        }

        ncclCcControlPlaneSleepUs(g_cc_control_plane_interval_us);
    }
    return NULL;
}

// Phase 2：仅在合法 CQE（pending-- 后 prev>0）路径更新；与 chunk_rtts（max）同源更新顺序
static void ncclIbChunkObservationOnValidCqe(struct CollectiveCC* cc, int chunk_slot, uint64_t wr_rtt, uint64_t now) {
    ncclIbUpdateChunkRTTOnly(cc, chunk_slot, wr_rtt);
    if (!ncclIbCcMetricsEnabled()) return;
    if (!cc || chunk_slot < 0 || chunk_slot >= cc->chunk_tracker.max_chunks) return;

    __atomic_fetch_add(&cc->chunk_tracker.chunk_lat_sum_us[chunk_slot], wr_rtt, __ATOMIC_RELAXED);
    __atomic_fetch_add(&cc->chunk_tracker.chunk_lat_samples[chunk_slot], 1u, __ATOMIC_RELAXED);

    uint64_t* minp = &cc->chunk_tracker.chunk_lat_min_us[chunk_slot];
    uint64_t oldm, newm;
    do {
        oldm = __atomic_load_n(minp, __ATOMIC_RELAXED);
        if (wr_rtt >= oldm) break;
        newm = wr_rtt;
    } while (!__atomic_compare_exchange_n(minp, &oldm, newm, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

    uint64_t* fp = &cc->chunk_tracker.chunk_first_cqe_us[chunk_slot];
    uint64_t oldf, nf;
    do {
        oldf = __atomic_load_n(fp, __ATOMIC_RELAXED);
        nf = (oldf == 0) ? now : ((oldf < now) ? oldf : now);
    } while (!__atomic_compare_exchange_n(fp, &oldf, nf, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

    uint64_t* lp = &cc->chunk_tracker.chunk_last_cqe_us[chunk_slot];
    uint64_t oldl, newl;
    do {
        oldl = __atomic_load_n(lp, __ATOMIC_RELAXED);
        newl = (now > oldl) ? now : oldl;
    } while (!__atomic_compare_exchange_n(lp, &oldl, newl, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

static void ncclIbAppendMetricsBucket(struct CollectiveCC* cc, const struct NcclCcChunkSample* s) {
    if (!cc || !s) return;
    pthread_mutex_lock(&cc->mutex);
    cc->metrics_bucket_sum_avg_us += s->avg_completion_us;
    cc->metrics_bucket_sum_max_us += s->max_completion_us;
    cc->metrics_bucket_sum_span_us += s->span_us;
    cc->metrics_bucket_sum_service_us += s->service_time_us;
    cc->metrics_bucket_chunk_count++;
    cc->metrics_bucket_last_finalize_us = ncclIbGetMicros();
    pthread_mutex_unlock(&cc->mutex);
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
        g_shadow_pool_shadow_gen_next[qp] = 0;
        pthread_mutex_init(&g_shadow_pool_mutex[qp], NULL);
    }
    
    g_shadow_pool_initialized = 1;
    return ncclSuccess;
}

int ncclIbAllocateShadowSlot(int qp_index, uint64_t original_wr_id,
                              int cc_index, int chunk_slot, uint64_t chunk_generation) {
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
            g_shadow_pool[qp_index][slot].chunk_slot = (uint32_t)chunk_slot;
            g_shadow_pool[qp_index][slot].chunk_generation = chunk_generation;
            g_shadow_pool[qp_index][slot].shadow_generation = g_shadow_pool_shadow_gen_next[qp_index]++;
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
    cc->hint_ttl_ns = 100000000ULL;
    {
        const char* htt = getenv(NCCL_CC_HINT_TTL_NS_ENV);
        if (htt && htt[0]) {
            uint64_t v = (uint64_t)strtoull(htt, NULL, 10);
            if (v >= 1000000ULL && v <= 10000000000ULL) cc->hint_ttl_ns = v;
        }
    }

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
    ncclCcStoreInt(&cc->lib_window, lib_w);
    cc->inflight_chunks = 0;
    cc->alpha = alpha_val;
    cc->beta = beta_val;
    cc->update_interval_us = uiu;
    cc->last_update_time = 0;

    cc->device_cap = max_w;
    cc->epoch_interval_ns = 1000000ULL;
    ncclCcStoreInt(&cc->effective_channels, 0);
    ncclCcStoreU32(&cc->effective_pacing_ns, 0);
    ncclCcStoreU64(&cc->last_inject_ns, 0ULL);
    ncclCcStoreU64(&cc->cq_posted_total, 0ULL);
    ncclCcStoreU64(&cc->cq_completed_total, 0ULL);
    ncclCcStoreInt(&cc->external_control_active, 0);
    cc->external_snapshot_ts_ns = 0;
    cc->external_target_window = 0;
    cc->external_target_channels = 0;
    cc->external_pacing_ns = 0;
    {
        const char* ein = getenv(NCCL_CC_EPOCH_INTERVAL_NS_ENV);
        if (ein && ein[0]) {
            uint64_t vn = (uint64_t)strtoull(ein, NULL, 10);
            if (vn >= 100000ULL && vn <= 1000000000ULL) cc->epoch_interval_ns = vn;
        }
    }

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
    if (max_chunks < 1) max_chunks = 1;
    if (max_chunks > 1024) {
        WARN("NET/IB: estimated_chunks=%d exceeds WR-ID chunk_slot bits, clamp to 1024", max_chunks);
        max_chunks = 1024;
    }
    cc->chunk_tracker.max_chunks = max_chunks;
    // I0/R1：在途 chunk 不应超过可区分 slot 数（见设计 §5.1）
    if (ncclCcLoadInt(&cc->lib_window) > max_chunks) ncclCcStoreInt(&cc->lib_window, max_chunks);
    cc->chunk_tracker.chunk_send_times = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_rtts = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_wr_counts = (int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_completed_wrs = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_status = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_cqes_pending = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_finalized_flag = (volatile int*)malloc(max_chunks * sizeof(int));
    cc->chunk_tracker.chunk_generation = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_original_wr_id = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_first_cqe_us = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_last_cqe_us = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_lat_sum_us = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_lat_min_us = (uint64_t*)malloc(max_chunks * sizeof(uint64_t));
    cc->chunk_tracker.chunk_lat_samples = (uint32_t*)malloc(max_chunks * sizeof(uint32_t));
    
    if (!cc->chunk_tracker.chunk_send_times || !cc->chunk_tracker.chunk_rtts ||
        !cc->chunk_tracker.chunk_wr_counts || !cc->chunk_tracker.chunk_completed_wrs ||
        !cc->chunk_tracker.chunk_status || !cc->chunk_tracker.chunk_cqes_pending ||
        !cc->chunk_tracker.chunk_finalized_flag || !cc->chunk_tracker.chunk_generation ||
        !cc->chunk_tracker.chunk_original_wr_id ||
        !cc->chunk_tracker.chunk_first_cqe_us || !cc->chunk_tracker.chunk_last_cqe_us ||
        !cc->chunk_tracker.chunk_lat_sum_us || !cc->chunk_tracker.chunk_lat_min_us ||
        !cc->chunk_tracker.chunk_lat_samples) {
        free(cc->chunk_tracker.chunk_send_times);
        free(cc->chunk_tracker.chunk_rtts);
        free(cc->chunk_tracker.chunk_wr_counts);
        free((void*)cc->chunk_tracker.chunk_completed_wrs);
        free((void*)cc->chunk_tracker.chunk_status);
        free((void*)cc->chunk_tracker.chunk_cqes_pending);
        free((void*)cc->chunk_tracker.chunk_finalized_flag);
        free(cc->chunk_tracker.chunk_generation);
        free(cc->chunk_tracker.chunk_original_wr_id);
        free(cc->chunk_tracker.chunk_first_cqe_us);
        free(cc->chunk_tracker.chunk_last_cqe_us);
        free(cc->chunk_tracker.chunk_lat_sum_us);
        free(cc->chunk_tracker.chunk_lat_min_us);
        free(cc->chunk_tracker.chunk_lat_samples);
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
    memset((void*)cc->chunk_tracker.chunk_finalized_flag, 0, max_chunks * sizeof(int));
    memset(cc->chunk_tracker.chunk_generation, 0, max_chunks * sizeof(uint64_t));
    memset(cc->chunk_tracker.chunk_original_wr_id, 0, max_chunks * sizeof(uint64_t));
    memset(cc->chunk_tracker.chunk_first_cqe_us, 0, max_chunks * sizeof(uint64_t));
    memset(cc->chunk_tracker.chunk_last_cqe_us, 0, max_chunks * sizeof(uint64_t));
    memset(cc->chunk_tracker.chunk_lat_sum_us, 0, max_chunks * sizeof(uint64_t));
    for (int zi = 0; zi < max_chunks; zi++)
        cc->chunk_tracker.chunk_lat_min_us[zi] = UINT64_MAX;
    memset(cc->chunk_tracker.chunk_lat_samples, 0, max_chunks * sizeof(uint32_t));
    
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

void ncclIbRecordChunkSend(struct CollectiveCC* cc, int chunk_slot, uint64_t chunk_generation, int num_wrs) {
    if (!cc || chunk_slot < 0 || chunk_slot >= cc->chunk_tracker.max_chunks) return;
    
    // CAS：仅抢到 0->1 或 2->1 的调用做初始化，防止重入重复初始化同一个 slot。
    int* status_ptr = (int*)&cc->chunk_tracker.chunk_status[chunk_slot];
    int expected = 0;
    if (__atomic_compare_exchange_n(status_ptr, &expected, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        // 从 idle 首次初始化
    } else {
        expected = 2;  // 复用：从 completed 再入
        if (!__atomic_compare_exchange_n(status_ptr, &expected, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED))
            return;  // 已是 1 (in-flight)，其他线程已初始化，直接返回
    }
    cc->chunk_tracker.chunk_generation[chunk_slot] = chunk_generation;
    cc->chunk_tracker.chunk_finalized_flag[chunk_slot] = 0;
    cc->chunk_tracker.chunk_rtts[chunk_slot] = 0;
    cc->chunk_tracker.chunk_wr_counts[chunk_slot] = num_wrs;
    cc->chunk_tracker.chunk_completed_wrs[chunk_slot] = 0;
    cc->chunk_tracker.chunk_cqes_pending[chunk_slot] = 0;
    cc->chunk_tracker.chunk_send_times[chunk_slot] = ncclIbGetMicros();
    if (cc->chunk_tracker.chunk_first_cqe_us) {
        cc->chunk_tracker.chunk_first_cqe_us[chunk_slot] = 0;
        cc->chunk_tracker.chunk_last_cqe_us[chunk_slot] = 0;
        cc->chunk_tracker.chunk_lat_sum_us[chunk_slot] = 0;
        cc->chunk_tracker.chunk_lat_min_us[chunk_slot] = UINT64_MAX;
        cc->chunk_tracker.chunk_lat_samples[chunk_slot] = 0;
    }
#ifdef AIMD_DEBUG
    /* 短期放宽 rate-limit 至 50ms，便于确认 cid 是否 0,1,2… 轮转；确认后可改回 500000(500ms) */
    { static uint64_t last=0; uint64_t n=ncclIbGetMicros();
      if (n-last>50000) {
        printf("[AIMD] RecordChunkSend: cc=%p slot=%d gen=%lu cqes_pending=%d\n",
               (void*)cc, chunk_slot, (unsigned long)chunk_generation, cc->chunk_tracker.chunk_cqes_pending[chunk_slot]);
        last=n;
      } }
#endif
}

void ncclIbUpdateChunkWRRTT(struct CollectiveCC* cc, int chunk_slot, uint64_t wr_rtt) {
    if (!cc || chunk_slot < 0 || chunk_slot >= cc->chunk_tracker.max_chunks) return;
    
    ncclIbUpdateChunkRTTOnly(cc, chunk_slot, wr_rtt);
    // 原子增加完成计数（仅当每个 WR 对应一个 CQE 时使用；Multi-QP 下改用 chunk_cqes_pending）
    __sync_fetch_and_add(&cc->chunk_tracker.chunk_completed_wrs[chunk_slot], 1);
}

void ncclIbUpdateChunkRTTOnly(struct CollectiveCC* cc, int chunk_slot, uint64_t wr_rtt) {
    if (!cc || chunk_slot < 0 || chunk_slot >= cc->chunk_tracker.max_chunks) return;
    
    // 使用CAS更新chunk RTT（取max），无锁
    uint64_t* chunk_rtt_ptr = &cc->chunk_tracker.chunk_rtts[chunk_slot];
    uint64_t old_rtt, new_rtt;
    do {
        old_rtt = __atomic_load_n(chunk_rtt_ptr, __ATOMIC_RELAXED);
        if (wr_rtt <= old_rtt) break;
        new_rtt = wr_rtt;
    } while (!__atomic_compare_exchange_n(chunk_rtt_ptr, &old_rtt, new_rtt, 
                                           0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

int ncclIbIsChunkComplete(struct CollectiveCC* cc, int chunk_slot) {
    if (!cc || chunk_slot < 0 || chunk_slot >= cc->chunk_tracker.max_chunks) return 0;
    
    pthread_mutex_lock(&cc->chunk_tracker.mutex);
    int complete = (cc->chunk_tracker.chunk_status[chunk_slot] == 1 &&
                    cc->chunk_tracker.chunk_completed_wrs[chunk_slot] >= 
                    cc->chunk_tracker.chunk_wr_counts[chunk_slot]);
    pthread_mutex_unlock(&cc->chunk_tracker.mutex);
    
    return complete;
}

void ncclIbFinalizeChunkRTT(struct CollectiveCC* cc, int chunk_slot) {
    if (!cc || chunk_slot < 0 || chunk_slot >= cc->chunk_tracker.max_chunks) return;
    
    uint64_t chunk_rtt = 0;
    int did_finalize = 0;
    struct NcclCcChunkSample metrics_sample;
    memset(&metrics_sample, 0, sizeof(metrics_sample));
    int have_metrics_sample = 0;

    // exactly-once finalize：CAS 抢到唯一执行权才 decrement inflight。
    if (!__sync_bool_compare_and_swap(&cc->chunk_tracker.chunk_finalized_flag[chunk_slot], 0, 1)) {
        return;
    }
    
    pthread_mutex_lock(&cc->chunk_tracker.mutex);
    if (cc->chunk_tracker.chunk_status[chunk_slot] == 1) {
        chunk_rtt = cc->chunk_tracker.chunk_rtts[chunk_slot];
        if (ncclIbCcMetricsEnabled() && cc->chunk_tracker.chunk_lat_samples) {
            uint64_t sum = __atomic_load_n(&cc->chunk_tracker.chunk_lat_sum_us[chunk_slot], __ATOMIC_RELAXED);
            uint32_t smps = __atomic_load_n(&cc->chunk_tracker.chunk_lat_samples[chunk_slot], __ATOMIC_RELAXED);
            uint64_t f = __atomic_load_n(&cc->chunk_tracker.chunk_first_cqe_us[chunk_slot], __ATOMIC_RELAXED);
            uint64_t l = __atomic_load_n(&cc->chunk_tracker.chunk_last_cqe_us[chunk_slot], __ATOMIC_RELAXED);
            uint64_t minlat = __atomic_load_n(&cc->chunk_tracker.chunk_lat_min_us[chunk_slot], __ATOMIC_RELAXED);
            int nwr = cc->chunk_tracker.chunk_wr_counts[chunk_slot];
            uint64_t send_ts = cc->chunk_tracker.chunk_send_times[chunk_slot];
            uint64_t gen = cc->chunk_tracker.chunk_generation[chunk_slot];

            if (smps == 0 || (uint32_t)nwr != smps || minlat == UINT64_MAX) {
                cc->metrics_o1_violations++;
            } else {
                metrics_sample.chunk_seq = 0;
                metrics_sample.chunk_generation = gen;
                metrics_sample.chunk_slot = chunk_slot;
                metrics_sample.max_completion_us = chunk_rtt;
                metrics_sample.min_completion_us = minlat;
                metrics_sample.avg_completion_us = sum / (uint64_t)smps;
                metrics_sample.samples = smps;
                metrics_sample.span_us = (f > 0 && l >= f) ? (l - f) : 0;
                metrics_sample.service_time_us = (send_ts > 0 && l >= send_ts) ? (l - send_ts) : 0;
                have_metrics_sample = 1;
            }
        }
        cc->chunk_tracker.chunk_status[chunk_slot] = 0;  // 置 0 以便 chunk_slot 复用
        did_finalize = 1;
    }
    pthread_mutex_unlock(&cc->chunk_tracker.mutex);

    if (have_metrics_sample)
        ncclIbAppendMetricsBucket(cc, &metrics_sample);
    
    // 必须始终扣减 inflight，否则 chunk_rtt==0（如 send_time 未设或 RTT 为 0）时 inflight 永不降→窗口死锁、卡死
    if (did_finalize) {
        int prev_inflight = __sync_fetch_and_sub(&cc->inflight_chunks, 1);
        // 诊断：rate-limit 打印，确认 Finalize 路径有被调用（若 blocked 时从无此打印，则 CQE 未进 OnCompletion 或 chunk_cqes_pending 未到 1）
        {
            static uint64_t last_fin_print = 0;
            uint64_t now_us = ncclIbGetMicros();
            if (now_us - last_fin_print > 1000000) {
                printf("[AIMD] FinalizeChunkRTT: cc=%p slot=%d inflight_before=%d\n", (void*)cc, chunk_slot, prev_inflight);
                fflush(stdout);
                last_fin_print = now_us;
            }
        }
        (void)__sync_fetch_and_add(&cc->finalized_chunks_total, 1ULL);
        if (ncclIbCcEpochEnabled()) {
            __atomic_store_n(&cc->epoch_due, 1u, __ATOMIC_RELEASE);
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
    
    // 控制面解耦：RTT 更新仅负责观测累积，不在 completion 热路径触发控制计算。
    // LIW/epoch/hint 统一由控制面线程定期执行。
}

// ============================================================================
// Phase 4：epoch 控制器（仅 ncclIbUpdateLIW 入口，design §5.2.2 (a)）
// ============================================================================

static void ncclCcEpochUpdate(struct CollectiveCC* cc, uint64_t now_ns) {
    pthread_mutex_lock(&cc->mutex);
    ncclCcRecomputeHintValid(cc, now_ns);

    uint64_t now = ncclIbGetMicros();

    /* queue_pressure 分母：上一轮已发布的 effective（设计 §4.3） */
    int prev_eff = (cc->effective_window > 0) ? cc->effective_window : cc->lib_window;
    if (prev_eff < 1) prev_eff = 1;
    int inflight = __atomic_load_n(&cc->inflight_chunks, __ATOMIC_RELAXED);
    (void)((double)inflight / (double)prev_eff); /* 预留：后续控制律可消费 queue_pressure */

    uint64_t current_max = __atomic_load_n(&cc->rtt_max, __ATOMIC_RELAXED);

    double alpha = 0.1;
    if (cc->rtt_ewma == 0) {
        cc->rtt_ewma = (double)current_max;
    } else {
        cc->rtt_ewma = alpha * (double)current_max + (1.0 - alpha) * cc->rtt_ewma;
    }

    double old_window = (double)ncclCcLoadInt(&cc->lib_window);
    double rtt_collective = cc->rtt_ewma;

    int hint_forbid_increase = 0;
    int hint_cap = cc->max_window;
    if (cc->hint_valid) {
        if (cc->hint.flags & XCCL_HINT_F_SEVERE) hint_forbid_increase = 1;
        float cnp = cc->hint.cnp_level;
        if (cnp < 0.f) cnp = 0.f;
        if (cnp > 1.f) cnp = 1.f;
        const double w = 0.5;
        double span = (double)(cc->max_window - cc->min_window);
        if (span < 0.) span = 0.;
        hint_cap = cc->min_window + (int)(span * (1.0 - w * (double)cnp));
        if (hint_cap < cc->min_window) hint_cap = cc->min_window;
        if (hint_cap > cc->max_window) hint_cap = cc->max_window;
    }

    if (cc->rtt_target_high > 0 && rtt_collective > (double)cc->rtt_target_high) {
        double new_window = old_window * cc->beta;
        if (new_window < cc->min_window) new_window = cc->min_window;
        ncclCcStoreInt(&cc->lib_window, (int)new_window);
        cc->congestion_events++;
        if (cc->congestion_events % 100 == 0) {
            INFO(NCCL_NET, "AIMD[%lu]: Congestion (RTT=%.1f > %.1f), LIW %.1f -> %.1f",
                 cc->collective_id, rtt_collective, (double)cc->rtt_target_high,
                 old_window, new_window);
        }
    } else if (!hint_forbid_increase && cc->rtt_target_low > 0 && rtt_collective < (double)cc->rtt_target_low) {
        double new_window = old_window + cc->alpha;
        if (new_window > cc->max_window) new_window = cc->max_window;
        ncclCcStoreInt(&cc->lib_window, (int)new_window);
        cc->increase_events++;
    }

    int eff = ncclCcLoadInt(&cc->lib_window);
    if (cc->device_cap < eff) eff = cc->device_cap;
    if (hint_cap < eff) eff = hint_cap;
    int mc = cc->chunk_tracker.max_chunks;
    if (eff > mc) {
        static int s_clamp_warn;
        if (!s_clamp_warn) {
            WARN("NET/IB: effective_window %d > max_chunks %d, clamp (Phase 4)", eff, mc);
            s_clamp_warn = 1;
        }
        eff = mc;
    }
    ncclCcStoreInt(&cc->effective_window, eff);

    cc->last_update_time = now;
    cc->total_updates++;
    cc->epoch_last_ts_ns = now_ns;

    if (ncclIbCcMetricsEnabled() && cc->metrics_bucket_chunk_count > 0) {
        uint64_t log_ms = 5000;
        const char* elog = getenv(NCCL_CC_METRICS_LOG_MS_ENV);
        if (elog && atoi(elog) > 0) log_ms = (uint64_t)atoi(elog);
        uint64_t now_mu = ncclIbGetMicros();
        if (cc->metrics_last_log_us == 0 || now_mu - cc->metrics_last_log_us >= log_ms * 1000ULL) {
            INFO(NCCL_NET, "AIMD[%lu] CC metrics: bucket_chunks=%u sum_avg=%lu sum_max=%lu sum_span=%lu sum_svc=%lu o1_violations=%lu",
                 (unsigned long)cc->collective_id,
                 (unsigned)cc->metrics_bucket_chunk_count,
                 (unsigned long)cc->metrics_bucket_sum_avg_us,
                 (unsigned long)cc->metrics_bucket_sum_max_us,
                 (unsigned long)cc->metrics_bucket_sum_span_us,
                 (unsigned long)cc->metrics_bucket_sum_service_us,
                 (unsigned long)cc->metrics_o1_violations);
            cc->metrics_last_log_us = now_mu;
        }
    }

    pthread_mutex_unlock(&cc->mutex);
}

static void ncclCcEpochUpdateIfDue(struct CollectiveCC* cc, uint64_t now_ns) {
    if (!cc || !cc->enabled) return;
    if (!ncclIbCcEpochEnabled()) return;

    int interval_elapsed = (cc->epoch_last_ts_ns == 0) ||
        (now_ns - cc->epoch_last_ts_ns >= cc->epoch_interval_ns);
    uint32_t due = __atomic_load_n(&cc->epoch_due, __ATOMIC_RELAXED);
    int due_set = (due != 0);
    if (!interval_elapsed && !due_set) return;

    (void)__atomic_exchange_n(&cc->epoch_due, 0u, __ATOMIC_ACQ_REL);
    ncclCcEpochUpdate(cc, now_ns);
}

// ============================================================================
// AIMD 控制
// ============================================================================

static void ncclIbUpdateLIWLocal(struct CollectiveCC* cc) {
    if (!cc || !cc->enabled) return;

    if (ncclIbCcHintEnabled()) {
        ncclCcRefreshHintIfNeeded(cc, ncclIbGetNanos());
    }

    if (ncclIbCcEpochEnabled()) {
        ncclCcEpochUpdateIfDue(cc, ncclIbGetNanos());
        return;
    }

    uint64_t now = ncclIbGetMicros();
    uint64_t time_since_update = (cc->last_update_time == 0) ?
        cc->update_interval_us + 1 : (now - cc->last_update_time);

    if (time_since_update < cc->update_interval_us) {
        return;
    }

    pthread_mutex_lock(&cc->mutex);

    uint64_t current_max = __atomic_load_n(&cc->rtt_max, __ATOMIC_RELAXED);

    double alpha = 0.1;
    if (cc->rtt_ewma == 0) {
        cc->rtt_ewma = (double)current_max;
    } else {
        cc->rtt_ewma = alpha * (double)current_max + (1.0 - alpha) * cc->rtt_ewma;
    }

    double old_window = (double)ncclCcLoadInt(&cc->lib_window);
    double rtt_collective = cc->rtt_ewma;

    if (cc->rtt_target_high > 0 && rtt_collective > (double)cc->rtt_target_high) {
        double new_window = old_window * cc->beta;
        if (new_window < cc->min_window) {
            new_window = cc->min_window;
        }
        ncclCcStoreInt(&cc->lib_window, (int)new_window);
        cc->congestion_events++;

        if (cc->congestion_events % 100 == 0) {
            INFO(NCCL_NET, "AIMD[%lu]: Congestion (RTT=%.1f > %.1f), LIW %.1f -> %.1f",
                 cc->collective_id, rtt_collective, (double)cc->rtt_target_high,
                 old_window, new_window);
        }
    }
    else if (cc->rtt_target_low > 0 && rtt_collective < (double)cc->rtt_target_low) {
        double new_window = old_window + cc->alpha;
        if (new_window > cc->max_window) {
            new_window = cc->max_window;
        }
        ncclCcStoreInt(&cc->lib_window, (int)new_window);
        cc->increase_events++;
    }

    cc->last_update_time = now;
    cc->total_updates++;

    if (ncclIbCcMetricsEnabled() && cc->metrics_bucket_chunk_count > 0) {
        uint64_t log_ms = 5000;
        const char* elog = getenv(NCCL_CC_METRICS_LOG_MS_ENV);
        if (elog && atoi(elog) > 0) log_ms = (uint64_t)atoi(elog);
        uint64_t now_mu = ncclIbGetMicros();
        if (cc->metrics_last_log_us == 0 || now_mu - cc->metrics_last_log_us >= log_ms * 1000ULL) {
            INFO(NCCL_NET, "AIMD[%lu] CC metrics: bucket_chunks=%u sum_avg=%lu sum_max=%lu sum_span=%lu sum_svc=%lu o1_violations=%lu",
                 (unsigned long)cc->collective_id,
                 (unsigned)cc->metrics_bucket_chunk_count,
                 (unsigned long)cc->metrics_bucket_sum_avg_us,
                 (unsigned long)cc->metrics_bucket_sum_max_us,
                 (unsigned long)cc->metrics_bucket_sum_span_us,
                 (unsigned long)cc->metrics_bucket_sum_service_us,
                 (unsigned long)cc->metrics_o1_violations);
            cc->metrics_last_log_us = now_mu;
        }
    }

    pthread_mutex_unlock(&cc->mutex);
}

void ncclIbUpdateLIW(struct CollectiveCC* cc) {
    if (!cc || !cc->enabled) return;
    if (ncclCcLoadInt(&cc->external_control_active)) return;
    if (ncclIbCcV2MinimalEnabled()) return; // v2-minimal 独占窗口控制权
    ncclIbUpdateLIWLocal(cc);
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
    NCCLCHECK(xcclTelemetryHintTransportInit());
    NCCLCHECK(xcclControlSnapshotTransportInit());
    NCCLCHECK(xcclHostSignalSnapshotInit());
    ncclCcExternalSnapshotTtlInitOnce();

    if (!ncclIbIsAIMDEnabled()) {
        return ncclSuccess;
    }

    NCCLCHECK(ncclIbInitShadowPool());
    NCCLCHECK(ncclIbInitCCPool());
    ncclCcControlPlaneIntervalInitOnce();

    if (g_cc_control_thread_started) return ncclSuccess;

    g_cc_control_thread_running = 1;
    if (pthread_create(&g_cc_control_thread, NULL, ncclCcControlPlaneThreadMain, NULL) != 0) {
        g_cc_control_thread_running = 0;
        WARN("NET/IB: failed to start CC control-plane thread");
        return ncclSystemError;
    }
    g_cc_control_thread_started = 1;

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
        if (old_inflight >= ncclCcGetEffectiveWindowForSend(cc)) {
            // 窗口已满，返回Success但让上层下一轮再试；限频日志确认是否因 Completion 未减 inflight 导致死锁
            static uint64_t last_print = 0;
            uint64_t now_us = ncclIbGetMicros();
            if (now_us - last_print > 1000000) {
                printf("[AIMD] PostSend blocked: inflight=%d window=%d chunk_id=%d\n",
                       old_inflight, ncclCcGetEffectiveWindowForSend(cc), chunk_id);
                fflush(stdout);
                last_print = now_us;
            }
            return ncclSuccess;  // 返回成功，但实际未发送（上层会重试）
        }
        new_inflight = old_inflight + 1;  // 预留1个chunk的空间
    } while (!__sync_bool_compare_and_swap(&cc->inflight_chunks, old_inflight, new_inflight));
    // 🔴 按 cc 单调分配 chunk_id_slot，与 p2p 一致，避免多路共 slot 导致 Finalize 少调、inflight 不降
    uint64_t seq = __sync_fetch_and_add(&cc->chunk_tracker.next_chunk_seq, 1);
    int chunk_id_slot = (int)((unsigned)seq % (unsigned)cc->chunk_tracker.max_chunks);
    uint64_t chunk_generation = seq / (uint64_t)cc->chunk_tracker.max_chunks;
    uint32_t chunk_generation_tag = (uint32_t)(chunk_generation & NCCL_CC_CHUNK_GENERATION_TAG_MASK);
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
    
    ncclIbRecordChunkSend(cc, chunk_id_slot, chunk_generation, num_wrs);
    
    // 为每个WR分配Shadow Slot并编码wr_id
    for (int r = 0; r < nreqs; r++) {
        // 🔴 关键：先保存NCCL原生的wr_id（完整的64位，可能是指针）
        uint64_t original_wr_id = wrs[r].wr_id;
        
        // 🔴 关键：分配Shadow Slot并存储原生ID（用 chunk_id_slot 保证 shadow->chunk_id 不越界）
        int shadow_slot = ncclIbAllocateShadowSlot(qp_index, original_wr_id, cc_index, chunk_id_slot, chunk_generation);
        if (shadow_slot < 0) {
            for (int j = 0; j < r; j++) {
                if (NCCL_CC_IS_MY_WR(wrs[j].wr_id)) {
                    int qp_i, cc_i, chunk_slot_i, gen_tag_i, sl;
                    NCCL_CC_DECODE_WR_ID(wrs[j].wr_id, &qp_i, &cc_i, &chunk_slot_i, &gen_tag_i, &sl);
                    (void)cc_i; (void)chunk_slot_i; (void)gen_tag_i;
                    if (sl >= 0 && qp_i >= 0)
                        ncclIbReleaseShadowSlot(qp_i, sl);
                }
            }
            __sync_fetch_and_sub(&cc->inflight_chunks, 1);
            return ncclSystemError;
        }
        wrs[r].wr_id = NCCL_CC_ENCODE_WR_ID(qp_index, cc_index, chunk_id_slot, chunk_generation_tag, shadow_slot);
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
            int qp_i, cc_i, chunk_slot_i, gen_tag_i, sl;
            NCCL_CC_DECODE_WR_ID(wrs[r].wr_id, &qp_i, &cc_i, &chunk_slot_i, &gen_tag_i, &sl);
            (void)cc_i; (void)chunk_slot_i; (void)gen_tag_i;
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
    
    (void)commBase;
    (void)devIndex;

    int qp_idx, cc_idx, chunk_slot, chunk_generation_tag, shadow_slot;
    NCCL_CC_DECODE_WR_ID(wc->wr_id, &qp_idx, &cc_idx, &chunk_slot, &chunk_generation_tag, &shadow_slot);
    
    if (qp_idx < 0 || cc_idx < 0 || chunk_slot < 0 || chunk_generation_tag < 0 || shadow_slot < 0 ||
        qp_idx >= NCCL_CC_MAX_QP || cc_idx >= NCCL_CC_MAX_COLLECTIVES ||
        shadow_slot >= NCCL_CC_MAX_SHADOW_SLOTS)
        return ncclSuccess;
    
    // 与 finalize 共用 g_cc_pool_mutex，保证 completion 持有 cc 指针期间不会被释放。
    pthread_mutex_lock(&g_cc_pool_mutex);
    struct CollectiveCC* cc = g_cc_pool[cc_idx];
    ncclResult_t ret = ncclSuccess;
    // C++：goto out_unlock_pool 不得跳过带初始化的局部变量，故提前声明
    int effective_chunk_slot = -1;
    uint64_t now = 0;
    uint64_t wr_rtt = 0;
    int prev = 0;
    
    uint64_t original_wr_id = 0;
    uint64_t send_time = 0;
    uint64_t shadow_chunk_generation = 0;
    uint32_t shadow_chunk_slot = 0;
    int shadow_was_active = 0;
    
    pthread_mutex_lock(&g_shadow_pool_mutex[qp_idx]);
    struct ShadowRequest* shadow = &g_shadow_pool[qp_idx][shadow_slot];
    if (shadow->active) {
        shadow_was_active = 1;
        original_wr_id = shadow->original_wr_id;
        // 强一致校验所需字段在释放 slot 前先抓取快照
        int shadow_cc_index = shadow->cc_index;
        shadow_chunk_slot = shadow->chunk_slot;
        shadow_chunk_generation = shadow->chunk_generation;
        send_time = shadow->send_timestamp;
        // wr_id 解码出的 cc/slot 必须与 shadow 一致，否则视为 stale。
        if (shadow_cc_index != cc_idx || (int)shadow_chunk_slot != chunk_slot) {
            shadow_was_active = 0;
        }
        if (shadow_was_active && cc && cc->chunk_tracker.chunk_original_wr_id &&
            (int)shadow_chunk_slot >= 0 && (int)shadow_chunk_slot < cc->chunk_tracker.max_chunks) {
            cc->chunk_tracker.chunk_original_wr_id[shadow_chunk_slot] = original_wr_id;
        }
        /* 不在此处释放 shadow：须先通过下方 stale/generation 校验且 pending-- 成功后再释放（Phase 1 §1.4） */
    }
    pthread_mutex_unlock(&g_shadow_pool_mutex[qp_idx]);
    
    // NCCL 的 wr_id 来自 req 数组偏移 (reqs[r]-base.reqs)<<(r*8)，第 0 个偏移为 0，故 original_wr_id==0 合法；
    if (!cc || !cc->enabled) {
        wc->wr_id = original_wr_id;
        goto out_unlock_pool;
    }

    // 如果 shadow slot 的 original_wr_id 恰好为 0，使用 chunk_original_wr_id 兜底。
    if (original_wr_id == 0 && cc->chunk_tracker.chunk_original_wr_id) {
        int fallback_slot = shadow_was_active ? (int)shadow_chunk_slot : chunk_slot;
        if (fallback_slot >= 0 && fallback_slot < cc->chunk_tracker.max_chunks) {
            original_wr_id = cc->chunk_tracker.chunk_original_wr_id[fallback_slot];
        }
    }
    wc->wr_id = original_wr_id;
    
    effective_chunk_slot = shadow_was_active ? (int)shadow_chunk_slot : chunk_slot;
    if (effective_chunk_slot < 0 || effective_chunk_slot >= cc->chunk_tracker.max_chunks)
        goto out_unlock_pool;
    
    // stale CQE：如果 shadow 已经不 active，或 generation/slot 强一致校验失败，则跳过 pending-- 与 finalize。
    if (!shadow_was_active) {
        goto out_unlock_pool;
    }

    // Fast-path（tag）预过滤
    if (((uint32_t)shadow_chunk_generation & NCCL_CC_CHUNK_GENERATION_TAG_MASK) != (uint32_t)chunk_generation_tag) {
        goto out_unlock_pool;
    }

    // Strong-path：shadow 与 chunk_slot 必须强一致
    if (cc->chunk_tracker.chunk_generation[effective_chunk_slot] != shadow_chunk_generation) {
        goto out_unlock_pool;
    }
    
    now = ncclIbGetMicros();
    wr_rtt = (send_time > 0 && now >= send_time) ? (now - send_time) : 0;

    prev = __sync_fetch_and_sub(&cc->chunk_tracker.chunk_cqes_pending[effective_chunk_slot], 1);
#ifdef AIMD_DEBUG
    if (prev <= 0)
        printf("[AIMD] BAD prev=%d cc=%p slot=%d\n", prev, (void*)cc, effective_chunk_slot);
    { static uint64_t last_p1=0; uint64_t n=ncclIbGetMicros(); if (prev==1 && n-last_p1>500000) { printf("[AIMD] prev==1 -> Finalize cc=%p slot=%d\n", (void*)cc, effective_chunk_slot); last_p1=n; } }
#endif
    if (prev <= 0) {
        // 发生 underflow 说明 pending/finalize 协议被破坏；尝试回滚 1，避免进一步负数扩散。
        if (prev == 0) __sync_fetch_and_add(&cc->chunk_tracker.chunk_cqes_pending[effective_chunk_slot], 1);
        goto out_unlock_pool;
    }
    /* 合法 pending 递减后才释放本 CQE 对应的 shadow（避免 stale CQE 误伤同槽位新 WR） */
    pthread_mutex_lock(&g_shadow_pool_mutex[qp_idx]);
    {
        struct ShadowRequest* sh = &g_shadow_pool[qp_idx][shadow_slot];
        if (sh->active && sh->chunk_generation == shadow_chunk_generation &&
            (int)sh->chunk_slot == (int)shadow_chunk_slot && sh->cc_index == cc_idx) {
            sh->active = 0;
        }
    }
    pthread_mutex_unlock(&g_shadow_pool_mutex[qp_idx]);

    // Phase 2：仅在 prev>0 时更新 RTT(max) 与观测字段（与 fetch_sub 语义一致）
    ncclIbChunkObservationOnValidCqe(cc, effective_chunk_slot, wr_rtt, now);
    if (prev == 1) {
        __sync_fetch_and_add(&cc->cq_completed_total, 1ULL);
        __sync_fetch_and_add(&cc->chunk_tracker.chunk_completed_wrs[effective_chunk_slot],
                            cc->chunk_tracker.chunk_wr_counts[effective_chunk_slot]);
        ncclIbFinalizeChunkRTT(cc, effective_chunk_slot);
    } else {
        __sync_fetch_and_add(&cc->cq_completed_total, 1ULL);
    }
out_unlock_pool:
    pthread_mutex_unlock(&g_cc_pool_mutex);
    return ret;
}

// ============================================================================
// 初始化/清理
// ============================================================================

ncclResult_t ncclIbAIMDFinalize(void) {
    if (g_cc_control_thread_started) {
        g_cc_control_thread_running = 0;
        pthread_join(g_cc_control_thread, NULL);
        g_cc_control_thread_started = 0;
    }

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
                if (cc->chunk_tracker.chunk_finalized_flag) free((void*)cc->chunk_tracker.chunk_finalized_flag);
                if (cc->chunk_tracker.chunk_generation) free(cc->chunk_tracker.chunk_generation);
                if (cc->chunk_tracker.chunk_original_wr_id) free(cc->chunk_tracker.chunk_original_wr_id);
                if (cc->chunk_tracker.chunk_first_cqe_us) free(cc->chunk_tracker.chunk_first_cqe_us);
                if (cc->chunk_tracker.chunk_last_cqe_us) free(cc->chunk_tracker.chunk_last_cqe_us);
                if (cc->chunk_tracker.chunk_lat_sum_us) free(cc->chunk_tracker.chunk_lat_sum_us);
                if (cc->chunk_tracker.chunk_lat_min_us) free(cc->chunk_tracker.chunk_lat_min_us);
                if (cc->chunk_tracker.chunk_lat_samples) free(cc->chunk_tracker.chunk_lat_samples);
                pthread_mutex_destroy(&cc->chunk_tracker.mutex);
                pthread_mutex_destroy(&cc->mutex);
                free(cc);
                g_cc_pool[i] = NULL;
            }
        }
        pthread_mutex_unlock(&g_cc_pool_mutex);
    }

    xcclTelemetryHintTransportFini();
    xcclControlSnapshotTransportFini();
    xcclHostSignalSnapshotFini();
    return ncclSuccess;
}
