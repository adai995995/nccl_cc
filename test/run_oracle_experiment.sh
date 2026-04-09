#!/usr/bin/env bash
# ============================================================================
# Oracle 最小实验驱动脚本 v2
#
# 核心改进：用 taskset 把 benchmark 绑到少量 CPU 核上，再用 stress-ng
# 竞争同一组核，制造真正的 host drain 压力。
#
# 实验矩阵（8 组）：
#   1) baseline         : 原生 NCCL（AIMD 关闭），无 stress
#   2) baseline+stress  : 原生 NCCL（AIMD 关闭），绑核 + CPU stress
#   3) aimd-native+stress : AIMD 开启原始窗口，绑核 + CPU stress
#   4) oracle-half+stress : AIMD + window*0.5，绑核 + CPU stress
#   5) oracle-third+stress: AIMD + window*0.33，绑核 + CPU stress
#   6) v2-minimal+stress : v2-minimal 自动闭环，绑核 + CPU stress
#   7) oracle-pacing-only+stress : 固定 chunk 注入 pacing（不缩窗），绑核 + stress
#   8) oracle-channels-only+stress: 限制可用 channel/QP 数（不 pacing），绑核 + stress
#
# 用法：
#   ./run_oracle_experiment.sh
#   NUM_GPUS=8 ITERS=500 PIN_CPUS=0-3 ./run_oracle_experiment.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---- 可配参数 ----
NUM_GPUS="${NUM_GPUS:-8}"
ITERS="${ITERS:-500}"
WARMUP="${WARMUP:-50}"
COUNT="${COUNT:-1048576}"
PIN_CPUS="${PIN_CPUS:-0-3}"          # benchmark 绑定的 CPU 核
STRESS_ON_SAME_CPUS="${STRESS_ON_SAME_CPUS:-1}"  # 1=stress 竞争同一组核
STRESS_EXTRA_WORKERS="${STRESS_EXTRA_WORKERS:-8}" # stress worker 数
STRESS_TIMEOUT="${STRESS_TIMEOUT:-180}"
OUTDIR="${OUTDIR:-oracle_results}"
# 消融：pacing-only / channels-only（纳秒 / 最大并行 channel 数）
ORACLE_PACING_NS="${ORACLE_PACING_NS:-5000}"
ORACLE_CHANNELS_CAP="${ORACLE_CHANNELS_CAP:-4}"
# 重复次数（整套 8 组作为一次 suite）
REPEAT="${REPEAT:-1}"

NCCL_BUILD="${NCCL_BUILD:-$(cd .. && pwd)/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${NCCL_BUILD}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ---- 前置检查 ----
if [[ ! -f "${NCCL_BUILD}/lib/libnccl.so" ]] && ! ls "${NCCL_BUILD}/lib"/libnccl.so.* >/dev/null 2>&1; then
    echo "ERROR: libnccl not found. Build NCCL first:"
    echo "  cd $(cd .. && pwd) && ./build_a100.sh"
    exit 1
fi

make NCCL_BUILD="${NCCL_BUILD}" oracle_bench

if ! command -v stress-ng &>/dev/null; then
    echo "WARNING: stress-ng not found. Install: apt install stress-ng"
    echo "         Will skip stress injection experiments."
    HAS_STRESS=0
else
    HAS_STRESS=1
fi

mkdir -p "$OUTDIR"

MASTER_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTDIR}/run_${MASTER_TS}"
mkdir -p "$RUN_DIR"
MASTER_SUMMARY="${RUN_DIR}/summary_master_${MASTER_TS}.txt"
echo "============================================" | tee "$MASTER_SUMMARY"
echo " Oracle Experiment v2 (REPEAT=$REPEAT)  $(date)" | tee -a "$MASTER_SUMMARY"
echo " RUN_DIR=$RUN_DIR" | tee -a "$MASTER_SUMMARY"
echo "============================================" | tee -a "$MASTER_SUMMARY"

# 计算绑核数量
IFS='-' read -r PIN_LO PIN_HI <<< "$PIN_CPUS"
PIN_COUNT=$(( PIN_HI - PIN_LO + 1 ))

run_suite_once() {
    local rep="$1"
    TIMESTAMP="${MASTER_TS}_r${rep}"
    SUMMARY="${RUN_DIR}/summary_${TIMESTAMP}.txt"

    echo "============================================" | tee "$SUMMARY"
    echo " Oracle Experiment v2  $(date)  (rep=$rep/$REPEAT)" | tee -a "$SUMMARY"
    echo " NUM_GPUS=$NUM_GPUS ITERS=$ITERS COUNT=$COUNT" | tee -a "$SUMMARY"
    echo " PIN_CPUS=$PIN_CPUS ($PIN_COUNT cores)" | tee -a "$SUMMARY"
    echo " STRESS workers=$STRESS_EXTRA_WORKERS on same cores=$STRESS_ON_SAME_CPUS" | tee -a "$SUMMARY"
    echo " Total machine cores: $(nproc)" | tee -a "$SUMMARY"
    echo " ORACLE_PACING_NS=$ORACLE_PACING_NS ORACLE_CHANNELS_CAP=$ORACLE_CHANNELS_CAP" | tee -a "$SUMMARY"
    echo "============================================" | tee -a "$SUMMARY"

    # ---- 公共 NCCL 环境：强制走网络 ----
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_DEBUG=WARN
    export NUM_GPUS ITERS WARMUP COUNT

    STRESS_PID=""

run_one() {
    local label="$1"
    local use_pin="$2"      # 1=taskset 绑核
    local use_stress="$3"   # 1=同时跑 stress
    local csv="${OUTDIR}/${label}_${TIMESTAMP}.csv"
    local log="${OUTDIR}/${label}_${TIMESTAMP}.log"

    echo ""
    echo "---- [$label] ----" | tee -a "$SUMMARY"
    echo "  CSV: $csv" | tee -a "$SUMMARY"
    if [[ "$use_pin" -eq 1 ]]; then
        echo "  PIN: taskset -c $PIN_CPUS" | tee -a "$SUMMARY"
    fi

    export CSV_OUTPUT="$csv"

    if [[ "$use_pin" -eq 1 ]]; then
        taskset -c "$PIN_CPUS" ./oracle_bench 2>&1 | tee "$log"
    else
        ./oracle_bench 2>&1 | tee "$log"
    fi

    grep -E "(Wall-clock|Collective max|Rank skew|Avg algo|p50|mean)" "$log" >> "$SUMMARY" 2>/dev/null || true
    echo "" >> "$SUMMARY"
}

start_stress() {
    if [[ "$HAS_STRESS" -eq 0 ]]; then return; fi
    if [[ "$STRESS_ON_SAME_CPUS" -eq 1 ]]; then
        echo "  Starting stress-ng: $STRESS_EXTRA_WORKERS workers pinned to CPU $PIN_CPUS for ${STRESS_TIMEOUT}s"
        taskset -c "$PIN_CPUS" stress-ng --cpu "$STRESS_EXTRA_WORKERS" --timeout "${STRESS_TIMEOUT}s" &>/dev/null &
    else
        echo "  Starting stress-ng: $STRESS_EXTRA_WORKERS workers (unpinned) for ${STRESS_TIMEOUT}s"
        stress-ng --cpu "$STRESS_EXTRA_WORKERS" --timeout "${STRESS_TIMEOUT}s" &>/dev/null &
    fi
    STRESS_PID=$!
    sleep 2
}

stop_stress() {
    if [[ -n "${STRESS_PID:-}" ]]; then
        kill "$STRESS_PID" 2>/dev/null || true
        # 避免 stress-ng 子进程/信号延迟导致卡住：最多等 ~3s，否则强杀
        for _ in 1 2 3 4 5 6; do
            if ! ps -p "$STRESS_PID" >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
        if ps -p "$STRESS_PID" >/dev/null 2>&1; then
            kill -9 "$STRESS_PID" 2>/dev/null || true
        fi
        wait "$STRESS_PID" 2>/dev/null || true
        STRESS_PID=""
        sleep 1
    fi
}

trap stop_stress EXIT

# ============================================================
# 实验 1：baseline 无 stress（绑核，作为绑核基线）
# ============================================================
export NCCL_AIMD_ENABLE=0
unset NCCL_CC_ORACLE_FACTOR 2>/dev/null || true
run_one "1_baseline_pinned_no_stress" 1 0

# ============================================================
# 实验 2：baseline + stress（绑核 + 同核 stress）
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=0
    unset NCCL_CC_ORACLE_FACTOR 2>/dev/null || true
    start_stress
    run_one "2_baseline_pinned_with_stress" 1 1
    stop_stress
fi

# ============================================================
# 实验 3：AIMD 开启（原始窗口）+ stress
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=1
    export NCCL_CC_ORACLE_FACTOR=1.0
    start_stress
    run_one "3_aimd_native_pinned_with_stress" 1 1
    stop_stress
fi

# ============================================================
# 实验 4：oracle window*0.5 + stress
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=1
    export NCCL_CC_ORACLE_FACTOR=0.5
    start_stress
    run_one "4_oracle_half_pinned_with_stress" 1 1
    stop_stress
fi

# ============================================================
# 实验 5：oracle window*0.33 + stress
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=1
    export NCCL_CC_ORACLE_FACTOR=0.33
    start_stress
    run_one "5_oracle_third_pinned_with_stress" 1 1
    stop_stress
fi

# ============================================================
# 实验 6：v2-minimal 自动闭环 + stress
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=1
    export NCCL_CC_EPOCH_ENABLE=1
    export NCCL_CC_V2_MINIMAL=1
    unset NCCL_CC_ORACLE_FACTOR 2>/dev/null || true
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=NET
    start_stress
    run_one "6_v2_minimal_pinned_with_stress" 1 1
    stop_stress
    unset NCCL_CC_V2_MINIMAL
    unset NCCL_CC_EPOCH_ENABLE
    export NCCL_DEBUG=WARN
fi

# ============================================================
# 实验 7：oracle pacing-only + stress（NCCL_CC_ORACLE_FACTOR=1，不缩窗）
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=1
    export NCCL_CC_EPOCH_ENABLE=1
    export NCCL_CC_ORACLE_FACTOR=1.0
    unset NCCL_CC_V2_MINIMAL 2>/dev/null || true
    export NCCL_CC_ORACLE_PACING_NS="${ORACLE_PACING_NS}"
    unset NCCL_CC_ORACLE_CHANNELS
    start_stress
    run_one "7_oracle_pacing_only_pinned_with_stress" 1 1
    stop_stress
    unset NCCL_CC_ORACLE_PACING_NS
fi

# ============================================================
# 实验 8：oracle channels-only + stress
# ============================================================
if [[ "$HAS_STRESS" -eq 1 ]]; then
    export NCCL_AIMD_ENABLE=1
    export NCCL_CC_EPOCH_ENABLE=1
    export NCCL_CC_ORACLE_FACTOR=1.0
    unset NCCL_CC_V2_MINIMAL 2>/dev/null || true
    unset NCCL_CC_ORACLE_PACING_NS 2>/dev/null || true
    export NCCL_CC_ORACLE_CHANNELS="${ORACLE_CHANNELS_CAP}"
    start_stress
    run_one "8_oracle_channels_only_pinned_with_stress" 1 1
    stop_stress
    unset NCCL_CC_ORACLE_CHANNELS
fi

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================" | tee -a "$SUMMARY"
echo " Experiment Complete" | tee -a "$SUMMARY"
echo " Results dir: $OUTDIR/" | tee -a "$SUMMARY"
echo " Summary: $SUMMARY" | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"
echo ""
echo "CSV files:"
ls -1 "${OUTDIR}"/*_${TIMESTAMP}.csv 2>/dev/null || echo "  (none)"
echo ""
echo "判定标准："
echo "  - 组 1 vs 2：stress 绑核后是否制造出明显 tail 恶化？"
echo "  - 组 2 vs 4/5：oracle 缩窗能否降低 max/p99？"
echo "  - 组 2 vs 3：仅开 AIMD 框架是否有副作用？"
echo "  - 组 6 vs 4/5：v2-minimal 自动闭环是否接近 oracle？"
echo "  - 组 2 vs 7/8：仅 pacing / 仅 channels 是否有收益（对比 window-only）"
echo "  - 若 oracle 都没正收益 → 该 anomaly 不适合 window 控制"

    echo "" | tee -a "$MASTER_SUMMARY"
    echo "rep $rep summary: $SUMMARY" | tee -a "$MASTER_SUMMARY"
}

for ((rep=1; rep<=REPEAT; rep++)); do
    run_suite_once "$rep"
done

echo "" | tee -a "$MASTER_SUMMARY"
echo "============================================" | tee -a "$MASTER_SUMMARY"
echo " All repeats complete" | tee -a "$MASTER_SUMMARY"
echo " MASTER_SUMMARY: $MASTER_SUMMARY" | tee -a "$MASTER_SUMMARY"
echo " RUN_DIR: $RUN_DIR" | tee -a "$MASTER_SUMMARY"
echo "============================================" | tee -a "$MASTER_SUMMARY"
