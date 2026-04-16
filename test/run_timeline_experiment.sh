#!/usr/bin/env bash
# ============================================================================
# Phase V3 时间线实验 v2
#
# 通过轮询 benchmark CSV 行数来判断进度，避免固定 sleep 导致时序偏差。
# 三阶段：CLEAN1 → STRESS → CLEAN2
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${NUM_GPUS:-8}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-50}"
COUNT="${COUNT:-1048576}"
PIN_CPUS="${PIN_CPUS:-0-3}"
STRESS_WORKERS="${STRESS_WORKERS:-8}"
OUTDIR="${OUTDIR:-timeline_results}"

PHASE1_END="${PHASE1_END:-300}"
PHASE2_END="${PHASE2_END:-700}"

NCCL_BUILD="${NCCL_BUILD:-$(cd .. && pwd)/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${NCCL_BUILD}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${NCCL_BUILD}/lib/libnccl.so" ]] && ! ls "${NCCL_BUILD}/lib"/libnccl.so.* >/dev/null 2>&1; then
    echo "ERROR: libnccl not found in ${NCCL_BUILD}/lib"
    exit 1
fi

make NCCL_BUILD="${NCCL_BUILD}" oracle_bench

if ! command -v stress-ng &>/dev/null; then
    echo "ERROR: stress-ng is required"
    exit 1
fi

mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NUM_GPUS ITERS WARMUP COUNT

# 阶段 0：可审计元数据（跑前写入；跑完再追加产物与 A/B 段 NCCL）
OUTDIR_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTDIR")"
GIT_ROOT="$(cd .. && pwd)"
GIT_REV="$(git -C "$GIT_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")"
GIT_SHORT="$(git -C "$GIT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

write_run_meta_start() {
    local meta="${OUTDIR_ABS}/run_meta.txt"
    {
        echo "# run_timeline_experiment.sh — run_meta (auto-generated)"
        echo "run_meta_version=1"
        echo "script_path=$(readlink -f "${BASH_SOURCE[0]}")"
        echo "date_iso=$(date -Is 2>/dev/null || date)"
        echo "hostname=$(hostname)"
        echo ""
        echo "[paths]"
        echo "outdir=${OUTDIR_ABS}"
        echo "outdir_basename=$(basename "${OUTDIR_ABS}")"
        echo "git_root=${GIT_ROOT}"
        echo "git_rev=${GIT_REV}"
        echo "git_short=${GIT_SHORT}"
        echo ""
        echo "[bench / phases]"
        echo "NUM_GPUS=${NUM_GPUS}"
        echo "ITERS=${ITERS}"
        echo "WARMUP=${WARMUP}"
        echo "COUNT=${COUNT}"
        echo "PHASE1_END=${PHASE1_END}"
        echo "PHASE2_END=${PHASE2_END}"
        echo "PIN_CPUS=${PIN_CPUS}"
        echo "STRESS_WORKERS=${STRESS_WORKERS}"
        echo ""
        echo "[build]"
        echo "NCCL_BUILD=${NCCL_BUILD}"
        echo "CUDA_HOME=${CUDA_HOME}"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo ""
        echo "[global NCCL / exports before phase A]"
        env | LC_ALL=C sort | grep -E '^NCCL_' || true
        echo ""
        echo "---"
        echo "status=started (append follows if run completes)"
    } > "${meta}"
}

write_run_meta_end() {
    local meta="${OUTDIR_ABS}/run_meta.txt"
    {
        echo ""
        echo "[artifacts timestamp=${TIMESTAMP}]"
        echo "baseline_latency=${OUTDIR_ABS}/baseline_latency_${TIMESTAMP}.csv"
        echo "baseline_log=${OUTDIR_ABS}/baseline_${TIMESTAMP}.log"
        echo "baseline_events=${OUTDIR_ABS}/baseline_events_${TIMESTAMP}.csv"
        echo "bench_latency=${OUTDIR_ABS}/bench_latency_${TIMESTAMP}.csv"
        echo "bench_log=${OUTDIR_ABS}/bench_${TIMESTAMP}.log"
        echo "events=${OUTDIR_ABS}/events_${TIMESTAMP}.csv"
        echo "nccl_timeline=${OUTDIR_ABS}/nccl_timeline_${TIMESTAMP}.csv"
        echo ""
        echo "[phase A — Baseline (no CC) NCCL intent]"
        echo "NCCL_AIMD_ENABLE=0"
        echo "NCCL_CC_V2_MINIMAL=(unset)"
        echo "NCCL_CC_V2_TIMELINE_FILE=(unset)"
        echo "NCCL_DEBUG=WARN"
        echo "CSV_OUTPUT=baseline_latency_${TIMESTAMP}.csv"
        echo ""
        echo "[phase B — v2-minimal NCCL intent]"
        echo "NCCL_AIMD_ENABLE=1"
        echo "NCCL_CC_EPOCH_ENABLE=1"
        echo "NCCL_CC_V2_MINIMAL=1"
        echo "NCCL_CC_V2_TIMELINE_FILE=nccl_timeline_${TIMESTAMP}.csv"
        echo "NCCL_DEBUG=INFO"
        echo "NCCL_DEBUG_SUBSYS=NET"
        echo "CSV_OUTPUT=bench_latency_${TIMESTAMP}.csv"
        echo ""
        echo "[stress-ng]"
        echo "cmd=taskset -c ${PIN_CPUS} stress-ng --cpu ${STRESS_WORKERS} --timeout 300s"
        echo ""
        echo "status=completed"
        echo "date_iso_end=$(date -Is 2>/dev/null || date)"
    } >> "${meta}"
}

write_run_meta_start

get_ts_us() {
    python3 -c "import time; print(int(time.monotonic() * 1e6))"
}

wait_for_csv_lines() {
    local csv="$1"
    local target="$2"
    while true; do
        if [[ -f "$csv" ]]; then
            local lines
            lines=$(wc -l < "$csv")
            if [[ "$lines" -ge "$target" ]]; then
                return 0
            fi
        fi
        sleep 0.2
    done
}

STRESS_PID=""
cleanup() {
    if [[ -n "${STRESS_PID:-}" ]]; then
        kill "$STRESS_PID" 2>/dev/null || true
        wait "$STRESS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "============================================"
echo " Phase V3 Timeline Experiment v2  $(date)"
echo " NUM_GPUS=$NUM_GPUS ITERS=$ITERS COUNT=$COUNT"
echo " PIN_CPUS=$PIN_CPUS"
echo " Phases: clean(0-${PHASE1_END}) stress(${PHASE1_END}-${PHASE2_END}) clean(${PHASE2_END}-${ITERS})"
echo "============================================"

# CSV 行号 = header(1) + warmup 无输出 + iter行
# target_lines = header(1) + iter_count  (warmup 不写 CSV)
P1_LINES=$(( PHASE1_END + 1 ))
P2_LINES=$(( PHASE2_END + 1 ))

# ---- 实验 A：Baseline ----
echo ""
echo "==== [A] Baseline (no CC) with phased stress ===="
BASELINE_CSV="${OUTDIR}/baseline_latency_${TIMESTAMP}.csv"
BASELINE_LOG="${OUTDIR}/baseline_${TIMESTAMP}.log"
BASELINE_EVENTS="${OUTDIR}/baseline_events_${TIMESTAMP}.csv"
echo "ts_us,event" > "$BASELINE_EVENTS"

export NCCL_AIMD_ENABLE=0
unset NCCL_CC_V2_MINIMAL 2>/dev/null || true
unset NCCL_CC_V2_TIMELINE_FILE 2>/dev/null || true
export NCCL_DEBUG=WARN
export CSV_OUTPUT="$BASELINE_CSV"

taskset -c "$PIN_CPUS" ./oracle_bench 2>&1 | tee "$BASELINE_LOG" &
BENCH_PID=$!

echo "  Waiting for iter ${PHASE1_END} (CSV lines >= ${P1_LINES})..."
wait_for_csv_lines "$BASELINE_CSV" "$P1_LINES"

echo "  Injecting stress..."
TS=$(get_ts_us)
echo "${TS},stress_start" >> "$BASELINE_EVENTS"
taskset -c "$PIN_CPUS" stress-ng --cpu "$STRESS_WORKERS" --timeout 300s &>/dev/null &
STRESS_PID=$!

echo "  Waiting for iter ${PHASE2_END} (CSV lines >= ${P2_LINES})..."
wait_for_csv_lines "$BASELINE_CSV" "$P2_LINES"

echo "  Removing stress..."
TS=$(get_ts_us)
echo "${TS},stress_stop" >> "$BASELINE_EVENTS"
kill "$STRESS_PID" 2>/dev/null || true
wait "$STRESS_PID" 2>/dev/null || true
STRESS_PID=""

echo "  Waiting for benchmark to finish..."
wait "$BENCH_PID" 2>/dev/null || true
echo "  [A] Baseline done."

# ---- 实验 B：v2-minimal ----
echo ""
echo "==== [B] v2-minimal with phased stress ===="

TIMELINE_CSV="${OUTDIR}/nccl_timeline_${TIMESTAMP}.csv"
BENCH_CSV="${OUTDIR}/bench_latency_${TIMESTAMP}.csv"
BENCH_LOG="${OUTDIR}/bench_${TIMESTAMP}.log"
EVENTS_LOG="${OUTDIR}/events_${TIMESTAMP}.csv"
echo "ts_us,event" > "$EVENTS_LOG"

export NCCL_AIMD_ENABLE=1
export NCCL_CC_EPOCH_ENABLE=1
export NCCL_CC_V2_MINIMAL=1
export NCCL_CC_V2_TIMELINE_FILE="$TIMELINE_CSV"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
export CSV_OUTPUT="$BENCH_CSV"

taskset -c "$PIN_CPUS" ./oracle_bench 2>&1 | tee "$BENCH_LOG" &
BENCH_PID=$!

echo "  Waiting for iter ${PHASE1_END} (CSV lines >= ${P1_LINES})..."
wait_for_csv_lines "$BENCH_CSV" "$P1_LINES"

echo "  Injecting stress..."
TS=$(get_ts_us)
echo "${TS},stress_start" >> "$EVENTS_LOG"
taskset -c "$PIN_CPUS" stress-ng --cpu "$STRESS_WORKERS" --timeout 300s &>/dev/null &
STRESS_PID=$!

echo "  Waiting for iter ${PHASE2_END} (CSV lines >= ${P2_LINES})..."
wait_for_csv_lines "$BENCH_CSV" "$P2_LINES"

echo "  Removing stress..."
TS=$(get_ts_us)
echo "${TS},stress_stop" >> "$EVENTS_LOG"
kill "$STRESS_PID" 2>/dev/null || true
wait "$STRESS_PID" 2>/dev/null || true
STRESS_PID=""

echo "  Waiting for benchmark to finish..."
wait "$BENCH_PID" 2>/dev/null || true
echo "  [B] v2-minimal done."

write_run_meta_end

unset NCCL_CC_V2_MINIMAL
unset NCCL_CC_V2_TIMELINE_FILE
export NCCL_DEBUG=WARN

echo ""
echo "============================================"
echo " Timeline Experiment Complete"
echo " NCCL timeline:  $TIMELINE_CSV"
echo " Bench (v2min):  $BENCH_CSV"
echo " Bench (base):   $BASELINE_CSV"
echo " Events (v2min): $EVENTS_LOG"
echo " Events (base):  $BASELINE_EVENTS"
echo "============================================"
echo ""
echo "分析命令："
echo "  python3 analyze_timeline.py $TIMELINE_CSV $BENCH_CSV $EVENTS_LOG $BASELINE_CSV $BASELINE_EVENTS"
