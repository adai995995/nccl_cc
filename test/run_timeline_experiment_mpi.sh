#!/usr/bin/env bash
# ============================================================================
# Phase V3 时间线实验（MPI 双机/多机版）
#
# 目标：复用单机 V3 的“CLEAN1 → STRESS → CLEAN2 + timeline 对齐”方法，
#      但将 benchmark 改为 MPI 多进程 + NCCL 跨节点（例如 2 节点 x 8 GPU = 16 ranks）。
#
# 产物（OUTDIR 下）：
#   - baseline_latency_<ts>.csv / baseline_<ts>.log / baseline_events_<ts>.csv
#   - bench_latency_<ts>.csv    / bench_<ts>.log    / events_<ts>.csv
#   - timelines/nccl_timeline_<ts>_rank<r>.csv      (每 rank 一份，避免冲突)
#
# 约定：
# - stress 仅在 STRESS_NODE 上注入（用 ssh），保持“单点 host pressure”
# - bench CSV 仅 rank0 输出（iter,ts_us,wall_us,max_gpu_us,min_gpu_us,skew_us）
#
# 用法示例：
#   OUTDIR=timeline_results_mpi \
#   HOSTS="192.168.3.20:8,192.168.3.172:8" \
#   STRESS_NODE=192.168.3.20 \
#   PIN_CPUS=8-31 \
#   ./run_timeline_experiment_mpi.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---- 可配参数 ----
HOSTS="${HOSTS:-192.168.3.20:8,192.168.3.172:8}"   # mpirun -H 的格式
STRESS_NODE="${STRESS_NODE:-192.168.3.20}"
SSH_OPTS="${SSH_OPTS:-}"                            # 例如：-o StrictHostKeyChecking=no

ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-50}"
COUNT="${COUNT:-1048576}"

PIN_CPUS="${PIN_CPUS:-8-31}"         # 只用于 stress 注入（bench 进程不强绑核）
STRESS_WORKERS="${STRESS_WORKERS:-8}"
OUTDIR="${OUTDIR:-timeline_results_mpi}"

PHASE1_END="${PHASE1_END:-300}"
PHASE2_END="${PHASE2_END:-700}"

NCCL_BUILD="${NCCL_BUILD:-$(cd .. && pwd)/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${NCCL_BUILD}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${NCCL_BUILD}/lib/libnccl.so" ]] && ! ls "${NCCL_BUILD}/lib"/libnccl.so.* >/dev/null 2>&1; then
  echo "ERROR: libnccl not found in ${NCCL_BUILD}/lib"
  exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: mpirun not found"
  exit 1
fi

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "ERROR: stress-ng is required on stress node"
  exit 1
fi

make NCCL_BUILD="${NCCL_BUILD}" oracle_bench_mpi
chmod +x ./mpi_rank_wrapper.sh

mkdir -p "$OUTDIR"
OUTDIR_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTDIR")"
mkdir -p "$OUTDIR_ABS"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

parse_hosts() {
  # Extract unique hostnames/IPs from HOSTS="h1:8,h2:8"
  local h
  IFS=',' read -ra HS <<< "$HOSTS"
  for h in "${HS[@]}"; do
    echo "${h%%:*}"
  done | awk '!seen[$0]++'
}

STAGE_SUBDIR_REMOTE="${STAGE_SUBDIR_REMOTE:-/tmp/xccl_timeline_${TIMESTAMP}}"
REMOTE_OUT_DIR="${STAGE_SUBDIR_REMOTE}/out"
REMOTE_TIMELINES_DIR="${REMOTE_OUT_DIR}/timelines"

deploy_to_hosts() {
  local host
  for host in $(parse_hosts); do
    echo "Staging payload to ${host}:${STAGE_SUBDIR_REMOTE} ..."
    # Create base dir + subdirs first (avoid relying on tar to create them).
    ssh ${SSH_OPTS} "$host" "rm -rf '${STAGE_SUBDIR_REMOTE}' && mkdir -p '${STAGE_SUBDIR_REMOTE}/bin' '${STAGE_SUBDIR_REMOTE}/lib'"

    # Ship binaries/scripts to remote/bin
    tar -C "$(pwd)" -cf - oracle_bench_mpi mpi_rank_wrapper.sh | \
      ssh ${SSH_OPTS} "$host" "tar -C '${STAGE_SUBDIR_REMOTE}/bin' -xf -"

    # Ship libnccl.so* to remote/lib (optional)
    (
      shopt -s nullglob
      libs=( "${NCCL_BUILD}/lib"/libnccl.so* )
      if [[ ${#libs[@]} -gt 0 ]]; then
        # Pass basenames to tar so globbing happens in this shell safely.
        lib_basenames=()
        for p in "${libs[@]}"; do lib_basenames+=( "$(basename "$p")" ); done
        tar -C "${NCCL_BUILD}/lib" -cf - "${lib_basenames[@]}" | \
          ssh ${SSH_OPTS} "$host" "tar -C '${STAGE_SUBDIR_REMOTE}/lib' -xf -"
      fi
    )

    # Verify payload exists
    ssh ${SSH_OPTS} "$host" "ls -la '${STAGE_SUBDIR_REMOTE}/bin' && test -x '${STAGE_SUBDIR_REMOTE}/bin/mpi_rank_wrapper.sh' && test -x '${STAGE_SUBDIR_REMOTE}/bin/oracle_bench_mpi'"
  done
}

prepare_remote_outdirs() {
  local host
  for host in $(parse_hosts); do
    ssh ${SSH_OPTS} "$host" "mkdir -p '${REMOTE_OUT_DIR}' '${REMOTE_TIMELINES_DIR}'"
  done
}

start_stress_remote() {
  # Start stress-ng on STRESS_NODE in background, print its PID.
  ssh ${SSH_OPTS} "$STRESS_NODE" "nohup taskset -c '${PIN_CPUS}' stress-ng --cpu '${STRESS_WORKERS}' --cpu-method all --timeout 300s >/tmp/xccl_stress_${TIMESTAMP}.log 2>&1 & echo \$!"
}

stop_stress_remote() {
  local pid="$1"
  if [[ -z "${pid:-}" ]]; then return 0; fi
  ssh ${SSH_OPTS} "$STRESS_NODE" "kill '${pid}' >/dev/null 2>&1 || true"
}

collect_outputs() {
  local label="$1"
  local rank0_host
  rank0_host="$(echo "$HOSTS" | awk -F',' '{print $1}' | awk -F':' '{print $1}')"

  mkdir -p "${OUTDIR_ABS}/timelines_${label}_${TIMESTAMP}"

  echo "Collecting bench CSV from rank0 host ${rank0_host}..."
  scp ${SSH_OPTS} "${rank0_host}:${REMOTE_OUT_DIR}/${label}_latency_${TIMESTAMP}.csv" "${OUTDIR_ABS}/" >/dev/null || true

  echo "Collecting timelines from all hosts..."
  local host
  for host in $(parse_hosts); do
    scp ${SSH_OPTS} "${host}:${REMOTE_TIMELINES_DIR}/nccl_timeline_${TIMESTAMP}_rank"*.csv \
      "${OUTDIR_ABS}/timelines_${label}_${TIMESTAMP}/" >/dev/null 2>&1 || true
  done
}

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

P1_LINES=$(( PHASE1_END + 1 ))
P2_LINES=$(( PHASE2_END + 1 ))

echo "============================================"
echo " Phase V3 Timeline Experiment (MPI)  $(date)"
echo " HOSTS=$HOSTS"
echo " STRESS_NODE=$STRESS_NODE PIN_CPUS=$PIN_CPUS workers=$STRESS_WORKERS"
echo " ITERS=$ITERS WARMUP=$WARMUP COUNT=$COUNT"
echo " Phases: clean(0-${PHASE1_END}) stress(${PHASE1_END}-${PHASE2_END}) clean(${PHASE2_END}-${ITERS})"
echo " OUTDIR=$OUTDIR"
echo "============================================"

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export ITERS WARMUP COUNT

echo "Preparing stage dir and deploying to all hosts..."
deploy_to_hosts
prepare_remote_outdirs
echo "Stage ready."

run_mpi() {
  local label="$1"
  local aimd_enable="$2"
  local v2_minimal="$3"
  local timeline_dir="$4"
  local bench_csv="$5"
  local log_path="$6"

  export NCCL_AIMD_ENABLE="$aimd_enable"
  if [[ "$v2_minimal" == "1" ]]; then
    export NCCL_CC_EPOCH_ENABLE=1
    export NCCL_CC_V2_MINIMAL=1
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=NET
  else
    unset NCCL_CC_EPOCH_ENABLE 2>/dev/null || true
    unset NCCL_CC_V2_MINIMAL 2>/dev/null || true
    export NCCL_DEBUG=WARN
    unset NCCL_DEBUG_SUBSYS 2>/dev/null || true
  fi

  export WRAP_BENCH_CSV="$bench_csv"
  export WRAP_TIMELINE_DIR="$timeline_dir"
  export WRAP_TAG="$TIMESTAMP"

  echo ""
  echo "==== [$label] mpirun ===="
  # Ensure remote nodes can find our staged libs first.
  export LD_LIBRARY_PATH="${STAGE_SUBDIR_REMOTE}/lib:${LD_LIBRARY_PATH}"
  mpirun --allow-run-as-root -np 16 -H "$HOSTS" \
    --wdir "${STAGE_SUBDIR_REMOTE}/bin" \
    --tag-output \
    -x LD_LIBRARY_PATH \
    -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
    -x NCCL_SOCKET_IFNAME -x NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE \
    -x NCCL_AIMD_ENABLE -x NCCL_CC_EPOCH_ENABLE -x NCCL_CC_V2_MINIMAL \
    -x WRAP_BENCH_CSV -x WRAP_TIMELINE_DIR -x WRAP_TAG \
    "${STAGE_SUBDIR_REMOTE}/bin/mpi_rank_wrapper.sh" 2>&1 | tee "$log_path"
}

# ---- 实验 A：Baseline ----
BASELINE_CSV_REMOTE="${REMOTE_OUT_DIR}/baseline_latency_${TIMESTAMP}.csv"
BASELINE_LOG="${OUTDIR}/baseline_${TIMESTAMP}.log"
BASELINE_EVENTS="${OUTDIR}/baseline_events_${TIMESTAMP}.csv"
echo "ts_us,event" > "$BASELINE_EVENTS"

run_mpi "A Baseline (no CC) with phased stress" 0 0 "" "$BASELINE_CSV_REMOTE" "$BASELINE_LOG" &
BENCH_PID=$!

echo "  Waiting for iter ${PHASE1_END} (CSV lines >= ${P1_LINES})..."
wait_for_csv_lines "$BASELINE_CSV_REMOTE" "$P1_LINES"

echo "  Injecting stress on ${STRESS_NODE}..."
TS=$(get_ts_us)
echo "${TS},stress_start" >> "$BASELINE_EVENTS"
STRESS_PID=$(start_stress_remote || true)

echo "  Waiting for iter ${PHASE2_END} (CSV lines >= ${P2_LINES})..."
wait_for_csv_lines "$BASELINE_CSV_REMOTE" "$P2_LINES"

echo "  Removing stress..."
TS=$(get_ts_us)
echo "${TS},stress_stop" >> "$BASELINE_EVENTS"
stop_stress_remote "$STRESS_PID"

echo "  Waiting for baseline to finish..."
wait "$BENCH_PID" 2>/dev/null || true
collect_outputs "baseline"
echo "  [A] Baseline done."

# ---- 实验 B：v2-minimal ----
TIMELINES_DIR_REMOTE="${REMOTE_TIMELINES_DIR}"
BENCH_CSV_REMOTE="${REMOTE_OUT_DIR}/bench_latency_${TIMESTAMP}.csv"
BENCH_LOG="${OUTDIR}/bench_${TIMESTAMP}.log"
EVENTS_LOG="${OUTDIR}/events_${TIMESTAMP}.csv"
echo "ts_us,event" > "$EVENTS_LOG"

run_mpi "B v2-minimal with phased stress" 1 1 "$TIMELINES_DIR_REMOTE" "$BENCH_CSV_REMOTE" "$BENCH_LOG" &
BENCH_PID=$!

echo "  Waiting for iter ${PHASE1_END} (CSV lines >= ${P1_LINES})..."
wait_for_csv_lines "$BENCH_CSV_REMOTE" "$P1_LINES"

echo "  Injecting stress on ${STRESS_NODE}..."
TS=$(get_ts_us)
echo "${TS},stress_start" >> "$EVENTS_LOG"
STRESS_PID=$(start_stress_remote || true)

echo "  Waiting for iter ${PHASE2_END} (CSV lines >= ${P2_LINES})..."
wait_for_csv_lines "$BENCH_CSV_REMOTE" "$P2_LINES"

echo "  Removing stress..."
TS=$(get_ts_us)
echo "${TS},stress_stop" >> "$EVENTS_LOG"
stop_stress_remote "$STRESS_PID"

echo "  Waiting for v2-minimal to finish..."
wait "$BENCH_PID" 2>/dev/null || true
collect_outputs "v2min"
echo "  [B] v2-minimal done."

echo ""
echo "============================================"
echo " Timeline Experiment (MPI) Complete"
echo " Baseline bench: ${OUTDIR_ABS}/baseline_latency_${TIMESTAMP}.csv"
echo " v2-min bench : ${OUTDIR_ABS}/bench_latency_${TIMESTAMP}.csv"
echo " Timelines dir: ${OUTDIR_ABS}/timelines_v2min_${TIMESTAMP} (and baseline)"
echo " Events (v2min): $EVENTS_LOG"
echo " Events (base):  $BASELINE_EVENTS"
echo "============================================"
echo ""
echo "注意：analyze_timeline.py 目前只接受单个 nccl_timeline.csv。"
echo "      MPI 模式下每 rank 一份，建议先选 rank0："
echo "  python3 analyze_timeline.py ${OUTDIR_ABS}/timelines_v2min_${TIMESTAMP}/nccl_timeline_${TIMESTAMP}_rank0.csv ${OUTDIR_ABS}/bench_latency_${TIMESTAMP}.csv $EVENTS_LOG ${OUTDIR_ABS}/baseline_latency_${TIMESTAMP}.csv $BASELINE_EVENTS"

