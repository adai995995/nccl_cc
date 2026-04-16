#!/usr/bin/env bash
# ============================================================================
# 阶段 1 + 2：同一配置重复 N 次（不同 OUTDIR），每次跑完后调用 analyze_timeline.py，
# 最后用 summarize_phase12_runs.py 汇总 timeline_stats 中的 detector 指标与可选 baseline 分阶段延迟。
#
# 环境变量：
#   N              重复次数（默认 5）
#   PREFIX         OUTDIR 前缀，实际目录为 ${PREFIX}_run1 .. ${PREFIX}_runN（默认 phase12_sweep）
#   SUMMARY_OUT    汇总 CSV 路径（默认 ${PREFIX}_summary_<timestamp>.csv）
#   START          起始序号（默认 1），用于断点续跑
#   ONLY_SUMMARIZE 若设为 1，不跑实验，只对已有 ${PREFIX}_run* 目录做汇总
#
# 其余 ITERS、PHASE*、PIN_CPUS 等传给 run_timeline_experiment.sh（与单机脚本一致）。
#
# 示例：
#   N=5 PREFIX=det_baseline_v1 ./run_phase12_sweep.sh
#   ONLY_SUMMARIZE=1 PREFIX=det_baseline_v1 ./run_phase12_sweep.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

N="${N:-5}"
PREFIX="${PREFIX:-phase12_sweep}"
START="${START:-1}"
ONLY_SUMMARIZE="${ONLY_SUMMARIZE:-0}"

analyze_one_outdir() {
  local D="$1"
  local TL BL BE BV EV
  TL=$(ls -1 "$D"/nccl_timeline_*.csv 2>/dev/null | head -1) || true
  BL=$(ls -1 "$D"/bench_latency_*.csv 2>/dev/null | head -1) || true
  BE=$(ls -1 "$D"/events_*.csv 2>/dev/null | head -1) || true
  BV=$(ls -1 "$D"/baseline_latency_*.csv 2>/dev/null | head -1) || true
  EV=$(ls -1 "$D"/baseline_events_*.csv 2>/dev/null | head -1) || true
  if [[ -z "${TL}" || -z "${BL}" || -z "${BE}" || -z "${BV}" || -z "${EV}" ]]; then
    echo "ERROR: missing CSVs in ${D} (need nccl_timeline, bench_latency, events, baseline_latency, baseline_events)"
    return 1
  fi
  echo "  [analyze] $D"
  python3 "$(pwd)/analyze_timeline.py" \
    "$(pwd)/${TL}" "$(pwd)/${BL}" "$(pwd)/${BE}" "$(pwd)/${BV}" "$(pwd)/${EV}"
}

collect_outdirs() {
  local pat="${PREFIX}_run"
  if ! compgen -G "${pat}*" > /dev/null; then
    echo "ERROR: no directories matching ${pat}*"
    return 1
  fi
  # shellcheck disable=SC2086
  ls -d ${PREFIX}_run* 2>/dev/null | sort -V
}

if [[ "$ONLY_SUMMARIZE" == "1" ]]; then
  mapfile -t OUTDIRS < <(collect_outdirs)
  if [[ ${#OUTDIRS[@]} -eq 0 ]]; then
    exit 1
  fi
  SUMMARY_OUT="${SUMMARY_OUT:-${PREFIX}_summary_$(date +%Y%m%d_%H%M%S).csv}"
  python3 "$(pwd)/summarize_phase12_runs.py" "${OUTDIRS[@]}" | tee "$SUMMARY_OUT"
  echo ""
  echo "Summary written to: $(pwd)/$SUMMARY_OUT"
  exit 0
fi

SUMMARY_OUT="${SUMMARY_OUT:-${PREFIX}_summary_$(date +%Y%m%d_%H%M%S).csv}"
RUN_LIST=()

for i in $(seq "$START" "$N"); do
  OD="${PREFIX}_run${i}"
  echo ""
  echo "============================================"
  echo " Phase12 sweep  run ${i}/${N}  OUTDIR=${OD}"
  echo "============================================"
  OUTDIR="$OD" ./run_timeline_experiment.sh
  analyze_one_outdir "$OD"
  RUN_LIST+=("$OD")
done

echo ""
echo "============================================"
echo " Summarizing ${#RUN_LIST[@]} runs -> ${SUMMARY_OUT}"
echo "============================================"
python3 "$(pwd)/summarize_phase12_runs.py" "${RUN_LIST[@]}" | tee "$SUMMARY_OUT"
echo ""
echo "Optional JSON:"
echo "  python3 summarize_phase12_runs.py --json ${PREFIX}_summary.json ${RUN_LIST[@]}"
echo "Done. Summary: $(pwd)/$SUMMARY_OUT"
