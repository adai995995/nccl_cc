#!/usr/bin/env bash
set -euo pipefail

# Per-rank wrapper to avoid multiple ranks writing the same CSV/timeline file.
# Env:
#   WRAP_BENCH_CSV   : path for rank0 bench CSV (others disabled)
#   WRAP_TIMELINE_DIR: directory for per-rank NCCL timeline CSVs
#   WRAP_TAG         : string tag (timestamp) used in filenames

RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"

if [[ -n "${WRAP_TIMELINE_DIR:-}" ]]; then
  mkdir -p "${WRAP_TIMELINE_DIR}"
  export NCCL_CC_V2_TIMELINE_FILE="${WRAP_TIMELINE_DIR}/nccl_timeline_${WRAP_TAG}_rank${RANK}.csv"
fi

if [[ "${RANK}" == "0" ]]; then
  if [[ -n "${WRAP_BENCH_CSV:-}" ]]; then
    export CSV_OUTPUT="${WRAP_BENCH_CSV}"
  fi
else
  unset CSV_OUTPUT 2>/dev/null || true
fi

exec ./oracle_bench_mpi

