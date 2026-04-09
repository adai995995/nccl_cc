#!/usr/bin/env bash
# 单机 8 卡 allreduce：尽量削弱 P2P/SHM，并打开 AIMD + Telemetry hint（需另开终端写 sidecar）。
# 注意：NVLink 拓扑下 NCCL 仍可能大量走非 IB 路径；多机才稳定走 NET/IB。
#
# 用法：
#   终端 A：python3 ../tools/xccl_telemetry_sidecar/xccl_telemetry_sidecar.py \
#             --file "$(pwd)/th_data/xccl_telemetry_hint.bin" --ifaces eth0
#   终端 B：./run_8gpu_net_hint.sh

set -euo pipefail
cd "$(dirname "$0")"

HINT_FILE="${NCCL_CC_HINT_MMAP_PATH:-$(pwd)/th_data/xccl_telemetry_hint.bin}"
NCCL_BUILD="${NCCL_BUILD:-$(cd .. && pwd)/build}"
export LD_LIBRARY_PATH="${NCCL_BUILD}/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"

# 削弱机内 P2P/SHM（不保证全部 payload 走 RoCE，见 README）
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# Telemetry hint + AIMD（读共享区 + 发送路径刷新 hint）
export NCCL_CC_HINT_ENABLE="${NCCL_CC_HINT_ENABLE:-1}"
export NCCL_CC_HINT_MMAP_PATH="$HINT_FILE"
export NCCL_CC_HINT_TTL_NS="${NCCL_CC_HINT_TTL_NS:-10000000000}"
export NCCL_AIMD_ENABLE="${NCCL_AIMD_ENABLE:-1}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-NET}"

export NUM_GPUS="${NUM_GPUS:-8}"
export COUNT="${COUNT:-1048576}"

echo "=== HINT_FILE=$HINT_FILE (sidecar 需写同一文件) ==="
echo "=== NUM_GPUS=$NUM_GPUS COUNT=$COUNT ==="
echo "=== NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE ==="

if [[ ! -x ./allreduce_test ]]; then
  echo "Building allreduce_test..."
  make NCCL_BUILD="$NCCL_BUILD" all
fi

exec ./allreduce_test
