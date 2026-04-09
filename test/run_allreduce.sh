#!/usr/bin/env bash
# 编译并运行 allreduce 冒烟测试。
# 用法：
#   ./run_allreduce.sh
#   NCCL_DEBUG=INFO NCCL_NET=IB ./run_allreduce.sh
#   NUM_GPUS=2 COUNT=4096 ./run_allreduce.sh

set -euo pipefail
cd "$(dirname "$0")"

NCCL_BUILD="${NCCL_BUILD:-$(cd .. && pwd)/build}"
export LD_LIBRARY_PATH="${NCCL_BUILD}/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${NCCL_BUILD}/lib/libnccl.so" ]] && ! ls "${NCCL_BUILD}/lib"/libnccl.so.* >/dev/null 2>&1; then
  echo "ERROR: libnccl not found under ${NCCL_BUILD}/lib"
  echo "Build NCCL first from repo root, e.g.:"
  echo "  cd $(cd .. && pwd) && ./build_a100.sh src.build"
  exit 1
fi

make NCCL_BUILD="${NCCL_BUILD}" all

echo "=== Running allreduce_test (NUM_GPUS=${NUM_GPUS:-1} COUNT=${COUNT:-1048576}) ==="
exec ./allreduce_test
