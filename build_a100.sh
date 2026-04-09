#!/usr/bin/env bash
# 仅针对 NVIDIA A100（Ampere，compute capability 8.0 / sm_80）编译，避免默认全架构 NVCC 耗时。
# 用法：在 nccl_cc 目录下执行：
#   ./build_a100.sh
#   ./build_a100.sh -j32 src.build
# 等价于自行设置：
#   export NVCC_GENCODE='-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80'
#
# 说明：第二段为 PTX，便于同驱动下 JIT；若确定仅部署在 A100 且要最小编译，可只保留 sm_80 一段。

set -euo pipefail
cd "$(dirname "$0")"

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80"

exec make -j"$(nproc)" src.build "$@"
