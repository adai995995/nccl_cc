# 依赖：先编译 NCCL，生成 ../../build/lib/libnccl.so* 与 ../../build/include/nccl.h
CUDA_HOME ?= /usr/local/cuda
NCCL_BUILD ?= $(abspath ../build)
NVCC ?= $(CUDA_HOME)/bin/nvcc

# 与仓库 build_a100 一致时可在外部设置 NVCC_GENCODE
NVCC_GENCODE ?=
NVCCFLAGS = -O2 -std=c++14 -I$(NCCL_BUILD)/include $(NVCC_GENCODE)
# 嵌入 rpath（nvcc 需用 -Xlinker，不能写 -Wl,-rpath,...）
LDFLAGS = -L$(NCCL_BUILD)/lib -Xlinker -rpath -Xlinker $(NCCL_BUILD)/lib \
	-L$(CUDA_HOME)/lib64 -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
	-lnccl -lcudart -lrt -lpthread

all: allreduce_test oracle_bench

allreduce_test: allreduce_test.cu
	@test -f $(NCCL_BUILD)/include/nccl.h || (echo "ERROR: $(NCCL_BUILD)/include/nccl.h not found. Build NCCL first: cd .. && ./build_a100.sh src.build" && exit 1)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

oracle_bench: oracle_bench.cu
	@test -f $(NCCL_BUILD)/include/nccl.h || (echo "ERROR: $(NCCL_BUILD)/include/nccl.h not found. Build NCCL first: cd .. && ./build_a100.sh src.build" && exit 1)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

# ---- Multi-node MPI benchmark (one GPU per rank) ----
# Build requirements:
# - OpenMPI (mpicxx/mpirun) installed
# - nvcc available
MPICXX ?= mpicxx
MPI_CFLAGS ?= $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_LDFLAGS ?= $(shell $(MPICXX) --showme:link 2>/dev/null)

oracle_bench_mpi: oracle_bench_mpi.cu
	@test -f $(NCCL_BUILD)/include/nccl.h || (echo "ERROR: $(NCCL_BUILD)/include/nccl.h not found. Build NCCL first: cd .. && ./build_a100.sh src.build" && exit 1)
	@command -v $(MPICXX) >/dev/null 2>&1 || (echo "ERROR: $(MPICXX) not found. Install OpenMPI (mpicxx/mpirun)." && exit 1)
	$(NVCC) $(NVCCFLAGS) $(MPI_CFLAGS) -o $@ $< $(LDFLAGS) $(MPI_LDFLAGS)

clean:
	rm -f allreduce_test oracle_bench oracle_bench_mpi

.PHONY: all clean
