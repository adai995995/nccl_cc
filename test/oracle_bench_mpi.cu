/*
 * oracle_bench_mpi: 多进程（MPI）版 Oracle benchmark
 *
 * - 每个 MPI rank 绑定 1 张 GPU（默认用 local rank 选卡）
 * - NCCL communicator 覆盖所有 ranks（支持双机 16 卡等）
 * - 只由 rank0 写 CSV（避免多进程写同一文件）
 *
 * 环境变量：
 *   ITERS       测量迭代数（默认 500）
 *   WARMUP      预热迭代数（默认 50）
 *   COUNT       float 元素数（默认 1048576）
 *   CSV_OUTPUT  rank0 输出逐迭代 CSV（iter,ts_us,wall_us,max_gpu_us,min_gpu_us,skew_us）
 *
 * 说明：
 * - wall_us 使用“全局最大 rank wall 时间”（MPI_Reduce max），更贴近 collective tail
 * - max_gpu_us / min_gpu_us / skew_us 使用“全局 max/min GPU event 时间”
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define CHECK_NCCL(call) do { \
  ncclResult_t res__ = (call); \
  if (res__ != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res__)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

static int getEnvInt(const char* name, int def) {
  const char* s = getenv(name);
  return (s && *s) ? atoi(s) : def;
}

static uint64_t getMonoUs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static int getLocalRank(int world_rank) {
  const char* s = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (s && *s) return atoi(s);
  s = getenv("MPI_LOCALRANKID"); // some MPI distros
  if (s && *s) return atoi(s);
  // fallback: assume ranks are mapped by node contiguously
  return world_rank;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int iters = getEnvInt("ITERS", 500);
  int warmup = getEnvInt("WARMUP", 50);
  size_t count = (size_t)getEnvInt("COUNT", 1048576);
  const char* csv_file = getenv("CSV_OUTPUT");

  int devCount = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if (devCount <= 0) {
    if (world_rank == 0) fprintf(stderr, "ERROR: cudaGetDeviceCount returned %d\n", devCount);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int local_rank = getLocalRank(world_rank);
  int dev = local_rank % devCount;
  CHECK_CUDA(cudaSetDevice(dev));

  if (world_rank == 0) {
    double dataMB = (double)(count * sizeof(float)) / 1e6;
    printf("oracle_bench_mpi: world_size=%d iters=%d warmup=%d count=%zu (%.2f MB)\n",
           world_size, iters, warmup, count, dataMB);
    printf("NCCL env: NCCL_AIMD_ENABLE=%s NCCL_CC_V2_MINIMAL=%s NCCL_CC_ORACLE_FACTOR=%s\n",
           getenv("NCCL_AIMD_ENABLE") ? getenv("NCCL_AIMD_ENABLE") : "(unset)",
           getenv("NCCL_CC_V2_MINIMAL") ? getenv("NCCL_CC_V2_MINIMAL") : "(unset)",
           getenv("NCCL_CC_ORACLE_FACTOR") ? getenv("NCCL_CC_ORACLE_FACTOR") : "(unset)");
    fflush(stdout);
  }

  ncclUniqueId id;
  if (world_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  MPI_Bcast((void*)&id, (int)sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  cudaStream_t stream;
  cudaEvent_t evStart, evStop;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&evStart));
  CHECK_CUDA(cudaEventCreate(&evStop));

  float* sendbuf = nullptr;
  float* recvbuf = nullptr;
  CHECK_CUDA(cudaMalloc(&sendbuf, count * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&recvbuf, count * sizeof(float)));
  CHECK_CUDA(cudaMemset(sendbuf, 0x3f, count * sizeof(float)));

  // Warmup
  for (int w = 0; w < warmup; w++) {
    CHECK_CUDA(cudaEventRecord(evStart, stream));
    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
    CHECK_CUDA(cudaEventRecord(evStop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
  }

  FILE* csv = nullptr;
  if (world_rank == 0 && csv_file && *csv_file) {
    csv = fopen(csv_file, "w");
    if (!csv) {
      fprintf(stderr, "ERROR: cannot open CSV_OUTPUT=%s\n", csv_file);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(csv, "iter,ts_us,wall_us,max_gpu_us,min_gpu_us,skew_us\n");
    fflush(csv);
  }

  for (int iter = 0; iter < iters; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t t0 = getMonoUs();

    CHECK_CUDA(cudaEventRecord(evStart, stream));
    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
    CHECK_CUDA(cudaEventRecord(evStop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    uint64_t t1 = getMonoUs();
    double wall_us_local = (double)(t1 - t0);

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, evStart, evStop));
    double gpu_us_local = (double)ms * 1000.0;

    double wall_us_max = 0.0;
    MPI_Reduce(&wall_us_local, &wall_us_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double gpu_us_max = 0.0, gpu_us_min = 0.0;
    MPI_Reduce(&gpu_us_local, &gpu_us_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gpu_us_local, &gpu_us_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (csv && world_rank == 0) {
      uint64_t ts_us = getMonoUs();
      double skew_us = gpu_us_max - gpu_us_min;
      fprintf(csv, "%d,%lu,%.1f,%.1f,%.1f,%.1f\n",
              iter, (unsigned long)ts_us, wall_us_max, gpu_us_max, gpu_us_min, skew_us);
      if ((iter % 50) == 0) fflush(csv);
    }
  }

  if (csv) fclose(csv);

  CHECK_CUDA(cudaFree(sendbuf));
  CHECK_CUDA(cudaFree(recvbuf));
  CHECK_CUDA(cudaEventDestroy(evStart));
  CHECK_CUDA(cudaEventDestroy(evStop));
  CHECK_CUDA(cudaStreamDestroy(stream));

  CHECK_NCCL(ncclCommDestroy(comm));

  MPI_Finalize();
  return 0;
}

