/*
 * 最小 NCCL AllReduce 冒烟测试（单机单进程，单卡或同进程多卡）。
 * 编译：在 test 目录 make
 * 运行：./run_allreduce.sh  或  NCCL_DEBUG=INFO ./allreduce_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(1); \
  } \
} while (0)

#define CHECK_NCCL(call) do { \
  ncclResult_t res__ = (call); \
  if (res__ != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res__)); \
    exit(1); \
  } \
} while (0)

static int getEnvInt(const char* name, int def) {
  const char* s = getenv(name);
  if (!s || !*s) return def;
  return atoi(s);
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  int nDev = getEnvInt("NUM_GPUS", 1);
  int devCount = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if (nDev < 1) nDev = 1;
  if (nDev > devCount) {
    fprintf(stderr, "NUM_GPUS=%d > cudaGetDeviceCount=%d, clamp\n", nDev, devCount);
    nDev = devCount;
  }
  if (nDev < 1) {
    fprintf(stderr, "No CUDA device\n");
    return 1;
  }

  const size_t count = (size_t)getEnvInt("COUNT", 1048576); /* 默认 1M float */
  if (count < 1) {
    fprintf(stderr, "COUNT invalid\n");
    return 1;
  }

  printf("allreduce_test: nDev=%d count=%zu floats op=Sum\n", nDev, count);

  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  ncclComm_t* comms = (ncclComm_t*)calloc((size_t)nDev, sizeof(ncclComm_t));
  cudaStream_t* streams = (cudaStream_t*)calloc((size_t)nDev, sizeof(cudaStream_t));
  float** sendbuff = (float**)calloc((size_t)nDev, sizeof(float*));
  float** recvbuff = (float**)calloc((size_t)nDev, sizeof(float*));
  if (!comms || !streams || !sendbuff || !recvbuff) {
    fprintf(stderr, "calloc failed\n");
    return 1;
  }

  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < nDev; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_NCCL(ncclCommInitRank(comms + i, nDev, id, i));
    CHECK_CUDA(cudaStreamCreate(streams + i));
  }
  CHECK_NCCL(ncclGroupEnd());

  for (int i = 0; i < nDev; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc((void**)(sendbuff + i), count * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)(recvbuff + i), count * sizeof(float)));
    /* 每卡 send: rank 作为常数，便于校验 */
    float* h = (float*)malloc(count * sizeof(float));
    if (!h) { fprintf(stderr, "malloc host failed\n"); return 1; }
    for (size_t j = 0; j < count; j++)
      h[j] = (float)(i + 1);
    CHECK_CUDA(cudaMemcpy(sendbuff[i], h, count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(recvbuff[i], h, count * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
  }

  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < nDev; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_NCCL(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], count,
                             ncclFloat, ncclSum, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());

  for (int i = 0; i < nDev; i++) {
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }

  /* 校验：sum 应为 nDev*(nDev+1)/2 * 1.0 每元素（每 rank 贡献 i+1） */
  float expected = 0.f;
  for (int r = 0; r < nDev; r++) expected += (float)(r + 1);
  int bad = 0;
  for (int i = 0; i < nDev; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    float* h = (float*)malloc(count * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h, recvbuff[i], count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < count; j++) {
      if (h[j] != expected) {
        if (bad < 5)
          fprintf(stderr, "mismatch dev=%d [%zu]: got %g expect %g\n", i, j, h[j], expected);
        bad++;
      }
    }
    free(h);
  }
  if (bad) {
    fprintf(stderr, "VERIFY FAIL: %d elements wrong (expected %g)\n", bad, expected);
    return 2;
  }
  printf("VERIFY OK: all elements == %g\n", expected);

  for (int i = 0; i < nDev; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    cudaFree(sendbuff[i]);
    cudaFree(recvbuff[i]);
    cudaStreamDestroy(streams[i]);
    ncclCommDestroy(comms[i]);
  }
  free(comms);
  free(streams);
  free(sendbuff);
  free(recvbuff);
  return 0;
}
