/*
 * Oracle 最小实验 benchmark：反复跑 AllReduce，采集逐次 latency。
 *
 * 编译：在 test/ 目录 make oracle_bench
 * 运行：NUM_GPUS=2 ITERS=500 ./oracle_bench
 *
 * 环境变量：
 *   NUM_GPUS    使用 GPU 数（默认 2）
 *   ITERS       测量迭代数（默认 500）
 *   WARMUP      预热迭代数（默认 50）
 *   COUNT       float 元素数（默认 1048576 = 1M float = 4MB）
 *   CSV_OUTPUT  若设置，逐迭代数据写入该 CSV 文件
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <vector>
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
    return (s && *s) ? atoi(s) : def;
}

static double getTimeUs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static double vecMean(const std::vector<double>& v) {
    if (v.empty()) return 0;
    double s = 0;
    for (auto x : v) s += x;
    return s / v.size();
}

static double vecPercentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (double)(v.size() - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= (int)v.size()) return v.back();
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static double vecStddev(const std::vector<double>& v) {
    double m = vecMean(v);
    double s = 0;
    for (auto x : v) s += (x - m) * (x - m);
    return sqrt(s / (double)v.size());
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    int nDev = getEnvInt("NUM_GPUS", 2);
    int iters = getEnvInt("ITERS", 500);
    int warmup = getEnvInt("WARMUP", 50);
    size_t count = (size_t)getEnvInt("COUNT", 1048576);
    const char* csv_file = getenv("CSV_OUTPUT");

    int devCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    if (nDev > devCount) nDev = devCount;
    if (nDev < 2) {
        fprintf(stderr, "ERROR: need >= 2 GPUs, got %d\n", nDev);
        return 1;
    }

    double dataMB = (double)(count * sizeof(float)) / 1e6;
    printf("oracle_bench: nDev=%d count=%zu (%.2f MB) iters=%d warmup=%d\n",
           nDev, count, dataMB, iters, warmup);
    printf("NCCL env: NCCL_AIMD_ENABLE=%s NCCL_CC_ORACLE_FACTOR=%s\n",
           getenv("NCCL_AIMD_ENABLE") ? getenv("NCCL_AIMD_ENABLE") : "(unset)",
           getenv("NCCL_CC_ORACLE_FACTOR") ? getenv("NCCL_CC_ORACLE_FACTOR") : "(unset)");
    fflush(stdout);

    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));

    ncclComm_t* comms = new ncclComm_t[nDev];
    cudaStream_t* streams = new cudaStream_t[nDev];
    float** sendbuf = new float*[nDev];
    float** recvbuf = new float*[nDev];
    cudaEvent_t* evStart = new cudaEvent_t[nDev];
    cudaEvent_t* evStop = new cudaEvent_t[nDev];

    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nDev; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclCommInitRank(comms + i, nDev, id, i));
    }
    CHECK_NCCL(ncclGroupEnd());

    for (int i = 0; i < nDev; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamCreate(streams + i));
        CHECK_CUDA(cudaEventCreate(evStart + i));
        CHECK_CUDA(cudaEventCreate(evStop + i));
        CHECK_CUDA(cudaMalloc(&sendbuf[i], count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recvbuf[i], count * sizeof(float)));
        CHECK_CUDA(cudaMemset(sendbuf[i], 0x3f, count * sizeof(float)));
    }

    // Warmup
    printf("warmup %d iters...\n", warmup);
    fflush(stdout);
    for (int w = 0; w < warmup; w++) {
        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < nDev; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_NCCL(ncclAllReduce(sendbuf[i], recvbuf[i], count,
                                     ncclFloat, ncclSum, comms[i], streams[i]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int i = 0; i < nDev; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }
    }
    printf("warmup done.\n");
    fflush(stdout);

    // Measured iterations
    std::vector<double> wall_lat(iters);
    std::vector<double> max_gpu_lat(iters);
    std::vector<double> min_gpu_lat(iters);
    std::vector<double> skew(iters);
    std::vector<std::vector<double>> per_gpu(nDev, std::vector<double>(iters));

    FILE* csv = csv_file ? fopen(csv_file, "w") : NULL;
    if (csv) {
        fprintf(csv, "iter,ts_us,wall_us,max_gpu_us,min_gpu_us,skew_us");
        for (int g = 0; g < nDev; g++) fprintf(csv, ",gpu%d_us", g);
        fprintf(csv, "\n");
    }

    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < nDev; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaEventRecord(evStart[i], streams[i]));
        }

        double t0 = getTimeUs();

        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < nDev; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_NCCL(ncclAllReduce(sendbuf[i], recvbuf[i], count,
                                     ncclFloat, ncclSum, comms[i], streams[i]));
        }
        CHECK_NCCL(ncclGroupEnd());

        for (int i = 0; i < nDev; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaEventRecord(evStop[i], streams[i]));
        }

        for (int i = 0; i < nDev; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }

        double t1 = getTimeUs();
        wall_lat[iter] = t1 - t0;

        double mx = 0, mn = 1e18;
        for (int i = 0; i < nDev; i++) {
            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, evStart[i], evStop[i]));
            double us = (double)ms * 1000.0;
            per_gpu[i][iter] = us;
            if (us > mx) mx = us;
            if (us < mn) mn = us;
        }
        max_gpu_lat[iter] = mx;
        min_gpu_lat[iter] = mn;
        skew[iter] = mx - mn;

        if (csv) {
            struct timespec csv_ts;
            clock_gettime(CLOCK_MONOTONIC, &csv_ts);
            uint64_t csv_ts_us = (uint64_t)csv_ts.tv_sec * 1000000ULL + (uint64_t)csv_ts.tv_nsec / 1000ULL;
            fprintf(csv, "%d,%lu,%.1f,%.1f,%.1f,%.1f",
                    iter, (unsigned long)csv_ts_us, wall_lat[iter], mx, mn, skew[iter]);
            for (int g = 0; g < nDev; g++)
                fprintf(csv, ",%.1f", per_gpu[g][iter]);
            fprintf(csv, "\n");
        }
    }

    if (csv) {
        fclose(csv);
        printf("CSV written to: %s\n", csv_file);
    }

    // Statistics
    double bytes = (double)count * sizeof(float);
    double algoBw = bytes * 2.0 * (double)(nDev - 1) / (double)nDev;

    printf("\n========================================\n");
    printf("  Oracle Experiment Results\n");
    printf("========================================\n");
    printf("Config: nDev=%d  count=%zu  data=%.2f MB  iters=%d\n",
           nDev, count, dataMB, iters);
    printf("NCCL_AIMD_ENABLE=%s  NCCL_CC_ORACLE_FACTOR=%s\n",
           getenv("NCCL_AIMD_ENABLE") ? getenv("NCCL_AIMD_ENABLE") : "(unset)",
           getenv("NCCL_CC_ORACLE_FACTOR") ? getenv("NCCL_CC_ORACLE_FACTOR") : "(unset)");
    printf("\n");

    printf("Wall-clock latency (us):\n");
    printf("  mean=%8.1f  stddev=%8.1f\n", vecMean(wall_lat), vecStddev(wall_lat));
    printf("  p50 =%8.1f  p99   =%8.1f  max=%8.1f\n",
           vecPercentile(wall_lat, 50), vecPercentile(wall_lat, 99),
           vecPercentile(wall_lat, 100));

    printf("\nCollective max GPU latency (us):\n");
    printf("  mean=%8.1f  stddev=%8.1f\n", vecMean(max_gpu_lat), vecStddev(max_gpu_lat));
    printf("  p50 =%8.1f  p99   =%8.1f  max=%8.1f\n",
           vecPercentile(max_gpu_lat, 50), vecPercentile(max_gpu_lat, 99),
           vecPercentile(max_gpu_lat, 100));

    printf("\nRank skew (us):\n");
    printf("  mean=%8.1f  stddev=%8.1f\n", vecMean(skew), vecStddev(skew));
    printf("  p50 =%8.1f  p99   =%8.1f  max=%8.1f\n",
           vecPercentile(skew, 50), vecPercentile(skew, 99),
           vecPercentile(skew, 100));

    double mean_wall = vecMean(wall_lat);
    double bw_GBs = (mean_wall > 0) ? algoBw / (mean_wall * 1e-6) / 1e9 : 0;
    printf("\nAvg algo bandwidth: %.2f GB/s\n", bw_GBs);

    printf("========================================\n\n");
    fflush(stdout);

    // Cleanup
    for (int i = 0; i < nDev; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        cudaFree(sendbuf[i]);
        cudaFree(recvbuf[i]);
        cudaEventDestroy(evStart[i]);
        cudaEventDestroy(evStop[i]);
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }
    delete[] comms;
    delete[] streams;
    delete[] sendbuf;
    delete[] recvbuf;
    delete[] evStart;
    delete[] evStop;

    return 0;
}
