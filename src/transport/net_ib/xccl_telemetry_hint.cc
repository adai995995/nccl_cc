/*************************************************************************
 * XCCL Phase 3：共享内存 Telemetry hint 读端（sidecar 写端见设计文档）
 ************************************************************************/

#include "xccl_telemetry_hint.h"
#include "common.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static void* g_telemetry_mmap = NULL;
static size_t g_telemetry_mmap_size = 0;
static int g_telemetry_init_attempted = 0;

int xcclTelemetryHintTransportIsMapped(void) {
  return (g_telemetry_mmap != NULL && g_telemetry_mmap != MAP_FAILED);
}

ncclResult_t xcclTelemetryHintTransportInit(void) {
  if (g_telemetry_init_attempted) {
    return ncclSuccess;
  }
  g_telemetry_init_attempted = 1;

  const char* en = getenv(NCCL_CC_HINT_ENABLE_ENV);
  if (!en || atoi(en) <= 0) {
    return ncclSuccess;
  }

  const char* path = getenv(NCCL_CC_HINT_MMAP_PATH_ENV);
  const char* shm_name = getenv(NCCL_CC_HINT_SHM_NAME_ENV);
  if (!shm_name || shm_name[0] == '\0') {
    shm_name = "/xccl_telemetry_hint";
  }

  int fd = -1;
  if (path && path[0] != '\0') {
    fd = open(path, O_RDONLY);
    if (fd < 0) {
      INFO(NCCL_NET, "XCCL hint: open %s failed: %s (hint disabled)", path, strerror(errno));
      return ncclSuccess;
    }
  } else {
    fd = shm_open(shm_name, O_RDONLY, 0);
    if (fd < 0) {
      INFO(NCCL_NET, "XCCL hint: shm_open %s failed: %s (hint disabled)", shm_name, strerror(errno));
      return ncclSuccess;
    }
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    INFO(NCCL_NET, "XCCL hint: fstat failed: %s", strerror(errno));
    return ncclSuccess;
  }
  if (st.st_size < (off_t)sizeof(TelemetryHintSnapshot)) {
    close(fd);
    INFO(NCCL_NET, "XCCL hint: region size %ld < %zu (hint disabled)", (long)st.st_size, sizeof(TelemetryHintSnapshot));
    return ncclSuccess;
  }

  g_telemetry_mmap_size = (size_t)st.st_size;
  g_telemetry_mmap = mmap(NULL, g_telemetry_mmap_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (g_telemetry_mmap == MAP_FAILED) {
    g_telemetry_mmap = NULL;
    g_telemetry_mmap_size = 0;
    INFO(NCCL_NET, "XCCL hint: mmap failed: %s", strerror(errno));
    return ncclSuccess;
  }

  INFO(NCCL_NET, "XCCL hint: telemetry mmap active (%zu bytes)", g_telemetry_mmap_size);
  return ncclSuccess;
}

void xcclTelemetryHintTransportFini(void) {
  if (g_telemetry_mmap && g_telemetry_mmap != MAP_FAILED) {
    munmap(g_telemetry_mmap, g_telemetry_mmap_size);
    g_telemetry_mmap = NULL;
    g_telemetry_mmap_size = 0;
  }
  g_telemetry_init_attempted = 0;
}

ncclResult_t xcclTelemetryHintReadSnapshot(TelemetryHintSnapshot* out) {
  if (!out) {
    return ncclInvalidArgument;
  }
  if (!xcclTelemetryHintTransportIsMapped()) {
    return ncclInternalError;
  }

  TelemetryHintSnapshot* snap = (TelemetryHintSnapshot*)g_telemetry_mmap;
  int max_retries = 8;
  const char* er = getenv(NCCL_CC_HINT_READ_RETRIES_ENV);
  if (er && atoi(er) > 0) {
    max_retries = atoi(er);
  }

  for (int i = 0; i < max_retries; i++) {
    uint64_t v1 = __atomic_load_n(&snap->version, __ATOMIC_ACQUIRE);
    if (v1 == 0) {
      return ncclInternalError;
    }
    if ((v1 % 2ULL) == 1ULL) {
      continue;
    }
    TelemetryHintSnapshot tmp;
    memcpy(&tmp, snap, sizeof(TelemetryHintSnapshot));
    uint64_t v2 = __atomic_load_n(&snap->version, __ATOMIC_ACQUIRE);
    if (v1 == v2 && v1 != 0 && (v1 % 2ULL) == 0ULL && tmp.version == v1) {
      *out = tmp;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}
