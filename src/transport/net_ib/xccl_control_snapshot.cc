/*************************************************************************
 * XCCL 控制快照（控制面 -> 数据面）共享内存读端
 ************************************************************************/

#include "xccl_control_snapshot.h"
#include "common.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static void* g_ctrl_mmap = NULL;
static size_t g_ctrl_mmap_size = 0;
static int g_ctrl_init_attempted = 0;
static int g_ctrl_enabled = -1;

int xcclControlSnapshotEnabled(void) {
  if (g_ctrl_enabled == -1) {
    const char* e = getenv(NCCL_CC_CONTROL_SNAPSHOT_ENABLE_ENV);
    g_ctrl_enabled = (e && atoi(e) > 0) ? 1 : 0;
  }
  return g_ctrl_enabled;
}

int xcclControlSnapshotTransportIsMapped(void) {
  return (g_ctrl_mmap != NULL && g_ctrl_mmap != MAP_FAILED);
}

ncclResult_t xcclControlSnapshotTransportInit(void) {
  if (g_ctrl_init_attempted) return ncclSuccess;
  g_ctrl_init_attempted = 1;

  if (!xcclControlSnapshotEnabled()) return ncclSuccess;

  const char* path = getenv(NCCL_CC_CONTROL_SNAPSHOT_MMAP_PATH_ENV);
  const char* shm_name = getenv(NCCL_CC_CONTROL_SNAPSHOT_SHM_NAME_ENV);
  if (!shm_name || shm_name[0] == '\0') shm_name = "/xccl_control_snapshot";

  int fd = -1;
  if (path && path[0] != '\0') {
    fd = open(path, O_RDONLY);
    if (fd < 0) {
      INFO(NCCL_NET, "XCCL ctrl snapshot: open %s failed: %s (external control disabled)",
           path, strerror(errno));
      return ncclSuccess;
    }
  } else {
    fd = shm_open(shm_name, O_RDONLY, 0);
    if (fd < 0) {
      INFO(NCCL_NET, "XCCL ctrl snapshot: shm_open %s failed: %s (external control disabled)",
           shm_name, strerror(errno));
      return ncclSuccess;
    }
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    INFO(NCCL_NET, "XCCL ctrl snapshot: fstat failed: %s", strerror(errno));
    return ncclSuccess;
  }
  if (st.st_size < (off_t)sizeof(XcclControlSnapshot)) {
    close(fd);
    INFO(NCCL_NET, "XCCL ctrl snapshot: region size %ld < %zu (external control disabled)",
         (long)st.st_size, sizeof(XcclControlSnapshot));
    return ncclSuccess;
  }

  g_ctrl_mmap_size = (size_t)st.st_size;
  g_ctrl_mmap = mmap(NULL, g_ctrl_mmap_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (g_ctrl_mmap == MAP_FAILED) {
    g_ctrl_mmap = NULL;
    g_ctrl_mmap_size = 0;
    INFO(NCCL_NET, "XCCL ctrl snapshot: mmap failed: %s", strerror(errno));
    return ncclSuccess;
  }

  INFO(NCCL_NET, "XCCL ctrl snapshot: mmap active (%zu bytes)", g_ctrl_mmap_size);
  return ncclSuccess;
}

void xcclControlSnapshotTransportFini(void) {
  if (g_ctrl_mmap && g_ctrl_mmap != MAP_FAILED) {
    munmap(g_ctrl_mmap, g_ctrl_mmap_size);
    g_ctrl_mmap = NULL;
    g_ctrl_mmap_size = 0;
  }
  g_ctrl_init_attempted = 0;
}

ncclResult_t xcclControlSnapshotRead(XcclControlSnapshot* out) {
  if (!out) return ncclInvalidArgument;
  if (!xcclControlSnapshotEnabled() || !xcclControlSnapshotTransportIsMapped()) {
    return ncclInternalError;
  }

  XcclControlSnapshot* snap = (XcclControlSnapshot*)g_ctrl_mmap;
  int max_retries = 8;
  const char* er = getenv(NCCL_CC_CONTROL_SNAPSHOT_READ_RETRIES_ENV);
  if (er && atoi(er) > 0) max_retries = atoi(er);

  for (int i = 0; i < max_retries; i++) {
    uint64_t v1 = __atomic_load_n(&snap->version, __ATOMIC_ACQUIRE);
    if (v1 == 0 || (v1 & 1ULL)) continue;
    XcclControlSnapshot tmp;
    memcpy(&tmp, snap, sizeof(tmp));
    uint64_t v2 = __atomic_load_n(&snap->version, __ATOMIC_ACQUIRE);
    if (v1 == v2 && v1 != 0 && ((v1 & 1ULL) == 0ULL) && tmp.version == v1 &&
        tmp.magic == XCCL_CONTROL_SNAPSHOT_MAGIC &&
        tmp.layout_version == XCCL_CONTROL_SNAPSHOT_LAYOUT_VERSION &&
        tmp.struct_size == (uint16_t)sizeof(XcclControlSnapshot)) {
      *out = tmp;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

