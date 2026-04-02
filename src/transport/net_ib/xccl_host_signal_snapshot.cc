/*************************************************************************
 * XCCL Host Signal Snapshot writer (NCCL side)
 ************************************************************************/

#include "xccl_host_signal_snapshot.h"
#include "common.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void* g_host_sig_mmap = NULL;
static size_t g_host_sig_mmap_size = 0;
static int g_host_sig_init_attempted = 0;
static int g_host_sig_enabled = -1;

int xcclHostSignalSnapshotEnabled(void) {
  if (g_host_sig_enabled == -1) {
    const char* e = getenv(NCCL_CC_HOST_SIGNAL_ENABLE_ENV);
    g_host_sig_enabled = (e && atoi(e) > 0) ? 1 : 0;
  }
  return g_host_sig_enabled;
}

int xcclHostSignalSnapshotIsMapped(void) {
  return (g_host_sig_mmap != NULL && g_host_sig_mmap != MAP_FAILED);
}

ncclResult_t xcclHostSignalSnapshotInit(void) {
  if (g_host_sig_init_attempted) return ncclSuccess;
  g_host_sig_init_attempted = 1;

  if (!xcclHostSignalSnapshotEnabled()) return ncclSuccess;

  const char* path = getenv(NCCL_CC_HOST_SIGNAL_MMAP_PATH_ENV);
  const char* shm_name = getenv(NCCL_CC_HOST_SIGNAL_SHM_NAME_ENV);
  if (!shm_name || shm_name[0] == '\0') shm_name = "/xccl_host_signal_snapshot";

  int fd = -1;
  if (path && path[0] != '\0') {
    fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
      INFO(NCCL_NET, "XCCL host signal: open %s failed: %s", path, strerror(errno));
      return ncclSuccess;
    }
  } else {
    fd = shm_open(shm_name, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
      INFO(NCCL_NET, "XCCL host signal: shm_open %s failed: %s", shm_name, strerror(errno));
      return ncclSuccess;
    }
  }

  g_host_sig_mmap_size = 4096;
  if (ftruncate(fd, (off_t)g_host_sig_mmap_size) != 0) {
    INFO(NCCL_NET, "XCCL host signal: ftruncate failed: %s", strerror(errno));
    close(fd);
    return ncclSuccess;
  }

  g_host_sig_mmap = mmap(NULL, g_host_sig_mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (g_host_sig_mmap == MAP_FAILED) {
    g_host_sig_mmap = NULL;
    g_host_sig_mmap_size = 0;
    INFO(NCCL_NET, "XCCL host signal: mmap failed: %s", strerror(errno));
    return ncclSuccess;
  }

  memset(g_host_sig_mmap, 0, g_host_sig_mmap_size);
  INFO(NCCL_NET, "XCCL host signal: mmap active (%zu bytes)", g_host_sig_mmap_size);
  return ncclSuccess;
}

void xcclHostSignalSnapshotFini(void) {
  if (g_host_sig_mmap && g_host_sig_mmap != MAP_FAILED) {
    munmap(g_host_sig_mmap, g_host_sig_mmap_size);
    g_host_sig_mmap = NULL;
    g_host_sig_mmap_size = 0;
  }
  g_host_sig_init_attempted = 0;
}

ncclResult_t xcclHostSignalSnapshotPublish(const XcclHostSignalSnapshot* in) {
  if (!in) return ncclInvalidArgument;
  if (!xcclHostSignalSnapshotEnabled() || !xcclHostSignalSnapshotIsMapped()) return ncclSuccess;

  XcclHostSignalSnapshot* dst = (XcclHostSignalSnapshot*)g_host_sig_mmap;
  uint64_t v0 = __atomic_load_n(&dst->version, __ATOMIC_RELAXED);
  uint64_t odd = v0 + 1;
  uint64_t even = v0 + 2;
  if (even == 0) { odd = 1; even = 2; }

  __atomic_store_n(&dst->version, odd, __ATOMIC_RELEASE);

  XcclHostSignalSnapshot tmp = *in;
  tmp.version = odd;
  memcpy(dst, &tmp, sizeof(tmp));

  __atomic_thread_fence(__ATOMIC_RELEASE);
  dst->version = even;
  __atomic_store_n(&dst->version, even, __ATOMIC_RELEASE);
  return ncclSuccess;
}

