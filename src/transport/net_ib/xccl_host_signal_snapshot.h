/*************************************************************************
 * XCCL Host Signal Snapshot (NCCL -> sidecar)
 ************************************************************************/

#ifndef XCCL_HOST_SIGNAL_SNAPSHOT_H_
#define XCCL_HOST_SIGNAL_SNAPSHOT_H_

#include "nccl.h"
#include <stdint.h>

#define NCCL_CC_HOST_SIGNAL_ENABLE_ENV "NCCL_CC_HOST_SIGNAL_ENABLE"
#define NCCL_CC_HOST_SIGNAL_MMAP_PATH_ENV "NCCL_CC_HOST_SIGNAL_MMAP_PATH"
#define NCCL_CC_HOST_SIGNAL_SHM_NAME_ENV "NCCL_CC_HOST_SIGNAL_SHM_NAME"

#define XCCL_HOST_SIGNAL_MAGIC 0x58484353U /* 'XHCS' */
#define XCCL_HOST_SIGNAL_LAYOUT_VERSION 1

#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint16_t layout_version;
  uint16_t struct_size;
  uint64_t comm_key;               /* 0 = aggregated */
  uint64_t version;                /* odd/even publish protocol */
  uint64_t ts_ns;

  uint32_t cq_posted;
  uint32_t cq_completed;
  uint32_t cq_backlog;

  uint32_t rtt_baseline_us;
  uint32_t rtt_ewma_us;
  float completion_stretch;
  float cpu_poll_delay_norm;       /* reserved, currently 0 */
} XcclHostSignalSnapshot;
#pragma pack(pop)

ncclResult_t xcclHostSignalSnapshotInit(void);
void xcclHostSignalSnapshotFini(void);
int xcclHostSignalSnapshotEnabled(void);
int xcclHostSignalSnapshotIsMapped(void);
ncclResult_t xcclHostSignalSnapshotPublish(const XcclHostSignalSnapshot* in);

#endif

