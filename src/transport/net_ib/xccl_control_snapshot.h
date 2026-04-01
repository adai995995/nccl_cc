/*************************************************************************
 * XCCL 控制快照（控制面 -> 数据面）共享内存读端
 ************************************************************************/

#ifndef XCCL_CONTROL_SNAPSHOT_H_
#define XCCL_CONTROL_SNAPSHOT_H_

#include "nccl.h"
#include <stdint.h>

#define NCCL_CC_CONTROL_SNAPSHOT_ENABLE_ENV "NCCL_CC_CONTROL_SNAPSHOT_ENABLE"
#define NCCL_CC_CONTROL_SNAPSHOT_SHM_NAME_ENV "NCCL_CC_CONTROL_SNAPSHOT_SHM_NAME"
#define NCCL_CC_CONTROL_SNAPSHOT_MMAP_PATH_ENV "NCCL_CC_CONTROL_SNAPSHOT_MMAP_PATH"
#define NCCL_CC_CONTROL_SNAPSHOT_READ_RETRIES_ENV "NCCL_CC_CONTROL_SNAPSHOT_READ_RETRIES"
#define NCCL_CC_CONTROL_SNAPSHOT_TTL_NS_ENV "NCCL_CC_CONTROL_SNAPSHOT_TTL_NS"

#define XCCL_CONTROL_SNAPSHOT_MAGIC 0x5843434cU /* 'XCCL' */
#define XCCL_CONTROL_SNAPSHOT_LAYOUT_VERSION 1

/* 与 docs/design_1.md 的推荐布局对齐（单写多读） */
#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint16_t layout_version;
  uint16_t struct_size;
  uint64_t comm_key;               /* 0=wildcard；否则必须匹配 CollectiveCC.comm_key */
  uint64_t version;
  uint64_t ts_ns;
  float    host_pressure_score;    /* [0,1] */
  float    network_pressure_score; /* [0,1] */
  uint8_t  mode;                   /* NORMAL/HOST/NET/MIXED/SEVERE 等 */
  uint8_t  severity;               /* 0..3 */
  uint16_t target_channels;
  uint16_t target_window;
  uint32_t pacing_ns;
  uint8_t  cooldown_level;
  uint8_t  stable_epochs;
  uint8_t  reserved[6];
} XcclControlSnapshot;
#pragma pack(pop)

ncclResult_t xcclControlSnapshotTransportInit(void);
void xcclControlSnapshotTransportFini(void);
int xcclControlSnapshotTransportIsMapped(void);
int xcclControlSnapshotEnabled(void);
ncclResult_t xcclControlSnapshotRead(XcclControlSnapshot* out);

#endif

