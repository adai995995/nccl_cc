/*************************************************************************
 * XCCL Phase 3：Telemetry hint 共享快照（与 design_docs 一致，36 字节）
 ************************************************************************/

#ifndef XCCL_TELEMETRY_HINT_H_
#define XCCL_TELEMETRY_HINT_H_

#include "nccl.h"
#include <stdint.h>

#define XCCL_TELEMETRY_HINT_LAYOUT_VERSION 1

#define NCCL_CC_HINT_ENABLE_ENV "NCCL_CC_HINT_ENABLE"
#define NCCL_CC_HINT_TTL_NS_ENV "NCCL_CC_HINT_TTL_NS"
#define NCCL_CC_HINT_SHM_NAME_ENV "NCCL_CC_HINT_SHM_NAME"
#define NCCL_CC_HINT_MMAP_PATH_ENV "NCCL_CC_HINT_MMAP_PATH"
#define NCCL_CC_HINT_READ_RETRIES_ENV "NCCL_CC_HINT_READ_RETRIES"
#define NCCL_CC_HINT_REFRESH_MIN_NS_ENV "NCCL_CC_HINT_REFRESH_MIN_NS"

#define XCCL_HINT_F_SEVERE        (1u << 0)
#define XCCL_HINT_F_CAUTION       (1u << 1)
#define XCCL_HINT_F_STALE_WRITER  (1u << 2)

#pragma pack(push, 1)
typedef struct {
  uint64_t version;
  uint64_t ts_ns;
  float    cnp_level;
  float    ce_level;
  float    pcie_stall_level;
  float    rnic_pressure;
  uint32_t flags;
} TelemetryHintSnapshot;
#pragma pack(pop)

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(TelemetryHintSnapshot) == 36, "TelemetryHintSnapshot must be 36 bytes");
#endif

ncclResult_t xcclTelemetryHintTransportInit(void);
void xcclTelemetryHintTransportFini(void);
int xcclTelemetryHintTransportIsMapped(void);

/* 成功写入 *out；失败不修改 *out */
ncclResult_t xcclTelemetryHintReadSnapshot(TelemetryHintSnapshot* out);

#endif
