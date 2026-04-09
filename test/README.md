# NCCL 冒烟测试（AllReduce）

## 前置条件

1. 已编译本仓库 NCCL，生成 `build/lib/libnccl.so*` 与 `build/include/nccl.h`：
   ```bash
   cd /path/to/nccl_cc
   ./build_a100.sh src.build
   ```
2. CUDA 工具链（`nvcc`）可用，环境变量 `CUDA_HOME` 默认 `/usr/local/cuda`。

## 编译与运行

```bash
cd test
chmod +x run_allreduce.sh
./run_allreduce.sh
```

或手动：

```bash
export NCCL_BUILD=/path/to/nccl_cc/build
make
./allreduce_test
```

Makefile 已为可执行文件写入 **rpath**（指向 `NCCL_BUILD/lib` 与 `CUDA_HOME/lib64`），一般无需再设 `LD_LIBRARY_PATH`。若你移动了 `build` 目录，需重新 `make`。

## 环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `NUM_GPUS` | 使用的 GPU 数量（单机多卡） | `1` |
| `COUNT` | 每个 tensor 元素个数（float） | `1048576` |
| `NCCL_DEBUG` | NCCL 日志，如 `INFO` | 未设置 |
| `NCCL_NET` | 如 `IB` 强制走 IB/RoCE 插件 | 未设置 |

示例：

```bash
NCCL_DEBUG=INFO NCCL_NET=IB NUM_GPUS=2 ./allreduce_test
```

## 期望结果

程序打印 `VERIFY OK` 且退出码为 `0`。若数值不一致会打印 `VERIFY FAIL` 并返回 `2`。

## 说明

- 本测试用于验证 **NCCL 库与驱动/RDMA** 基本可用；**不**替代 `nccl-tests` 性能压测。
- 多卡时需保证 GPU peer 访问与 NCCL 拓扑正常（单机多 GPU 常见场景）。

---

## 验证 Telemetry Hint（Phase 3）

读端实现见 `src/transport/net_ib/xccl_telemetry_hint.{h,cc}`，在 **`ncclIbAIMDInit`** 里 `mmap`，在 **`ncclCcOnCollectiveBegin`**（需 **AIMD 已启用** 且走 IB 发送路径）里读快照。

### 1. 传输层是否挂上（最先看这条）

```bash
export NCCL_CC_HINT_ENABLE=1
# 二选一：文件 mmap（推荐）
python3 write_telemetry_hint.py --path /tmp/xccl_hint.bin --once
export NCCL_CC_HINT_MMAP_PATH=/tmp/xccl_hint.bin
# 或：默认 posix 名（脚本写 /dev/shm/xccl_telemetry_hint）
python3 write_telemetry_hint.py --shm --once
# 不设 MMAP_PATH 时 NCCL 会 shm_open("/xccl_telemetry_hint")，对应 /dev/shm/xccl_telemetry_hint
```

运行：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET ./allreduce_test
```

**期望**出现一行：`XCCL hint: telemetry mmap active (... bytes)`。  
若失败会看到 `open ... failed` / `shm_open ... failed` / `region size ... < 36`。

### 2. Hint 是否参与控制（需 AIMD + IB 发送）

- 设置 **`NCCL_AIMD_ENABLE=1`**，否则 `ncclCcOnCollectiveBegin` 不会在发送路径里刷新 hint。
- 单机多卡时日志里若大量是 **`via P2P/direct pointer`**，集合通信主要走 NVLink，**IB 发送可能很少**，hint 刷新频率会低；**多机**或明显走 **NET/IB** 的路径更容易观察到 hint 对 `effective_window` 的影响（见 `aimd_cc.cc` 里 `hint_cap` / `XCCL_HINT_F_SEVERE`）。
- 可调大 TTL、降低刷新节流：`NCCL_CC_HINT_TTL_NS`（默认约 100ms 量级，见代码）、`NCCL_CC_HINT_REFRESH_MIN_NS`。

### 3. 结构体与协议

- 快照 **36 字节** packed，字段顺序见 `xccl_telemetry_hint.h`。
- 写端须遵守 **奇数=写入中 / 非零偶数=稳定帧**（`write_telemetry_hint.py` 已按 §5.3 发布）。

### 4. 与日志里 “CC On/Off” 的区别

`init.cc` 里 **`CC On/Off`** 指的是 **workFifo / 另一套 CC**，**不是** 本仓库 AIMD；不要用它判断 Telemetry Hint 是否开启。

---

## 单机 8 卡 + 尽量走网 / 验证 hint 读数

**说明**：单机 NVLink 拓扑下，NCCL **通常仍以 NVLink/P2P 为主**；`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1` **不保证**「所有机内流量都走 RoCE」，但可削弱直连，便于观察与 IB 相关的日志。

**终端 A**（先启动 sidecar，与 hint 文件一致）：

```bash
cd /path/to/nccl_cc/tools/xccl_telemetry_sidecar
export XCCL_TELEMETRY_FILE=/path/to/nccl_cc/test/th_data/xccl_telemetry_hint.bin
python3 xccl_telemetry_sidecar.py --file "$XCCL_TELEMETRY_FILE" --ifaces eth0
```

**终端 B**：

```bash
cd /path/to/nccl_cc/test
chmod +x run_8gpu_net_hint.sh
./run_8gpu_net_hint.sh
```

脚本内已包含：`NUM_GPUS=8`、`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`、`NCCL_NET=IB`、`NCCL_CC_HINT_*`、`NCCL_AIMD_ENABLE=1`、`NCCL_DEBUG_SUBSYS=NET`。

**如何看「读到数据」**：

1. 日志中有 **`XCCL hint: telemetry mmap active`** → 映射成功。  
2. sidecar 终端里 **`cnp` / `rnic` 等** 在压测时是否非零（`--dry-run` 仅打印）。  
3. 日志里是否仍有大量 **`via P2P`**：若有，属单机拓扑常态；要稳定走 IB 需 **多机**。
