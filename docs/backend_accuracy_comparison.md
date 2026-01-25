# SHUD Backend Accuracy Comparison Report

> 本文档记录了 CPU Serial、OpenMP、CUDA 三种后端的数值精度对比结果。
> 生成时间: 2026-01-25

## 概述

在进行 GPU 性能优化之前，需要确保各后端的数值一致性。理论上：
- **CPU vs OMP**: 应该几乎完全一致（仅浮点舍入误差，约 1e-10）
- **CPU vs CUDA**: 允许较小差异（单精度影响，约 1e-6）

## 测试配置

- Project: `ccw`
- CPU dir: `output/ccw_cpu`
- OMP dir: `output/ccw_omp`
- CUDA dir: `output/ccw_cuda`

### 容差标准

| 对比组 | 期望 rel_max | 实际结果 |
|--------|-------------|---------|
| CPU vs OMP | ≤ 1e-10 | **最差 6.748e+00 (674.8%)** |
| CPU vs CUDA | ≤ 1e-6 | **最差 6.291e+04 (6291000%)** |

## 结果摘要

### 总体统计

- 文件扫描: 25 个 `*.dat` 文件
- 有效对比: 24 个文件
- 跳过: 1 个 (`DY.dat` 为空)

| 对比组 | PASS | FAIL | MISSING | ERROR |
|--------|------|------|---------|-------|
| CPU vs OMP | 7 | **17** | 0 | 0 |
| CPU vs CUDA | 3 | **20** | 0 | 1 |

### 关键状态变量差异

| 变量 | 文件 | CPU vs OMP rel_max | CPU vs CUDA rel_max | 备注 |
|------|------|-------------------:|--------------------:|------|
| yGw (地下水) | `ccw.eleygw.dat` | 24.56% | 23.32% | OMP worst@t=1650d; CUDA worst@t=0d |
| ySf (地表水深) | `ccw.eleysurf.dat` | 7.03% | 142.7% | 显著差异 |
| yUs (非饱和带) | `ccw.eleyunsat.dat` | 67.88% | 7.50% | CUDA 有 6 个 NaN |
| yRiv (河道水位) | `ccw.rivystage.dat` | 38.44% | 5.63% | OMP 误差随时间累积 |

## 问题分析

### CPU vs OMP 差异过大（异常）

OpenMP 版本应该产生与串行版本几乎相同的结果，但实际差异巨大：

1. **最严重**: `ccw.elevexfil.dat` (rel_max=6.748, 即 674.8%)
2. **状态变量**: yUs 差异 67.88%，yRiv 差异 38.44%
3. **误差累积**: 最差点多在模拟后期（t=1600-1800 天）

**可能原因**:
- 并行循环中的竞态条件
- 浮点运算顺序差异导致的误差累积
- 归约操作未正确处理

### CPU vs CUDA 差异极大

CUDA 版本显示严重问题：

1. **最严重**: `ccw.elevprcp.dat` (rel_max=6.291e+04，即 6291000%)
2. **初始化问题**: 多数最大误差出现在 **t=0**（模拟开始时刻）
3. **Forcing 传输**: prcp/netprcp 误差极大，说明 `gpuUpdateForcing()` 可能有错
4. **NaN 问题**: `ccw.eleyunsat.dat` CUDA 输出含 6 个 NaN

**可能原因**:
- GPU 端 forcing 数据初始化错误
- Host→Device 数据传输遗漏或顺序错误
- 内存布局不匹配

## 详细数据

### CPU vs OMP (全部 .dat 文件)

| 类型 | 文件 | 形状(TxN) | 尺度(max\|CPU\|) | rel_max | rel_mean | rel_std | abs_max | 状态 | 最差位置 |
|------|------|----------:|-----------------:|--------:|---------:|--------:|--------:|------|----------|
| Flux/Diag | `ccw.elevexfil.dat` | 1827x1147 | 3.412e-04 | 6.748e+00 | 1.758e-03 | 6.314e-02 | 2.303e-03 | FAIL | t=1611d, id=411 |
| Flux/Diag | `ccw.elevrech.dat` | 1827x1147 | 1.083e-01 | 1.550e+00 | 4.962e-04 | 7.956e-03 | 1.679e-01 | FAIL | t=718d, id=539 |
| Flux/Diag | `ccw.eleveta.dat` | 1827x1147 | 1.043e-02 | 1.000e+00 | 1.334e-01 | 1.113e-01 | 1.043e-02 | FAIL | t=1648d, id=511 |
| Flux/Diag | `ccw.elevetev.dat` | 1827x1147 | 5.763e-03 | 1.000e+00 | 6.962e-02 | 4.609e-02 | 5.763e-03 | FAIL | t=1251d, id=232 |
| Flux/Diag | `ccw.elevettr.dat` | 1827x1147 | 8.289e-03 | 1.000e+00 | 1.142e-01 | 1.159e-01 | 8.289e-03 | FAIL | t=1648d, id=511 |
| Flux/Diag | `ccw.elevinfil.dat` | 1827x1147 | 1.192e-01 | 8.987e-01 | 5.817e-03 | 1.724e-02 | 1.072e-01 | FAIL | t=732d, id=539 |
| Flux/Diag | `ccw.elevetic.dat` | 1827x1147 | 1.808e-03 | 7.378e-01 | 2.272e-02 | 6.606e-02 | 1.334e-03 | FAIL | t=106d, id=1053 |
| Flux(Q*) | `ccw.eleqsurf.dat` | 1827x1147 | 2.724e+04 | 7.105e-01 | 1.123e-03 | 6.986e-03 | 1.935e+04 | FAIL | t=1080d, id=256 |
| State(y*) | `ccw.eleyunsat.dat` | 1827x1147 | 1.208e+01 | 6.788e-01 | 2.400e-01 | 1.570e-01 | 8.200e+00 | FAIL | t=1826d, id=243 |
| Flux(Q*) | `ccw.rivqsurf.dat` | 1827x103 | 2.454e+05 | 4.404e-01 | 7.518e-04 | 7.416e-03 | 1.081e+05 | FAIL | t=1078d, id=65 |
| Flux(Q*) | `ccw.eleqsub.dat` | 1827x1147 | 3.905e+04 | 3.961e-01 | 5.909e-04 | 3.115e-03 | 1.547e+04 | FAIL | t=1803d, id=1136 |
| State(y*) | `ccw.rivystage.dat` | 1827x103 | 1.022e+00 | 3.844e-01 | 7.579e-04 | 4.577e-03 | 3.930e-01 | FAIL | t=1803d, id=1 |
| Flux(Q*) | `ccw.rivqsub.dat` | 1827x103 | 1.099e+05 | 3.784e-01 | 9.638e-04 | 5.011e-03 | 4.158e+04 | FAIL | t=1803d, id=1 |
| Flux(Q*) | `ccw.rivqdown.dat` | 1827x103 | 6.802e+06 | 2.839e-01 | 3.183e-04 | 4.395e-03 | 1.931e+06 | FAIL | t=1078d, id=1 |
| Flux(Q*) | `ccw.rivqup.dat` | 1827x103 | 6.833e+06 | 2.834e-01 | 2.940e-04 | 4.333e-03 | 1.936e+06 | FAIL | t=1078d, id=2 |
| State(y*) | `ccw.eleygw.dat` | 1827x1147 | 3.007e+01 | 2.456e-01 | 9.665e-04 | 5.513e-03 | 7.383e+00 | FAIL | t=1650d, id=411 |
| State(y*) | `ccw.eleysurf.dat` | 1827x1147 | 2.064e-01 | 7.029e-02 | 2.675e-04 | 5.457e-04 | 1.451e-02 | FAIL | t=732d, id=539 |
| Other | `ccw.eleysnow.dat` | 1827x1147 | 1.708e-01 | ~0 | ~0 | 0 | ~0 | PASS | - |
| Flux/Diag | `ccw.elevetp.dat` | 1827x1147 | 1.917e-02 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.elevnetprcp.dat` | 1827x1147 | 9.169e-02 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.elevprcp.dat` | 1827x1147 | 9.169e-02 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.rn_factor.dat` | 1827x1147 | 9.940e-01 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.rn_h.dat` | 1827x1147 | 3.844e+02 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.rn_t.dat` | 1827x1147 | 3.847e+02 | 0 | 0 | 0 | 0 | PASS | - |

### CPU vs CUDA (全部 .dat 文件)

| 类型 | 文件 | 形状(TxN) | 尺度(max\|CPU\|) | rel_max | rel_mean | rel_std | abs_max | 状态 | 最差位置 |
|------|------|----------:|-----------------:|--------:|---------:|--------:|--------:|------|----------|
| Flux/Diag | `ccw.elevprcp.dat` | 1827x1147 | 9.169e-02 | 6.291e+04 | 2.097e+01 | 9.161e+02 | 5.768e+03 | FAIL | t=0d, id=390 |
| Flux/Diag | `ccw.elevnetprcp.dat` | 1827x1147 | 9.169e-02 | 3.320e+04 | 1.045e+01 | 4.560e+02 | 3.044e+03 | FAIL | t=0d, id=120 |
| Flux/Diag | `ccw.elevexfil.dat` | 1827x1147 | 3.412e-04 | 2.931e+04 | 6.408e+00 | 2.771e+02 | 1.000e+01 | FAIL | t=0d, id=5 |
| Flux/Diag | `ccw.eleveta.dat` | 1827x1147 | 1.043e-02 | 2.877e+04 | 4.400e-01 | 5.335e+01 | 3.000e+02 | FAIL | t=0d, id=8 |
| Other | `ccw.eleysnow.dat` | 1827x1147 | 1.708e-01 | 6.118e+01 | 2.410e-03 | 3.000e-01 | 1.045e+01 | FAIL | t=0d, id=495 |
| Flux/Diag | `ccw.elevrech.dat` | 1827x1147 | 1.083e-01 | 4.354e+01 | 2.046e-02 | 8.807e-01 | 4.716e+00 | FAIL | t=0d, id=881 |
| State(y*) | `ccw.eleysurf.dat` | 1827x1147 | 2.064e-01 | 1.427e+00 | 2.157e-03 | 3.018e-02 | 2.945e-01 | FAIL | t=714d, id=794 |
| Flux(Q*) | `ccw.eleqsub.dat` | 1827x1147 | 3.905e+04 | 1.000e+00 | 9.994e-04 | 5.559e-03 | 3.905e+04 | FAIL | t=1078d, id=1136 |
| Flux(Q*) | `ccw.eleqsurf.dat` | 1827x1147 | 2.724e+04 | 1.000e+00 | 1.538e-03 | 1.133e-02 | 2.724e+04 | FAIL | t=1077d, id=183 |
| Flux/Diag | `ccw.elevetev.dat` | 1827x1147 | 5.763e-03 | 1.000e+00 | 6.962e-02 | 4.609e-02 | 5.763e-03 | FAIL | t=1251d, id=232 |
| Flux/Diag | `ccw.elevettr.dat` | 1827x1147 | 8.289e-03 | 1.000e+00 | 1.142e-01 | 1.159e-01 | 8.289e-03 | FAIL | t=1648d, id=511 |
| Flux/Diag | `ccw.elevinfil.dat` | 1827x1147 | 1.192e-01 | 1.000e+00 | 2.117e-02 | 5.696e-02 | 1.192e-01 | FAIL | t=1822d, id=900 |
| Flux(Q*) | `ccw.rivqdown.dat` | 1827x103 | 6.802e+06 | 1.000e+00 | 8.015e-04 | 9.888e-03 | 6.802e+06 | FAIL | t=1078d, id=4 |
| Flux(Q*) | `ccw.rivqsub.dat` | 1827x103 | 1.099e+05 | 1.000e+00 | 3.434e-03 | 1.170e-02 | 1.099e+05 | FAIL | t=1078d, id=1 |
| Flux(Q*) | `ccw.rivqsurf.dat` | 1827x103 | 2.454e+05 | 1.000e+00 | 1.415e-03 | 1.580e-02 | 2.454e+05 | FAIL | t=1078d, id=65 |
| Flux(Q*) | `ccw.rivqup.dat` | 1827x103 | 6.833e+06 | 1.000e+00 | 7.370e-04 | 9.768e-03 | 6.833e+06 | FAIL | t=1078d, id=2 |
| Flux/Diag | `ccw.elevetic.dat` | 1827x1147 | 1.808e-03 | 7.378e-01 | 2.272e-02 | 6.606e-02 | 1.334e-03 | FAIL | t=106d, id=1053 |
| Flux/Diag | `ccw.elevetp.dat` | 1827x1147 | 1.917e-02 | 3.912e-01 | 1.920e-04 | 8.223e-03 | 7.500e-03 | FAIL | t=0d, id=770 |
| State(y*) | `ccw.eleygw.dat` | 1827x1147 | 3.007e+01 | 2.332e-01 | 1.498e-04 | 3.520e-03 | 7.010e+00 | FAIL | t=0d, id=858 |
| State(y*) | `ccw.rivystage.dat` | 1827x103 | 1.022e+00 | 5.627e-02 | 6.651e-05 | 6.151e-04 | 5.752e-02 | FAIL | t=796d, id=1 |
| Flux/Diag | `ccw.rn_factor.dat` | 1827x1147 | 9.940e-01 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.rn_h.dat` | 1827x1147 | 3.844e+02 | 0 | 0 | 0 | 0 | PASS | - |
| Flux/Diag | `ccw.rn_t.dat` | 1827x1147 | 3.847e+02 | 0 | 0 | 0 | 0 | PASS | - |
| State(y*) | `ccw.eleyunsat.dat` | 1827x1147 | 1.208e+01 | 7.502e-02 | 8.203e-04 | 3.215e-03 | 9.061e-01 | ERROR | t=1177d, id=405; 含 6 个 NaN |

## 根因分析（codeagent 深度分析结果）

### OpenMP 后端问题

#### 1. 竞态条件（致命）

**位置**: `src/ModelData/MD_f_omp.cpp:9-17`

```cpp
void Model_Data::f_applyDY_omp(double *DY, double t){
    double area;
    int isf, ius, igw, i;
#pragma omp parallel  default(shared) private(i) num_threads(CS.num_threads)
```

**问题**: `#pragma omp parallel default(shared) private(i)` 只把 `i` 设为 private，但 `area/isf/ius/igw` 都在并行域外声明，默认 shared。

**后果**: 多线程同时写 `isf/ius/igw/area`，导致 DY 写入错误索引、用错 `area`，RHS 完全错乱。这种错乱在 CVODE 长时间积分中"看起来能跑但逐步偏离"，符合误差在模拟后期累积的现象。

#### 2. RHS 逻辑不等价（缺失 ET 和湖泊处理）

| 功能 | Serial 版本 | OMP 版本 |
|------|------------|---------|
| ET 通量 | `f_etFlux(i,t)` (MD_f.cpp:18) | **缺失** |
| 湖泊处理 | `updateLakeElement()/fun_Ele_lakeVertical()` | **缺失** |

**后果**:
- ET 相关输出 (`eta/etev/ettr`) 出现 `rel_max=1.0`
- ET 通量进入状态方程，缺失导致水量平衡偏离

#### 3. 河道方程实现不一致

| 项目 | Serial (MD_f.cpp:119-129) | OMP (MD_f_omp.cpp:54-65) |
|------|--------------------------|-------------------------|
| 归一化 | `/ Riv[i].Length` | `/ Riv[i].u_TopArea` |
| 负面积限制 | 有 | **无** |
| `fun_dAtodY` 转换 | 有 | **无** |

**后果**: 河道状态 `yRivStg` 系统性偏离，符合 yRiv 差异 38% 的量级。

#### 4. 湖单元 DY 处理缺失

Serial 明确对湖单元 `DY=0` (MD_f.cpp:108-112)，OMP 版本缺失对应处理。

---

### CUDA 后端问题

#### 1. 输出 buffer 未初始化（解释 t=0 巨大误差）

**位置**: `src/classes/Model_Control.cpp:516-552`

```cpp
buffer = new double[NumVar];  // 没有初始化为 0！
```

**问题**: `Print_Ctrl::Init()` 分配 buffer 后没有清零初始化。

**机制**:
- 第一次 `PrintData()` 执行 `buffer[i] += *(PrintVar[i])`
- 如果 buffer 初值是随机残留内存，第一个输出时间点带着残留值一起写出
- 后续时间点 buffer 被重置为 0，恢复正常

**为什么 CUDA 更明显**: CUDA 路径引入更多堆分配（NVECTOR_CUDA、gpuInit、device buffers），让 `new double[NumVar]` 更容易拿到非零旧内存。

**这解释了**:
- 多数最大误差集中在 `t=0`
- `prcp/netprcp` 这种本应很小的量，首点污染后相对误差放大到 1e4~1e5
- `yUs` 的 NaN（buffer 初值包含 NaN bit-pattern）

#### 2. 诊断量未同步回 host（解释 rel_max=1.0、后期 FAIL）

**问题**: GPU RHS 在 device 上计算并写入诊断量，但 host 侧用于输出的数组**从未从 device 同步回来**。

| 计算位置 | 受影响变量 |
|---------|-----------|
| `rhs_kernels.cu:594` | `qEleInfil/qEleExfil/qEleRecharge/qEs/qEu/qEg/qTu/qTg` |
| `rhs_kernels.cu:795/874` | `QeleSurf/QeleSub` |
| `rhs_kernels.cu:955` | `Qe2r_*/Qriv*` |

`gpuSyncStateFromDevice()` 只同步状态向量 `y`，不同步这些诊断量。

**后果**: 通量/诊断量（`qEleInfil/qEleExfil/Q*` 等）host 侧保持初值 0，导致大量输出文件 `rel_max=1.0` 或后期大偏差。

#### 3. 初始 forcing 可能未传输到 device

`SetCVODE()` 在主循环前调用 `CVodeInit`，但第一次 `gpuUpdateForcing()` 在主循环内。如果 CVodeInit 触发 RHS 评估，CUDA RHS 可能用到全 0 的 forcing。

---

## 修复建议优先级

### P0（阻塞项 - 不修则对比数据不可信）

| 优先级 | 问题 | 文件位置 | 修复方向 |
|--------|------|---------|---------|
| P0-1 | 输出 buffer 未初始化 | `Model_Control.cpp:516,554` | 初始化 `buffer[i]=0.0` |
| P0-2 | OMP 竞态条件 | `MD_f_omp.cpp:9` | 声明 `area/isf/ius/igw` 为 private |
| P0-3 | OMP 缺失 ET/湖泊逻辑 | `MD_f_omp.cpp:69` | 补齐与 Serial 等价的逻辑 |
| P0-4 | OMP 河道公式不一致 | `MD_f_omp.cpp:54` | 对齐 Serial（Length、dA 限制、fun_dAtodY） |
| P0-5 | CUDA 诊断量未同步 | `DeviceContext.cu` | 添加 D2H 同步或修改输出直接读 device |

### P1（在 P0 之后查真正的数值差异）

| 优先级 | 问题 | 修复方向 |
|--------|------|---------|
| P1-1 | CUDA 状态漂移 | 关闭预处理器测试是否收敛 |
| P1-2 | realtype 精度一致性 | 确认 CPU/CUDA 使用相同精度 |
| P1-3 | RHS 分量分叉点 | 使用 DEBUG_GPU_VERIFY 定位 |

---

## 结论

### 核心发现

1. **CPU vs OMP 差异主因**: OpenMP 变量作用域错误导致竞态条件，加上缺失 ET/湖泊逻辑和河道公式不一致
2. **CUDA t=0 巨大误差**: 输出 buffer 未初始化，不是 forcing 传输问题
3. **CUDA 后期 FAIL**: 诊断量只在 device 计算，从未同步回 host 用于输出
4. **CUDA NaN**: 输出 buffer 初值包含 NaN bit-pattern

### 行动计划

**在进行 GPU 性能优化（Epic #40）之前，必须先完成 P0 修复项。**

## 备注

- 相对误差使用每个文件的 `max(|CPU|)` 作为归一化尺度
- 对比在 CPU 和目标后端的时间步和列 ID 交集上进行
- 分析由 codeagent (codex backend) 于 2026-01-25 完成
