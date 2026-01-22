# SHUD GPU 加速落地 Spec / Backlog / 字段映射表 / Kernel I/O 表（v0.1）

> 面向：直接落地实现（工程开发 + 科研可复现实验）。  
> 范围：对 **Coupled SHUD（CVODE + 单向量状态）** 做端到端 GPU 加速，第一阶段聚焦 RHS（`f_update/f_loop/f_applyDY`）搬到 CUDA；第二阶段加入 GPU 预条件器（Block-Jacobi）等进一步提速。  
> 兼容目标：默认 **数值行为与当前 CPU-Serial 路径一致**（同输入、同容差下输出差异可控），并提供“修复已知问题”的可选模式。

---

## 0. 现有代码风格与结构审计（从“怎么写代码”到“怎么接 GPU”）

### 0.1 代码组织与主调用链

**目录结构（SHUD.zip / SHUD-up-master）：**
- `src/Model/`：求解器入口（`shud.cpp`）、RHS 回调（`f.cpp`）、宏与全局变量（`Macros.hpp`）
- `src/ModelData/`：`Model_Data` 大对象，负责读入/内存分配/forcing 更新/ET/通量/ApplyDY/输出汇总等
- `src/classes/`：领域对象（`_Element/_River/_Lake/...`）+ TimeSeries
- `src/Equations/`：数值/物理公式（`ManningEquation`, `effKH`, `SoilMoistureStress`, ...）

**Coupled 模式下调用链（关键）：**
`main.cpp → Model_Data.initialize() → SHUD() → CVode() → f(t,y,ydot) → MD->f_update → MD->f_loop → MD->f_applyDY`

> 性能瓶颈“落点”非常清晰：`CVode()` 内会多次调用 RHS `f()`，而 `f()` 的主体就是 `f_update/f_loop/f_applyDY` 三段循环与清零。把这三段搬到 GPU 才能获得数量级收益（否则容易被隐式 memcpy/UVM 拖垮）。

### 0.2 项目风格画像（你现在的“默认写法”）

以下是你代码中非常一致的风格特征（GPU 接入必须尊重，否则维护成本会爆炸）：

1) **宏驱动的布局与快捷索引**  
- `Macros.hpp` 定义了状态向量布局与索引宏（`iSF/iUS/iGW/iRIV/iLAKE`），大量代码依赖。  
- 常量（`GRAV/MAXYSURF/...`）也以宏/全局常量存在。

2) **“大对象 + 裸指针数组”**  
- `Model_Data` 持有大量 `double* / double**`，初始化阶段一次性 `new[]`。  
- 通量 `QeleSurf/QeleSub` 用 `double**`（每 element 3 邻边），更偏 CPU 友好。

3) **OO 对象用于读入与派生量更新，但 RHS 其实是“数组计算”**  
- RHS 主耗时部分本质是按 element/edge/segment/river/lake 的规则计算通量并汇总。  
- GPU 适配的关键，不是把 `_Element` 整个搬上 GPU，而是把 RHS 需要的字段拆成 SoA。

4) **预处理器选择 backend（Serial / OpenMP）**  
- `_OPENMP_ON` 控制不同的 N_Vector 与 f-loop 实现。  
- 现状：Serial 与 OpenMP 路径在行为上存在差异（尤其 ET/Lake/River 导数公式），这会影响 GPU 的“对齐参照”。

5) **错误处理偏“科研原型风格”**  
- `printf/fprintf + myexit()`，更偏直接中止，而不是异常/错误码传播。  
- 对 GPU 来说要增加 `CUDA_CHECK()` 宏与可选 debug dump，但尽量不引入复杂框架。

### 0.3 对 GPU 方案落地的直接影响（必须先定的“工程约束”）

- **必须将 RHS 的数据访问从 host-only 对象迁移到 device-resident SoA**（`Element/River` 仍可留在 host 做初始化/输出）。  
- **RHS 回调不得触碰 host 指针**（包括 `N_VGetHostArrayPointer_Cuda`），否则会触发隐式搬运，性能直接归零。  
- **forcing/LAI/BC 如继续 CPU 计算，只能按 forcing step 批量传到 GPU**，绝不能在每次 RHS 内来回拷贝。

---

## 1. GPU 加速总体方案（端到端可落地）

### 1.1 总体原则（与 SUNDIALS GPU 模型一致）

- 控制逻辑在 CPU；数值向量与 RHS 所需数据常驻 GPU；host/device 一致性由用户负责。  
- CVODE + Krylov（SPGMR）可以与 NVECTOR_CUDA 协同工作，但前提是回调只访问 device 数据。  
- 避免 UVM 的隐式迁移：第一版建议使用 **unmanaged device memory**（`N_VNew_Cuda`），只有在确实频繁读 host 时再评估 managed。

### 1.2 总体架构（新增模块 + 与现有代码的关系）

**新增目录（建议与现有风格兼容）：**
```
src/GPU/
  ShudCudaContext.hpp/.cu        // stream、错误检查、memset/拷贝封装
  ShudDeviceData.hpp/.cu         // SoA 数据结构 + Host→Device 初始化/释放
  ShudRhsKernels.cu              // RHS kernels（local/edge/seg/river/lake/applyDY）
  ShudPrecKernels.cu             // 预条件器 setup/solve（Phase-2）
  ShudJtimesKernels.cu           // 自定义 Jv（Phase-3，可选）
src/Model/f_gpu.cpp              // CVODE RHS 回调薄封装（取 device ptr + launch kernels）
```

**运行时对象关系：**
- `Model_Data`（host）继续负责：读入、I/O、主循环、输出控制
- `DeviceModel`（device 常驻）分三块：
  - `DeviceStatic`：拓扑与常量参数（初始化一次上传）
  - `DeviceForcing`：forcing/LAI 等（每 forcing step 更新）
  - `DeviceScratch`：通量数组与临时派生量（常驻 GPU，每次 RHS 复用+必要 memset）

### 1.3 backend 选择策略（最小改动可落地）

**推荐分两步：**
- **Step-1（最稳）**：单独编译 `shud_cuda` 可执行文件，编译期选择 backend；避免把所有 backend 混在一起的复杂性。
- **Step-2（工程更优雅）**：加 CLI `--backend cpu|omp|cuda`，运行时选择（仍需编译时启用 CUDA 代码）。

### 1.4 “参照语义”与兼容模式（科研落地的关键）

由于当前仓库存在多条 RHS 实现路径（Serial / OpenMP / GPU-dev），为了保证可复现与验证：

- **基准语义（Baseline）**：以 `ModelData/MD_f.cpp + MD_update.cpp` 的 Serial 路径为准（含 ET/Lake/River 的完整逻辑）。  
- **兼容模式开关（建议新增）**
  - `LEGACY_COMPAT=1`：严格复现现有 Serial 路径（包括已知的小瑕疵，便于回归）
  - `PHYSICS_FIX=1`：修复明确问题（例如 lake evap 缩放、time series 指针移动等），并在文档中标记为“需重新标定/回归”

---

## 2. 详细实现 Spec（文件/接口/生命周期/数据驻留）

### 2.1 构建与依赖（Makefile + configure）

#### 2.1.1 SUNDIALS（CVODE + NVECTOR_CUDA）构建要求
- `configure`（或新增 `configure_cuda`）需要启用 `ENABLE_CUDA=ON`，并设置 `CMAKE_CUDA_ARCHITECTURES`。
- 安装产物必须包含：
  - `libsundials_cvode`
  - `libsundials_nveccuda`

#### 2.1.2 Makefile 增加目标
新增 target：`shud_cuda`（或 `shud_gpu`）：
- 编译器：`nvcc`（第一版最稳）
- 关键链接：`-lsundials_cvode -lsundials_nveccuda -lcudart`
- 增加宏：`-D_CUDA_ON`

> 若你希望继续保留 `g++` 作为主编译器：  
> - `.cpp` 用 `g++` 编译  
> - `.cu` 用 `nvcc` 编译为 `.o`  
> - 最终链接可用 `nvcc`（或 `g++` + `-lcudart`）  

### 2.2 N_Vector 创建与 Host 同步策略

在 `src/Model/shud.cpp` 中：
- CUDA backend：`udata = N_VNew_Cuda(NY, sunctx)`（建议 unmanaged），`du = N_VNew_Cuda(NY, sunctx)`  
- 输出/诊断前：调用 `N_VCopyFromDevice_Cuda(udata)`，再用 `N_VGetHostArrayPointer_Cuda(udata)` 获取 host 指针进行 `summary/ExportResults`  
- 若 CPU 侧修改了 `udata`（比如重新写 IC），需要 `N_VCopyToDevice_Cuda(udata)`

### 2.3 RHS 回调分发

在 `src/Model/f.cpp`：
- 保留现有 serial/omp 分支
- 增加 `#ifdef _CUDA_ON` 分支，直接调用 `f_gpu(t, CV_Y, CV_Ydot, DS)`（由 `src/Model/f_gpu.cpp` 实现）

### 2.4 GPU 侧数据结构 Spec（SoA）

#### 2.4.1 状态向量布局（保持不变）
```
Y = [ Ysf(NumEle),
      Yus(NumEle),
      Ygw(NumEle),
      Yriv(NumRiv),
      Ylake(NumLake) ]
NY = 3*NumEle + NumRiv + NumLake
```

GPU 上通过 NVECTOR_CUDA 的 device pointer 直接访问 `Y` 与 `Ydot`，并在 `DeviceModel` 内部提供切片别名指针（不额外分配）。

#### 2.4.2 DeviceModel 顶层结构（建议）

```cpp
struct DeviceModel {
  int NumEle, NumRiv, NumSeg, NumLake;
  int CloseBoundary;     // from Control_Data
  // ---- static element arrays ----
  const double* ele_area;
  const double* ele_z_surf;
  const double* ele_z_bottom;
  const double* ele_depression;
  const double* ele_Rough;
  const double* ele_avgRough;    // [NumEle*3]
  const double* ele_edge;        // [NumEle*3]
  const double* ele_Dist2Nabor;  // [NumEle*3]
  const double* ele_Dist2Edge;   // [NumEle*3]
  const int*    ele_nabr;        // [NumEle*3] (0-based, -1 = boundary)
  const int*    ele_lakenabr;    // [NumEle*3] (0-based, -1 = none)
  const int*    ele_iLake;       // [NumEle]   (0-based, -1 = not lake element)

  // soil / geology / landcover
  const double* ele_AquiferDepth;
  const double* ele_infD;
  const double* ele_infKsatV;
  const double* ele_hAreaF;
  const double* ele_macKsatV;
  const double* ele_KsatV;
  const double* ele_ThetaS;
  const double* ele_ThetaR;
  const double* ele_ThetaFC;
  const double* ele_Alpha;
  const double* ele_Beta;
  const double* ele_KsatH;
  const double* ele_macKsatH;
  const double* ele_macD;
  const double* ele_geo_vAreaF;
  const double* ele_Sy;

  const double* ele_VegFrac;
  const double* ele_ImpAF;
  const double* ele_WetlandLevel;
  const double* ele_RootReachLevel;

  // BC / SS（建议压缩为 type+value，避免 GPU 再查 TimeSeries）
  const int8_t* ele_bc_type;     // 0 none, +1 fixed head, -1 flux
  const double* ele_bc_value;    // head or flux
  const int8_t* ele_ss_type;     // 0 none, +1 surface SS, -1 gw SS
  const double* ele_ss_value;

  // ---- static river arrays ----
  const int*    riv_down_raw;    // keep raw (<=0 boundary codes)
  const int*    riv_toLake;      // 0-based, -1 none
  const int8_t* riv_bc_type;
  const double* riv_bc_value;    // stage or flux
  const double* riv_Length;
  const double* riv_depth;
  const double* riv_BankSlope;
  const double* riv_BottomWidth;
  const double* riv_BedSlope;
  const double* riv_rivRough;
  const double* riv_avgRough;
  const double* riv_zbank;
  const double* riv_KsatH;
  const double* riv_BedThick;
  const double* riv_Dist2Down;

  // ---- static segment arrays ----
  const int*    seg_iEle;     // 0-based
  const int*    seg_iRiv;     // 0-based
  const double* seg_length;
  const double* seg_Cwr;

  // ---- static lake arrays ----
  const double* lake_zmin;
  const double* lake_invNumEle;   // 1.0/NumEleLake
  // bathymetry packed arrays
  const int*    lake_bathy_off;
  const int*    lake_bathy_n;
  const double* bathy_yi;
  const double* bathy_ai;

  // ---- forcing arrays (per forcing step update) ----
  double* qEleNetPrep;
  double* qPotEvap;
  double* qPotTran;
  double* qEleE_IC;
  double* t_lai;
  double* fu_Surf;
  double* fu_Sub;

  // ---- persistent/scratch (per RHS) ----
  double* ele_satn;       // persistent, used by ET (matches CPU calling order)
  double* ele_effKH;

  double* qEleInfil;
  double* qEleExfil;
  double* qEleRecharge;

  double* qEs; double* qEu; double* qEg;
  double* qTu; double* qTg;
  double* qEleEvapo; double* qEleTrans; double* qEleETA;

  double* QeleSurf;      // [NumEle*3]
  double* QeleSub;       // [NumEle*3]
  double* Qe2r_Surf;     // [NumEle]
  double* Qe2r_Sub;      // [NumEle]

  // rivers
  double* QrivSurf; double* QrivSub;
  double* QrivUp;   double* QrivDown;
  double* riv_CSarea;
  double* riv_CSperem;
  double* riv_topWidth;

  // lakes
  double* QLakeSurf; double* QLakeSub;
  double* QLakeRivIn; double* QLakeRivOut;
  double* qLakePrcp; double* qLakeEvap;
  double* y2LakeArea;
};
```

> 注：字段不是越多越好。**只放 RHS 高频访问字段**。其余用于输出/诊断的可以留在 host。

### 2.5 生命周期与 API

建议在 `Model_Data` 中增加 GPU 资源管理接口：

```cpp
class Model_Data {
public:
  // ...
  int  gpu_enabled = 0;
  void gpuInit(SUNContext sunctx);         // allocate + upload static + allocate forcing/scratch
  void gpuFree();                          // free device allocations
  void gpuUpdateForcing();                 // copy qEleNetPrep/qPotEvap/... to device (per forcing step)
  void gpuSyncStateFromDevice(N_Vector y); // before summary/output
  void gpuSyncStateToDevice(N_Vector y);   // after CPU sets IC or overwrites y
};
```

---

## 3. GPU RHS kernel 设计（与现有 f_update/f_loop/f_applyDY 对齐）

### 3.1 kernel 流水（第一版建议拆成 7~10 个 kernel，易验证）

每次 `f_gpu(t,y,ydot)`：

0) **memset / init**
- `cudaMemsetAsync(Qe2r_Surf/Qe2r_Sub, 0)`
- `cudaMemsetAsync(QrivSurf/QrivSub/QrivUp, 0)`
- `cudaMemsetAsync(QLake*, 0)`, `cudaMemsetAsync(qLake*, 0)`
- `cudaMemsetAsync(ydot, 0)`（或后续 kernel 全覆盖写入）

1) `k_apply_bc_and_sanitize_state`（NY 或分段）
- 对 fixed-head BC：直接把 `Ygw` 或 `Yriv` 写成 BC 值（匹配 CPU f_update 的“覆盖状态”语义）
- 可选：ClampPolicy（非负截断）在此做

2) `k_ele_local`（NumEle threads）
- lake element：`updateLakeElement` + `fun_Ele_lakeVertical`（设置 qEleEvapo/qEleETA 等）+ 原子累加到 `qLakePrcp/qLakeEvap`
- land element：按 CPU 顺序：
  - 用 **上一轮存下的 satn（ele_satn）** 做 `SoilMoistureStress`（匹配 CPU：ET 在 updateElement 之前）
  - 计算 `f_etFlux` 产物：`qEs/qEu/qEg/qTu/qTg/qEleEvapo/qEleTrans/qEleETA`（并按需修改 `qEleE_IC`）
  - 计算 `updateElement`（得到当前 satn/effKH 等）并写回 `ele_satn/ele_effKH`
  - 计算 `Flux_Infiltration` 与 `Flux_Recharge`，得到 `qEleInfil/qEleExfil/qEleRecharge`

3) `k_ele_edge_surface`（NumEle*3 threads）
- 计算 `QeleSurf[i,edge]`
- 若 edge 对应 lake neighbor：原子累加 `QLakeSurf[lake]`

4) `k_ele_edge_sub`（NumEle*3 threads）
- 计算 `QeleSub[i,edge] = Q * fu_Sub[i]`
- 若 edge 对应 lake neighbor：原子累加 `QLakeSub[lake]`（注意：CPU 累加的是 *未乘 fu_Sub 的 Q，需匹配）

5) `k_seg_exchange`（NumSeg threads）
- 计算 element↔river 的 `QsegSurf/QsegSub`（可不存数组，直接 atomicAdd 到聚合量）
- 原子累加：
  - `QrivSurf[r] += QsegSurf`
  - `Qe2r_Surf[e] += -QsegSurf`
  - `QrivSub[r] += QsegSub`
  - `Qe2r_Sub[e] += -QsegSub`

6) `k_river_down_and_up`（NumRiv threads）
- 根据 `Yriv` 计算 river 横断面派生量：`CSarea/CSperem/topWidth`
- 计算 `QrivDown[i]`（Manning 等）
- 若 `toLake>=0`：原子累加 `QLakeRivIn[toLake] += QrivDown[i]`
- 若 `down>0`：原子累加 `QrivUp[down-1] += -QrivDown[i]`

7) `k_lake_toparea_and_scale`（NumLake threads）
- `y2LakeArea[l] = bathymetry.toparea(Ylake[l] + lake_zmin[l])`
- `qLakePrcp[l] *= y2LakeArea[l]`
- `qLakeEvap[l] *= y2LakeArea[l] * lake_invNumEle[l]`（严格复现 CPU 逻辑；PHYSICS_FIX 可改）

8) `k_apply_dy_element`（NumEle threads）
- 汇总 `QeleSurfTot = Qe2r_Surf + Σ QeleSurf[3]`
- 汇总 `QeleSubTot = Qe2r_Sub  + Σ QeleSub[3]`
- 按现有公式写 `ydot[iSF/iUS/iGW]`，处理 `iBC/iSS/isLakeEle`
- `ydot_us/gw` 做 `/Sy`

9) `k_apply_dy_river`（NumRiv threads）
- 若 BC fixed stage：`ydot=0`
- 否则：`dA = (-Up - Surf - Sub - Down + qBC)/Length`，截断，`dY = fun_dAtodY(dA, topWidth, BankSlope)`，写回 `ydot`

10) `k_apply_dy_lake`（NumLake threads）
- `ydot_lake = qLakePrcp - qLakeEvap + (QLakeRivIn - QLakeRivOut + QLakeSub + QLakeSurf)/y2LakeArea`

> 第一版建议不做 kernel 融合，优先保证正确性与可对照。  
> Phase-1 通过后再做融合/减少 atomic/减少 memset。

---

## 4. Backlog（可直接落地实施）

> 说明：优先级 P0 必做、P1 强烈建议、P2 可选；每项给出验收标准（AC）与依赖（Dep）。

### Epic 0：基线对齐与测试框架（P0）

**E0-1 建立 CPU-Serial “RHS 回归基准”**
- 内容：固定输入工程 + 固定编译选项；输出关键指标（逐时刻 y、若干通量）作为 golden
- AC：相同输入，多次运行输出差异 ≤ 1e-12（若非确定则给容差）
- Dep：无

**E0-2 明确“Baseline 是 Serial 还是 OMP”并写入文档**
- 内容：说明为何选择 Serial（含 ET/Lake/River 完整逻辑）
- AC：spec 文档 + README 中明确
- Dep：E0-1

**E0-3 增加 `ClampPolicy` 开关**
- 内容：为 state 非负截断提供统一开关（CPU/GPU 共用），默认保持旧行为
- AC：开启/关闭开关均可跑通；关闭时回归不变
- Dep：E0-1

### Epic 1：构建系统与依赖（P0）

**E1-1 升级/新增 SUNDIALS CUDA 构建脚本**
- 内容：在 `configure` 或 `configure_cuda` 中启用 NVECTOR_CUDA
- AC：本机/集群可生成 `libsundials_nveccuda` 并被工程链接
- Dep：无

**E1-2 Makefile 增加 `shud_cuda`**
- 内容：增加 nvcc 编译 `.cu`，链接 `-lsundials_nveccuda -lcudart`
- AC：`make shud_cuda` 产出可执行文件并能启动（哪怕暂时跑 CPU RHS）
- Dep：E1-1

### Epic 2：运行时 backend 选择（P0/P1）

**E2-1（P0）新增独立可执行 `shud_cuda`**
- 内容：最小改动，让 CUDA 版先能跑
- AC：能读入工程、能完成一个短仿真并输出
- Dep：E1-2

**E2-2（P1）CLI 增加 `--backend cpu|omp|cuda`**
- 内容：在 `CommandIn.*` 增加解析，写入 `global_backend`
- AC：同一个可执行可根据参数切换 backend（前提：编译启用 CUDA）
- Dep：E2-1

### Epic 3：DeviceModel 数据驻留（P0）

**E3-1 设计并实现 `DeviceModel`（SoA）**
- 内容：从 `Model_Data::Ele/Riv/RivSeg/lake` 抽取 RHS 必要字段
- AC：gpuInit 后设备侧字段与 host 一致（抽样检查 50 个 element/river）
- Dep：E2-1

**E3-2 实现 `gpuUpdateForcing()`**
- 内容：每 forcing step 批量拷贝 `qEleNetPrep/qPotEvap/qPotTran/qEleE_IC/t_lai/fu_*`
- AC：拷贝发生次数等于 forcing step 数量，不随 RHS 调用次数增长
- Dep：E3-1

**E3-3 实现输出前 `gpuSyncStateFromDevice()`**
- 内容：输出/summary 前从 device 拷回 y（只在需要时）
- AC：CUDA 版输出与 CPU 版格式一致且不崩溃
- Dep：E3-1

### Epic 4：GPU RHS（P0）

**E4-1 实现 `f_gpu()` wrapper**
- 内容：从 NVECTOR_CUDA 获取 dY/dYdot，launch kernel pipeline
- AC：CVODE 可调用，且不触碰 host 指针
- Dep：E3-1

**E4-2 实现 Phase-1 kernel pipeline（local/edge/seg/river/lake/applyDY）**
- 内容：按 3.1 设计逐个实现 kernel
- AC：短仿真在 GPU 可跑通；与 CPU-Serial 输出相对误差 ≤ 1e-6（容差可调）
- Dep：E4-1

**E4-3 引入 Debug 校验模式（GPU vs CPU 同步对比）**
- 内容：可选：每 N 步把 GPU 中间数组拷回，与 CPU 同时计算对比
- AC：定位误差到单 kernel/单字段
- Dep：E4-2

### Epic 5：性能与稳定性（P1）

**E5-1 Profiling（Nsight Systems/Compute）**
- 内容：确认热点（atomic/memset/访存不合并/小 kernel 过多）
- AC：给出 Profile 报告与 Top-5 bottleneck
- Dep：E4-2

**E5-2 Kernel 融合与原子优化**
- 内容：合并 `k_river_down_and_up`、减少 `cudaMemset` 范围、必要时用 warp-level reduction
- AC：RHS walltime 降低 ≥ 30%
- Dep：E5-1

### Epic 6：GPU 预条件器（P1/P2）

**E6-1 Block-Jacobi PSetup/PSolve（GPU kernel）**
- 内容：每 element 构建 3×3 近似块（SF/US/GW），river/lake 1×1
- AC：GMRES 迭代次数显著下降（≥2×），总耗时下降
- Dep：E4-2

**E6-2 CVODE 接入 Preconditioner**
- 内容：`CVodeSetPreconditioner` 绑定 PSetup/PSolve
- AC：开启预条件器仿真稳定，结果与未开启一致
- Dep：E6-1

### Epic 7：文档与可复现实验（P0/P1）

**E7-1 README：GPU 构建/运行/依赖说明**
- AC：新同学可按文档从零跑通
- Dep：E1-2

**E7-2 可复现实验脚本**
- 内容：固定 seed/固定输入/自动对比 CPU vs GPU
- AC：CI 或本地脚本一键产出对比报告
- Dep：E4-2

---

## 5. 字段映射表（Host→Device）

> 说明：这里给出 **RHS 相关字段** 的映射（不是把整个 Element/River 类搬过去）。  
> 约定：device 侧索引尽量统一为 0-based；host 侧如果是 1-based（如 `nabr`），在上传时转换。

### 5.1 状态向量映射（Y）

| 变量块 | Host 位置 | Device 位置 | 类型 | 长度 | 读/写 | 参与 kernel |
|---|---|---|---|---:|---|---|
| `Ysf` | `N_Vector y` | `d_y + 0` | double | NumEle | R/W | local / edge / seg / applyDY |
| `Yus` | `N_Vector y` | `d_y + NumEle` | double | NumEle | R/W | local / applyDY |
| `Ygw` | `N_Vector y` | `d_y + 2*NumEle` | double | NumEle | R/W | local / edge_sub / seg_sub / applyDY |
| `Yriv` | `N_Vector y` | `d_y + 3*NumEle` | double | NumRiv | R/W | seg / river / applyDY_river |
| `Ylake` | `N_Vector y` | `d_y + 3*NumEle + NumRiv` | double | NumLake | R/W | edge_lake / lake_toparea / applyDY_lake |

### 5.2 Element 静态字段（geometry + topology）

| Host 字段（`Ele[i].*`） | Device 数组名 | 类型 | 形状 | 更新频率 | 使用 kernel | 备注 |
|---|---|---|---|---|---|---|
| `area` | `ele_area[i]` | double | NumEle | init | applyDY | |
| `z_surf` | `ele_z_surf[i]` | double | NumEle | init | edge/seg | |
| `z_bottom` | `ele_z_bottom[i]` | double | NumEle | init | edge_sub/seg_sub | |
| `depression` | `ele_depression[i]` | double | NumEle | init | edge_surface/seg_surface | |
| `Rough` | `ele_Rough[i]` | double | NumEle | init | edge_surface boundary | |
| `nabr[0..2]` | `ele_nabr[i*3+j]` | int | NumEle*3 | init | edge_* | upload 时 `-1=boundary` |
| `lakenabr[0..2]` | `ele_lakenabr[i*3+j]` | int | NumEle*3 | init | edge_* | `-1=none` |
| `edge[0..2]` | `ele_edge[i*3+j]` | double | NumEle*3 | init | edge_* | |
| `Dist2Nabor[0..2]` | `ele_Dist2Nabor[i*3+j]` | double | NumEle*3 | init | edge_* | |
| `Dist2Edge[0..2]` | `ele_Dist2Edge[i*3+j]` | double | NumEle*3 | init | edge_* boundary | |
| `avgRough[0..2]` | `ele_avgRough[i*3+j]` | double | NumEle*3 | init | edge_surface | |
| `iLake` | `ele_iLake[i]` | int | NumEle | init | local/edge/applyDY | 0-based, -1 none |

### 5.3 Element 土壤/地质/地表参数（用于 updateElement/infil/recharge/ET）

| Host 字段 | Device 数组名 | 类型 | 形状 | 更新频率 | 使用 kernel |
|---|---|---|---|---|---|
| `AquiferDepth` | `ele_AquiferDepth[i]` | double | NumEle | init | local |
| `Sy` | `ele_Sy[i]` | double | NumEle | init | applyDY |
| `infD` | `ele_infD[i]` | double | NumEle | init | local |
| `infKsatV` | `ele_infKsatV[i]` | double | NumEle | init | local |
| `hAreaF` | `ele_hAreaF[i]` | double | NumEle | init | local |
| `macKsatV` | `ele_macKsatV[i]` | double | NumEle | init | local |
| `KsatV` | `ele_KsatV[i]` | double | NumEle | init | local |
| `ThetaS` | `ele_ThetaS[i]` | double | NumEle | init | local |
| `ThetaR` | `ele_ThetaR[i]` | double | NumEle | init | local |
| `ThetaFC` | `ele_ThetaFC[i]` | double | NumEle | init | local |
| `Alpha` | `ele_Alpha[i]` | double | NumEle | init | local |
| `Beta` | `ele_Beta[i]` | double | NumEle | init | local |
| `KsatH` | `ele_KsatH[i]` | double | NumEle | init | local (effKH) |
| `macKsatH` | `ele_macKsatH[i]` | double | NumEle | init | local (effKH) |
| `macD` | `ele_macD[i]` | double | NumEle | init | local (effKH) |
| `geo_vAreaF` | `ele_geo_vAreaF[i]` | double | NumEle | init | local (effKH) |
| `VegFrac` | `ele_VegFrac[i]` | double | NumEle | init | local (ET) |
| `ImpAF` | `ele_ImpAF[i]` | double | NumEle | init | local (ET) |
| `WetlandLevel` | `ele_WetlandLevel[i]` | double | NumEle | init | local/applyDY |
| `RootReachLevel` | `ele_RootReachLevel[i]` | double | NumEle | init | local/applyDY |

### 5.4 Element BC/SS（建议上传为 type+value）

| Host 字段 | Device 字段 | 说明 |
|---|---|---|
| `Ele[i].iBC` + `tsd_eyBC/eqBC` | `ele_bc_type[i]`, `ele_bc_value[i]` | `+1` fixed head（写 Ygw），`-1` flux（加到 DYgw） |
| `Ele[i].iSS` + `tsd_esBC/egBC` | `ele_ss_type[i]`, `ele_ss_value[i]` | `+1` surface SS（加到 DYsf），`-1` gw SS（加到 DYgw） |

> 当前 TimeSeriesData 是“分段常值”模型，且 BC/SS 的 `movePointer` 未在 forcing 更新中调用：  
> - `LEGACY_COMPAT`：保持现状（BC/SS 常值）  
> - `PHYSICS_FIX`：在 forcing step 同步移动 BC/SS 指针后再上传

### 5.5 Segment 映射（Element↔River exchange）

| Host 字段（`RivSeg[i].*`） | Device 数组名 | 类型 | 形状 | 更新频率 | 使用 kernel |
|---|---|---|---|---|---|
| `iEle` | `seg_iEle[s]` | int | NumSeg | init | seg_exchange |
| `iRiv` | `seg_iRiv[s]` | int | NumSeg | init | seg_exchange |
| `length` | `seg_length[s]` | double | NumSeg | init | seg_exchange |
| `Cwr` | `seg_Cwr[s]` | double | NumSeg | init | seg_exchange |

### 5.6 River 映射（topology + geometry + BC）

| Host 字段（`Riv[i].*`） | Device 数组名 | 类型 | 形状 | 更新频率 | 使用 kernel |
|---|---|---|---|---|---|
| `down` | `riv_down_raw[i]` | int | NumRiv | init | river_down |
| `toLake` | `riv_toLake[i]` | int | NumRiv | init | river_down |
| `BC` + `tsd_ryBC/rqBC` | `riv_bc_type[i]`, `riv_bc_value[i]` | int8/double | NumRiv | forcing/init | apply_bc, applyDY_river |
| `Length` | `riv_Length[i]` | double | NumRiv | init | river_down/applyDY_river |
| `depth` | `riv_depth[i]` | double | NumRiv | init | seg_surface/river_down |
| `BankSlope` | `riv_BankSlope[i]` | double | NumRiv | init | river_down/applyDY_river |
| `BottomWidth` | `riv_BottomWidth[i]` | double | NumRiv | init | river_down |
| `BedSlope` | `riv_BedSlope[i]` | double | NumRiv | init | river_down |
| `rivRough` | `riv_rivRough[i]` | double | NumRiv | init | river_down |
| `avgRough` | `riv_avgRough[i]` | double | NumRiv | init | river_down |
| `zbank` | `riv_zbank[i]` | double | NumRiv | init | seg_surface |
| `KsatH` | `riv_KsatH[i]` | double | NumRiv | init | seg_sub |
| `BedThick` | `riv_BedThick[i]` | double | NumRiv | init | seg_sub |
| `Dist2DownStream` | `riv_Dist2Down[i]` | double | NumRiv | init | river_down |

**River 派生量（scratch，每 RHS 更新）：**
| 量 | Device 数组 | 说明 |
|---|---|---|
| `u_CSarea` | `riv_CSarea[i]` | 由 `Yriv` 更新 |
| `u_CSperem` | `riv_CSperem[i]` | 由 `Yriv` 更新 |
| `u_topWidth` | `riv_topWidth[i]` | 由 `Yriv` 更新 |

### 5.7 Lake 映射（bathymetry + 聚合通量）

| Host 字段（`lake[i].*`） | Device 数组名 | 类型 | 形状 | 更新频率 | 使用 kernel |
|---|---|---|---|---|---|
| `zmin`（=`bathymetry.yi[0]`） | `lake_zmin[l]` | double | NumLake | init | edge/ lake_toparea |
| `NumEleLake` | `lake_invNumEle[l]` | double | NumLake | init | local / scale |
| `bathymetry.yi/ai` | `bathy_yi/bathy_ai` + `off/n` | double/int | packed | init | lake_toparea |

---

## 6. Kernel I/O 表（Phase-1）

> 约定：`Y`/`Ydot` 为 NVECTOR_CUDA device pointer。

### 6.1 Kernel 列表

| Kernel | Grid（建议） | 主要输入 | 主要输出 | 原子操作 | 备注 |
|---|---|---|---|---|---|
| `k_apply_bc_and_sanitize_state` | `NY` 或分段 | `Y`, `ele_bc_*`, `riv_bc_*` | `Y`（原地修正） | 否 | 固定水位 BC 覆盖状态（匹配 CPU） |
| `k_ele_local` | `NumEle` | `Ysf/Yus/Ygw`, forcing, ele_* | `qEle*`, `qE*`, `ele_satn/effKH` | `atomicAdd(qLake*)` | 关键：ET 用上一轮 satn |
| `k_ele_edge_surface` | `NumEle*3` | `Ysf`, `Ylake`, ele topo/geom | `QeleSurf` | `atomicAdd(QLakeSurf)` | lake neighbor 走 weir |
| `k_ele_edge_sub` | `NumEle*3` | `Ygw`, `Ylake`, `effKH`, topo/geom | `QeleSub` | `atomicAdd(QLakeSub)` | 注意 CPU 的 QLakeSub 未乘 fu_Sub |
| `k_seg_exchange` | `NumSeg` | `Ysf/Ygw/Yriv`, `qEleInfil/Exfil`, `effKH`, seg/riv/ele | `Qe2r_*`, `QrivSurf/Sub` | 多处 `atomicAdd` | 可选：保留 Qseg 数组用于 debug |
| `k_river_down_and_up` | `NumRiv` | `Yriv`, riv geom/topo | `QrivDown`, `QrivUp`, `QLakeRivIn`, `riv_*derived` | `atomicAdd(QrivUp)`, `atomicAdd(QLakeRivIn)` | 同时更新派生横断面 |
| `k_lake_toparea_and_scale` | `NumLake` | `Ylake`, bathy, `qLake*` | `y2LakeArea`, scaled `qLake*` | 否 | 保持 legacy 缩放逻辑 |
| `k_apply_dy_element` | `NumEle` | `qEle*`, `Qe2r_*`, `Qele*`, `BC/SS`, `Sy` | `Ydot(sf/us/gw)` | 否 | `/Sy` 与 BC/SS 逻辑对齐 |
| `k_apply_dy_river` | `NumRiv` | `Qriv*`, `riv_*derived`, `BC` | `Ydot(riv)` | 否 | `fun_dAtodY` |
| `k_apply_dy_lake` | `NumLake` | `QLake*`, `qLake*`, `y2LakeArea` | `Ydot(lake)` | 否 | |

---

## 7. Phase-2（建议）：GPU 预条件器 Spec（Block-Jacobi）

### 7.1 目标
降低 GMRES 迭代次数，减少全局规约与 RHS 额外调用，避免“GPU 被 GMRES 拖死”。

### 7.2 形式
- Element：3×3 局部块（`[SF, US, GW]`）
- River：1×1
- Lake：1×1

### 7.3 数据结构
- `prec_inv[NumEle][9]`（或 LU 分解形式）
- River/Lake：`prec_diag_inv[NumRiv+NumLake]`

### 7.4 接口
- `PSetup(t, y, fy, jok, jcurPtr, gamma, user_data)`：GPU kernel 计算并写入块逆
- `PSolve(t, y, fy, r, z, gamma, delta, lr, user_data)`：GPU kernel 做 `z = M^{-1} r`

---

## 8. 关键风险点与对策（落地必须看）

1) **Serial/OMP 行为不一致**  
- 对策：明确 baseline（Serial），GPU 首先对齐 Serial；OMP 后续可重构为调用同一套公式（只是并行化）。

2) **TimeSeriesData 的 BC/SS 指针更新缺失**  
- 对策：PHYSICS_FIX 模式下修复；LEGACY_COMPAT 保持旧行为以做回归。

3) **Lake evap/precip 缩放可能有瑕疵**  
- 对策：Phase-1 复现，Phase-2 提供修复开关并重新标定。

4) **大量 atomicAdd（Lake 与 QrivUp/Qe2r 汇总）**  
- 对策：Phase-1 先正确；Phase-2 做分块归约（block-level reduction）减少原子冲突。

5) **输出/summary 读 host 数据**  
- 对策：严格在输出前 `CopyFromDevice`；禁止在 RHS 内读 host。

---

## 9. 建议的最小落地顺序（“能跑→对齐→跑快”）

1) shud_cuda 编译通过 + NVECTOR_CUDA 创建（不崩）  
2) gpuInit/gpuUpdateForcing/gpuSyncStateFromDevice 跑通  
3) f_gpu + 最小 kernel（只把 ydot=0）跑通 CVODE  
4) 逐段实现 kernel：local → edge → seg → river → lake → applyDY  
5) CPU vs GPU 回归对齐  
6) 性能 profiling + 优化  
7) 预条件器 Block-Jacobi

---

## 10. 附：参考文档（你已提供）
- 《SHUD重构方案设计》
- 《仓库改动清单》
- 《参考资料清单》
- 《GPU加速方案》
