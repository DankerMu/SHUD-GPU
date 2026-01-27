# CUDA 输出/诊断链路说明（source of truth + 同步点）

本文档说明 SHUD 的 CUDA backend（`shud_cuda` / `--backend cuda`）下，输出文件中各类变量的“数据来源（source of truth）”以及 host↔device 的同步位置，避免出现“旧值/零值/未更新”的假象。

## 总体策略（混合策略）

- **状态量（ySf/yUs/yGw/yRiv/yLake）**：以 **CVODE 的状态向量**为准。
  - CUDA backend：状态向量驻留 device，输出前需要 D2H 同步。
- **通量/诊断量**：以 **device 计算结果**为准（必要时在输出前同步到 host）。
  - CUDA backend：flux/diagnostics 在 RHS kernel 中更新在 device 侧数组；输出前用 D2H 同步到 host 侧 `Model_Data` 数组。
- **forcing/ET 中间量**：以 **host 计算**为准（同时会按需复制到 device 供 RHS 使用）。

## 关键同步点（输出前）

在耦合模式主循环中，CUDA backend 会在每个输出步（outer step）结束、写文件前执行：

- `Model_Data::gpuSyncStateFromDevice(udata)`：把 NVECTOR_CUDA 的 state 从 device 拷回 host（供 `Model_Data::summary()` 使用）。
- `Model_Data::gpuSyncDiagnosticsFromDevice(udata)`：把输出所需的 flux/diagnostics 从 device 拷回 host（供 `Control_Data::ExportResults()` 使用）。

实现位置：
- `src/Model/shud.cpp`
- `src/GPU/DeviceContext.cu`

## 输出变量的 source of truth（按类别）

### 1) 状态输出（state）

**source of truth**：CVODE 状态向量（CUDA：device）

host 输出链路：
1. D2H：`gpuSyncStateFromDevice(udata)`
2. 提取：`Model_Data::summary(udata)`（把 NVECTOR 的 host data 写入 `yEleSurf/yEleUnsat/yEleGW/yRivStg/yLakeStg` 等数组）
3. 写盘：`Control_Data::ExportResults(t)`（`Print_Ctrl` 绑定的是 `yEle* / yRivStg / yLakeStg`）

典型变量：
- `yEleSurf` / `yEleUnsat` / `yEleGW`
- `yRivStg`
- `yLakeStg`

### 2) forcing/ET（host 计算 + 可选 H2D）

**source of truth**：host（由 forcing/ET 更新逻辑计算）

CUDA backend 下，这些数组会通过 `Model_Data::gpuUpdateForcing()` 按需 H2D 复制，供 RHS kernels 使用；同时它们本身也可直接用于输出（`Print_Ctrl` 指向 host 数组）。

典型变量：
- `qElePrep` / `qEleNetPrep`
- `qEleETP` / `qPotEvap` / `qPotTran`
- `qEleE_IC`
- `t_lai`
- `fu_Surf` / `fu_Sub`

### 3) device 侧通量（flux）

**source of truth**：device（RHS kernels 更新）

host 输出前需要 D2H 同步：
- `Model_Data::gpuSyncDiagnosticsFromDevice(udata)` 会按输出控制间隔（`Control_Data` 的 `dt_*`）选择性同步所需数组。

典型变量（element / river / lake）：
- `qEleInfil` / `qEleExfil` / `qEleRecharge`
- `qEs` / `qEu` / `qEg` / `qTu` / `qTg`
- `QeleSurf` / `QeleSub`
- `Qe2r_Surf` / `Qe2r_Sub`
- `QrivSurf` / `QrivSub` / `QrivUp` / `QrivDown`
- `QLakeSurf` / `QLakeSub` / `QLakeRivIn` / `QLakeRivOut` / `qLakePrcp` / `qLakeEvap` / `y2LakeArea`

### 4) device 侧派生诊断（derived diagnostics）

**source of truth**：device（在同步前派生）

为了兼容既有输出接口（`Print_Ctrl` 直接绑定 host 数组），CUDA backend 会在 `gpuSyncDiagnosticsFromDevice` 内部先在 device 上派生，再 D2H：

典型变量：
- `qEleTrans` / `qEleEvapo` / `qEleETA`
- `QeleSurfTot` / `QeleSubTot`

## Print_Ctrl 指针绑定位置

输出绑定发生在 `Model_Data::initialize_output()`：`Control_Data::PCtrl[ip++].Init(...)` / `InitIJ(...)`，把 `double*` / `double**` 指针直接绑定到 `Model_Data` 的 host 数组上。

因此 CUDA backend 的关键要求是：**在 `ExportResults()` 之前必须完成对应 host 数组的同步/更新**（上文“关键同步点”即为此目的）。

