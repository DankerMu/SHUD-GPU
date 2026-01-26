# SHUD-GPU：OMP / CUDA 后端精度问题分析报告

生成时间：2026-01-25  
范围：`/mnt/sdc/SHUD-GPU`（当前仓库工作区）  
说明：本报告**仅做源码差异分析与问题梳理**，不包含任何修复实现。

---

## 0. 背景与现状（以 `ccw` 对比结果为例）

本轮“修复后”的精度对比结果来自 `post_analysis/accuracy_report_ccw.md`：

- CPU vs OMP：`PASS 8 / FAIL 16`，最差 `rel_max ≈ 0.479 (47.9%)`
- CPU vs CUDA：`PASS 7 / FAIL 17`，最差 `rel_max ≈ 5.992 (599.2%)`

关键变量（`rel_max`）：

- `ySf`（地表水深）  
  - OMP：`15.05%`（最差点：`t=1494d, id=508`）  
  - CUDA：`599%`（最差点：`t=1081d, id=1072`）
- `yUs`（非饱和带）  
  - OMP：`0.80%`（`t=1495d, id=508`）  
  - CUDA：`11.94%`（`t=1175d, id=1019`）
- `yRiv`（河道水位）  
  - OMP：`1.68%`（`t=15d, id=1`）  
  - CUDA：`8.42%`（`t=1174d, id=1`）

此外，CUDA 对比中多类通量/诊断输出出现 `rel_max=1.0`（例如 `eleqsurf/eleqsub/rivq*/eta/etev/ettr/infil/exfil/rech`），这是本报告重点单独拆分的一类问题（见 §6）。

> 注意：对比脚本使用“**每个文件按 `max(|CPU|)` 归一化**”的 `rel_max`（见 `post_analysis/accuracy_comparison.py` 与 `post_analysis/accuracy_report_ccw.md`），因此：
> - `rel_max=1.0` 往往意味着“某处差值达到 CPU 文件全局尺度”，常见于“目标侧输出为 0/未更新”这类问题；
> - `rel_max>1` 则意味着差值超过 CPU 全局尺度，通常更像“数值路径已经明显偏离/出现额外峰值”。

---

## 1. 三个版本的执行路径与职责划分

### 1.1 CPU 串行（基线）

RHS 入口：`src/Model/f.cpp` 中 `f()`，当 `global_backend == BACKEND_CPU` 时：

1. `Model_Data::f_update()`（`src/ModelData/MD_update.cpp`）  
   - 从状态向量 `Y` 生成工作数组 `uYsf/uYus/uYgw/uYriv/yLakeStg`（带 clamp & BC）  
   - 重置/清零本步用到的各类通量累积数组（`Qe2r_*`, `Qriv*`, `QLake*` 等）
2. `Model_Data::f_loop(t)`（`src/ModelData/MD_f.cpp`）  
   - 逐元素：ET 通量、入渗/回灌、坡面/地下侧向通量、河道-单元交换、河道路由  
   - 末尾 `PassValue()`：聚合河道-单元交换与上下游通量（关键的“通量累积/归约”点）
3. `Model_Data::f_applyDY(DY, t)`（`src/ModelData/MD_f.cpp`）  
   - 将本步通量写回微分 `DYdot`

### 1.2 OpenMP（CPU 并行）

当 `global_backend == BACKEND_OMP` 时（仍在 `src/Model/f.cpp`）：

1. `Model_Data::f_update_omp()`（`src/ModelData/MD_f_omp.cpp`）
2. `Model_Data::f_loop_omp()`（`src/ModelData/MD_f_omp.cpp`）
3. `Model_Data::f_applyDY_omp()`（`src/ModelData/MD_f_omp.cpp`）

整体逻辑与串行一致，关键差别在于：

- 若 `NumLake>0`，部分 lake 相关累加使用 `#pragma omp atomic`（例如 `qLakeEvap/qLakePrcp/QLakeSurf/QLakeSub/QLakeRivIn`）。
- CVODE 使用的向量实现不同：OMP 后端使用 `NVECTOR_OPENMP`，其内部 `dot/norm` 等归约的浮点求和顺序与串行不同，可能改变自适应步长/迭代路径（这是“即便 RHS 数学等价也可能长期偏离”的重要来源，见 §5）。

### 1.3 CUDA（GPU 驻留 + CVODE NVECTOR_CUDA）

当 `global_backend == BACKEND_CUDA` 时：

- RHS 入口：`src/GPU/f_cuda.cpp` 的 `f_gpu()` → `launch_rhs_kernels()`（`src/GPU/rhs_kernels.cu`）
- 核心：在 device 上实现一套“等价于 CPU f_update + f_loop + f_applyDY”的 kernel pipeline：
  1. `k_zero_flux_accumulators`
  2. `k_apply_bc_and_sanitize_state`
  3. `k_ele_local`（ET 各分量、入渗/回灌、更新 `ele_satn/ele_effKH` 等）
  4. `k_ele_edge_surface`
  5. `k_ele_edge_sub`
  6. `k_seg_exchange`
  7. `k_river_down_and_up`
  8. `k_lake_toparea_and_scale`（若有湖泊）
  9. `k_apply_dy_element` / `k_apply_dy_river` / `k_apply_dy_lake`

此外 CUDA 版本默认启用预条件器（见 `src/Equations/cvode_config.cpp`），调用 `PSetup_cuda/PSolve_cuda`（`src/GPU/precond_kernels.cu`）。这会使 CUDA 的线性求解路径与 CPU/OMP 不同（见 §5.3）。

---

## 2. ySf（地表水深）方程与三后端实现对齐检查

### 2.1 数学形式（代码中的实现）

三后端对 `ySf` 的 RHS 形式一致，核心为（以单元 `i` 为例）：

```
QeleSurfTot[i] = Qe2r_Surf[i] + Σ_j QeleSurf[i][j]   (j=0..2)

DYsf = qEleNetPrep[i]
    - qEleInfil[i] + qEleExfil[i]
    - QeleSurfTot[i] / area
    - qEs[i]
    + (surface SS if any)
```

对应位置：

- CPU：`src/ModelData/MD_f.cpp` → `f_applyDY()`
- OMP：`src/ModelData/MD_f_omp.cpp` → `f_applyDY_omp()`
- CUDA：`src/GPU/rhs_kernels.cu` → `k_apply_dy_element()`

### 2.2 ySf 相关的“误差放大点”

即便公式一致，`ySf` 的数值对齐非常敏感，因为其上游通量包含多个阈值/分段逻辑：

- 入渗/出渗：`_Element::Flux_Infiltration()`（`src/classes/Element.cpp`）  
  - 由 `Ygw+Yus`、`deficit>infD`、`av=Ysf+netprcp` 等条件切换不同流态
- 坡面侧向流：`Model_Data::fun_Ele_surface()`（`src/ModelData/MD_ElementFlux.cpp`）  
  - `avgY_sf()` + `Ymean<=0` 分支 + `MAXYSURF` 上限  
  - 边界/封闭域/抑制条件分支
- 河道-单元水面交换：`Model_Data::fun_Seg_surface()`（`src/ModelData/MD_RiverFlux.cpp`）  
  - `WeirFlow_jtoi()` 内部 `threshold` / `bank` 分支

这些条件使得：

- 小的浮点差异（来自求和顺序、不同硬件 `pow/cbrt/sqrt`、不同的 CVODE 步长与误差控制）  
  → 可能触发“不同分支”  
  → 造成通量突变  
  → 最终在 `ySf` 上表现为显著偏离（尤其当某些点恰好处在阈值附近时）。

---

## 3. 通量累积/归约差异（PassValue vs 原子累加）

### 3.1 CPU/OMP：`PassValue()` 的串行聚合

CPU 与 OMP 在 `f_loop()/f_loop_omp()` 末尾都调用 `Model_Data::PassValue()`（`src/ModelData/MD_f.cpp`），其职责是：

1. 清零 `QrivSurf/QrivSub/QrivUp` 与 `Qe2r_Surf/Qe2r_Sub`
2. 遍历所有 river segment：从 `QsegSurf/QsegSub` 聚合到 river 与 element
3. 遍历所有河道：由 `QrivDown` 更新下游的 `QrivUp`

这在 CPU 侧是确定性的（固定遍历顺序），但在 OMP 后端中，**segment 通量本身**是并行计算的（见 §4.2），因此“先算后聚合”的整体顺序仍确定，但中间过程可能存在不必要的并发写入（见 §4.2 的潜在 UB）。

### 3.2 CUDA：直接在 kernel 中用 `atomicAdd` 聚合

CUDA 版本通过 `k_seg_exchange` 与 `k_river_down_and_up` 直接对 `QrivSurf/QrivSub/Qe2r_*/QrivUp` 做 `atomicAdd`（以及 warp 聚合原子），并在每次 RHS 开始时 `k_zero_flux_accumulators` 清零。

优点：

- 避免把 `Qseg*` 拉回 host 再 `PassValue()`

代价（精度相关）：

- **求和顺序不可控**：不同 block/warp 的原子加顺序不确定  
  → 浮点非结合性带来可观的“末位差异”  
  → 在阈值敏感系统中可能被放大

---

## 4. OMP 后端：实现差异与剩余风险点

### 4.1 RHS 计算路径基本等价

从 `src/ModelData/MD_f.cpp` 与 `src/ModelData/MD_f_omp.cpp` 对照看：

- 逐元素 ET、入渗/回灌、坡面/地下侧向流的调用顺序一致
- 河道/segment 的调用顺序一致
- `f_applyDY` 的公式一致

因此“理论上” OMP 与串行应极接近。

### 4.2 潜在的并发写入/未定义行为（需确认是否仍存在）

`fun_Seg_surface()` / `fun_Seg_sub()`（`src/ModelData/MD_RiverFlux.cpp`）内部会写：

- `QsegSurf[seg]` / `QsegSub[seg]`（按 seg 独占索引 → 线程安全）
- **同时也会 `+=` 写 `QrivSurf/QrivSub/Qe2r_Surf/Qe2r_Sub`**（按 river/element 多对一 → 并发写入风险）

在 CPU/OMP 主路径中，这些 `Qriv*`/`Qe2r*` 最终会被 `PassValue()` 清零后重算，因此这类并发写入从“逻辑结果”上看是冗余的，但它仍然属于 C/C++ 层面的数据竞争（UB），会导致：

- 非确定性（每次运行可能不同）
- 极端情况下可能污染邻近内存/触发难以复现的偏差（尤其在高优化等级）

这类问题的优先级通常应高于“纯舍入误差”，因为它可能造成远大于 1e-10 的偏离。

### 4.3 CVODE + NVECTOR_OPENMP 的归约顺序差异

即便 RHS 完全等价，只要：

- CVODE 使用自适应步长
- 且向量范数/点积等内部运算的归约顺序不同

就可能改变：

- 误差估计与步长选择
- Newton/GMRES 的收敛路径
- RHS 被调用的时间点序列

在包含阈值/分段逻辑（§2.2）的水文系统中，这些差异可能被放大到 “% 级别”。

这解释了为何当前 OMP 的误差“集中在少数敏感点”（例如 `id=508` 多次成为最差点），而不是均匀地出现微小随机扰动。

---

## 5. CUDA 后端：实现差异与可能导致状态偏离的机制

### 5.1 RHS 方程层面的对齐情况

CUDA RHS 关键公式与 CPU 基本一致：

- `ManningEquation/avgY_sf/avgY_gw/WeirFlow_jtoi/flux_R2E_GW/fun_dAtodY` 均在 `src/GPU/rhs_kernels.cu` 内提供 device 版本，与 `src/Equations/*` 与 `src/ModelData/*` 的实现逐段对齐（本次静态审阅未发现明显的符号/条件反转错误）。

因此，若出现显著状态偏离（如 `ySf` 的 599%），更可能来自：

1) **求和/归约顺序**与硬件数学差异导致的“微小差异被阈值放大”  
2) **与 CPU 语义不完全等价的边界处理/状态更新细节**  
3) **CVODE 迭代路径差异（尤其是 CUDA 独有的预条件器）**

### 5.2 一个明确的语义不一致点：`iBC < 0` 的 uYgw 赋值

CPU/OMP 的 `f_update()` / `f_update_omp()`（`src/ModelData/MD_update.cpp`、`src/ModelData/MD_f_omp.cpp`）在 `Ele[i].iBC < 0`（固定通量边界）分支中：

- 只更新 `Ele[i].QBC`
- **没有显式设置 `uYgw[i]`**

而 CUDA 的 `k_apply_bc_and_sanitize_state`（`src/GPU/rhs_kernels.cu`）对 GW：

- `bc > 0`：用 `ele_yBC` 覆盖（Dirichlet）
- 否则（包含 `bc < 0`）：`uYgw = clamp(Y[iGW])`

这意味着：一旦模型存在 `iBC<0` 的单元，CPU/OMP 与 CUDA 对同一时刻 `uYgw` 的取值可能不一致，进而影响：

- 地下侧向通量（`fun_Ele_sub`）
- 回灌/出渗判据（`Flux_Recharge` / `Flux_Infiltration`）
- 最终反映到 `yUs/yGw/ySf`

该点非常值得优先核实：`ccw` 的最大误差点之一反复落在 `id=508`，而 `id=508` 可能恰好是 BC/SS 敏感单元（需要结合输入数据确认）。

### 5.3 CUDA 独有预条件器导致的“求解路径差异”

`src/Equations/cvode_config.cpp` 中：

- CPU/OMP：`SUNLinSol_SPGMR(..., PREC_NONE, ...)`
- CUDA：默认 `PREC_LEFT` 且 `CVodeSetPreconditioner(PSetup_cuda, PSolve_cuda)`

即使 RHS 一致，预条件器也可能改变：

- GMRES 的迭代次数与停止条件触发点
- Newton 迭代的线性化误差传播
- CVODE 的步长选择与拒绝步频率

此外，`precond_kernels.cu` 中的局部 RHS/雅可比近似与真实 RHS 不完全一致（例如 ET 使用 `satn_new` 而 RHS 使用“上一轮 satn”的策略），会降低预条件器质量并放大“路径差异”。

建议在精度定位阶段提供“关闭预条件器”的 A/B 对比（见 §7 的 issue 拆分）。

### 5.4 CUDA 的原子累加与 GPU 数学实现差异

即使不考虑预条件器，CUDA 仍有额外来源：

- `atomicAdd` 的非确定累加顺序（尤其对 river/segment 汇总量）
- GPU 的 `pow/cbrt/sqrt` 与 CPU libc 实现不同；GPU 更可能触发 FMA（融合乘加）路径  
  → 位级不一致  
  → 在阈值敏感系统中被放大

---

## 6. “rel_max = 1.0”的通量/诊断输出：优先判断为 **输出同步问题**（非 RHS 计算错误的直接证据）

### 6.1 输出系统使用的是 host 侧数组指针

输出映射在 `src/ModelData/MD_initialize.cpp::initialize_output()`：

- `elevinfil/elevexfil/elevrech` 对应 `qEleInfil/qEleExfil/qEleRecharge`
- `elevetev/elevettr/eleveta` 对应 `qEleEvapo/qEleTrans/qEleETA`
- `eleqsurf/eleqsub` 对应 `QeleSurfTot/QeleSubTot`
- `rivq*` 对应 `QrivUp/QrivDown/QrivSurf/QrivSub`

这些 `Print_Ctrl` 记录的都是 host 指针。

### 6.2 CUDA RHS 在 device 上计算了多数通量，但并不回写 host

CUDA 的 `rhs_kernels.cu` 会在 device 上生成：

- `qEleInfil/qEleExfil/qEleRecharge/qEs/qEu/qEg/qTu/qTg`
- `QeleSurf/QeleSub/Qe2r_* / Qriv*`

但在主循环（`src/Model/shud.cpp`）中，每次输出前只调用：

- `Model_Data::gpuSyncStateFromDevice(udata)`：**仅同步状态向量 y**（`src/GPU/DeviceContext.cu`）

并没有将上述诊断数组做 D2H 同步，因此：

- `qEleInfil/qEleExfil/qEleRecharge/Qriv*/Qele*` 等输出文件在 CUDA 后端很可能仍是“初值/旧值/零”  
  → 直接表现为 `rel_max=1.0`

另外，CUDA RHS 当前也没有显式维护某些“CPU 侧用于输出的派生诊断数组”：

- 例如 `qEleEvapo/qEleTrans/qEleETA` 在 CPU/OMP 是由 `f_etFlux()` 写入，但 CUDA 的 `k_ele_local` 只写了分量 `qEs/qEu/qEg/qTu/qTg`，没有合成写回这些派生量。  
  → 即使做了 D2H 同步，也需要明确“在 device 上生成并同步”或“在 host 上复算并输出”的策略。

结论：在当前架构下，`Q* / eta / etev / ettr` 的 `rel_max=1.0` 更像是 **CUDA 后端输出链路未闭合**，而非证明 RHS 本身错误。

---

## 7. 可能问题原因（按优先级）

### P0（最高优先级）：CUDA 输出链路未闭合（诊断数组缺失/未同步）

**症状**：大量 `rel_max=1.0` 出现在通量/诊断文件（`Q*`, `eta/etev/ettr`, `infil/exfil/rech`）。  
**根因候选**：输出系统读取 host 数组，但 CUDA 计算结果驻留在 device；且部分派生诊断量在 CUDA 侧未生成。  
**影响**：虽然不一定影响状态积分，但会严重影响“精度对比”与“定位 ySf 偏离的来源”。

### P1：CUDA 的 `ySf` 状态出现 599% 偏离（需要定位“第一次偏离发生在 RHS 的哪个环节”）

**症状**：`ccw.eleysurf.dat` CUDA 相对 CPU 出现远大于 1 的 `rel_max`，表明 CUDA 产生了 CPU 未出现的峰值/事件。  
**可能机制**：

- NVECTOR_CUDA / 原子累加 / GPU math 差异 → 微小差异触发不同阈值分支（§2.2）  
- 预条件器改变迭代/步长路径（§5.3）  
- 某些边界/状态更新细节不等价（例如 `iBC<0` 的 uYgw，§5.2）

**建议的定位方法（不改算法，仅做诊断）**：

- 使用仓库内置的 GPU 验证框架（`src/GPU/f_cuda.cpp` + `src/GPU/rhs_kernels.cu` 的 `DEBUG_GPU_VERIFY`）  
  - 在 `t≈1081d`（报告中最差点附近）观察：
    - `uYsf/uYus/uYgw` 是否先偏离
    - `qEleInfil/qEleExfil/qEs` 是否先偏离（直接决定 `DYsf`）
    - `QeleSurf/Qe2r_Surf` 是否先偏离（决定侧向出入流）
    - `DYdot` 哪个变量最先超阈值

### P2：CPU/OMP vs CUDA 在 GW 固定通量 BC（`iBC<0`）的工作态可能不一致

**症状**：同一时间点 `uYgw` 的来源不同（§5.2）。  
**影响**：可能造成局部（BC 单元）偏离并逐步传播到 `yUs/ySf`。  
**建议**：明确 SHUD 语义：`iBC<0` 时 `yGW` 是否仍应由状态向量决定（通常应是“状态仍演化，只是在 RHS 加一个通量源项”）。若是，则 CPU/OMP 的 `f_update` 更可能存在遗漏。

### P3：CUDA 预条件器导致的求解路径差异（并可能放大阈值敏感性）

**症状**：CUDA 的状态变量偏离显著高于 OMP。  
**候选原因**：预条件器近似不等价 + NVECTOR_CUDA 归约顺序差异共同影响 CVODE 自适应路径。  
**建议**：提供“关闭预条件器”对比，判断偏离是否显著收敛。

### P4：OMP segment 计算中的并发写入（UB）可能导致非确定性偏差

**症状**：CPU vs OMP 仍有 16 个 FAIL，且最差点集中在特定 element（例如 `id=508`）。  
**候选原因**：`fun_Seg_surface/sub` 内的冗余 `+=` 写入在 OMP 下存在数据竞争（§4.2）。  
**建议**：即便最终会被 `PassValue` 覆盖，也应移除或保护该并发写入，以避免 UB 对结果与可复现性的影响。

### P5：浮点运算顺序与硬件数学差异（“正常但会被放大”）

**症状**：ySf/yUs/yRiv 在 OMP/CUDA 上的误差幅度比“纯舍入误差”大得多。  
**解释**：非线性 + 分段阈值系统对微小差异敏感；不同后端导致 CVODE 子步/迭代路径不同。  
**建议**：在设定目标时区分：

- “数值一致性（bitwise/近似 bitwise）”  
- “物理可接受误差（在 reltol/abstol 允许范围内）”

否则会出现“期望 1e-10/1e-6”但系统本质上对路径极敏感的矛盾。

---

## 8. 可拆分的 GitHub Issues 建议（按优先级）

> 以下 issue 以“可验收、可分工”为目标编写；不包含具体实现代码。

### Issue 1（P0）：[CUDA] 输出链路闭合：同步/生成所有被 `Print_Ctrl` 依赖的诊断数组

**问题**：`initialize_output()` 输出的是 host 指针（如 `qEleInfil/QeleSurfTot/QrivDown/qEleETA/...`），但 CUDA RHS 在 device 上计算且未同步；部分派生量在 CUDA 侧未生成。  
**涉及文件**：

- `src/ModelData/MD_initialize.cpp`（输出映射）
- `src/GPU/rhs_kernels.cu`（device 端可用的原始分量）
- `src/GPU/DeviceContext.cu`（D2H 同步能力目前仅限 state vector）

**验收标准**：

- CUDA 模式下 `eleqsurf/eleqsub/rivq*` 不再出现“全局尺度上的 0/旧值”（`rel_max=1.0`）  
- `eta/etev/ettr/infil/exfil/rech` 等诊断输出能够随时间变化且与 CPU/OMP 一致性显著提升（至少可用于定位状态偏离原因）

### Issue 2（P1）：[CUDA] 定位 `ySf` 599% 偏离的最早发生点（kernel 级别对齐检查）

**问题**：当前仅从输出文件观察到 `ySf` 偏离，但无法判定偏离来自哪个通量项/哪个 kernel 阶段。  
**建议工具**：利用已有的 GPU verify 框架（`DEBUG_GPU_VERIFY` 路径）。  
**验收标准**：

- 给出“首次 mismatch 的字段（例如 `qEleInfil` / `QeleSurf` / `Qe2r_Surf` / `DYdot`）”  
- 给出最早 mismatch 的时间点与 element/riv 索引范围（至少包含 `t≈1081d, id=1072` 这一最差点周边）

### Issue 3（P2）：[CPU/OMP vs CUDA] 明确并对齐 `iBC < 0`（GW 固定通量 BC）的 `uYgw` 语义

**问题**：CPU/OMP 在 `iBC<0` 分支未显式更新 `uYgw`，CUDA 会从 `Y` 派生 `uYgw`。  
**验收标准**：

- 明确文档化：`iBC<0` 时 `yGW` 是否仍应来自状态向量  
- 三后端在该语义下对齐（至少在包含 `iBC<0` 的小算例中 `uYgw` 与 `DYgw` 的行为一致）

### Issue 4（P3）：[CUDA] 评估预条件器对精度的影响（提供可开关对比与数据）

**问题**：CUDA 默认启用 `PREC_LEFT + PSetup/PSolve`，CPU/OMP 不启用；这可能改变 CVODE 的收敛路径并放大阈值敏感性。  
**验收标准**：

- 在同一输入上给出 “precond ON vs OFF” 的对比报告（至少包含 `ySf/yUs/yRiv`）  
- 若 OFF 显著改善一致性，则进一步分析并提出预条件器近似改进方向（例如与 RHS 的 satn 策略对齐）

### Issue 5（P4）：[OMP] 消除 `fun_Seg_surface/sub` 的并发写入 UB（保证可复现性）

**问题**：OMP 下 segment 循环并行，`fun_Seg_*` 内部对 `Qriv* / Qe2r_*` 的 `+=` 存在数据竞争，尽管随后 `PassValue` 会重算，但 UB 可能导致非确定性。  
**验收标准**：

- OMP 多次重复运行在相同输入下输出可复现（bitwise 或至少 `rel_max` 显著稳定）  
- CPU vs OMP 的最差误差进一步收敛，或至少不会出现“偶发性放大”

### Issue 6（P5）：[All] 给出“精度目标”的可实现分级（避免 1e-10/1e-6 的不现实预期）

**问题**：当前目标（OMP≈1e-10、CUDA≈1e-6）在含阈值/分段逻辑 + 自适应积分器的系统中可能不现实。  
**验收标准**：

- 定义两套指标：  
  - 物理一致性（例如水量平衡、峰值/事件一致性）  
  - 数值一致性（逐点误差阈值）  
- 给出推荐 tolerances 与验证集（小算例 + 真实流域）

---

## 9. 结论（面向当前数据的简要判断）

1. **CUDA 的大量 `rel_max=1.0` 通量/诊断文件，更像是“输出链路未闭合/诊断量未同步或未生成”，不是 RHS 错误的直接证据。**  
2. **`ySf` 的 599% 偏离属于“状态层面的严重偏离”，需要用 kernel 级别 verify 找到最早偏离字段，再判断是语义不一致（如 BC）还是数值路径差异（原子/预条件器/归约顺序）。**  
3. **OMP 的 15% `ySf` 与 48% 的某些通量差异，既可能来自 CVODE+NVECTOR_OPENMP 的求解路径差异，也可能存在残留 UB（segment 并发写）。**  

---

## 10. 修复后验证（#64）

本节基于合入以下修复后的重新验证结果：

- #65：`iBC < 0` 语义对齐  
- #67：OMP 并发写入修复  
- #66：CUDA 预条件器开关（默认 ON，可用 `SHUD_CUDA_PRECOND=0` 关闭）

目标：重新运行 CUDA `DEBUG_GPU_VERIFY`，确认“首次 mismatch”是否发生变化，并评估剩余差异。

### 10.1 运行配置

- 编译：`make shud_cuda DEBUG_GPU_VERIFY=1`
- 运行窗口（仅在 `t≈1079-1083d` 执行 verify）：
  - `SHUD_GPU_VERIFY=1`
  - `SHUD_GPU_VERIFY_INTERVAL=1`
  - `SHUD_GPU_VERIFY_T_MIN_DAY=1079`
  - `SHUD_GPU_VERIFY_T_MAX_DAY=1083`
  - `SHUD_GPU_VERIFY_STOP_ON_MISMATCH=1`
- 容差（来自日志）：`atol=1e-8`, `rtol=1e-6`

### 10.2 首次 mismatch：修复前 vs 修复后

- 修复前（Issue #64 初步定位）：`t_day=1079.0`, kernel=`k_ele_local`, field=`qEu`, ele=`936`
- 修复后（`gpu_verify_post_fix.log`）：
  - `t=1553760.62 min`（≈ `1079.00043 day`）
  - kernel=`k_ele_local`, field=`qEu`
  - worst idx=`935`（ele=`936`）
  - mismatches=`994/1147`
  - `max_abs=1.44e-7`，threshold=`1e-8`

结论：**首次 mismatch 的时间点/字段/element 与修复前一致**，说明 #65/#67/#66 未覆盖该偏离源头。

### 10.3 当前剩余差异分析

从 mismatch 的数值形态看：

- CPU 侧 `qEu` 为 0，而 GPU 侧出现 `~1e-8~1e-7` 的非零值（单次 verify 即有 `994/1147` 元素超阈值）
- worst 发生在 `ele=936`，但 mismatch 并非单点，而是“大片元素的小量非零”

该形态更像是以下两类问题之一（或叠加）：

1. **device 侧 `qEu` 未覆盖写/未清零**：在 `k_ele_local` 的某些分支下未写 `qEu`，导致残留值在后续被当作本步结果参与计算。  
2. **CPU/GPU 阈值/截断分支不一致**：CPU 侧分段逻辑将值截断为 0，而 GPU 侧由于浮点路径差异保留了极小正值，并在阈值敏感系统中被后续放大。

与 `ySf` 599% 偏离的关系：`qEu` 属于影响状态更新（`DYdot`）的上游通量项；首次 mismatch 出现在最差点 `t≈1081d` 之前，符合“早期小差异→后期事件分叉”的传播链假设。

### 10.4 结论与建议

- 结论：在当前容差下，CUDA 的首次 mismatch 仍由 `k_ele_local:qEu` 触发（`t≈1079d, ele=936`），修复后未前移/未消失。
- 建议：
  - 对 `src/GPU/rhs_kernels.cu::k_ele_local` 内 `qEu` 增加“全覆盖写/显式清零”的保障，避免分支遗漏导致残留值。
  - 在更窄的窗口（如 `t≈1079-1081d`）增加 `DYdot`/`uYsf` 的 verify，以确认 `qEu` mismatch 是否确实是 `ySf` 偏离的第一传播源。
  - 若要对准报告最差点（`t≈1081d, ele=1072`），建议将 verify 窗口进一步缩窄到 `1080.5-1081.5d` 并提高 `SHUD_GPU_VERIFY_MAX_PRINT` 以覆盖该 element 的打印。
