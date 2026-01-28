

---

## 1. 目标与边界：什么叫“可靠的 GPU 加速”

### 1.1 目标（可验收）

1. **端到端 GPU 驻留**：CVODE 的状态向量 `y`、`ydot`、RHS 所需的几何/参数/拓扑/forcing/BC/临时数组**常驻 GPU**，RHS 回调只做 kernel launch，不再把 `y` 拷回 CPU 做计算。
    
2. **结果可信**：GPU 与 CPU（Serial 或 OpenMP）在相同输入下：
    
    - 状态变量（SF/US/GW/RIV/LAKE）输出在可设阈值内一致（例如 `1e-8~1e-6` 相对误差，按变量量级分档）。
        
    - 守恒量（流域总水量收支、河网累计出流）误差曲线一致（允许 GPU 并行导致的极小舍入差异）。
        
3. **性能可证明**：给出至少 3 个规模（小/中/大）的 benchmark：wall time、CVODE 统计（RHS 调用数、线性迭代次数、非线性迭代次数等），并用 Nsight/NVTX 证明瓶颈从 “CPU RHS + 拷贝” 迁移到 “GPU kernel/线性代数”。
    

### 1.2 边界与假设（先讲清楚，避免“做了也不快”）

- **硬件假设**：第一版优先面向 NVIDIA CUDA GPU（A 系列/RTX/数据中心卡均可），走 NVCC + SUNDIALS NVECTOR_CUDA。
    
- **数值精度**：默认 double（`realtype` 通常是 double），第一版不做混合精度，避免科学结果争议。
    
- **问题规模现实**：如果你的典型网格只有几万 unknown（例如 3×NumEle + NumRiv + NumLake ≈ 3~5 万），GPU 可能“未必大幅领先 OpenMP”，所以必须依赖 **(a) 完全消除 host/device 往返** + **(b) 合理预条件器降低 GMRES 迭代** 才可能体现优势。
    

---

## 2. 先把“为什么现在 CUDA 版更慢”说透：否则路线会走歪

你现在的 GPU-dev 思路本质是：

- CVODE 用了 NVECTOR_CUDA（甚至 UVM/managed），但 **RHS 仍调用 `MD->f_update_omp / f_loop_omp / f_applyDY_omp` 在 CPU 上跑**；
    
- 每次 RHS 回调里还会做 `cudaMemcpy`（或触发 UVM page migration）把 `ydot` 来回搬。
    

这等价于：**把 CVODE 的向量运算放 GPU（dot/norm/axpy），但最重的物理 RHS 仍在 CPU**，且每次回调引入同步与数据迁移 —— 这在 SUNDIALS GPU 指南里被明确认为是 GPU 版最常见的“性能杀手”。正确方向必须是：**RHS 本身 kernel 化**，并保证回调只访问 device 数据。

---

## 3. 总体架构：后端抽象 + 端到端 device 驻留（方案 A）

### 3.1 运行时 Backend 三态（CPU / OMP / CUDA）

建议加统一开关（CLI 或配置）：

- `--backend cpu|omp|cuda`  
    并把它写入一个全局/配置变量（类似你们已有的 `global_implicit_mode` 风格）。
    

这样做的价值：

- **同一套输入**可以直接对比三种后端，便于回归与验证；
    
- CUDA 版出问题时可快速切回 OMP，不影响科研产出。
    

### 3.2 CUDA 后端的“数据驻留”原则（最核心）

CUDA 后端的运行期只允许出现两类 host-device 交互：

1. **forcing / BC 更新时**：CPU 读时间序列文件、算出本步 forcing 数组 → `cudaMemcpyAsync` 到 GPU（每个 forcing step 一次）。
    
2. **输出/诊断时**：把必要的状态/通量数组从 GPU 拷回 host，做 I/O（按输出间隔发生，不在每次 RHS 内发生）。
    

除此以外：

- `f(t,y,ydot)` 回调内 **禁止** `N_VGetHostArrayPointer_Cuda()` + CPU 计算 + CopyBack 这种模式；
    
- 所有 RHS 需要的数组（几何、参数、拓扑、forcing、scratch、flux）要么常驻 device，要么在 forcing step 时一次性更新到 device。
    

---

## 4. 数据结构改造：从“复杂 C++ 对象图”到 GPU 可用的扁平 SoA

### 4.1 状态向量布局保持不变（利于与现有 CVODE 对齐）

继续使用你现在的 global layout（GPU/CPU 一致）：

- `SF[i] = y[i]`
    
- `US[i] = y[NumEle + i]`
    
- `GW[i] = y[2*NumEle + i]`
    
- `RIV[i] = y[3*NumEle + i]`
    
- `LAKE[i] = y[3*NumEle + NumRiv + i]`
    

这点非常关键：

- 既能复用现有 `SetIC2Y / summary / IO` 的逻辑（只需增加 CUDA 分支的拷贝/指针访问），
    
- 也便于实现“同一份输出对比三后端”。
    

### 4.2 设备侧常驻数据：DeviceContext 设计（建议新增模块）

新增 `src/GPU/DeviceContext.{hpp,cpp,cu}`（或类似目录）维护 CUDA 侧所有资源：

**(A) Element 静态参数（SoA）**  
把 `_Element` 里 RHS 必需字段抽出来，按字段分数组：

- 几何：`area[i]`, `z_surf[i]`, `z_bottom[i]`, `edge[3*i+j]`, `dist2nabor[3*i+j]`, `dist2edge[3*i+j]`
    
- 粗糙/边界：`avgRough[3*i+j]`, `depression[i]`, `closeBoundaryFlag`
    
- 土壤/地质/土地利用关键参数（用于 infiltration/recharge/ET）：例如 `Sy[i]`、`infD[i]`、`infKsatV[i]`、`macKsatV[i]`、`hAreaF[i]`、`RzD[i]`、`ThetaS[i]`/`ThetaR[i]` 等（具体要以你 RHS 公式用到的为准）
    
- 拓扑索引：`nabr[3*i+j]`, `lakenabr[3*i+j]`, `isLakeEle[i]`, `lakeId[i]`（如需要）
    

**(B) River 静态参数（SoA）**

- `down[i]`, `toLake[i]`, `type[i]`, `BC[i]`
    
- `depth[i]`, `bankslope[i]`, `bottomWidth[i]`, `length[i]`, `bedSlope[i]`, `zbank[i]`, `zbed[i]`, `avgRough[i]`, `dist2Down[i]`, `KsatH[i]`, `BedThick[i]`, `Cwr[i]` 等  
    （实际用到哪些以 `Flux_RiverDown / fun_Seg_*` 为准）
    

**(C) Segment 拓扑与参数（SoA）**

- `segEle[i]`, `segRiv[i]`, `segLen[i]`, `segCwr[i]`, `segBedThick[i]`, `segEqDist[i]`（如用）
    

**(D) Lake 参数与 bathymetry**

- `zmin[l]`, `lakeBottom[l]`（你在 element-lake GW flux 中会用）
    
- bathymetry 建议存成 **拼接数组 + offset/len**：
    
    - `bath_yi[]`, `bath_ai[]`, `bath_off[l]`, `bath_len[l]`
        
    - kernel 内对每个 lake 做线性搜索或二分搜索求 toparea
        

> 这样做的核心理由：GPU kernel 不应“穿透”到 `_Element/_River/_Lake` 这种继承层级很深的对象，否则无法保证可控的内存访问模式与可维护性。

### 4.3 动态 scratch / flux 数组也要 device 常驻（否则没有意义）

device 侧至少需要常驻（按你现有 RHS）：

- 元素通量：`qEleInfil/qEleExfil/qEleRecharge/qEs/qEu/qEg/qTu/qTg/qEleETA...`
    
- 元素边通量：`QeleSurf[3*NumEle]`, `QeleSub[3*NumEle]`
    
- 河段通量：`QsegSurf[NumSeg]`, `QsegSub[NumSeg]`
    
- 汇总：`QrivSurf/Sub/Up/Down[NumRiv]`, `Qe2r_Surf/Sub[NumEle]`
    
- 湖相关：`QLakeSurf/Sub`, `QLakeRivIn/Out`, `qLakePrcp/Evap`, `y2LakeArea`
    

输出时再按需把其中一部分拷回 host。

---

## 5. RHS GPU 化：把现有 `f_update -> f_loop -> f_applyDY` 映射为 kernel pipeline

### 5.1 Host 侧 wrapper：`f_cuda(t, y, ydot, user_data)` 的职责要“薄”

新增：

- `src/GPU/f_cuda.cpp`（host wrapper）
    
- `src/GPU/rhs_kernels.cu`（实际 kernels）
    

`f_cuda` 做三件事即可：

1. 从 NVECTOR_CUDA 取 device 指针：
    
    - `dY = N_VGetDeviceArrayPointer_Cuda(y)`
        
    - `dYdot = N_VGetDeviceArrayPointer_Cuda(ydot)`
        
2. 使用与 NVECTOR_CUDA 一致的 stream / exec policy（避免隐式同步）。
    
3. 顺序 launch kernels（不在 wrapper 里做任何 CPU 侧数组遍历/拷贝）。
    

### 5.2 推荐的 kernel 拆分（第一版以“正确可跑 + 易 debug”为主）

结合你当前 CPU RHS 的结构，建议拆 4~6 个 kernel（后续再融合）：

#### Kernel A：`rhs_unpack_and_zero`

- 输入：`dY`
    
- 输出：
    
    - `uYsf/uYus/uYgw/uYriv/uYlake`（可选：也可以不单独存，直接用 dY 的切片）
        
    - 初始化/清零各种累加数组：
        
        - `QeleSurf/QeleSub`、`QsegSurf/QsegSub`、`QrivSurf/Sub/Up`、`Qe2r_*`、`QLake*`、`DY` 等
            
- 在这里处理 **BC（固定水位）** 的“覆盖”逻辑（element/river/lake），确保后续 kernel 用的是“实际有效状态”。
    

> 清零数组建议优先用 `cudaMemsetAsync`（对大数组更快），而不是 kernel 里每线程清零很多元素。

#### Kernel B：`rhs_vertical_processes`（每 element 1 线程）

对应 CPU 的：

- `Ele[i].updateElement(...)`
    
- `fun_Ele_Infiltration`
    
- `fun_Ele_Recharge`
    
- `f_etFlux`（蒸散分配）
    

输出写入：

- `qEleInfil/qEleExfil/qEleRecharge/qEs/qEu/qEg/qTu/qTg/...`
    
- element 的一些派生量（如 `effKH/satn/deficit`）可选择写到 scratch 或局部寄存器（权衡重算 vs 带宽）
    

湖区 element 特例：

- `fun_Ele_lakeVertical` 等逻辑在这里分支处理（isLakeEle 走另一套路径）。
    

#### Kernel C：`rhs_element_lateral`（每 element 1 线程）

对应 CPU 的：

- `fun_Ele_surface`
    
- `fun_Ele_sub`
    

做法建议沿用你 CPU 的“按 element 定向边”计算（每 element 计算 3 条边，写 `QeleSurf[3*i+j]`、`QeleSub[3*i+j]`）：

- 优点：无写冲突，不需要原子；
    
- 与 CPU 行为更一致（尤其你们在 depression / Dist2Nabor 等上可能不是严格对称）。
    

湖边界（lakenabr）需要对 `QLakeSurf/Sub[lake]` 做累加，则用 `atomicAdd`（湖数量通常小，可接受）。

#### Kernel D：`rhs_segment_flux`（每 segment 1 线程）

对应 CPU 的：

- `fun_Seg_surface`
    
- `fun_Seg_sub`
    

只写：

- `QsegSurf[i]`, `QsegSub[i]`  
    不要在这里直接加到 `QrivSurf/Qe2r`（否则需要原子且难 debug），汇总留给下一步。
    

#### Kernel E：`rhs_river_down`（每 river 1 线程）

对应 CPU 的：

- `Flux_RiverDown`
    

写：

- `QrivDown[i]`  
    若 `toLake`：对 `QLakeRivIn[lake]` 做 `atomicAdd`。
    

#### Kernel F：`rhs_passvalue_and_applyDY`

对应 CPU 的：

- `PassValue()` + `f_applyDY()`
    

建议分两段（也可两 kernel）：

1. **PassValue 部分**：
    
    - `QrivSurf/Sub/Up`、`Qe2r_*` 清零
        
    - 遍历 `NumSeg`：`atomicAdd(QrivSurf[segRiv], QsegSurf)`；`atomicAdd(Qe2r_Surf[segEle], -QsegSurf)`；sub 同理
        
    - 遍历 `NumRiv`：按 downstream 把 `-QrivDown[i]` 累加到 `QrivUp[down]`（需要 atomic）
        
2. **ApplyDY 部分**：
    
    - element：`DYsf/DYus/DYgw` 按公式写 `dYdot`
        
    - river：按你希望的物理一致版本计算（建议采用 serial 版的 `Q/Length -> dA/dt -> dY/dt` 逻辑，保证与几何一致）
        
    - lake：按 `qLakePrcp - qLakeEvap + (QLakeRivIn-QLakeRivOut+QLakeSub+QLakeSurf)/y2LakeArea`等写入
        

> 这一段是数值正确性关键：建议你先做一个 **debug 模式**，在同一个时间点把 GPU 版 `dYdot` 拷回 host，与 CPU 版 `dYdot` 对比，定位差异来源（是 vertical、lateral、segment、riverdown 还是 applyDY）。

---

## 6. 线性求解与预条件器：决定“隐式模式 GPU 能不能赢”的上限

### 6.1 第一版：先保持 SUNLinSol_SPGMR（矩阵自由）

你现状就是 SPGMR（GMRES）+ matrix-free。CUDA 版可以继续沿用，NVECTOR_CUDA 会把向量运算放到 GPU。

但注意：如果你不做预条件器，GMRES 迭代次数一高，GPU 版会被大量 dot/norm/axpy 这种“带宽型小 kernel”拖死（尤其 unknown 只有几万时更明显）。

### 6.2 推荐预条件器：Block-Jacobi（按 element 3×3 小块）

这和你《重构方案设计》里建议一致：把最“硬”的局部垂向过程（infil/recharge/ET/GW 释放等）作为局部块近似，侧向耦合忽略或弱化。

**实现要点**（落地版）：

- 每个 element 存一个 3×3（或 LU/逆）：
    
    - 块对应 `[SF, US, GW]`
        
- river/lake 先做 1×1（或简单对角缩放）
    

**PSetup（GPU kernel）**：

- 输入：`t, y, fy, gamma`
    
- 输出：每 element 的 `M_i ≈ I - gamma * J_local` 的分解结果（例如 3×3 LU）
    
- 关键：完全局部、无通信，特别适合 GPU。
    

**PSolve（GPU kernel）**：

- 对 `r` 做 `z = M^{-1} r`：每 element 解 3×3，小而快；
    
- river/lake 直接除法。
    

### 6.3 J_local 怎么来（建议两阶段，保证可落地）

- **阶段 1（最快落地）**：先做“对角/近对角”的局部缩放（不追求最强，但能明显降迭代）
    
- **阶段 2（更强）**：做“局部有限差分”近似 Jacobian：  
    每 element 在 PSetup 内只计算 **local RHS（不含侧向与河网）**，对 3 个分量做扰动差分，构成 3×3。  
    这是工程上最稳的“强预条件器”路线：实现成本可控、效果通常明显。
    

---

## 7. forcing / ET / BC：第一版怎么做才能“不碰 host、但又不大改业务逻辑”

你给的设计建议是“第一版 forcing/LAI/BC 仍由 CPU 计算，每 forcing step 一次性拷贝到 device”。这非常适合落地。

### 7.1 关键点：不要在 forcing 更新里依赖 GPU 状态

现有 `updateforcing()` 里有 `Ele[i].updateElement(uYsf,uYus,uYgw)` 这样的状态依赖调用，如果你照搬到 CUDA 版，就会迫使你每 forcing step 把 `y` 从 device 拷回 host —— 会直接毁掉驻留策略。

因此 CUDA 版建议拆分：

- **CPU 侧保留**：纯时间序列/气象驱动计算（不读 `y`）
    
    - `tsd_weather/mf/lai` 的 `movePointer(t)`
        
    - `tReadForcing()` 产生 `qElePrep/qPotEvap/qPotTran/qEleETP...`
        
    - `ET(t,tnext)` 更新 `yEleSnow/yEleIS/qEleNetPrep/qEleE_IC...`（这些本来就不在 ODE 状态里）
        
- **GPU 侧完成**：所有需要 `y` 的更新（`updateElement`、ET 分配 `f_etFlux`、infiltration/recharge 等）在 RHS kernels 里做。
    

### 7.2 BC/SS 时间序列怎么办（落地处理）

你现在的 `_TimeSeriesData::getX()` 实际返回的是“当前指针值”，是否随时间变只取决于你是否调用了 `movePointer(t)`。因此一个很稳的落地策略是：

- 在每个 forcing step 开始时，CPU 对 **所有会在 RHS 用到的 time series** 做 `movePointer(t)`（weather/LAI/MF/BC/SS…）
    
- 然后把“当前时刻有效值”预计算成数组（例如 `Ele_QBC[i]`, `Riv_qBC[i]`），一次性拷贝到 device
    
- RHS kernel 内只读取这些数组，不做文件/指针逻辑
    

这既符合你现有代码风格，也避免把复杂 TimeSeriesData 逻辑搬上 GPU。

---

## 8. 构建系统与工程改造：确保“能编出来、能跑、可切换”

### 8.1 SUNDIALS 必须开启 CUDA 特性

你们现有 `configure` 脚本只开了 OpenMP。CUDA 版需要：

- `-DENABLE_CUDA=ON`
    
- `-DCMAKE_CUDA_ARCHITECTURES=<xx>`  
    并确保安装出来有 `libsundials_nveccuda`。
    

### 8.2 Makefile 增加 `shud_cuda`，或直接迁移到 CMake（更推荐）

《仓库改动清单》里给了 Makefile 增量方案：nvcc 编译 `.cu`，链接 `-lsundials_nveccuda -lcudart`。

但从“长期科研可维护”角度，更推荐把 SHUD 本体也迁到 CMake：

- 后端选择变成 CMake option：`-DSHUD_ENABLE_CUDA=ON` / `-DSHUD_ENABLE_OPENMP=ON`
    
- `.cu` 与 `.cpp` 统一管理
    
- 便于 CI/benchmark 脚本化、便于不同平台（Linux/Windows）一致构建
    

### 8.3 NVECTOR_CUDA 的 coherency 与访问方式必须规范

CUDA 后端要用：

- `N_VGetDeviceArrayPointer_Cuda()` 取 device 指针
    
- 输出/诊断前用 `N_VCopyFromDevice_Cuda()` 保证 host 读到的是新数据  
    反之如在 host 修改了 `y`（初始化/重设 IC），要 `N_VCopyToDevice_Cuda()`。
    

---

## 9. 科学研究落地：验证、回归、benchmark、profiling 的“必做清单”

### 9.1 验证策略（建议按“层级”做，不要一口吃成胖子）

**Level 0：RHS 单点评估**

- 固定某个 `t` 和 `y`（从一次 CPU run dump 出来）
    
- GPU 版只跑 `f_cuda(t,y,ydot)`，把 `ydot` 拷回，与 CPU `f()` 的 `ydot` 对比
    
- 这一步能把问题快速定位到某个物理过程 kernel
    

**Level 1：短时积分一致性**

- 用 ccw 这类小流域跑 1~2 天
    
- 对比：
    
    - 输出水位/流量时间序列
        
    - 总水量守恒误差曲线
        
- 允许少量浮点差异，但趋势与量级必须一致
    

**Level 2：中尺度/大尺度回归**

- 选择你们已有 benchmark（如 ChenHai）
    
- 对比 CPU/OMP/CUDA 三后端的结果一致性与性能
    

### 9.2 性能评估必须输出“可解释指标”

每个 benchmark 输出至少：

- wall time（总耗时、每 simulated day 耗时）
    
- `CVodeGetNumRhsEvals`（nfe）
    
- `CVodeGetNumLinIters`（nli）
    
- `CVodeGetNumNonlinSolvIters`（nni）
    
- `CVodeGetNumErrTestFails`（netf）
    
- 以及你们已有的 `MD->nFCall` 等计数
    

否则“快/慢”的原因无法解释（是 RHS kernel 慢？还是 GMRES 迭代多？还是 host/device 拷贝？）。

### 9.3 Profiling：必须用 Nsight + NVTX

- 在 `f_cuda`、PSetup、PSolve 周围加 NVTX range，Nsight Systems 一眼能看到时间线瓶颈。
    
- Nsight Compute 针对最重的 1~2 个 kernel 看：
    
    - 全局内存吞吐/合并访问（coalescing）
        
    - 原子开销（若 PassValue/湖汇总用 atomic）
        
    - 分支发散
        

---

## 10. 风险与对策（提前把坑填上）

1. **GPU 版仍不快**（最常见原因：GMRES 迭代高、problem size 太小）
    
    - 对策：优先实现 Block-Jacobi 预条件器；benchmark 时选足够大规模；对比 nli/nfe。
        
2. **输出/诊断导致频繁拷贝**
    
    - 对策：只在输出时刻拷贝必要数组；给用户开关减少高频输出项。
        
3. **数值差异引发科研质疑**（并行顺序不同、atomic 累加不同）
    
    - 对策：建立“误差可接受标准”与守恒量检查；对关键累加（如湖/河汇总）考虑两段归约减少不确定性。
        
4. **UVM/managed 内存不确定性**
    
    - 对策：第一版优先 unmanaged + 显式 copy；UVM 作为可选优化而不是默认。
        

---

## 11. 里程碑交付（按可验收拆分，确保“每一步都有成果”）

> 这里按你材料里的 Milestone 思路细化，并补上“验收标准”。

### Milestone A0：CUDA 构建跑通

- SUNDIALS 编译出 NVECTOR_CUDA
    
- SHUD 能创建 `N_VNew_Cuda` 并跑一个最小例子（哪怕 RHS 先是空）
    

### Milestone A1：GPU RHS 跑通（无预条件器）

- 完成 `f_cuda + rhs_kernels` 基本 pipeline（A~F）
    
- 小案例（几十~几百单元）`ydot` 单点评估与 CPU 对齐
    
- 1~2 天短时积分对齐
    

### Milestone A2：Block-Jacobi 预条件器上线

- `CVodeSetPreconditioner` 接入
    
- 观测到 `nli` 明显下降、减速问题得到缓解
    

### Milestone A3：输出/forcing 传输优化

- forcing/BC/SS 数组一次性异步拷贝，输出时按需拷贝
    
- 加 NVTX + Nsight 报告，形成可发表/可复现的性能证据链
    

### Milestone A4：性能深挖（可选）

- kernel 融合、减少 atomic、结构化归约
    
- 自定义 Jv（如果 profiling 发现差分 Jv 导致 RHS 调用爆炸）
    

---

## 12. 从你当前仓库出发的“最小改动落地路线”（非常具体）

结合《仓库改动清单》给出的目录级拆分，我建议你按下面文件/模块推进（每一步都可单独跑通）：

1. **新增 GPU 目录与 DeviceContext**
    
    - `src/GPU/DeviceContext.hpp/.cpp/.cu`：负责扁平化数据的 `cudaMalloc/cudaFree`、H2D/D2H、stream、exec policy
        
2. **改 `shud.cpp`：创建 CUDA NVECTOR + 初始化 GPU 数据**
    
    - backend==cuda 时：`N_VNew_Cuda(NY)`
        
    - `MD->gpuInit(...)`：把 Ele/Riv/Seg/Lake 静态参数与拓扑扁平化上传
        
3. **新增 `f_cuda` + kernels**
    
    - `f.cpp` 变成分发器：cpu/omp/cuda 三分支
        
4. **输出与 summary 改造**
    
    - backend==cuda：输出前 `N_VCopyFromDevice_Cuda`，并把必要数组 D2H
        
5. **预条件器**
    
    - 新增 `src/GPU/precond.cu`，实现 PSetup/PSolve kernel
        
    - `SetCVODE()` 增加 `CVodeSetPreconditioner()`
        
6. **benchmark 脚本 + Nsight**
    
    - `scripts/validate_cpu_omp_cuda.sh`
        
    - NVTX 标记加到 `f_cuda` 与 precond 上
        

---

如果你愿意把这套方案进一步变成“研发实施清单”（例如：把每个 kernel 需要的输入/输出数组列成表、把 `_Element::updateElement / Flux_Infiltration / Flux_Recharge / fun_Ele_surface / fun_Ele_sub / Flux_RiverDown / PassValue` 各自需要的字段逐项抽出来，形成 **DeviceContext 字段白名单**），我也可以基于你当前代码结构把这张“字段映射表 + kernel I/O 表”完整列出来，确保你们实现时不会遗漏字段或拷贝过多导致带宽浪费。
