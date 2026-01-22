
---

## 0. 方案核心约束（为什么要这么设计）

SUNDIALS 的 GPU 编程模型要求：**控制逻辑在 CPU，数据常驻在 GPU，SUNDIALS 不会自动帮你搬数据**；你的 RHS/预条件器等回调必须只访问 device 数据，否则会发生隐式拷贝导致性能崩掉。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/GPU_link.html "1.8. Features for GPU Accelerated Computing — Documentation for SUNDIALS  documentation"))

这直接决定了：

1. `N_Vector` 必须换成 **NVECTOR_CUDA**（或 RAJA/Kokkos/HIP 等，但这里先锁定 CUDA）。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/GPU_link.html "1.8. Features for GPU Accelerated Computing — Documentation for SUNDIALS  documentation"))
    
2. `f(t,y,ydot)` **不能再走当前 Element/River C++ OO 方法**（那套数据结构全在 host），必须把 RHS 计算改成 CUDA kernel + SoA 数据。
    
3. CVODE 的 Krylov 线性求解（你现在用 `SUNLinSol_SPGMR`）是可以和 GPU N_Vector 一起工作的（迭代解法继承 N_Vector 的 GPU 能力）。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/GPU_link.html "1.8. Features for GPU Accelerated Computing — Documentation for SUNDIALS  documentation"))
    

---

## 1. 你当前最重的计算在哪里（从代码结构“落点”）

在一次 `CVode()` 内部会多次调用 RHS `f()`，而你的 `f()` 主要做：

- `Model_Data::f_update(Y,DY,t)`：
    
    - 拆分 `Y` → `uYsf/uYus/uYgw/uYriv/yLakeStg`（并处理负值、BC 等）
        
    - 大量数组清零（`Qe2r_*`, `Qriv*`, `QLake*`, `DY` 等）
        
- `Model_Data::f_loop(t)`：
    
    - `Ele[i].updateElement(t)`（逐单元的本地水力学/土壤参数更新）
        
    - 单元—单元侧向通量（每单元 3 邻边）
        
    - 河段（segment）交换（`NumSegmt`）
        
    - 河道汇流（`NumRiv`）
        
- `Model_Data::f_applyDY(t)`：把各通量汇总成最终 `DY`（也包含湖、河的更新）
    

这意味着 GPU 化的切入点非常明确：**把 f_update / f_loop / f_applyDY 的主体改成 device kernels**，并把 Element/River 里的常量参数展开成 SoA 数组常驻 GPU。

---

## 2. 总体架构（新增模块与职责拆分）

### 2.1 新增目录/模块（建议）

```
src/GPU/
  ShudCudaContext.hpp/.cu        // CUDA stream、错误检查、memset 封装
  ShudDeviceData.hpp/.cu         // SoA 数据结构 + Host→Device 初始化
  ShudRhsKernels.cu              // RHS 主 kernels（update/flux/applyDY）
  ShudPrecKernels.cu             // 预条件器 setup/solve kernels（可选但强烈建议）
  ShudJtimesKernels.cu           // Jv（可选优化项）
src/Model/f_gpu.cpp              // CVODE 回调入口（CPU 侧薄封装）
```

### 2.2 运行时对象关系

- `Model_Data`（host）仍然负责：读入、I/O、控制循环、写结果
    
- 新增 `DeviceModel`（device 常驻）：
    
    - `DeviceStatic`：网格拓扑、土壤/地质/几何常量（初始化一次上传）
        
    - `DeviceStateScratch`：通量数组、临时量（常驻 GPU、每次 RHS memset/复用）
        
    - `DeviceForcing`：随时间变化的 forcing/LAI/BC 等（按需要更新）
        

CPU 与 GPU 的交互原则：

- **只在输出时拷回 state（y）**，其它尽量不回传
    
- forcing/BC 若必须在 CPU 端算，则 **每个 forcing step 批量拷到 GPU**（不要在 RHS 内 host/device 来回）
    

---

## 3. 数据布局：从 OO（Element/River）转 SoA（GPU 友好）

你现在的 `Element` 类内部混了大量参数、派生状态与方法，不可能原封不动丢进 GPU。方案 A 要求做一次“结构性抽取”：

### 3.1 State 向量（保持 SUNDIALS 的单向量布局）

继续沿用你已有的布局（`Macros.hpp`）：

- `Y = [Ysf(NumEle), Yus(NumEle), Ygw(NumEle), Yriv(NumRiv), Ylake(NumLake)]`
    

GPU 上 RHS kernel 通过 NVECTOR_CUDA 提供的 device pointer 访问 y/ydot。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/Install_link.html "1.1. Installing SUNDIALS — Documentation for SUNDIALS  documentation"))

### 3.2 Element/River/Segment 常量参数：SoA

初始化阶段从 `Model_Data::Ele/Riv/RivSeg/lake` 抽取字段，生成 SoA：

**Elements（长度 NumEle）**（示例，不是全量）

- 几何：`area, zmax, zmin, zbed, AquiferDepth, ...`
    
- 邻接：`nabr0,nabr1,nabr2`；`Dist2Nabor0..2`；`edge0..2`；`avgRough0..2`
    
- 属性：`isLakeEle, iBC, iSS, ...`
    
- 土壤/地质参数：`KsatH, infKsatV, ThetaS, ThetaR, Alpha, Beta, macD, macKsatH, ...`
    
- 植被/地表：`vegFrac, impAF, albedo, rootDepth, ...`
    

**Rivers（长度 NumRiv）**

- 拓扑：`down, toLake, ...`
    
- 几何/糙率：`length, bankslope, rough, zmin, ...`
    
- BC 标志：`BC, ...`
    

**Segments（长度 NumSegmt）**

- 关联：`seg_ele, seg_riv, ...`
    
- 交换几何：`length, eqDistance, zbottom, zbank, ...`
    
- 类型：`seg_type`（你代码里 `SegType`）
    

> 关键：**把 RHS 中需要频繁访问的字段都拉平**，避免 kernel 内追指针/虚函数/复杂对象。

---

## 4. RHS（f）在 GPU 上的 kernel 设计

### 4.1 RHS 的 kernel 分解（推荐 5 段流水）

为了减少全局内存读写、减少 kernel 启动开销，可以“融合”，但第一版建议按逻辑清晰拆 5 段，先跑通再融合。

#### Kernel A：State sanitize + 清零（替代 f_update 的大部分）

输入：`y`  
输出：`ysf,yus,ygw,yriv,ylake`（其实就是 y 的切片视图）+ 清零 scratch arrays

- 做你现在的：
    
    - `max(y,0)` 截断
        
    - 处理 `iBC/BC`（如果保留）
        
- 对 `DY`、`Qriv*`、`Qe2r_*`、`QLake*`、`Qseg*` 等做 `cudaMemsetAsync`
    

> 你现在在 `f_update` 里有很多 for 循环清零；GPU 上统一变成 **memset + 少量 kernel**。

#### Kernel B：Per-element 本地更新（替代 `Ele[i].updateElement` + ET/inf/recharge 本地部分）

一线程一单元（或一 warp 一单元，看后续优化）：

- 读：`Ysf,Yus,Ygw` + element 常量参数 + forcing（如 `qEleNetPrep/qPotEvap`）
    
- 算：你现在 `Element::updateElement()` 里算的
    
    - `u_effKH, u_effKV, u_effDmac, u_effKmac, u_infil_capacity, u_deficit, ...`
        
- 同时算本地通量（无需邻居）：
    
    - `qEleInfil, qEleRecharge, qEleExfil, qEleET, ...`
        
- 写：这些通量数组和必要的中间量
    

#### Kernel C：Element-Element 侧向通量（替代 `fun_Ele_surface/sub`）

一线程一“单元-边”（总线程数 `3*NumEle`），避免每线程做 3 次邻居访问导致分支：

- 读：`i, edgeId, nabr`, `Ysf/Ygw`、距离、糙率等
    
- 算：`ManningEquation`/`flux_Ele2Ele_GW` 等（把这些函数改成 `__device__` 可内联）
    
- 写：`QeleSurf[i,edge]`、`QeleSub[i,edge]`
    

> 这一步完全没有写冲突（每个 thread 写自己位置），非常适合 GPU。

#### Kernel D：Element-River segment 交换 + RiverDown（替代 `fun_Seg_*` + `Flux_RiverDown` + `PassValue`）

这里有两种实现：

**D1（简单版，第一版推荐）**：两阶段 + atomic

1. `SegExchangeKernel`：每段算 `QsegSurf/QsegSub`，并 `atomicAdd` 到：
    
    - `Qe2r_Surf[ele]`, `Qe2r_Sub[ele]`
        
    - `QrivSurf[riv]`, `QrivSub[riv]`
        
2. `RiverDownKernel`：每河道算 `QrivDown[i]`（只写自己，不冲突）
    
3. `RiverUpAccKernel`：每河道把 `QrivDown[i]` 加到下游 `QrivUp[down]`（对 `down` 用 atomic）
    

**D2（更快但复杂）**：按 river 拓扑层级做 prefix/scan，减少 atomic（后续优化）

#### Kernel E：ApplyDY（替代 `f_applyDY`）

一线程一个状态分量（或分段：elements/rivers/lakes 三个 kernel）：

- element：
    
    - `DYsf, DYus, DYgw` 按你现有公式汇总
        
    - 处理 lake element（`isLakeEle`）设置 0
        
    - 处理 `iBC/iSS`（若保留）
        
- river：
    
    - 用 `QrivUp/Surf/Sub/Down` 更新 `DYriv`
        
- lake：
    
    - 汇总 `QLake*` 更新 `DYlake`
        

---

## 5. CVODE / SUNDIALS 侧的对接方式（关键点）

### 5.1 构建 SUNDIALS：启用 CUDA

SUNDIALS 官方 CMake 流程下，开启 CUDA 需要 `ENABLE_CUDA=ON`，并设置 `CMAKE_CUDA_ARCHITECTURES`。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/Install_link.html "1.1. Installing SUNDIALS — Documentation for SUNDIALS  documentation"))

你仓库的 `configure` 脚本当前是下载并编译 SUNDIALS（6.0.0）但只开了 OpenMP。方案 A 要把它升级/改造为支持 CUDA 的构建参数（保持“自动下载编译”也可以）。

### 5.2 选择 NVECTOR_CUDA

- 在创建 `N_Vector y` 时用 `N_VNew_Cuda(...)`（或 `N_VNew_CudaManaged`）。NVECTOR_CUDA 文档明确支持 unmanaged/UVM 两种内存模型，并提供 device pointer 的访问接口。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/Install_link.html "1.1. Installing SUNDIALS — Documentation for SUNDIALS  documentation"))
    
- **建议第一版用 unmanaged（device memory）**，避免 UVM 的隐式迁移风险；按 SUNDIALS GPU 模型，你本来也应该自己保证数据在 device 上是最新的。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/GPU_link.html "1.8. Features for GPU Accelerated Computing — Documentation for SUNDIALS  documentation"))
    

> 备注：NVECTOR_CUDA 文档也提醒：某些“依赖 host array pointer 的直接法/预条件器”在 unmanaged 下不适用；你现在用的是 Krylov 迭代法路线，本身是 OK 的，但这也提示我们：**预条件器最好自己写成 GPU kernel**，不要指望 CVBANDPRE 之类模块（它也不支持 CUDA NVector）。([SUNDIALS](https://sundials.readthedocs.io/en/v6.1.0/cvode/Usage/index.html "4.4. Using CVODE for IVP Solution — User Documentation for SUNDIALS  documentation"))

### 5.3 Linear solver

继续用 `SUNLinSol_SPGMR`（你当前配置），它在 GPU 环境下是“继承 NVECTOR 的 GPU 能力”的那类模块。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/GPU_link.html "1.8. Features for GPU Accelerated Computing — Documentation for SUNDIALS  documentation"))

---

## 6. 预条件器（强烈建议做，不然 GPU 可能被 GMRES 拖死）

你现在是 **无预条件器**。在 GPU 上，GMRES 的瓶颈很常见是：

- 多次全局 dot/norm（规约）
    
- 默认差分 Jv 可能导致额外 RHS 调用次数
    

### 6.1 推荐的预条件器：Block-Jacobi（按 element/riv 局部块）

**思路**：近似 Jacobian 的“最硬部分”来自本地垂向过程（infil/recharge/ET、GW 释放等），而侧向耦合先忽略（或弱化）。  
这样预条件器是**完全局部**的：

- 每个 element：3×3 小块（`[SF, US, GW]`）
    
- 每个 river：1×1
    
- 每个 lake：1×1
    

#### PSetup（GPU kernel）

对每个 element 计算一个近似 `M_i ≈ I - γ * J_local`（γ 为 CVODE 给的系数），并把 3×3 逆（或 LU）存起来。

- 存储：每 element 存 9 个 double（或 6 个下三角+对角），总开销可控。
    

#### PSolve（GPU kernel）

对向量 `r` 做 `z = M^{-1} r`：每 element 解 3×3，小而快；river/lake 直接除法。

> 这个预条件器的优势：**没有跨元素通信、没有稀疏矩阵构建、非常适合 GPU**；通常能显著降低 GMRES 迭代次数。

---

## 7. 可选优化：自定义 Jv（Jacobian-vector product），减少“差分 RHS 次数”

如果你继续使用 CVODE 默认的差分 Jv（矩阵自由），在某些设置下会增加 RHS 调用次数（从而增加 kernel 调用次数）。你可以考虑做一个“够用版”的 `J*v`：

- 预先在 RHS 的某个阶段计算出“局部导数系数”（比如对 US/GW 的线性化项）
    
- 在 `JtimesKernel(v)` 里按稀疏邻接结构把 `v` 映射到 `Jv`
    

这部分工作量较大，但如果你发现 profile 里 RHS 调用次数异常多，这是最值得的深挖点之一。

---

## 8. forcing/BC/LAI：GPU 常驻策略（避免每次 RHS 触碰 host）

### 8.1 最实用（第一版）

- forcing/LAI/BC 仍由 CPU 计算（沿用 `updateforcing/ET/updateLAI`）
    
- **每个 forcing step**把结果数组（如 `qEleNetPrep/qPotEvap/...`）一次性 `cudaMemcpyAsync` 到 `DeviceForcing`
    

优点：改动小  
缺点：forcing step 很小、步数巨大时，PCIe 传输会变明显

### 8.2 最大化（第二版）

把 `tReadForcing/ET/updateLAI` 也搬到 GPU：

- forcing 原始时间序列（`TimeSeriesData`）改造成 GPU 可用的紧凑数组
    
- GPU 上用“分段常值”或插值更新 forcing
    

这会更接近 SUNDIALS GPU 模型里倡导的“回调只访问 device 数据”。([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/GPU_link.html "1.8. Features for GPU Accelerated Computing — Documentation for SUNDIALS  documentation"))

---

## 9. 构建系统改造（你当前是 Makefile + configure 脚本）

### 9.1 你的现状

- `configure` 下载 SUNDIALS 并用 CMake 编译
    
- Makefile 编译 SHUD 本体（g++）
    

### 9.2 方案 A 的建议改造路径

1. `configure` 增加 CUDA 选项：`-DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=...` ([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/Install_link.html "1.1. Installing SUNDIALS — Documentation for SUNDIALS  documentation"))
    
2. Makefile 增加 nvcc 编译 `.cu`（或改用 CMake 管理 SHUD 本体，长期更省心）
    
3. 链接额外库：
    
    - `libsundials_nveccuda`
        
    - 可能还需要 `cudart` 等（取决于你安装方式）
        

---

## 10. 逐步交付里程碑（不“画饼”，按可验收拆）

### Milestone A0：SUNDIALS+CUDA 构建跑通

- SUNDIALS 能编译出 NVECTOR_CUDA ([SUNDIALS](https://sundials.readthedocs.io/en/develop/sundials/Install_link.html "1.1. Installing SUNDIALS — Documentation for SUNDIALS  documentation"))
    
- SHUD 编译链接通过，能创建 `N_VNew_Cuda` 并跑一个“空 RHS”示例
    

### Milestone A1：GPU RHS 跑通（不做预条件器）

- `f_gpu.cpp` 用 device kernels 计算 RHS
    
- 与 CPU 版对比：小案例（NumEle~几十）结果一致到可接受误差
    

### Milestone A2：加入 Block-Jacobi 预条件器

- 统计 `CVodeGetNumLinIters / PrecSolves` 明显下降
    
- 总时间下降
    

### Milestone A3：forcing/ET GPU 化（可选）

- forcing step 很密集的场景下进一步提速
    

---

## 11. 风险与对策（提前说清楚）

1. **数值差异**：GPU 并行顺序不同（尤其 atomic 汇总）会引入微小差异
    
    - 对策：先做小网格回归 + 守恒量检查（总水量误差曲线）
        
2. **性能不达预期**：若 GMRES 迭代多、默认差分 Jv 造成 RHS 次数暴涨
    
    - 对策：优先上 Block-Jacobi；必要时再做自定义 Jv
        
3. **内存压力**：SoA 常量 + scratch 数组很多
    
    - 对策：按 kernel 需要“最小集合”设计 scratch；能融合的数组融合；输出相关的统计量尽量延后/按需计算
        
