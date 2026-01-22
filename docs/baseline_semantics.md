# RHS Baseline 语义（Serial 参照标准）

本文档用于**明确 SHUD/SHUD-GPU 的 RHS（Right-Hand Side，常微分方程组右端项）基准语义**，并规定：

- **基准（Baseline）= CPU Serial（无 OpenMP）路径**
- 任何其它实现路径（OpenMP / GPU / 未来的多后端）在做“对齐/验证”时，应以该 Baseline 为参照

> 目标：让“同一份输入 + 同一份代码（Serial Baseline）”在数值上可复现，并作为 GPU/并行实现的 golden/回归标准。

---

## 0. 构建命令（`make shud` vs `make shud_omp`）

本仓库 `Makefile` 提供两条主要 CPU 构建路径：

- **Serial Baseline（推荐用于对齐/回归）**：不定义 `_OPENMP_ON`，生成 `./shud`

```bash
make clean
make shud
```

- **OpenMP（并行路径，当前不作为 Baseline）**：编译期定义 `_OPENMP_ON`（`Makefile` 通过 `-D_OPENMP_ON`），生成 `./shud_omp`

```bash
make clean
make shud_omp
```

> 提示：`validation/baseline` 工作流默认要求使用 `make shud` 的 Serial 版本。

## 1. 为什么选择 Serial 作为 Baseline（含 ET / Lake / River 完整逻辑）

### 1.1 Serial 路径具备“完整 RHS 逻辑闭环”

在当前仓库中，**Serial 路径**（未定义 `_OPENMP_ON`）的 RHS 回调 `f(...)` 会按如下顺序执行：

1. **状态与通量初始化 + 边界条件（BC）装配**：`Model_Data::f_update(...)`
2. **物理过程通量计算（包含 ET/Lake/River 的完整逻辑）**：`Model_Data::f_loop(...)`
3. **把通量装配为 ydot（DY）**：`Model_Data::f_applyDY(...)`

对应源码入口：

- `src/Model/f.cpp`：Serial vs OpenMP 的 RHS 分发
- `src/ModelData/MD_update.cpp`：`f_update(...)`
- `src/ModelData/MD_f.cpp`：`f_loop(...)`
- `src/ModelData/MD_ET.cpp`：ET 相关（forcing、雪/截留、蒸散通量）

其中 `MD_f.cpp:f_loop(...)` 明确包含：

- **ET 通量**（`f_etFlux(...)`）
- **Lake 元素的垂向/水平处理**（`fun_Ele_lakeVertical / fun_Ele_lakeHorizon`）
- **River routing**（`Flux_RiverDown(...)`）
- **Lake 蒸发上限**（`qLakeEvap = min(qLakeEvap, qLakePrcp + yLakeStg)`）

这使得 Serial RHS 对“陆面-河道-湖泊”的耦合逻辑闭合，适合作为语义基准。

### 1.2 Serial 更易做到“确定性（determinism）”与可复现

作为基准语义，最关键的是可对照与可复现：

- Serial 的循环顺序固定，浮点累加次序固定，更容易做到**多次运行误差接近机器精度**
- 仓库已提供 **CPU-Serial RHS baseline 回归**（见 `validation/baseline/README.md`），默认容差目标为 `1e-12`

相比之下，多线程并行通常会引入：

- 归约/累加顺序变化（浮点非结合律）
- 非原子写入共享数组导致的数据竞争（race）

因此我们把 Serial 定义为基准语义，OpenMP/GPU 以“对齐 Serial”为第一优先级。

---

## 2. Baseline 执行链路（时间推进 + RHS 语义）

### 2.1 时间推进与 forcing / ET 的位置（重要）

在 `src/Model/shud.cpp` 的主循环中，每个求解子步（`SolverStep` 或可选更小的 `ETStep`）会执行：

1. `Model_Data::updateAllTimeSeries(t)`：把各类时间序列指针移动到当前时间 `t`（提升查询效率）
2. `Model_Data::updateforcing(t)`：读取气象/LAI/MF 等 forcing，并更新潜在蒸散等派生量
3. `Model_Data::ET(t, tout)`：按 `DT_min=tout-t` 更新雪与冠层截留，产出 `qEleNetPrep`、`qEleE_IC` 等
4. 调用 `CVode(...)`：由 CVODE 在区间内多次调用 RHS `f(...)`

**Baseline 语义假设**：在一个 CVODE 子步区间内，forcing/ET 的输入已在外层按上述步骤更新；RHS 负责把这些输入与当前状态 `Y(t)` 转换为 `DY(t)`。

### 2.2 RHS（Serial Baseline）三阶段语义

当 CVODE 调用 `f(t, Y, DY, MD)` 时（Serial）：

#### 阶段 A：`f_update(Y, DY, t)` —— 初始化/BC/清零

- 从 `Y` 填充工作态变量：`uYsf/uYus/uYgw/uYriv/yLakeStg`
  - **状态非负截断（ClampPolicy）**：当 `CLAMP_POLICY=1` 时，以上工作态变量在装配时会被截断到 `>=0`（BC 固定水头/水位的元素/河道除外）；当 `CLAMP_POLICY=0` 时，保持使用原始 `Y`（用于严格复现历史 Serial baseline）
- 应用边界条件：
  - 元素 GW：`iBC>0` 固定水头（`yBC`），`iBC<0` 固定通量（`QBC`）
  - 河道：`BC>0` 固定水位（`yBC`），`BC<0` 固定入流（`qBC`）
- 清零（为本次 RHS 调用服务）：
  - 单元侧向通量数组：`QeleSurf/QeleSub`（及 Tot）
  - 河道交换聚合量：`QrivSurf/QrivSub/QrivUp`、`Qe2r_Surf/Qe2r_Sub`
  - 湖泊聚合量：`QLakeSurf/QLakeSub/QLakeRivIn/QLakeRivOut`、`qLakeEvap/qLakePrcp`
  - `DY` 全部清零

#### 阶段 B：`f_loop(t)` —— 计算通量（ET/Lake/River 全部在此闭合）

`MD_f.cpp:f_loop(...)` 的核心顺序如下（按源码的实际循环结构）：

1. **元素循环（垂向过程）**
   - 若 `lakeon && Ele[i].iLake>0`：湖泊元素
     - `Ele[i].updateLakeElement()`
     - `fun_Ele_lakeVertical(i,t)`：湖面蒸发等（当前实现中 `qEleEvapo=qPotEvap`）
     - 按 `NumEleLake` 汇总：`qLakeEvap/qLakePrcp`
   - 否则：陆面元素
     - `f_etFlux(i,t)`：计算 `qEs/qEu/qEg/qTu/qTg` 等实际蒸散通量
     - `Ele[i].updateElement(...)`
     - `fun_Ele_Infiltraion(i,t)`、`fun_Ele_Recharge(i,t)`

2. **元素循环（水平侧向过程）**
   - 湖泊元素：`fun_Ele_lakeHorizon(i,t)`（侧向通量置零）
   - 陆面元素：`fun_Ele_surface(i,t)`、`fun_Ele_sub(i,t)`

3. **河段循环（element↔river 交换）**
   - `fun_Seg_surface(...)`、`fun_Seg_sub(...)`，产出 `QsegSurf/QsegSub`

4. **河道循环（下泄与汇流）**
   - `Flux_RiverDown(t,i)`：计算 `QrivDown[i]`，并在 `toLake` 情况下累加 `QLakeRivIn`

5. **湖泊蒸发约束**
   - `qLakeEvap[i] = min(qLakeEvap[i], qLakePrcp[i] + yLakeStg[i])`
   - `qLakeEvap[i] = max(0, qLakeEvap[i])`

6. **聚合更新（PassValue）**
   - `PassValue()` 以 `QsegSurf/QsegSub` 为准重新汇总：
     - `QrivSurf/QrivSub`（river 收到的交换）
     - `Qe2r_Surf/Qe2r_Sub`（element→river 的交换，符号相反）
     - `QrivUp`（由 `QrivDown` 推导上游来水）

> 注：`PassValue()` 会重置并重算 `QrivSurf/QrivSub/QrivUp/Qe2r_*`，因此 **Baseline 语义以 `PassValue()` 的汇总结果为准**。

#### 阶段 C：`f_applyDY(DY, t)` —— 装配 ydot（DY）

1. **元素（3 个状态：SF/US/GW）**
   - 汇总：
     - `QeleSurfTot = Qe2r_Surf + Σ(QeleSurf[3])`
     - `QeleSubTot  = Qe2r_Sub  + Σ(QeleSub[3])`
   - 写入：
     - `DY[sf] = qEleNetPrep - qEleInfil + qEleExfil - QeleSurfTot/area - qEs`
     - `DY[us] = qEleInfil - qEleRecharge - qEu - qTu`
     - `DY[gw] = qEleRecharge - qEleExfil - QeleSubTot/area - qEg - qTg`
   - 应用 BC/SS + `/Sy`（对 US/GW）
   - 若 `Ele[i].iLake>0`：该 element 的三个状态 `DY=0`（由湖泊状态方程接管）

2. **河道（RIV）**
   - 若固定水位：`DY=0`
   - 否则先算横断面积变化率：
     - `dA = (-Up - Surf - Sub - Down + qBC)/Length`
     - 截断：`dA >= -u_CSarea`
   - 再由几何把 `dA` 转为 `dY`：`DY = fun_dAtodY(dA, topWidth, bankslope)`

3. **湖泊（LAKE）**
   - `DY_lake = qLakePrcp - qLakeEvap + (QLakeRivIn - QLakeRivOut + QLakeSub + QLakeSurf)/y2LakeArea`

---

## 3. Serial 与 OpenMP 路径的已知差异（当前仓库事实）

当编译期开启 `_OPENMP_ON` 时，`src/Model/f.cpp` 会切换到：

- `Model_Data::f_update_omp(...)`（`src/ModelData/MD_f_omp.cpp`）
- `Model_Data::f_loop_omp(...)`（`src/ModelData/MD_f_omp.cpp`）
- `Model_Data::f_applyDY_omp(...)`（`src/ModelData/MD_f_omp.cpp`）

与 Serial Baseline 相比，当前已知差异包括（不穷举）：

1. **Lake 逻辑缺失/不完整**
   - `f_loop_omp(...)` 未包含 `lakeon && Ele[i].iLake>0` 的分支处理（无 `fun_Ele_lakeVertical/Horizon`、无 `qLakeEvap/qLakePrcp` 汇总）
   - `f_applyDY_omp(...)` 不计算 `LAKE` 状态的 `DY`
   - 因此：OpenMP 路径不能视为“Lake-ON 的基准实现”

2. **ET 通量路径差异**
   - Serial 的 `f_loop(...)` 会对陆面元素调用 `f_etFlux(i,t)`（影响 `qEs/qEu/qEg/qTu/qTg` 与 `DY`）
   - OpenMP 的 `f_loop_omp(...)` 当前未调用 `f_etFlux(...)`

3. **River 方程装配差异（物理含义不同）**
   - Serial `f_applyDY(...)`：以 `Length` 计算 `dA`，截断后用 `fun_dAtodY(...)` 转换为 `dY`
   - OpenMP `f_applyDY_omp(...)`：使用 `u_TopArea` 做归一化，且未执行 `dA->dY` 转换与截断逻辑

4. **状态非负裁剪策略不同**
   - 现已提供统一开关 `CLAMP_POLICY`（CPU/GPU 共用接口）控制 `uYsf/uYus/uYgw/uYriv/yLakeStg` 的非负截断：
     - `CLAMP_POLICY=1`：启用截断（默认）
     - `CLAMP_POLICY=0`：关闭截断（保持历史 Serial baseline 语义）
   - 设置方式（二选一即可）：
     - 配置文件（`*.cfg.para`）：`CLAMP_POLICY 0/1`
     - CLI：`./shud -C 0/1 ...`

5. **并行写共享聚合量的风险**
   - OpenMP `#pragma omp for` 循环中存在对共享聚合数组的 `+=` 写入（例如 `QLakeSurf/QLakeSub/QLakeRivIn` 等）时，若未做原子/分块归约，会引入非确定性甚至错误结果
   - `PassValue()` 能覆盖一部分（segment→river 的聚合），但无法覆盖所有共享累加（尤其是 lake 相关聚合量）

> 结论：**OpenMP 路径目前不作为 Baseline**。如需对齐与验证，请使用 Serial build（无 `_OPENMP_ON`）及 `validation/baseline` 工作流。

---

## 4. 兼容模式定义：`LEGACY_COMPAT` vs `PHYSICS_FIX`

这两个模式用于后续多后端（尤其 GPU）开发时区分“严格复现”与“物理修复”。目前它们主要作为**语义约定/规格**：

- **现状**：代码中尚未实现对应开关（既没有编译时宏，也没有运行时 flag）；文档中仅用于描述期望行为。
- **约定（建议的落地方式）**：未来实现时建议使用**编译时预处理宏**（例如编译参数 `-DLEGACY_COMPAT=1` / `-DPHYSICS_FIX=1`），而不是运行时 flag；两者应互斥，缺省行为应与 `LEGACY_COMPAT` 一致（避免破坏既有回归/golden）。

### 4.1 `LEGACY_COMPAT`（严格复现 Baseline）

**目标**：让非 Serial 后端（GPU/OpenMP 等）在数值上尽可能复现 Serial Baseline 的行为，便于回归与 golden 对比。

语义要求（原则）：

- 计算流程、分支条件与装配公式与 Serial Baseline 一致（见第 2 节）
- 保持 Serial 的“已知小瑕疵/约定”以便做逐项对齐（例如湖泊蒸发/降水的汇总方式、若干数组的装配顺序等）
- 优先追求可复现：固定归约顺序或使用确定性归约策略

### 4.2 `PHYSICS_FIX`（允许改变结果的物理修复）

**目标**：修复明确的物理/数值问题，哪怕会改变回归结果；需要明确标注“会破坏 legacy baseline，对应需要重新生成 golden/重新回归/可能需重新标定”。

语义要求（原则）：

- 每一个“修复项”必须被文档化：修复动机、改动点、影响范围、如何验证
- 允许输出与 legacy baseline 不同，但应提供新的对照基准（新 golden 或新指标）

典型（示例）修复方向：

- 纠正湖泊通量的尺度/加权方式（例如按面积加权而非简单 `NumEleLake` 平均）
- 冻融因子 `fu_Sub/fu_Surf` 在 lake exchange 等路径上的一致性
- 统一 River `DY` 的装配公式（`dA/Length` + `fun_dAtodY` + 截断）以避免后端间物理含义不一致
- 强化并行路径的原子/归约策略，避免共享数组竞态（在保证语义一致的前提下）

---

## 5. 与验证工作流的关系

- CPU-Serial RHS baseline 回归：`validation/baseline/README.md`
- 任何新后端（GPU/OpenMP）的验收建议：
  1. `LEGACY_COMPAT`：先做到与 Serial golden 在给定容差内一致
  2. `PHYSICS_FIX`：再逐项引入修复，并为每项修复建立新的对照或指标
