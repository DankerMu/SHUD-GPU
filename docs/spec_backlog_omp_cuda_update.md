# spec_backlog.md — SHUD-GPU 后续优化 Backlog（面向 ccw / Tesla T4 / double）

> 目标：把「可验证的精度收敛」与「可解释的性能提升」拆成一组可交付、可验收的工作项（issues/PR）。
>
> 本 backlog 以 **ccw**（NumEle=1147, NumRiv=103, NY≈3544, NumLake=0）为最小可复现实例；GPU 为 **Tesla T4**，SUNDIALS `realtype=double`。

---

## 0. 当前基线（Baseline）

### 0.1 精度基线（post-fix）
- CPU vs OMP：**24/24 PASS（rel_max=0）**
- CPU vs CUDA：**7 PASS / 17 FAIL**
  - 最差：`ccw.eleqsurf.dat rel_max=4.287`
  - `ySf`：`ccw.eleysurf.dat rel_max=2.421`
  - `yUs`：`ccw.eleyunsat.dat rel_max=1.098e-01`
  - `yRiv`：`ccw.rivystage.dat rel_max=4.532e-02`

### 0.2 性能基线（端到端 wall time, ccw）
- CPU：**862s**
- OMP：**798s（~1.08×）**
- CUDA：**1175s（比 CPU 慢）**

### 0.3 CUDA 精度偏离的最早定位（GPU verify）
- 首次 mismatch：`k_ele_local` 阶段的 `qEu`
- 时间：约 `t≈1079 day`（窗口内第一次出现）
- 形态：`~994/1147` 元素超阈值，`max_abs~1e-7` 级别（CPU 为 0）

---

## 1. 总体目标与验收口径（Definition of Done）

> 说明：ccw 规模非常小，且在 T4 上 double 性能不占优。**因此：**
> - ccw 上 CUDA 的“性能目标”以 **降低开销/不比 CPU 差太多** 为主；
> - CUDA 的“精度目标”优先以 **定位并消除可疑的语义不一致/未覆盖写** 为主；
> - 真正的 GPU 性能优势需要在 **更大规模（NY ≫ 1e5）** 才有机会体现。

### 1.1 精度目标分级（建议）
- **L0：可解释一致性（Debug 级）**
  - GPU verify：首个 mismatch 被定位到“具体字段 + 具体 kernel + 具体分支”，并能用测试用例稳定复现
- **L1：工程一致性（Regression 级）**
  - CPU vs CUDA：关键状态量（ySf/yUs/yGw/yRiv）达到 `rel_max <= 1e-3`（ccw 全时段）
  - 通量/诊断输出“非零且随时间变化”，且不再出现“明显未同步/全零”模式
- **L2：数值一致性（严格级，可选）**
  - CPU vs CUDA：`rel_max <= 1e-6`（或按文件类别分别设阈值）
  - 需要明确接受：可能牺牲性能（确定性归约 / 关闭 FMA / 更严格步长）

### 1.2 性能目标分级（建议）
- **P0：ccw 上不再“明显更慢”**
  - CUDA wall time ≤ CPU * 1.1（允许 10% 回归空间）
- **P1：中等规模（NY~1e5）**
  - CUDA ≥ CPU * 2（端到端） 或 RHS ≥ CPU * 5（RHS-only）
- **P2：大规模（NY≥1e6）**
  - CUDA ≥ CPU * 5（端到端） 或 RHS ≥ CPU * 10（RHS-only）

> 注：性能目标必须同时报告 CVODE 统计（nfe/nli/nni 等）以避免“快是因为求解更粗”。

---

## 2. Backlog 结构说明

- **优先级**
  - **P0**：阻塞精度/性能评估或明显错误/回归
  - **P1**：能显著提升性能或稳定性（中等投入）
  - **P2**：长期收益（较大改动/需要更多实验）
  - **P3**：研究/探索项

- **工作项格式**
  - 背景/问题
  - 交付物（Deliverables）
  - 验收标准（Acceptance）
  - 风险/依赖（Risks/Deps）

---

## 3. P0 Backlog（必须先做）

### ACC-CUDA-001 — 修复/解释 `k_ele_local:qEu` 首次 mismatch（全覆盖写 + 分支对齐）
**背景/问题**
- GPU verify 显示最早偏离来自 `qEu`，CPU=0，GPU=O(1e-7)，并且是“大片元素出现小非零”。

**交付物**
1. 在 `k_ele_local` 内保证 `qEu` **每个元素每次 RHS 都会被显式写入**（即使为 0）。
2. 将 `qEu` 的分支逻辑与 CPU 逐行对齐（阈值/钳制/湿地条件/ET 开关等）。
3. 增加一个“局部单元测试/最小复现”：
   - 固定一个时刻的 forcing + 一个 element（或少量元素）输入，单步调用 ET，比较 CPU vs device 输出。

**验收**
- GPU verify：`qEu` 首次 mismatch 消失，或首个 mismatch 明确前移/变为其他字段且有解释。
- accuracy_report：与 `qEu` 强相关的文件（`elevetev/elevettr/eleveta/eleyunsat` 等）`rel_max` 显著下降。

**风险/依赖**
- 若 `qEu` 偏离是“阈值敏感 + CVODE 步长路径差异”放大导致，则仅修复覆盖写不够，需要进入 ACC-CUDA-004/005。

---

### ACC-CUDA-002 — CUDA 输出/诊断链路“闭环”与一致性策略（host-sync vs host-recompute）
**背景/问题**
- 诊断/通量输出在 CUDA 模式下容易出现“旧值/零值/未更新”的假象，影响精度评估与定位。

**交付物**
1. 设计并实现一套明确策略（两选一，或混合）：
   - **方案 A（device 生成 + D2H 同步）**：所有 `Print_Ctrl` 依赖数组在 device 更新，并在输出前同步到 host。
   - **方案 B（host 复算）**：输出时在 host 用已同步的状态量复算诊断（需保证复算代价可接受）。
2. 为每个输出变量标注“source of truth”（device/host）。

**验收**
- CUDA 输出文件不再出现明显的“常零/不变化”。
- 同一时刻 CPU/OMP/CUDA 输出结构一致（维度、列 id、时间步一致）。

---

### PERF-BENCH-001 — 统一 benchmark & profiling 入口（端到端 + RHS-only + CVODE stats）
**背景/问题**
- 目前 wall time 不能分辨瓶颈是 RHS、CVODE 迭代、还是 I/O/同步。

**交付物**
1. 新增 `scripts/bench/run_bench.sh`：
   - 支持 backend=cpu/omp/cuda
   - 输出：wall time、RHS time、I/O time（粗粒度即可）
2. 在每次 run 输出 CVODE 统计（nfe/nli/nni/netf/npe 等）到 `bench.log`。
3. 提供 `nvprof/nsys` 的可选入口（仅 CUDA）。

**验收**
- 同一机器同一输入，重复 3 次 benchmark 的波动 < 5%。
- PR 合并前自动生成一份 summary（markdown）。

---

### PERF-CUDA-001 — 为小规模问题添加“后端自适应选择/门槛”（ccw 默认不走 GPU）
**背景/问题**
- ccw 的 NY≈3544，T4 + double 下 GPU 常被 kernel launch / 归约开销支配，反而更慢。

**交付物**
1. 增加运行时门槛（可配置）：
   - 例如：`SHUD_BACKEND_AUTO=1` 时，若 `NY < NY_GPU_MIN` 则强制用 OMP/CPU。
2. 输出日志明确提示选择原因。

**验收**
- ccw 默认运行不再落入 CUDA 慢路径；用户显式指定 `--backend cuda` 时仍可强制使用。

---

## 4. P1 Backlog（中期收益显著）

### PERF-OMP-001 — OMP 性能：减少并行开销 + 绑核建议 + schedule 优化
**背景/问题**
- ccw 中 element/riv 循环很短，OMP 调度与 barrier 开销占比高。

**交付物**
1. 在 OMP 关键循环使用 `schedule(static)`，避免动态调度开销。
2. 增加运行时提示：推荐 `OMP_PROC_BIND/OMP_PLACES`。
3. 将 NVECTOR_OPENMP 线程数与 `-n` 参数联动（避免只并行 RHS，不并行向量归约）。

**验收**
- 在 ccw 上 OMP 至少达到 CPU * 1.15（或证明主要瓶颈在 I/O，给出证据）。

---

### PERF-CUDA-002 — 减少 RHS kernel launch 次数（kernel fusion / CUDA Graph）
**背景/问题**
- 小规模问题 launch overhead 极高；CVODE 调用 RHS 频繁，开销累计巨大。

**交付物**
1. 合并/融合明显可融合的 kernel（例如清零+sanitize+部分局部计算）。
2. 评估并引入 **CUDA Graph capture**（对固定 kernel pipeline 特别有效）。
3. 在 benchmark 中报告：每次 RHS 的平均 kernel 数、平均 launch 时间。

**验收**
- ccw 上 CUDA wall time 至少改善 20%（相对当前 CUDA 基线）。
- RHS-only time 显著下降，且结果不回归。

---

### ACC-CUDA-003 — 原子累加热点治理（两段式归约替代 atomicAdd）
**背景/问题**
- river/segment 汇总中的 atomicAdd 引入：
  - 性能上的争用与序列化
  - 精度上的非确定求和顺序

**交付物**
1. 构建 CSR/offset 索引（river→segments 映射），实现两段式归约：
   - kernel1：写每段结果
   - kernel2：按固定顺序归约到 river/element
2. 可选：Kahan/pairwise summation（用于严格精度模式）。

**验收**
- 精度：相关通量（`rivq*`、`eleq*`）差异下降；
- 性能：atomic hotspot 的 kernel 时间下降（profile 佐证）。

---

### ACC-CUDA-004 — CVODE 路径差异控制：预条件器开关、步长限制、统计对比
**背景/问题**
- CUDA 默认预条件器（PREC_LEFT）与 CPU/OMP 不同，可能改变 GMRES/Newton 路径并放大阈值敏感性。

**交付物**
1. 让预条件器变成“可配置 + 可记录”：
   - on/off/auto（基于 NY 或基于历史统计）
2. 提供两组对比报告：
   - precond ON vs OFF 的 CVODE 统计 + 精度差异
3. 若 precond 质量不足，记录改进方向（例如雅可比近似对齐 RHS）。

**验收**
- 给出结论：对 ccw（NY≈3544）是否应默认关闭 precond；
- 对中等规模 case（NY~1e5）是否应默认开启。

---

## 5. P2 Backlog（大改动/长期项）

### ACC-CUDA-005 — “严格一致性模式”（deterministic / strict-fp）
**背景/问题**
- 若需要 `rel_max<=1e-6` 级别一致性，需要牺牲一定性能换确定性。

**交付物**
1. 增加 build/runtime 模式：
   - `SHUD_STRICT_FP=1`：禁用 FMA、使用更精确 sqrt/div
   - `SHUD_DETERMINISTIC_REDUCE=1`：关键归约使用固定顺序
2. 文档说明：严格模式的性能代价、适用场景。

**验收**
- 在 ccw 上 CPU vs CUDA 的关键状态量达到 L2 级目标（或在文档中证明不现实并给出替代指标）。

---

### PERF-IO-001 — 输出 I/O 性能优化（减少端到端时间噪声）
**背景/问题**
- ccw 输出矩阵 1827x1147，多文件写盘会显著影响端到端时间，掩盖计算优化。

**交付物**
1. 统一输出缓冲（batch write）或可选减少输出变量集合。
2. 增加参数：输出间隔/输出开关分组（state / flux / diag）。
3. benchmark 增加 “I/O off” 模式用于纯计算对比。

**验收**
- 在 I/O-heavy 配置下端到端时间下降；
- I/O off 模式下能更清晰看出 RHS/solver 的真实加速比。

---

### PERF-CUDA-003 — forcing/参数更新的 device 常驻与异步流水
**背景/问题**
- forcing 更新与状态同步可能引入 CPU↔GPU 同步点，阻碍 overlap。

**交付物**
1. forcing 数据常驻 device（或 pinned host + async memcpy）。
2. RHS 使用 stream + event，避免不必要 `cudaDeviceSynchronize()`。

**验收**
- profile 显示同步点减少；
- wall time 改善并可重复。

---

## 6. P3 Backlog（探索/研究）

### SCALE-001 — 更大规模算例/合成算例生成器（NY 扩展到 1e5/1e6）
**动机**
- ccw 太小，无法代表 GPU 真正优势；需要标准化大规模验证集。

**交付物**
1. 合成网格/参数生成脚本（保持物理合理的同时可扩展规模）。
2. 提供至少 3 个规模档：1e4、1e5、1e6（NY）。

---

### PERF-MIXED-001 — 混合精度探索（precond/部分通量 float）
**动机**
- T4 的 double 性能弱，混合精度可能带来数量级性能改善。

**交付物**
1. 仅在 precond 或某些中间数组使用 float，最终状态仍 double。
2. 误差评估报告（与 L1/L2 指标对齐）。

---

## 7. 里程碑建议（Milestones）

- **M0（稳定可测）**：ACC-CUDA-001/002 + PERF-BENCH-001 + PERF-CUDA-001
- **M1（可改进）**：PERF-CUDA-002 + ACC-CUDA-003 + ACC-CUDA-004 + PERF-OMP-001
- **M2（可选严格一致）**：ACC-CUDA-005 + PERF-IO-001
- **M3（规模与研究）**：SCALE-001 + PERF-MIXED-001

---

## 8. 附：运行/验证建议（命令范例）

### 8.1 精度对比
```bash
make clean && make shud && make shud_omp && make shud_cuda
./shud ccw
./shud_omp -n 16 ccw
./shud_cuda --backend cuda ccw

python3 post_analysis/accuracy_comparison.py \
  output/ccw_cpu output/ccw_omp output/ccw_cuda \
  --tol_omp 1e-10 --tol_cuda 1e-6
```

### 8.2 GPU verify（缩小时间窗）
```bash
make shud_cuda DEBUG_GPU_VERIFY=1
export SHUD_GPU_VERIFY=1
export SHUD_GPU_VERIFY_T_MIN_DAY=1079
export SHUD_GPU_VERIFY_T_MAX_DAY=1083
export SHUD_GPU_VERIFY_INTERVAL=1
export SHUD_GPU_VERIFY_STOP_ON_MISMATCH=1
./shud_cuda --backend cuda ccw
```

---
