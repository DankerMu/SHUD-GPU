# CPU-Serial RHS baseline (ccw)

此目录提供 **CPU Serial**（无 OpenMP）路径的 RHS 回归基准，用于验证 GPU 实现的正确性。

## Golden 数据

- 位置：`validation/baseline/ccw/`
  - `golden.npz`：numpy 二进制数据（包含 `time_min`、`y`、关键通量数组、CVODE 统计数组）
  - `metadata.json`：JSON 元数据（case/config/git/environment/hash 等）

## 一键回归测试

从仓库根目录执行：

```bash
bash validation/baseline/run_baseline_test.sh
```

脚本会：
1. `make clean && make shud`（确保 Serial build）
2. 跑 Python 单测并强制 coverage ≥ 90%
3. 运行 `ccw` 并与 golden 数据对比（容差 `1e-12`）

## 生成/更新 golden

确保仓库根目录已有 `./shud` 可执行文件（Serial build）。

```bash
python3 validation/baseline/generate_golden.py
```

默认设置：
- case：`ccw`
- `END=2` 天
- 输出间隔 `DT_* = 60` 分钟
- `TERRAIN_RADIATION=0`
- 自动重复运行 2 次并校验差异 ≤ `1e-12`（确保确定性）

自定义示例：

```bash
python3 validation/baseline/generate_golden.py --end-days 1 --dt-min 30 --terrain-radiation 1
```

## 对比已有输出目录

如果你已经手动运行了模型并得到输出目录：

```bash
python3 validation/baseline/compare_baseline.py --use-output-dir output/ccw.base
```

