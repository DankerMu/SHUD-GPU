#!/usr/bin/env python3
"""
GPU 加速盈亏平衡点分析

基于 benchmark 数据，理论估算多大问题规模下 GPU 才能体现加速效果。

Usage:
    python3 post_analysis/gpu_breakeven_analysis.py
    python3 post_analysis/gpu_breakeven_analysis.py --report output/benchmark_logs/report_ccw.md
"""

import argparse
import re
from pathlib import Path


def _split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _extract_md_table(text: str, heading: str) -> list[dict[str, str]]:
    """
    提取形如:

    ## <heading>
    | ... |
    |---|...|
    | ... |

    的 Markdown 表格，返回每行的 dict。
    """
    # 定位 section
    section_re = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = section_re.search(text)
    if not match:
        return []

    lines = text[match.end() :].splitlines()

    # 收集紧随其后的表格行（以 '|' 开头），遇到空行或非表格行停止
    table_lines: list[str] = []
    started = False
    for raw in lines:
        line = raw.strip()
        if not started:
            if not line:
                continue
            if line.startswith("|"):
                started = True
                table_lines.append(line)
            continue

        if not line or not line.startswith("|"):
            break
        table_lines.append(line)

    if len(table_lines) < 2:
        return []

    headers = _split_md_row(table_lines[0])
    rows: list[dict[str, str]] = []
    for row_line in table_lines[2:]:
        cells = _split_md_row(row_line)
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _parse_float(value: str) -> float | None:
    v = value.strip()
    if not v or v.upper() == "N/A":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    v = value.strip()
    if not v or v.upper() == "N/A":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def parse_report(report_path: Path) -> dict:
    """解析 benchmark 报告获取数据"""
    text = report_path.read_text(encoding="utf-8", errors="replace")

    data = {}

    # 解析 Wall time 表格（必须限定在该 section 内，避免匹配到其他表格）
    for row in _extract_md_table(text, "Wall time"):
        backend_raw = row.get("Backend", "")
        backend = backend_raw.lower().replace(" ", "_")
        time_s = _parse_float(row.get("Time(s)", ""))
        speedup = _parse_float(row.get("Speedup", ""))
        if time_s is not None:
            data[f"{backend}_time"] = time_s
        if speedup is not None:
            data[f"{backend}_speedup"] = speedup

    # 解析 CVODE stats 表格
    for row in _extract_md_table(text, "CVODE stats"):
        backend_raw = row.get("Backend", "")
        backend = backend_raw.lower().replace(" ", "_")
        for key in ("nfe", "nli", "nni", "netf", "npe", "nps"):
            parsed = _parse_int(row.get(key, ""))
            if parsed is not None:
                data[f"{backend}_{key}"] = parsed

    return data


def analyze_breakeven(data: dict, num_ele: int = 1147, num_riv: int = 567):
    """分析 GPU 盈亏平衡点"""

    print("=" * 60)
    print("GPU 加速盈亏平衡点分析")
    print("=" * 60)
    print()

    # 问题规模
    ny = 3 * num_ele + num_riv
    print(f"【当前问题规模 (ccw)】")
    print(f"  NumEle (元素数): {num_ele}")
    print(f"  NumRiv (河段数): {num_riv}")
    print(f"  NY (状态变量数): {ny}")
    print()

    # 提取数据
    defaults = {
        "cpu_serial_time": 866.12,
        "cuda_time": 1190.43,
        "cpu_serial_nfe": 264417,
        "cuda_nfe": 263782,
        "cuda_npe": 13261,
        "cuda_nps": 740907,
        "cuda_nli": 477142,
        "cpu_serial_nli": 437466,
    }
    missing: list[str] = []

    def _get(key: str):
        value = data.get(key)
        if value is None:
            missing.append(key)
            return defaults[key]
        return value

    cpu_time = float(_get("cpu_serial_time"))
    cuda_time = float(_get("cuda_time"))
    cpu_nfe = int(_get("cpu_serial_nfe"))
    cuda_nfe = int(_get("cuda_nfe"))
    cuda_npe = int(_get("cuda_npe"))
    cuda_nps = int(_get("cuda_nps"))
    cuda_nli = int(_get("cuda_nli"))
    cpu_nli = int(_get("cpu_serial_nli"))

    if missing:
        print("【警告】报告字段缺失，使用默认值进行估算 (可能不准确):")
        print("  " + ", ".join(missing))
        print()

    print(f"【Benchmark 数据】")
    print(f"  CPU 总时间: {cpu_time:.2f}s")
    print(f"  CUDA 总时间: {cuda_time:.2f}s")
    print(f"  CUDA/CPU 比值: {cuda_time/cpu_time:.2f}x (>1 表示 GPU 更慢)")
    print()

    # 每次调用时间
    cpu_per_rhs = cpu_time / cpu_nfe * 1000  # ms
    cuda_per_rhs = cuda_time / cuda_nfe * 1000  # ms

    print(f"【每次 RHS 调用时间】")
    print(f"  CPU: {cpu_per_rhs:.3f} ms/次")
    print(f"  CUDA: {cuda_per_rhs:.3f} ms/次 (含预处理)")
    print()

    # 预处理开销分析
    print(f"【CUDA 预处理开销】")
    print(f"  npe (PSetup 调用): {cuda_npe}")
    print(f"  nps (PSolve 调用): {cuda_nps}")
    print(f"  nli (Krylov 迭代): {cuda_nli} (CPU: {cpu_nli})")
    print(f"  Krylov 迭代增加: {(cuda_nli/cpu_nli - 1)*100:.1f}%")
    print()

    # GPU 时间分解模型
    # T_gpu = T_fixed + T_rhs_compute + T_precond
    #
    # 假设:
    # - T_fixed: kernel 启动、数据同步等固定开销
    # - T_rhs_compute: 与问题规模成正比的 RHS 计算
    # - T_precond: 预处理器开销

    # 估算预处理开销 (假设每次 PSolve ~0.05-0.2ms)
    psolve_time_per_call = 0.1  # ms, 估算值
    precond_overhead = cuda_nps * psolve_time_per_call / 1000  # seconds

    print(f"【GPU 时间分解估算】")
    print(f"  预处理开销 (估算): {precond_overhead:.1f}s ({precond_overhead/cuda_time*100:.1f}%)")
    print(f"  剩余时间 (RHS + 固定开销): {cuda_time - precond_overhead:.1f}s")
    print()

    # 理论模型
    # T_cpu(N) = α * N
    # T_gpu(N) = T_fixed + β * N
    #
    # GPU 加速条件: T_gpu < T_cpu
    # T_fixed + β * N < α * N
    # N > T_fixed / (α - β)

    alpha = cpu_time / num_ele  # CPU 每元素时间

    print(f"【理论模型参数】")
    print(f"  α (CPU 每元素时间): {alpha:.4f} s/元素")
    print()

    # 说明：只有一个规模的数据点，无法同时拟合 T_fixed 和 β。
    # 这里用“固定开销占 CUDA 总时间的比例 f”为参数做敏感性分析：
    #   T_fixed = f * T_cuda(current)
    #   β = (T_cuda(current) - T_fixed) / N_current
    # 再计算盈亏平衡点 N_be。

    diff = cuda_time - cpu_time
    min_fixed_ratio = diff / cuda_time if cuda_time > 0 else 0.0

    print(f"【盈亏平衡点估算】")
    print()
    print("  模型假设:")
    print("    - CPU/CUDA 的可扩展部分随问题规模近似线性增长")
    print("    - CUDA 额外固定开销 T_fixed 与规模无关 (或远弱相关)")
    print()
    print(f"  必要条件 (GPU 最终可能跑赢 CPU):")
    print(f"    - T_fixed > T_cuda - T_cpu = {diff:.2f}s")
    print(f"    - 等价于固定开销比例 f > {min_fixed_ratio*100:.1f}%")
    print()

    scenarios = [
        ("轻微固定开销", 0.30),
        ("较常见固定开销", 0.33),
        ("偏高固定开销", 0.35),
        ("很高固定开销", 0.40),
        ("极高固定开销", 0.50),
    ]

    for name, fixed_ratio in scenarios:
        if fixed_ratio <= min_fixed_ratio:
            print(f"  {name} (f={fixed_ratio*100:.0f}%): 固定开销低于下限，GPU 难以靠放大规模跑赢 CPU")
            continue

        T_fixed = cuda_time * fixed_ratio
        T_var_gpu_current = cuda_time - T_fixed
        beta = T_var_gpu_current / num_ele

        denom = alpha - beta
        if denom <= 0:
            print(f"  {name} (f={fixed_ratio*100:.0f}%): β>=α，无法实现加速")
            continue

        N_breakeven = T_fixed / denom
        scale_factor = N_breakeven / num_ele

        # 假设 NumRiv 与 NumEle 按比例放大
        num_ele_be = int(round(num_ele * scale_factor))
        num_riv_be = int(round(num_riv * scale_factor))
        ny_be = 3 * num_ele_be + num_riv_be

        print(f"  {name} (f={fixed_ratio*100:.0f}%):")
        print(f"    - T_fixed≈{T_fixed:.1f}s, β≈{beta:.4f} s/元素")
        print(f"    - 盈亏平衡点: NumEle≈{num_ele_be} (×{scale_factor:.1f}), NumRiv≈{num_riv_be}, NY≈{ny_be}")
        print()

    # 实际建议
    print(f"【结论与建议】")
    print()
    print(f"  当前状态:")
    print(f"    - ccw 规模 (NumEle={num_ele}) 下 GPU 比 CPU 慢 {(cuda_time/cpu_time-1)*100:.0f}%")
    print(f"    - 预处理器开销显著 (nps={cuda_nps})")
    print()
    print(f"  GPU 加速预计需要 (基于上面的线性+固定开销模型):")
    print(f"    - 若固定开销约 33%~35%: NumEle ≈ 5,000~7,000 (约 4~6x 当前规模)")
    print(f"    - 若固定开销更低 (~30%): NumEle 可能需要 ≈ 12,000 (约 11x 当前规模)")
    print()
    print(f"  优化方向:")
    print(f"    1. 减少 kernel 启动次数 (kernel fusion)")
    print(f"    2. 优化预处理器 (Block-Jacobi 可能不适合小规模)")
    print(f"    3. 使用更大规模测试用例验证")
    print(f"    4. 考虑禁用预处理器 (--no-precond) 对比")


def main():
    parser = argparse.ArgumentParser(description="GPU 加速盈亏平衡点分析")
    parser.add_argument(
        "--report",
        default="output/benchmark_logs/report_ccw.md",
        help="Benchmark 报告路径"
    )
    parser.add_argument("--num-ele", type=int, default=1147, help="元素数量")
    parser.add_argument("--num-riv", type=int, default=567, help="河段数量")
    args = parser.parse_args()

    report_path = Path(args.report)
    if report_path.exists():
        data = parse_report(report_path)
    else:
        print(f"警告: 报告文件不存在 ({report_path})，使用默认数据")
        data = {}

    analyze_breakeven(data, args.num_ele, args.num_riv)


if __name__ == "__main__":
    main()
