#!/usr/bin/env python3
"""
辐射对比图：水平辐射 vs 地形修正辐射 (不同坡向单元)

用法: python plot_radiation_comparison.py [project_name]
默认: ccw
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 添加 shud_reader 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "validation" / "tsr" / "py"))
from shud_reader import read_dat

# 配置
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "ccw"
INPUT_DIR = Path(f"input/{PROJECT}")
TSR_OUTPUT_DIR = Path(f"output/{PROJECT}_tsr")
OUTPUT_DIR = Path("post_analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模拟起始日期 (从 forcing 文件获取)
START_DATE = datetime(2000, 1, 1)  # 默认值，实际应从配置读取


def read_mesh(mesh_file: Path):
    """读取 mesh 文件，返回单元和节点数据"""
    with open(mesh_file, 'r') as f:
        lines = f.readlines()

    n_elements, _ = map(int, lines[0].split())

    elements = []
    for i in range(2, 2 + n_elements):
        parts = lines[i].split()
        elem_id = int(parts[0])
        node1, node2, node3 = int(parts[1]), int(parts[2]), int(parts[3])
        elements.append((elem_id, node1, node2, node3))

    node_header_line = 2 + n_elements
    n_nodes, _ = map(int, lines[node_header_line].split())

    nodes = {}
    for i in range(node_header_line + 2, node_header_line + 2 + n_nodes):
        parts = lines[i].split()
        node_id = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[4])
        nodes[node_id] = (x, y, z)

    return elements, nodes


def compute_slope_aspect(elements, nodes):
    """计算每个单元的坡度和坡向"""
    elem_terrain = {}

    for elem_id, n1, n2, n3 in elements:
        p1 = np.array(nodes[n1])
        p2 = np.array(nodes[n2])
        p3 = np.array(nodes[n3])

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        if normal[2] < 0:
            normal = -normal

        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len

        slope_deg = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))

        if slope_deg < 0.1:
            aspect = -1
        else:
            aspect_rad = np.arctan2(normal[0], normal[1])
            aspect = np.degrees(aspect_rad)
            if aspect < 0:
                aspect += 360

        elem_terrain[elem_id] = {'slope': slope_deg, 'aspect': aspect}

    return elem_terrain


def aspect_to_direction(aspect):
    """将坡向角度转换为方向名称"""
    if aspect < 0:
        return "Flat"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((aspect + 22.5) % 360 / 45)
    return dirs[idx]


def find_representative_elements(elem_terrain, target_slope_min=25):
    """为每个坡向找一个代表性单元（坡度较大的）"""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    aspect_ranges = {
        "N": (337.5, 22.5),
        "NE": (22.5, 67.5),
        "E": (67.5, 112.5),
        "SE": (112.5, 157.5),
        "S": (157.5, 202.5),
        "SW": (202.5, 247.5),
        "W": (247.5, 292.5),
        "NW": (292.5, 337.5)
    }

    representatives = {}

    for direction in directions:
        lo, hi = aspect_ranges[direction]
        candidates = []

        for elem_id, terrain in elem_terrain.items():
            slope = terrain['slope']
            aspect = terrain['aspect']

            if slope < target_slope_min:
                continue

            if direction == "N":
                if aspect >= lo or aspect < hi:
                    candidates.append((elem_id, slope, aspect))
            else:
                if lo <= aspect < hi:
                    candidates.append((elem_id, slope, aspect))

        if candidates:
            # 选择坡度最大的
            candidates.sort(key=lambda x: x[1], reverse=True)
            elem_id, slope, aspect = candidates[0]
            representatives[direction] = {
                'elem_id': elem_id,
                'slope': slope,
                'aspect': aspect
            }

    return representatives


def plot_radiation_comparison(project, representatives, rn_h_df, rn_t_df, n_days=30):
    """绘制辐射对比图"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle("Net Radiation: Horizontal (Baseline) vs Terrain (TSR)", fontsize=14, fontweight='bold')

    # 时间轴 (取前 n_days 天)
    times_min = rn_h_df.index.values[:n_days]
    dates = [START_DATE + timedelta(minutes=float(t)) for t in times_min]

    # 按方向分组: 左列北向(270-360, 0-90), 右列南向(90-270)
    north_facing = ["NW", "N", "NE", "E"]
    south_facing = ["SE", "S", "SW", "W"]

    def plot_element(ax, direction, col_side):
        if direction not in representatives:
            ax.text(0.5, 0.5, f"No {direction}-facing element found",
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{direction}-facing")
            return

        info = representatives[direction]
        elem_id = info['elem_id']
        slope = info['slope']
        aspect = info['aspect']

        col_name = f"X{elem_id}"
        if col_name not in rn_h_df.columns:
            ax.text(0.5, 0.5, f"Column {col_name} not found",
                    transform=ax.transAxes, ha='center', va='center')
            return

        rn_h = rn_h_df[col_name].values[:n_days]
        rn_t = rn_t_df[col_name].values[:n_days]

        ax.plot(dates, rn_h, 'b-', linewidth=1.5, label='Baseline (Horizontal)')
        ax.plot(dates, rn_t, 'r-', linewidth=1.5, label='TSR (Terrain)')

        ax.set_title(f"Element {elem_id} - {direction}-facing\nSlope: {slope:.1f}°, Aspect: {aspect:.1f}°",
                     fontsize=10)
        ax.set_ylabel("Net Radiation [W/m²]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))

        if col_side == 0:
            ax.legend(loc='upper right', fontsize=8)

    # 绘制北向单元 (左列)
    for i, direction in enumerate(north_facing):
        plot_element(axes[i, 0], direction, 0)

    # 绘制南向单元 (右列)
    for i, direction in enumerate(south_facing):
        plot_element(axes[i, 1], direction, 1)

    # 添加列标题
    axes[0, 0].annotate("North-Facing (270-360°, 0-90°)", xy=(0.5, 1.15),
                        xycoords='axes fraction', fontsize=11, ha='center', fontweight='bold')
    axes[0, 1].annotate("South-Facing (90-270°)", xy=(0.5, 1.15),
                        xycoords='axes fraction', fontsize=11, ha='center', fontweight='bold')

    # 底部标签
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 旋转日期标签
    for ax_row in axes:
        for ax in ax_row:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    output_file = OUTPUT_DIR / f"{project}_radiation_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def main():
    global START_DATE

    print(f"处理项目: {PROJECT}")

    # 检查 TSR 输出
    rn_h_file = TSR_OUTPUT_DIR / f"{PROJECT}.rn_h.dat"
    rn_t_file = TSR_OUTPUT_DIR / f"{PROJECT}.rn_t.dat"

    if not rn_h_file.exists() or not rn_t_file.exists():
        print(f"错误: 找不到 TSR 输出文件")
        print(f"  需要: {rn_h_file}")
        print(f"  需要: {rn_t_file}")
        print("请先运行: ./shud -o output/{PROJECT}_tsr {PROJECT} (启用 TERRAIN_RADIATION=1)")
        sys.exit(1)

    # 读取 mesh 并计算地形
    mesh_file = INPUT_DIR / f"{PROJECT}.sp.mesh"
    print("读取 mesh 文件...")
    elements, nodes = read_mesh(mesh_file)
    print(f"  单元数: {len(elements)}")

    print("计算坡度和坡向...")
    elem_terrain = compute_slope_aspect(elements, nodes)

    print("选择代表性单元...")
    representatives = find_representative_elements(elem_terrain, target_slope_min=25)
    for direction, info in representatives.items():
        print(f"  {direction}: Element {info['elem_id']} (slope={info['slope']:.1f}°, aspect={info['aspect']:.1f}°)")

    # 读取辐射数据
    print("读取辐射数据...")
    rn_h_df = read_dat(rn_h_file, time_mode='index')
    rn_t_df = read_dat(rn_t_file, time_mode='index')
    print(f"  时间步数: {len(rn_h_df)}")
    print(f"  单元数: {len(rn_h_df.columns)}")

    # 尝试获取起始日期
    try:
        forc_file = INPUT_DIR / f"{PROJECT}.tsd.forc"
        with open(forc_file, 'r') as f:
            for line in f:
                if 'ForcStartTime' in line or line.strip().startswith('2'):
                    parts = line.split()
                    for part in parts:
                        if len(part) == 8 and part.isdigit():
                            START_DATE = datetime.strptime(part, '%Y%m%d')
                            print(f"  起始日期: {START_DATE.strftime('%Y-%m-%d')}")
                            break
                    break
    except Exception:
        print(f"  使用默认起始日期: {START_DATE.strftime('%Y-%m-%d')}")

    print("绘制辐射对比图...")
    plot_radiation_comparison(PROJECT, representatives, rn_h_df, rn_t_df, n_days=30)

    print("完成!")


if __name__ == "__main__":
    main()
