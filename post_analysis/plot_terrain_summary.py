#!/usr/bin/env python3
"""
地形总结图：坡度分布、坡向分布、坡向玫瑰图

用法: python plot_terrain_summary.py [project_name]
默认: ccw
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, ListedColormap
import numpy as np

# 配置
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "ccw"
INPUT_DIR = Path(f"input/{PROJECT}")
OUTPUT_DIR = Path("post_analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_mesh(mesh_file: Path):
    """读取 mesh 文件，返回单元和节点数据"""
    with open(mesh_file, 'r') as f:
        lines = f.readlines()

    # 第一行：单元数量和列数
    n_elements, n_cols = map(int, lines[0].split())

    # 单元数据 (跳过表头)
    elements = []
    for i in range(2, 2 + n_elements):
        parts = lines[i].split()
        elem_id = int(parts[0])
        node1, node2, node3 = int(parts[1]), int(parts[2]), int(parts[3])
        zmax = float(parts[7])
        elements.append((elem_id, node1, node2, node3, zmax))

    # 节点头部位置
    node_header_line = 2 + n_elements
    n_nodes, _ = map(int, lines[node_header_line].split())

    # 节点数据 (跳过表头)
    nodes = {}
    for i in range(node_header_line + 2, node_header_line + 2 + n_nodes):
        parts = lines[i].split()
        node_id = int(parts[0])
        x, y = float(parts[1]), float(parts[2])
        z = float(parts[4])
        nodes[node_id] = (x, y, z)

    return elements, nodes


def compute_slope_aspect(elements, nodes):
    """计算每个单元的坡度和坡向"""
    slopes = []
    aspects = []
    centroids = []
    polygons = []

    for elem_id, n1, n2, n3, zmax in elements:
        # 获取三个顶点坐标
        p1 = np.array(nodes[n1])
        p2 = np.array(nodes[n2])
        p3 = np.array(nodes[n3])

        # 计算中心点
        centroid = (p1 + p2 + p3) / 3
        centroids.append(centroid[:2])

        # 多边形顶点 (用于绘图)
        polygons.append([p1[:2], p2[:2], p3[:2]])

        # 计算法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # 确保法向量朝上
        if normal[2] < 0:
            normal = -normal

        # 归一化
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len

        # 坡度 (度)
        slope_rad = np.arccos(np.clip(normal[2], -1, 1))
        slope_deg = np.degrees(slope_rad)
        slopes.append(slope_deg)

        # 坡向 (度，北为0，顺时针)
        if slope_deg < 0.1:  # 近乎平坦
            aspect = -1  # 无坡向
        else:
            # 投影到水平面
            aspect_rad = np.arctan2(normal[0], normal[1])  # E-W / N-S
            aspect = np.degrees(aspect_rad)
            if aspect < 0:
                aspect += 360
        aspects.append(aspect)

    return np.array(slopes), np.array(aspects), centroids, polygons


def aspect_to_direction(aspect):
    """将坡向角度转换为方向名称"""
    if aspect < 0:
        return "Flat"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((aspect + 22.5) % 360 / 45)
    return dirs[idx]


def create_aspect_colormap():
    """创建循环坡向色图"""
    # HSV 色环
    n = 256
    hsv = np.ones((n, 3))
    hsv[:, 0] = np.linspace(0, 1, n)  # Hue
    hsv[:, 1] = 0.8  # Saturation
    hsv[:, 2] = 0.9  # Value

    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    return ListedColormap(rgb)


def plot_terrain_summary(project, elements, nodes, slopes, aspects, polygons):
    """绘制地形总结图"""
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Terrain Summary: {project}", fontsize=16, fontweight='bold')

    # 转换坐标为 km
    polygons_km = [[(p[0]/1000, p[1]/1000) for p in poly] for poly in polygons]

    # ========== 1. 坡度分布图 ==========
    ax1 = fig.add_subplot(2, 2, 1)

    coll1 = PolyCollection(polygons_km, array=slopes, cmap='YlOrRd', edgecolors='face', linewidths=0.1)
    coll1.set_clim(0, 30)
    ax1.add_collection(coll1)
    ax1.autoscale()
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_title('Slope Distribution')
    cbar1 = plt.colorbar(coll1, ax=ax1, label='Slope Distribution [°]')

    # ========== 2. 坡向分布图 ==========
    ax2 = fig.add_subplot(2, 2, 2)

    # 使用循环色图
    aspect_cmap = create_aspect_colormap()
    valid_aspects = np.where(aspects >= 0, aspects, np.nan)

    coll2 = PolyCollection(polygons_km, array=valid_aspects, cmap=aspect_cmap,
                           edgecolors='face', linewidths=0.1)
    coll2.set_clim(0, 360)
    ax2.add_collection(coll2)
    ax2.autoscale()
    ax2.set_aspect('equal')
    ax2.set_xlabel('X [km]')
    ax2.set_ylabel('Y [km]')
    ax2.set_title('Aspect Distribution')

    # 坡向图例
    flat_count = np.sum(aspects < 0)
    sloped_count = np.sum(aspects >= 0)
    ax2.text(0.02, 0.98, f"Flat: {flat_count}\nSloped: {sloped_count}",
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 添加坡向色环图例
    ax_legend = fig.add_axes([0.91, 0.55, 0.07, 0.07], projection='polar')
    theta = np.linspace(0, 2*np.pi, 256)
    r = np.linspace(0.6, 1, 2)
    Theta, R = np.meshgrid(theta, r)
    colors = (theta / (2*np.pi) * 360)
    ax_legend.pcolormesh(Theta, R, np.tile(colors, (2, 1)), cmap=aspect_cmap, shading='auto')
    ax_legend.set_yticks([])
    ax_legend.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax_legend.set_xticklabels(['N\n(0°)', 'E\n(90°)', 'S\n(180°)', 'W\n(270°)'], fontsize=7)
    ax_legend.set_title('Aspect\n(angle)', fontsize=8)

    # ========== 3. 坡向玫瑰图 ==========
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')

    # 统计各方向单元数
    directions = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]  # 从南开始，顺时针
    angles_center = np.array([180, 225, 270, 315, 0, 45, 90, 135])  # 度

    counts = []
    for i, (lo, hi) in enumerate([(157.5, 202.5), (202.5, 247.5), (247.5, 292.5),
                                   (292.5, 337.5), (337.5, 360), (0, 22.5),
                                   (22.5, 67.5), (67.5, 112.5), (112.5, 157.5)]):
        if i == 4:  # N: 337.5-360 和 0-22.5
            c = np.sum((aspects >= 337.5) | ((aspects >= 0) & (aspects < 22.5)))
        elif i == 5:  # 已经在 N 中处理
            continue
        else:
            c = np.sum((aspects >= lo) & (aspects < hi))
        counts.append(c)

    # 重新计算（简化）
    dir_bins = [(-22.5, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5),
                (157.5, 202.5), (202.5, 247.5), (247.5, 292.5), (292.5, 337.5)]
    dir_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    counts = []
    for lo, hi in dir_bins:
        if lo < 0:
            c = np.sum(((aspects >= lo + 360) | (aspects < hi)) & (aspects >= 0))
        else:
            c = np.sum((aspects >= lo) & (aspects < hi))
        counts.append(c)

    # 玫瑰图 (从南开始，顺时针排列)
    theta_rose = np.deg2rad([180, 225, 270, 315, 0, 45, 90, 135])
    width = np.deg2rad(45)

    # 重排 counts 以匹配 theta_rose (S, SW, W, NW, N, NE, E, SE)
    rose_order = [4, 5, 6, 7, 0, 1, 2, 3]  # N, NE, E, SE, S, SW, W, NW -> S, SW, W, NW, N, NE, E, SE
    counts_rose = [counts[i] for i in [4, 5, 6, 7, 0, 1, 2, 3]]

    bars = ax3.bar(theta_rose, counts_rose, width=width, bottom=0, alpha=0.7, edgecolor='black')

    # 添加百分比标签
    total = sum(counts_rose)
    for bar, cnt, theta in zip(bars, counts_rose, theta_rose):
        pct = cnt / total * 100
        ax3.text(theta, bar.get_height() + 5, f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)

    ax3.set_theta_zero_location('S')
    ax3.set_theta_direction(-1)
    ax3.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                       ["S", "SW", "W", "NW", "N", "NE", "E", "SE"])
    ax3.set_title(f"Aspect Distribution (n={len(aspects)})", pad=20)
    ax3.set_ylabel("Number of Elements", labelpad=30)

    # ========== 4. 统计信息文本框 ==========
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = f"""Terrain Statistics
{'='*40}

Slope Statistics:
    Mean:   {np.mean(slopes):.2f}°
    Min:    {np.min(slopes):.2f}°
    Max:    {np.max(slopes):.2f}°
    Std:    {np.std(slopes):.2f}°

Aspect Distribution:
"""
    for i, name in enumerate(dir_names):
        stats_text += f"    {name:3s}:   {counts[i]:4d} ({counts[i]/total*100:5.1f}%)\n"

    stats_text += f"\nTotal Elements: {len(aspects)}"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title("Aspect Rose Diagram", fontsize=12, pad=20)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    output_file = OUTPUT_DIR / f"{project}_terrain_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def main():
    print(f"处理项目: {PROJECT}")

    mesh_file = INPUT_DIR / f"{PROJECT}.sp.mesh"
    if not mesh_file.exists():
        print(f"错误: 找不到 {mesh_file}")
        sys.exit(1)

    print("读取 mesh 文件...")
    elements, nodes = read_mesh(mesh_file)
    print(f"  单元数: {len(elements)}, 节点数: {len(nodes)}")

    print("计算坡度和坡向...")
    slopes, aspects, centroids, polygons = compute_slope_aspect(elements, nodes)

    print("绘制地形总结图...")
    plot_terrain_summary(PROJECT, elements, nodes, slopes, aspects, polygons)

    print("完成!")


if __name__ == "__main__":
    main()
