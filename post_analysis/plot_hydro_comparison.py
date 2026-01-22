#!/usr/bin/env python3
"""
水文变量对比图：Baseline (TSR=OFF) vs TSR (TSR=ON)

比较两次模拟运行的水文变量差异，包括：
- ET (实际蒸散发)
- PET (潜在蒸散发)
- 地下水位
- 河道流量

用法: python plot_hydro_comparison.py [project_name]
默认: ccw
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 添加 shud_reader 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "validation" / "tsr" / "py"))
from shud_reader import read_dat

# 配置
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "ccw"
BASE_OUTPUT_DIR = Path(f"output/{PROJECT}_base.out")
TSR_OUTPUT_DIR = Path(f"output/{PROJECT}_tsr")
OUTPUT_DIR = Path("post_analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模拟起始日期
START_DATE = datetime(2000, 1, 1)


def time_to_dates(times_min, start_date):
    """将分钟时间转换为日期"""
    return [start_date + timedelta(minutes=float(t)) for t in times_min]


def compute_spatial_mean(df):
    """计算所有单元的空间均值时间序列"""
    return df.mean(axis=1)


def compute_spatial_stats(df):
    """计算空间统计量"""
    return {
        'mean': df.mean(axis=1),
        'std': df.std(axis=1),
        'min': df.min(axis=1),
        'max': df.max(axis=1),
        'q25': df.quantile(0.25, axis=1),
        'q75': df.quantile(0.75, axis=1),
    }


def plot_variable_comparison(ax, dates, base_data, tsr_data, var_name, unit,
                             show_legend=True, show_diff=True):
    """绘制单个变量的对比图"""
    # 主图：时间序列对比
    ax.plot(dates, base_data, 'b-', linewidth=1.2, label='Baseline (TSR=OFF)', alpha=0.8)
    ax.plot(dates, tsr_data, 'r-', linewidth=1.2, label='TSR (TSR=ON)', alpha=0.8)

    ax.set_ylabel(f'{var_name} [{unit}]')
    ax.set_title(var_name, fontsize=11, fontweight='bold')

    if show_legend:
        ax.legend(loc='upper right', fontsize=8)

    # 添加统计信息
    base_mean = np.nanmean(base_data)
    tsr_mean = np.nanmean(tsr_data)
    diff_pct = (tsr_mean - base_mean) / base_mean * 100 if base_mean != 0 else 0

    stats_text = f'Base: {base_mean:.4e}\nTSR: {tsr_mean:.4e}\nΔ: {diff_pct:+.2f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def plot_difference_panel(ax, dates, base_data, tsr_data, var_name):
    """绘制差异图 (TSR - Baseline)"""
    diff = tsr_data - base_data

    # 用颜色区分正负差异
    ax.fill_between(dates, 0, diff, where=(diff >= 0), color='red', alpha=0.5, label='TSR > Base')
    ax.fill_between(dates, 0, diff, where=(diff < 0), color='blue', alpha=0.5, label='TSR < Base')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_ylabel(f'Δ{var_name}')
    ax.legend(loc='upper right', fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def plot_hydro_comparison(project, variables_data, n_days=None):
    """绘制水文变量对比总图"""
    n_vars = len(variables_data)

    fig, axes = plt.subplots(n_vars, 2, figsize=(16, 3.5 * n_vars))
    fig.suptitle(f'Hydrological Variables Comparison: Baseline vs TSR\nProject: {project}',
                 fontsize=14, fontweight='bold')

    for i, (var_name, var_info) in enumerate(variables_data.items()):
        dates = var_info['dates']
        base_data = var_info['base']
        tsr_data = var_info['tsr']
        unit = var_info['unit']

        if n_days:
            dates = dates[:n_days]
            base_data = base_data[:n_days]
            tsr_data = tsr_data[:n_days]

        # 左列：时间序列对比
        plot_variable_comparison(axes[i, 0], dates, base_data, tsr_data,
                                 var_name, unit, show_legend=(i == 0))

        # 右列：差异图
        plot_difference_panel(axes[i, 1], dates, base_data, tsr_data, var_name)
        axes[i, 1].set_title(f'{var_name} Difference (TSR - Baseline)', fontsize=11)

    # 设置底部 x 轴标签
    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 旋转日期标签
    for ax_row in axes:
        for ax in ax_row:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    suffix = f'_{n_days}days' if n_days else '_full'
    output_file = OUTPUT_DIR / f'{project}_hydro_comparison{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'已保存: {output_file}')
    plt.close()


def plot_scatter_comparison(project, variables_data):
    """绘制散点对比图 (1:1 线)"""
    n_vars = len(variables_data)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    fig.suptitle(f'Scatter Comparison: Baseline vs TSR\nProject: {project}',
                 fontsize=14, fontweight='bold')

    for i, (var_name, var_info) in enumerate(variables_data.items()):
        ax = axes[i]
        base_data = var_info['base']
        tsr_data = var_info['tsr']
        unit = var_info['unit']

        # 散点图
        ax.scatter(base_data, tsr_data, alpha=0.3, s=5, c='steelblue')

        # 1:1 线
        lims = [min(base_data.min(), tsr_data.min()), max(base_data.max(), tsr_data.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='1:1 line')

        ax.set_xlabel(f'Baseline [{unit}]')
        ax.set_ylabel(f'TSR [{unit}]')
        ax.set_title(var_name, fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_aspect('equal', adjustable='box')

        # 相关系数
        valid_mask = np.isfinite(base_data) & np.isfinite(tsr_data)
        if np.sum(valid_mask) > 10:
            r = np.corrcoef(base_data[valid_mask], tsr_data[valid_mask])[0, 1]
            ax.text(0.05, 0.95, f'r = {r:.4f}', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = OUTPUT_DIR / f'{project}_hydro_scatter.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'已保存: {output_file}')
    plt.close()


def plot_monthly_comparison(project, variables_data):
    """绘制月均值对比柱状图"""
    n_vars = len(variables_data)

    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3.5 * n_vars))
    if n_vars == 1:
        axes = [axes]

    fig.suptitle(f'Monthly Mean Comparison: Baseline vs TSR\nProject: {project}',
                 fontsize=14, fontweight='bold')

    for i, (var_name, var_info) in enumerate(variables_data.items()):
        ax = axes[i]
        dates = var_info['dates']
        base_data = var_info['base']
        tsr_data = var_info['tsr']
        unit = var_info['unit']

        # 按月聚合
        import pandas as pd
        df = pd.DataFrame({
            'date': dates,
            'base': base_data,
            'tsr': tsr_data
        })
        df['month'] = df['date'].apply(lambda x: x.strftime('%Y-%m'))
        monthly = df.groupby('month').agg({'base': 'mean', 'tsr': 'mean'}).reset_index()

        x = np.arange(len(monthly))
        width = 0.35

        bars1 = ax.bar(x - width/2, monthly['base'], width, label='Baseline', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, monthly['tsr'], width, label='TSR', color='coral', alpha=0.8)

        ax.set_ylabel(f'{var_name} [{unit}]')
        ax.set_title(var_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x[::3])  # 每3个月显示一个标签
        ax.set_xticklabels(monthly['month'].values[::3], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8)

        # 计算总体差异
        total_diff = (monthly['tsr'].mean() - monthly['base'].mean()) / monthly['base'].mean() * 100
        ax.text(0.02, 0.95, f'Mean Δ: {total_diff:+.2f}%', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = OUTPUT_DIR / f'{project}_hydro_monthly.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'已保存: {output_file}')
    plt.close()


def main():
    global START_DATE

    print(f'处理项目: {PROJECT}')
    print(f'Baseline 目录: {BASE_OUTPUT_DIR}')
    print(f'TSR 目录: {TSR_OUTPUT_DIR}')

    # 检查目录
    if not BASE_OUTPUT_DIR.exists():
        print(f'错误: Baseline 目录不存在 {BASE_OUTPUT_DIR}')
        sys.exit(1)
    if not TSR_OUTPUT_DIR.exists():
        print(f'错误: TSR 目录不存在 {TSR_OUTPUT_DIR}')
        sys.exit(1)

    # 要比较的变量
    variables = [
        ('eleveta', 'ET (Actual Evapotranspiration)', 'm/day'),
        ('elevetp', 'PET (Potential Evapotranspiration)', 'm/day'),
        ('elevettr', 'Transpiration', 'm/day'),
        ('elevetev', 'Evaporation', 'm/day'),
        ('eleygw', 'Groundwater Level', 'm'),
        ('eleyunsat', 'Unsaturated Zone Storage', 'm'),
    ]

    # 河流变量单独处理（维度不同）
    river_variables = [
        ('rivqdown', 'River Discharge', 'm³/day'),
    ]

    print('\n读取数据...')
    variables_data = {}

    for var_code, var_name, unit in variables:
        base_file = BASE_OUTPUT_DIR / f'{PROJECT}.{var_code}.dat'
        tsr_file = TSR_OUTPUT_DIR / f'{PROJECT}.{var_code}.dat'

        if not base_file.exists() or not tsr_file.exists():
            print(f'  跳过 {var_name}: 文件不存在')
            continue

        print(f'  读取 {var_name}...')
        base_df = read_dat(base_file, time_mode='index')
        tsr_df = read_dat(tsr_file, time_mode='index')

        # 计算空间均值
        base_mean = compute_spatial_mean(base_df)
        tsr_mean = compute_spatial_mean(tsr_df)

        # 时间轴
        times_min = base_df.index.values
        dates = time_to_dates(times_min, START_DATE)

        variables_data[var_name] = {
            'base': base_mean.values,
            'tsr': tsr_mean.values,
            'dates': dates,
            'unit': unit,
        }

    # 河流流量（取出口）
    for var_code, var_name, unit in river_variables:
        base_file = BASE_OUTPUT_DIR / f'{PROJECT}.{var_code}.dat'
        tsr_file = TSR_OUTPUT_DIR / f'{PROJECT}.{var_code}.dat'

        if not base_file.exists() or not tsr_file.exists():
            print(f'  跳过 {var_name}: 文件不存在')
            continue

        print(f'  读取 {var_name}...')
        base_df = read_dat(base_file, time_mode='index')
        tsr_df = read_dat(tsr_file, time_mode='index')

        # 取第一列（通常是出口）
        base_outlet = base_df.iloc[:, 0]
        tsr_outlet = tsr_df.iloc[:, 0]

        times_min = base_df.index.values
        dates = time_to_dates(times_min, START_DATE)

        variables_data[var_name] = {
            'base': base_outlet.values,
            'tsr': tsr_outlet.values,
            'dates': dates,
            'unit': unit,
        }

    print(f'\n共加载 {len(variables_data)} 个变量')

    # 绘图
    print('\n绘制时间序列对比图...')
    plot_hydro_comparison(PROJECT, variables_data)

    print('\n绘制前180天详细对比图...')
    plot_hydro_comparison(PROJECT, variables_data, n_days=180)

    print('\n绘制散点对比图...')
    plot_scatter_comparison(PROJECT, variables_data)

    print('\n绘制月均值对比图...')
    plot_monthly_comparison(PROJECT, variables_data)

    # 输出统计摘要
    print('\n' + '=' * 60)
    print('统计摘要')
    print('=' * 60)
    print(f"{'变量':<35} {'Baseline':>12} {'TSR':>12} {'差异%':>10}")
    print('-' * 60)

    for var_name, var_info in variables_data.items():
        base_mean = np.nanmean(var_info['base'])
        tsr_mean = np.nanmean(var_info['tsr'])
        diff_pct = (tsr_mean - base_mean) / base_mean * 100 if base_mean != 0 else 0
        print(f'{var_name:<35} {base_mean:>12.4e} {tsr_mean:>12.4e} {diff_pct:>+10.2f}')

    print('\n完成!')


if __name__ == '__main__':
    main()
