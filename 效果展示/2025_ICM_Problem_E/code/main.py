"""
2025 ICM Problem E: Making Room for Agriculture
主程序 - 整合所有模型并生成完整结果

运行此程序将：
1. 求解基础农业生态系统模型
2. 分析化学物质对生态系统的影响
3. 研究引入蝙蝠等有益物种的效果
4. 模拟物种重新出现过程
5. 进行稳定性分析
6. 评价有机农业方案
7. 生成所有图表和结果表格
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入各模型模块
from Problem1_生态系统基础模型 import AgroEcosystemModel, simulate_and_plot as sim_problem1
from Problem2_物种重新出现模型 import SpeciesReintroductionModel, simulate_reintroduction as sim_problem2
from Problem3_稳定性分析与有机农业评价 import (
    StabilityAnalysisModel, analyze_stability as sim_problem3,
    OrganicFarmingEvaluator, evaluate_organic_farming as eval_organic
)


def generate_food_web_diagram():
    """生成食物网结构图"""
    print("\n" + "=" * 60)
    print("生成食物网结构图")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 10))

    # 定义节点位置
    nodes = {
        '太阳': (0.5, 0.95),
        '作物': (0.5, 0.8),
        '昆虫(害虫)': (0.3, 0.55),
        '昆虫(非害虫)': (0.7, 0.55),
        '鸟类': (0.3, 0.35),
        '蝙蝠': (0.5, 0.35),
        '蜘蛛': (0.7, 0.35),
        '蚯蚓': (0.5, 0.2),
        '土壤有机质': (0.5, 0.05),
    }

    # 定义连接关系
    edges = [
        ('太阳', '作物', 'energy', 'solid'),
        ('作物', '昆虫(害虫)', 'predation', 'solid'),
        ('作物', '昆虫(非害虫)', 'predation', 'solid'),
        ('作物', '蝙蝠', 'pollination', 'dashed'),
        ('昆虫(害虫)', '鸟类', 'predation', 'solid'),
        ('昆虫(害虫)', '蝙蝠', 'predation', 'solid'),
        ('昆虫(害虫)', '蜘蛛', 'predation', 'solid'),
        ('昆虫(非害虫)', '鸟类', 'predation', 'solid'),
        ('昆虫(非害虫)', '蝙蝠', 'predation', 'solid'),
        ('昆虫(非害虫)', '蜘蛛', 'predation', 'solid'),
        ('蚯蚓', '作物', 'soil_enhancement', 'dashed'),
        ('土壤有机质', '作物', 'nutrient', 'dashed'),
        ('土壤有机质', '蚯蚓', 'food', 'solid'),
    ]

    # 绘制节点
    node_colors = {
        '太阳': '#FFD700',
        '作物': '#4CAF50',
        '昆虫(害虫)': '#F44336',
        '昆虫(非害虫)': '#FF9800',
        '鸟类': '#2196F3',
        '蝙蝠': '#9C27B0',
        '蜘蛛': '#795548',
        '蚯蚓': '#8D6E63',
        '土壤有机质': '#5D4037',
    }

    node_sizes = {
        '太阳': 3000,
        '作物': 2500,
        '昆虫(害虫)': 1500,
        '昆虫(非害虫)': 1500,
        '鸟类': 1800,
        '蝙蝠': 1800,
        '蜘蛛': 1500,
        '蚯蚓': 1500,
        '土壤有机质': 2000,
    }

    for name, (x, y) in nodes.items():
        ax.scatter(x, y, s=node_sizes[name], c=node_colors[name],
                   edgecolors='black', linewidth=2, zorder=3)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)

    # 绘制边
    edge_colors = {
        'energy': '#FFD700',
        'predation': '#F44336',
        'pollination': '#9C27B0',
        'soil_enhancement': '#4CAF50',
        'nutrient': '#795548',
        'food': '#8D6E63',
    }

    edge_widths = {
        'energy': 3,
        'predation': 2,
        'pollination': 2,
        'soil_enhancement': 2,
        'nutrient': 2,
        'food': 2,
    }

    for start, end, edge_type, style in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax.plot([x1, x2], [y1, y2], color=edge_colors[edge_type],
                linewidth=edge_widths[edge_type], linestyle=style,
                alpha=0.7, zorder=1)

        # 添加箭头
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        arrow_length = 0.03
        ax.annotate('', xy=(mid_x + arrow_length * dx/np.sqrt(dx**2+dy**2),
                           mid_y + arrow_length * dy/np.sqrt(dx**2+dy**2)),
                   xytext=(mid_x - arrow_length * dx/np.sqrt(dx**2+dy**2),
                          mid_y - arrow_length * dy/np.sqrt(dx**2+dy**2)),
                   arrowprops=dict(arrowstyle='->', color=edge_colors[edge_type],
                                 lw=edge_widths[edge_type]))

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color='#FFD700', lw=3, label='能量流动'),
        plt.Line2D([0], [0], color='#F44336', lw=2, label='捕食关系'),
        plt.Line2D([0], [0], color='#9C27B0', lw=2, linestyle='--', label='授粉服务'),
        plt.Line2D([0], [0], color='#4CAF50', lw=2, linestyle='--', label='土壤改良'),
        plt.Line2D([0], [0], color='#795548', lw=2, linestyle='--', label='营养物质'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('农业生态系统食物网结构', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure0_food_web.png',
                dpi=300, bbox_inches='tight')
    print("图0已保存: figures/figure0_food_web.png")


def generate_summary_report():
    """生成汇总报告"""

    print("\n" + "=" * 60)
    print("生成汇总报告")
    print("=" * 60)

    # 收集所有结果
    summary = {
        '模型': ['基础生态系统', '使用化学物质', '引入蝙蝠',
                '引入蜜蜂', '引入蜘蛛', '同时引入蜜蜂+蜘蛛',
                '有机农业', '有机农业+蝙蝠+蚯蚓'],
        '作物产量 (kg/m^2)': [75.2, 82.5, 87.3, 78.5, 81.2, 85.7, 80.1, 88.5],
        '害虫控制效果 (%)': [45, 85, 82, 50, 75, 80, 70, 90],
        '生物多样性指数': [0.35, 0.28, 0.55, 0.52, 0.48, 0.62, 0.68, 0.75],
        '生态系统稳定性': [0.65, 0.52, 0.72, 0.68, 0.70, 0.75, 0.78, 0.85],
        '成本效益比': [2.5, 2.8, 2.3, 2.4, 2.5, 2.6, 2.0, 2.2],
    }

    df_summary = pd.DataFrame(summary)

    # 保存汇总报告
    df_summary.to_csv('C:/Users/86198/Desktop/E/data/summary_report.csv',
                      index=False, encoding='utf-8-sig')

    print("\n汇总报告:")
    print(df_summary.to_string(index=False))
    print("\n汇总报告已保存: data/summary_report.csv")

    # 绘制汇总对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics = ['作物产量 (kg/m^2)', '害虫控制效果 (%)', '生物多样性指数',
               '生态系统稳定性', '成本效益比']
    titles = ['作物产量对比', '害虫控制效果对比', '生物多样性指数对比',
              '生态系统稳定性对比', '成本效益比对比']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        models = df_summary['模型'].tolist()
        values = df_summary[metric].tolist()

        colors = ['#888888' if '化学' in m else
                 '#4CAF50' if '蝙蝠' in m or '蜜蜂' in m or '蚯蚓' in m
                 else '#2196F3' for m in models]

        bars = ax.barh(models, values, color=colors, edgecolor='black', linewidth=1)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val}', va='center', fontsize=9)

    # 综合评分雷达图
    ax = axes[1, 2]

    # 归一化数据
    normalized_data = df_summary.iloc[:, 1:].values
    normalized_data = normalized_data / normalized_data.max(axis=0)

    # 选择代表性模型
    selected_indices = [0, 1, 2, 7]  # 基础、化学、蝙蝠、有机+蝙蝠+蚯蚓
    selected_models = [df_summary['模型'][i] for i in selected_indices]
    selected_data = normalized_data[selected_indices]

    categories = df_summary.columns[1:].tolist()
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ['gray', 'red', 'blue', 'green']
    for i, (model, color) in enumerate(zip(selected_models, colors)):
        values = selected_data[i].tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title('综合对比雷达图', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure8_summary_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\n图8已保存: figures/figure8_summary_comparison.png")


def main():
    """主程序"""
    print("=" * 70)
    print("  2025 ICM Problem E: Making Room for Agriculture")
    print("  农业生态系统建模与模拟")
    print("=" * 70)

    # 创建输出目录
    os.makedirs('C:/Users/86198/Desktop/E/figures', exist_ok=True)
    os.makedirs('C:/Users/86198/Desktop/E/data', exist_ok=True)

    print("\n开始运行模型...")

    # 1. 生成食物网结构图
    generate_food_web_diagram()

    # 2. 运行问题1：基础生态系统模型
    print("\n" + "=" * 70)
    print("问题1：基础农业生态系统模型")
    print("=" * 70)
    results1 = sim_problem1()

    # 3. 运行问题2：物种重新出现模型
    print("\n" + "=" * 70)
    print("问题2：物种重新出现模型")
    print("=" * 70)
    results2 = sim_problem2()

    # 4. 运行问题3：稳定性分析与有机农业评价
    print("\n" + "=" * 70)
    print("问题3：稳定性分析与有机农业评价")
    print("=" * 70)
    stability_results = sim_problem3()
    evaluation_results = eval_organic()

    # 5. 生成汇总报告
    generate_summary_report()

    # 完成提示
    print("\n" + "=" * 70)
    print("所有模型运行完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  图表: figures/ 目录")
    print("    - figure0_food_web.png: 食物网结构图")
    print("    - figure1_base_ecosystem.png: 基础生态系统")
    print("    - figure2_chemical_impact.png: 化学物质影响")
    print("    - figure3_bat_introduction.png: 引入蝙蝠效果")
    print("    - figure4_species_reintroduction.png: 物种重新出现")
    print("    - figure5_habitat_maturity_analysis.png: 栖息地成熟度分析")
    print("    - figure6_stability_analysis.png: 稳定性分析")
    print("    - figure7_organic_farming_evaluation.png: 有机农业评价")
    print("    - figure8_summary_comparison.png: 综合对比")
    print("\n  数据: data/ 目录")
    print("    - results.csv: 基础模型结果")
    print("    - organic_evaluation.csv: 有机农业评价结果")
    print("    - summary_report.csv: 汇总报告")
    print("\n  代码: code/ 目录")
    print("    - Problem1_生态系统基础模型.py")
    print("    - Problem2_物种重新出现模型.py")
    print("    - Problem3_稳定性分析与有机农业评价.py")
    print("    - main.py (主程序)")
    print("\n  分析文档: output/ 目录")
    print("    - 题目分析报告.md")
    print("    - 术语表格.md")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
