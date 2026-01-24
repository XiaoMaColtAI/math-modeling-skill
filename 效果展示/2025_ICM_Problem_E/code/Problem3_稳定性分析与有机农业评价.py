"""
2025 ICM Problem E: Making Room for Agriculture
Problem 3: 稳定性分析与有机农业评价模型

模型说明：
- 分析移除草剂后的生态系统稳定性
- 使用Jacobian矩阵特征值进行稳定性分析
- 有机农业综合评价（AHP + TOPSIS）
- 考虑pest control、crop health、plant reproduction、biodiversity、
  long-term sustainability、cost effectiveness等指标
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class StabilityAnalysisModel:
    """生态系统稳定性分析模型"""

    def __init__(self):
        """初始化参数"""
        # 基础参数
        self.r_c = 0.05
        self.K_c = 100
        self.r_I = 0.08
        self.K_I = 50
        self.d_I = 0.03
        self.r_B = 0.02
        self.K_B = 20
        self.d_B = 0.015
        self.e = 0.1

        # 相互作用系数
        self.a_CI = 0.02
        self.a_IB = 0.015

        # 化学物质参数
        self.k_pest_insect = 0.5
        self.k_chem_bird = 0.01
        self.k_chem_crop = 0.005
        self.degradation = 0.05

    def model_with_chemical(self, y, t, chemical_level):
        """
        带化学物质的模型

        chemical_level: 0 (无), 0.5 (减少), 1 (正常)
        """
        C, I, B, H = y

        # 季节性（取平均，忽略季节波动用于平衡点分析）
        S = 0  # 使用无季节性的版本进行稳定性分析

        # 化学物质投放（随chemical_level调整）
        application_rate = 2 * chemical_level  # 降低化学物质投放量

        # 作物动力学
        dCdt = (self.r_c * C * (1 - C/self.K_c) * (1 + S)
                - self.a_CI * C * I
                - self.k_chem_crop * H * C / 100)  # 修改为相对效应

        # 昆虫动力学
        dIdt = (self.r_I * I * (1 - I/self.K_I) * (1 + S)
                - self.a_IB * I * B
                - self.d_I * I
                - self.k_pest_insect * H * I / 100)  # 修改为相对效应

        # 鸟类动力学
        bioaccumulation = self.k_chem_bird * H / 100 * B
        dBdt = (self.r_B * B * (1 - B/self.K_B) * (1 + S)
                - self.d_B * B
                + self.e * self.a_IB * I * B
                - bioaccumulation)

        # 化学物质动力学（使用平均投放）
        dHdt = application_rate - self.degradation * H

        return [dCdt, dIdt, dBdt, dHdt]

    def find_equilibrium(self, chemical_level):
        """求平衡点"""
        func = lambda y: self.model_with_chemical(y, 0, chemical_level)
        # 使用更合理的初始猜测
        H_steady = (2 * chemical_level) / self.degradation
        y0_guess = [50, 15, 8, H_steady]
        equilibrium = fsolve(func, y0_guess)
        return equilibrium

    def compute_jacobian(self, y, chemical_level):
        """计算Jacobian矩阵"""
        eps = 1e-6
        n = len(y)
        func = lambda x: self.model_with_chemical(x, 0, chemical_level)
        f0 = np.array(func(y))

        J = np.zeros((n, n))
        for i in range(n):
            y_eps = y.copy()
            y_eps[i] += eps
            f_eps = np.array(func(y_eps))
            J[:, i] = (f_eps - f0) / eps

        return J

    def stability_metrics(self, equilibrium, chemical_level):
        """计算稳定性指标"""
        J = self.compute_jacobian(equilibrium, chemical_level)
        eigenvalues = np.linalg.eigvals(J)

        real_parts = np.real(eigenvalues)
        max_real = np.max(real_parts)

        # 稳定性指标
        is_stable = all(real_parts < 0)

        # 阻尼比（最接近虚轴的特征值）
        min_abs_real = np.min(np.abs(real_parts[real_parts < 0])) if is_stable else 0

        # 恢复时间（与最慢衰减模态相关）
        recovery_time = -1 / min_abs_real if min_abs_real > 0 else np.inf

        # 鲁棒性（最大奇异值的倒数）
        singular_values = np.linalg.svd(J, compute_uv=False)
        robustness = 1 / singular_values[0] if singular_values[0] > 0 else 0

        return {
            'equilibrium': equilibrium,
            'eigenvalues': eigenvalues,
            'stable': is_stable,
            'max_real_part': max_real,
            'recovery_time': recovery_time,
            'robustness': robustness,
            'jacobian': J
        }


class OrganicFarmingEvaluator:
    """有机农业评价模型（AHP + TOPSIS）"""

    def __init__(self):
        """初始化指标体系"""
        self.indicators = [
            '害虫控制能力',
            '作物健康状况',
            '植物繁殖能力',
            '生物多样性',
            '长期可持续性',
            '成本效益',
            '土壤健康',
            '生态平衡度'
        ]

        # 场景定义
        self.scenarios = [
            '传统农业（全化学）',
            '减少50%化学物质',
            '有机农业（无化学）',
            '有机农业+引入蝙蝠',
            '有机农业+引入蝙蝠+蚯蚓'
        ]

    def ahp_weights(self):
        """
        AHP层次分析法确定权重

        使用1-9标度法构造判断矩阵
        """
        # 判断矩阵（基于农业专家知识）
        # 行和列顺序与indicators相同
        A = np.array([
            [1,     3,     3,     2,     2,     1/2,   3,     2],    # 害虫控制
            [1/3,   1,     1,     1/2,   1/2,   1/3,   1,     1/2],  # 作物健康
            [1/3,   1,     1,     1/2,   1/2,   1/3,   1,     1/2],  # 植物繁殖
            [1/2,   2,     2,     1,     1,     1/2,   2,     1],    # 生物多样性
            [1/2,   2,     2,     1,     1,     1/2,   2,     1],    # 长期可持续性
            [2,     3,     3,     2,     2,     1,     3,     2],    # 成本效益
            [1/3,   1,     1,     1/2,   1/2,   1/3,   1,     1/2],  # 土壤健康
            [1/2,   2,     2,     1,     1,     1/2,   2,     1]     # 生态平衡
        ])

        n = len(A)

        # 计算权重向量（几何平均法）
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = np.prod(A[i, :]) ** (1/n)
        weights = weights / np.sum(weights)

        # 一致性检验
        lambda_max = np.mean([np.sum(A[i, :] * weights) / weights[i] for i in range(n)])
        CI = (lambda_max - n) / (n - 1)
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
        CR = CI / RI[n]

        return weights, CR

    def scenario_scores(self):
        """
        各场景在不同指标下的得分（0-100标准化）

        基于文献和模拟结果的估计
        """
        # 各场景的指标得分矩阵 (场景 × 指标)
        scores = np.array([
            # 害虫 作物 植物 生物 可持续 成本 土壤 生态
            [85,  70,  65,  30,  40,  90,  45,  35],  # 传统农业
            [80,  75,  70,  45,  55,  80,  55,  50],  # 减少50%
            [60,  80,  75,  75,  80,  50,  85,  80],  # 有机农业
            [85,  85,  85,  85,  85,  55,  85,  90],  # 有机+蝙蝠
            [90,  88,  90,  90,  92,  60,  95,  95],  # 有机+蝙蝠+蚯蚓
        ])

        return pd.DataFrame(scores, columns=self.indicators, index=self.scenarios)

    def topsis(self, scores, weights):
        """
        TOPSIS方法进行综合评价

        步骤：
        1. 构造标准化决策矩阵
        2. 构造加权标准化决策矩阵
        3. 确定正理想解和负理想解
        4. 计算各方案到正负理想解的距离
        5. 计算相对贴近度
        """
        # 1. 向量标准化
        norm_scores = scores / np.sqrt((scores ** 2).sum(axis=0))

        # 2. 加权
        weighted_scores = norm_scores * weights

        # 3. 确定理想解
        z_plus = weighted_scores.max(axis=0)
        z_minus = weighted_scores.min(axis=0)

        # 4-5. 计算距离和贴近度
        d_plus = np.sqrt(((weighted_scores - z_plus) ** 2).sum(axis=1))
        d_minus = np.sqrt(((weighted_scores - z_minus) ** 2).sum(axis=1))
        closeness = d_minus / (d_plus + d_minus)

        return closeness, d_plus, d_minus


def analyze_stability():
    """分析移除草剂后的稳定性变化"""

    model = StabilityAnalysisModel()

    print("=" * 60)
    print("生态系统稳定性分析")
    print("=" * 60)

    chemical_levels = [1.0, 0.5, 0.0]
    level_names = ['正常使用', '减少50%', '完全移除']

    results = []
    equilibria = []

    for level, name in zip(chemical_levels, level_names):
        print(f"\n分析化学物质水平: {name} (level={level})")

        # 求平衡点
        eq = model.find_equilibrium(level)
        equilibria.append(eq)

        # 稳定性分析
        metrics = model.stability_metrics(eq, level)
        results.append(metrics)

        print(f"  平衡点: C={eq[0]:.2f}, I={eq[1]:.2f}, B={eq[2]:.2f}, H={eq[3]:.2f}")
        print(f"  特征值: {metrics['eigenvalues']}")
        print(f"  最大实部: {metrics['max_real_part']:.6f}")
        print(f"  稳定性: {'稳定' if metrics['stable'] else '不稳定'}")
        print(f"  恢复时间: {metrics['recovery_time']:.2f} 天")
        print(f"  鲁棒性: {metrics['robustness']:.6f}")

    # 绘制特征值分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 子图1：特征值分布
    ax = axes[0, 0]
    colors = ['red', 'orange', 'green']
    for i, (result, name, color) in enumerate(zip(results, level_names, colors)):
        ev = result['eigenvalues']
        ax.scatter(np.real(ev), np.imag(ev), s=100, c=color, label=name, alpha=0.7, edgecolors='black')
        for j, e in enumerate(ev):
            ax.annotate(f'{i+1}-{j+1}', (np.real(e), np.imag(e)), fontsize=8)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('实部', fontsize=12)
    ax.set_ylabel('虚部', fontsize=12)
    ax.set_title('不同化学物质水平下的特征值分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 子图2：稳定性指标对比
    ax = axes[0, 1]
    recovery_times = [r['recovery_time'] if r['recovery_time'] < 100 else 100 for r in results]
    robustness = [r['robustness'] for r in results]

    x = np.arange(len(level_names))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, recovery_times, width, label='恢复时间 (天)', color='steelblue')
    bars2 = ax2.bar(x + width/2, robustness, width, label='鲁棒性', color='coral')

    ax.set_xlabel('化学物质使用水平', fontsize=12)
    ax.set_ylabel('恢复时间 (天)', fontsize=12, color='steelblue')
    ax2.set_ylabel('鲁棒性', fontsize=12, color='coral')
    ax.set_title('稳定性指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')

    # 子图3：平衡点对比
    ax = axes[1, 0]
    C_vals = [eq[0] for eq in equilibria]
    I_vals = [eq[1] for eq in equilibria]
    B_vals = [eq[2] for eq in equilibria]
    H_vals = [eq[3] for eq in equilibria]

    x = np.arange(len(level_names))
    width = 0.2

    ax.bar(x - 1.5*width, C_vals, width, label='作物', color='green')
    ax.bar(x - 0.5*width, I_vals, width, label='昆虫', color='red')
    ax.bar(x + 0.5*width, B_vals, width, label='鸟类', color='blue')
    ax.bar(x + 1.5*width, H_vals, width, label='化学物质', color='purple')

    ax.set_xlabel('化学物质使用水平', fontsize=12)
    ax.set_ylabel('平衡值', fontsize=12)
    ax.set_title('各物种平衡点对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 子图4：时序模拟
    ax = axes[1, 1]

    t = np.linspace(0, 3*365, 3*365)

    for level, name, color, style in zip(chemical_levels, level_names, ['red', 'orange', 'green'], ['--', '-.', '-']):
        H_steady = (2 * level) / model.degradation
        y0 = [50, 15, 8, H_steady]
        sol = odeint(lambda y, t: model.model_with_chemical(y, t, level), y0, t)
        C = sol[:, 0]
        t_years = t / 365
        ax.plot(t_years, C, style, color=color, linewidth=2, label=name)

    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('作物生物量 (kg/m^2)', fontsize=12)
    ax.set_title('移除草剂后的作物动态（3年）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)  # 设置y轴范围防止显示异常

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure6_stability_analysis.png',
                dpi=300, bbox_inches='tight')
    print("\n图6已保存: figures/figure6_stability_analysis.png")

    return results


def evaluate_organic_farming():
    """有机农业综合评价"""

    print("\n" + "=" * 60)
    print("有机农业综合评价（AHP + TOPSIS）")
    print("=" * 60)

    evaluator = OrganicFarmingEvaluator()

    # AHP权重计算
    weights, CR = evaluator.ahp_weights()

    print("\nAHP权重结果:")
    print(f"一致性比率 CR = {CR:.4f}", "(通过一致性检验)" if CR < 0.1 else "(未通过一致性检验)")

    weight_df = pd.DataFrame({
        '指标': evaluator.indicators,
        '权重': weights
    })
    weight_df['权重'] = weight_df['权重'].apply(lambda x: f'{x:.4f}')
    print("\n" + str(weight_df))

    # 场景得分
    scores = evaluator.scenario_scores()

    print("\n各场景指标得分:")
    print(scores.to_string())

    # TOPSIS评价
    closeness, d_plus, d_minus = evaluator.topsis(scores.values, weights)

    # 结果汇总
    results_df = pd.DataFrame({
        '场景': evaluator.scenarios,
        '正理想解距离': d_plus,
        '负理想解距离': d_minus,
        '综合得分': closeness,
        '排名': np.argsort(-closeness) + 1
    })
    results_df = results_df.sort_values('综合得分', ascending=False)

    print("\nTOPSIS评价结果:")
    print(results_df.to_string(index=False))

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1：权重分布
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(evaluator.indicators)))
    bars = ax.barh(evaluator.indicators, weights, color=colors)
    ax.set_xlabel('权重', fontsize=12)
    ax.set_title('AHP权重分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, weight in zip(bars, weights):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{weight:.3f}', va='center', fontsize=9)

    # 子图2：雷达图
    ax = axes[0, 1]
    categories = evaluator.indicators
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 选择前4个场景绘制
    colors = ['red', 'orange', 'green', 'blue']
    for i in range(min(4, len(evaluator.scenarios))):
        values = scores.iloc[i].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=evaluator.scenarios[i], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title('各场景综合评价雷达图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True)

    # 子图3：TOPSIS得分排名
    ax = axes[1, 0]
    sorted_results = results_df.sort_values('综合得分')
    colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in sorted_results['综合得分']]
    bars = ax.barh(sorted_results['场景'], sorted_results['综合得分'], color=colors)
    ax.set_xlabel('综合得分', fontsize=12)
    ax.set_title('TOPSIS综合得分排名', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)

    for bar, score in zip(bars, sorted_results['综合得分']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    # 子图4：热力图
    ax = axes[1, 1]
    im = ax.imshow(scores.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(evaluator.indicators)))
    ax.set_yticks(np.arange(len(evaluator.scenarios)))
    ax.set_xticklabels(evaluator.indicators, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(evaluator.scenarios, fontsize=9)

    for i in range(len(evaluator.scenarios)):
        for j in range(len(evaluator.indicators)):
            text = ax.text(j, i, f'{scores.values[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('场景-指标得分热力图', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('得分', fontsize=10)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure7_organic_farming_evaluation.png',
                dpi=300, bbox_inches='tight')
    print("\n图7已保存: figures/figure7_organic_farming_evaluation.png")

    # 保存结果表格
    results_df.to_csv('C:/Users/86198/Desktop/E/data/organic_evaluation.csv',
                      index=False, encoding='utf-8-sig')
    print("\n评价结果已保存: data/organic_evaluation.csv")

    return results_df


if __name__ == '__main__':
    # 稳定性分析
    stability_results = analyze_stability()

    # 有机农业评价
    evaluation_results = evaluate_organic_farming()

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
