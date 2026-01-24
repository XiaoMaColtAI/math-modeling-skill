"""
2025 ICM Problem E: Making Room for Agriculture
Problem 2: 物种重新出现模型

模型说明：
- 模拟边缘栖息地成熟过程中物种的回归
- 纳入两个不同物种分析其影响
- 物种1: 蜜蜂 (Bees) - 授粉者
- 物种2: 蜘蛛 (Spiders) - 害虫捕食者
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SpeciesReintroductionModel:
    """物种重新引入模型"""

    def __init__(self):
        """初始化参数"""
        # 栖息地成熟度参数
        self.M0 = 0.1          # 初始成熟度
        self.alpha_M = 0.002   # 成熟度自然增长率
        self.beta_M = 0.001    # 人为活动对成熟度的负面影响

        # 蜜蜂参数 (授粉者)
        self.r_bee = 0.05      # 增长率
        self.K_bee = 100       # 环境容纳量
        self.d_bee = 0.02      # 死亡率
        self.im_bee = 0.8      # 最大迁入率
        self.M_crit_bee = 0.3  # 临界成熟度

        # 蜘蛛参数 (害虫捕食者)
        self.r_spider = 0.04
        self.K_spider = 50
        self.d_spider = 0.025
        self.im_spider = 0.6
        self.M_crit_spider = 0.4

        # 授粉效益参数
        self.pollination_benefit = 0.01  # 蜜蜂对作物产量的贡献系数

        # 捕食效益参数
        self.spider_predation = 0.03     # 蜘蛛对害虫的捕食系数

    def migration_rate(self, M, species_type):
        """
        计算物种迁入率

        使用sigmoid函数表示栖息地成熟度对迁入的影响
        """
        if species_type == 'bee':
            M_crit = self.M_crit_bee
            im_max = self.im_bee
        elif species_type == 'spider':
            M_crit = self.M_crit_spider
            im_max = self.im_spider
        else:
            return 0

        # Sigmoid函数
        k = 10  # 陡度参数
        im = im_max / (1 + np.exp(-k * (M - M_crit)))
        return im

    def reintroduction_model(self, y, t, include_bees=False, include_spiders=False):
        """
        物种重新引入模型

        状态变量：
        y[0]: M - 栖息地成熟度
        y[1]: C - 作物生物量
        y[2]: I - 昆虫(害虫)种群
        y[3]: Bee - 蜜蜂种群 (可选)
        y[4]: Spider - 蜘蛛种群 (可选)
        """
        M, C, I = y[0], y[1], y[2]

        # 从y中提取蜜蜂和蜘蛛种群
        Bee = y[3] if include_bees and len(y) > 3 else 0
        Spider = y[4] if include_spiders and len(y) > 4 else 0

        # 基础生态系统参数
        r_c = 0.05
        K_c = 100
        a_CI = 0.02
        r_I = 0.08
        K_I = 50
        d_I = 0.03

        # 季节性
        S = 0.2 * np.sin(2 * np.pi * t / 365)

        # 栖息地成熟度动力学
        dMdt = self.alpha_M * M * (1 - M)

        # 作物动力学（加入蜜蜂授粉效应）
        pollination_effect = self.pollination_benefit * Bee * C if include_bees else 0
        dCdt = r_c * C * (1 - C/K_c) * (1 + S) - a_CI * C * I + pollination_effect

        # 昆虫动力学（加入蜘蛛捕食效应）
        spider_effect = self.spider_predation * Spider * I if include_spiders else 0
        dIdt = r_I * I * (1 - I/K_I) * (1 + S) - d_I * I - spider_effect

        result = [dMdt, dCdt, dIdt]

        # 蜜蜂动力学
        if include_bees:
            im_bee = self.migration_rate(M, 'bee')
            dBeedt = (self.r_bee * Bee * (1 - Bee/self.K_bee)
                      - self.d_bee * Bee
                      + im_bee * (1 - Bee/self.K_bee))
            result.append(dBeedt)

        # 蜘蛛动力学
        if include_spiders:
            im_spider = self.migration_rate(M, 'spider')
            dSpiderdt = (self.r_spider * Spider * (1 - Spider/self.K_spider)
                         - self.d_spider * Spider
                         + im_spider * (1 - Spider/self.K_spider))
            result.append(dSpiderdt)

        return result

    def solve(self, t_span=10*365, y0=None, include_bees=False, include_spiders=False):
        """求解模型"""
        t = np.linspace(0, t_span, t_span)

        if y0 is None:
            y0 = [self.M0, 20, 10]  # [M, C, I]

        # 根据要包含的物种正确构建y0
        y0 = list(y0[:3])  # 基础的M, C, I
        if include_bees:
            y0.append(0)  # Bee初始值
        if include_spiders:
            y0.append(0)  # Spider初始值

        func = lambda y, t: self.reintroduction_model(y, t, include_bees, include_spiders)
        solution = odeint(func, y0, t)

        return t, solution


def simulate_reintroduction():
    """模拟物种重新引入过程"""

    model = SpeciesReintroductionModel()

    print("=" * 60)
    print("物种重新引入模拟")
    print("=" * 60)

    # ===== 场景1：无重新引入 =====
    print("\n场景1：无物种重新引入")
    y0_base = [model.M0, 20, 10]
    t1, sol1 = model.solve(t_span=10*365, y0=y0_base, include_bees=False, include_spiders=False)

    M1, C1, I1 = sol1[:, 0], sol1[:, 1], sol1[:, 2]
    t_years = t1 / 365

    # ===== 场景2：引入蜜蜂 =====
    print("\n场景2：引入蜜蜂（授粉者）")
    y0_bee = [model.M0, 20, 10, 0]
    t2, sol2 = model.solve(t_span=10*365, y0=y0_bee, include_bees=True, include_spiders=False)

    M2, C2, I2, Bee2 = sol2[:, 0], sol2[:, 1], sol2[:, 2], sol2[:, 3]

    # ===== 场景3：引入蜘蛛 =====
    print("\n场景3：引入蜘蛛（害虫捕食者）")
    y0_spider = [model.M0, 20, 10, 0]
    t3, sol3 = model.solve(t_span=10*365, y0=y0_spider, include_bees=False, include_spiders=True)

    M3, C3, I3, Spider3 = sol3[:, 0], sol3[:, 1], sol3[:, 2], sol3[:, 3]

    # ===== 场景4：同时引入蜜蜂和蜘蛛 =====
    print("\n场景4：同时引入蜜蜂和蜘蛛")
    y0_both = [model.M0, 20, 10, 0, 0]
    t4, sol4 = model.solve(t_span=10*365, y0=y0_both, include_bees=True, include_spiders=True)

    M4, C4, I4, Bee4, Spider4 = sol4[:, 0], sol4[:, 1], sol4[:, 2], sol4[:, 3], sol4[:, 4]

    # ===== 绘图 =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 子图1：栖息地成熟度
    ax = axes[0, 0]
    ax.plot(t_years, M1, 'k-', label='无引入', linewidth=2)
    ax.plot(t_years, M2, 'b-', label='引入蜜蜂', linewidth=2, alpha=0.7)
    ax.plot(t_years, M3, 'r-', label='引入蜘蛛', linewidth=2, alpha=0.7)
    ax.plot(t_years, M4, 'g-', label='同时引入', linewidth=2)
    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('栖息地成熟度', fontsize=12)
    ax.set_title('栖息地成熟度演化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 子图2：蜜蜂种群动态
    ax = axes[0, 1]
    ax.plot(t_years, Bee2, 'b-', linewidth=2)
    ax.fill_between(t_years, 0, Bee2, alpha=0.3, color='blue')
    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('蜜蜂种群数量', fontsize=12)
    ax.set_title('蜜蜂种群动态（授粉者）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 子图3：蜘蛛种群动态
    ax = axes[0, 2]
    ax.plot(t_years, Spider3, 'r-', linewidth=2)
    ax.fill_between(t_years, 0, Spider3, alpha=0.3, color='red')
    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('蜘蛛种群数量', fontsize=12)
    ax.set_title('蜘蛛种群动态（害虫捕食者）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 子图4：作物产量对比
    ax = axes[1, 0]
    ax.plot(t_years, C1, 'k--', label='无引入', linewidth=2)
    ax.plot(t_years, C2, 'b-', label='引入蜜蜂', linewidth=2)
    ax.plot(t_years, C3, 'r-', label='引入蜘蛛', linewidth=2)
    ax.plot(t_years, C4, 'g-', label='同时引入', linewidth=2)
    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('作物生物量 (kg/m^2)', fontsize=12)
    ax.set_title('物种引入对作物产量的影响', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 子图5：害虫种群对比
    ax = axes[1, 1]
    ax.plot(t_years, I1, 'k--', label='无引入', linewidth=2)
    ax.plot(t_years, I2, 'b-', label='引入蜜蜂', linewidth=2)
    ax.plot(t_years, I3, 'r-', label='引入蜘蛛', linewidth=2)
    ax.plot(t_years, I4, 'g-', label='同时引入', linewidth=2)
    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('害虫种群数量', fontsize=12)
    ax.set_title('物种引入对害虫种群的影响', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 子图6：协同效应分析
    ax = axes[1, 2]
    expected_sum = C2[-1] + C3[-1] - C1[-1]  # 期望的单独效应之和
    actual_combined = C4[-1] - C1[-1]        # 实际的联合效应
    synergy = actual_combined - expected_sum

    scenarios = ['无引入', '仅蜜蜂', '仅蜘蛛', '同时引入']
    final_crops = [C1[-1], C2[-1], C3[-1], C4[-1]]
    colors = ['gray', 'blue', 'red', 'green']

    bars = ax.bar(scenarios, final_crops, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('最终作物生物量 (kg/m^2)', fontsize=12)
    ax.set_title('10年后各场景作物产量对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure4_species_reintroduction.png',
                dpi=300, bbox_inches='tight')
    print("\n图4已保存: figures/figure4_species_reintroduction.png")

    # ===== 详细分析图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 栖息地成熟度与物种引入的关系
    ax = axes[0]
    ax.plot(t_years, M1, 'k-', linewidth=2, label='栖息地成熟度')
    ax.axhline(y=model.M_crit_bee, color='b', linestyle='--', linewidth=1.5, label='蜜蜂临界成熟度')
    ax.axhline(y=model.M_crit_spider, color='r', linestyle='--', linewidth=1.5, label='蜘蛛临界成熟度')
    ax.fill_between(t_years, 0, model.M_crit_bee, alpha=0.1, color='blue')
    ax.fill_between(t_years, model.M_crit_bee, model.M_crit_spider, alpha=0.1, color='purple')
    ax.fill_between(t_years, model.M_crit_spider, 1, alpha=0.1, color='red')
    ax.text(2, model.M_crit_bee + 0.05, '蜜蜂适生区', fontsize=10, color='blue')
    ax.text(2, model.M_crit_spider + 0.05, '蜘蛛适生区', fontsize=10, color='red')
    ax.set_xlabel('时间 (年)', fontsize=12)
    ax.set_ylabel('栖息地成熟度', fontsize=12)
    ax.set_title('栖息地成熟度与物种引入阈值', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 迁入率曲线
    ax = axes[1]
    M_range = np.linspace(0, 1, 100)
    im_bee_curve = [model.migration_rate(m, 'bee') for m in M_range]
    im_spider_curve = [model.migration_rate(m, 'spider') for m in M_range]

    ax.plot(M_range, im_bee_curve, 'b-', linewidth=2, label='蜜蜂迁入率')
    ax.plot(M_range, im_spider_curve, 'r-', linewidth=2, label='蜘蛛迁入率')
    ax.axvline(x=model.M_crit_bee, color='b', linestyle='--', alpha=0.5)
    ax.axvline(x=model.M_crit_spider, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('栖息地成熟度', fontsize=12)
    ax.set_ylabel('迁入率', fontsize=12)
    ax.set_title('物种迁入率与栖息地成熟度的关系', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure5_habitat_maturity_analysis.png',
                dpi=300, bbox_inches='tight')
    print("图5已保存: figures/figure5_habitat_maturity_analysis.png")

    # ===== 统计结果 =====
    print("\n" + "=" * 60)
    print("物种重新引入统计结果（10年）")
    print("=" * 60)

    results_summary = pd.DataFrame({
        '场景': ['无引入', '引入蜜蜂', '引入蜘蛛', '同时引入'],
        '最终栖息地成熟度': [f'{M1[-1]:.3f}', f'{M2[-1]:.3f}', f'{M3[-1]:.3f}', f'{M4[-1]:.3f}'],
        '最终作物产量 (kg/m^2)': [f'{C1[-1]:.2f}', f'{C2[-1]:.2f}', f'{C3[-1]:.2f}', f'{C4[-1]:.2f}'],
        '最终害虫种群': [f'{I1[-1]:.2f}', f'{I2[-1]:.2f}', f'{I3[-1]:.2f}', f'{I4[-1]:.2f}'],
        '蜜蜂种群': ['0', f'{Bee2[-1]:.2f}', '0', f'{Bee4[-1]:.2f}'],
        '蜘蛛种群': ['0', '0', f'{Spider3[-1]:.2f}', f'{Spider4[-1]:.2f}'],
    })

    print("\n" + str(results_summary))

    # 计算协同效应
    print(f"\n协同效应分析:")
    print(f"期望的单独效应之和: {expected_sum:.2f} kg/m^2")
    print(f"实际的联合效应: {actual_combined:.2f} kg/m^2")
    print(f"协同效应: {synergy:.2f} kg/m^2 ({synergy/expected_sum*100:.1f}%)")

    return {
        'base': (t_years, sol1),
        'bee': (t_years, sol2),
        'spider': (t_years, sol3),
        'both': (t_years, sol4)
    }


if __name__ == '__main__':
    results = simulate_reintroduction()
    print("\n" + "=" * 60)
    print("物种重新引入模拟完成！")
    print("=" * 60)
