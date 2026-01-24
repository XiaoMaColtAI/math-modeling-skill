"""
2025 ICM Problem E: Making Room for Agriculture
Problem 1: 基础农业生态系统模型

模型说明：
- 建立农业生态系统的食物网模型
- 使用改进的Lotka-Volterra方程描述物种间相互作用
- 考虑农业周期和季节性因素
- 分析除草剂和杀虫剂对生态系统的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pandas as pd
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 10
rcParams['lines.linewidth'] = 2
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14


class AgroEcosystemModel:
    """农业生态系统动力学模型"""

    def __init__(self, params=None):
        """
        初始化模型参数

        参数说明：
        - r_c: 作物增长率
        - K_c: 作物环境容纳量
        - a_CI: 昆虫对作物的影响系数
        - r_I: 昆虫增长率
        - K_I: 昆虫环境容纳量
        - a_IB: 鸟类对昆虫的捕食系数
        - d_I: 昆虫自然死亡率
        - r_B: 鸟类增长率
        - K_B: 鸟类环境容纳量
        - d_B: 鸟类自然死亡率
        - e: 能量转化效率
        """
        if params is None:
            # 默认参数值（基于文献估计）
            self.params = {
                # 作物参数
                'r_c': 0.05,      # 作物日增长率
                'K_c': 100,       # 作物生物量上限 (kg/m^2)
                'a_CI': 0.02,     # 昆虫对作物的影响系数

                # 昆虫参数
                'r_I': 0.08,      # 昆虫日增长率
                'K_I': 50,        # 昆虫环境容纳量
                'd_I': 0.03,      # 昆虫日死亡率
                'a_IB': 0.015,    # 鸟类对昆虫的捕食系数

                # 鸟类参数
                'r_B': 0.02,      # 鸟类日增长率
                'K_B': 20,        # 鸟类环境容纳量
                'd_B': 0.015,     # 鸟类日死亡率

                # 能量转化
                'e': 0.1,         # 能量转化效率

                # 季节性参数
                'A_c': 0.3,       # 作物季节性波动幅度
                'A_I': 0.2,       # 昆虫季节性波动幅度
                'A_B': 0.15,      # 鸟类季节性波动幅度
                'T': 365,         # 季节周期（天）
            }
        else:
            self.params = params

    def seasonality(self, t, species):
        """
        季节性函数

        S(t) = A * sin(2*pi*t/T + phase)
        """
        T = self.params['T']
        if species == 'crop':
            A = self.params['A_c']
            phase = 0  # 春季开始生长
        elif species == 'insect':
            A = self.params['A_I']
            phase = np.pi/4  # 稍后出现
        elif species == 'bird':
            A = self.params['A_B']
            phase = np.pi/2  # 夏季活跃
        else:
            return 0

        return A * np.sin(2 * np.pi * t / T + phase)

    def base_model(self, y, t):
        """
        基础农业生态系统模型（不使用化学物质）

        状态变量：
        y[0]: C - 作物生物量 (kg/m^2)
        y[1]: I - 昆虫种群数量
        y[2]: B - 鸟类种群数量
        """
        C, I, B = y

        p = self.params

        # 季节性调整
        S_C = self.seasonality(t, 'crop')
        S_I = self.seasonality(t, 'insect')
        S_B = self.seasonality(t, 'bird')

        # 作物动力学：Logistic增长 - 昆虫取食
        dCdt = p['r_c'] * C * (1 - C/p['K_c']) * (1 + S_C) - p['a_CI'] * C * I

        # 昆虫动力学：Logistic增长 - 鸟类捕食 - 自然死亡
        dIdt = p['r_I'] * I * (1 - I/p['K_I']) * (1 + S_I) - p['a_IB'] * I * B - p['d_I'] * I

        # 鸟类动力学：Logistic增长 - 自然死亡 + 从捕食中获得能量
        dBdt = p['r_B'] * B * (1 - B/p['K_B']) * (1 + S_B) - p['d_B'] * B + p['e'] * p['a_IB'] * I * B

        return [dCdt, dIdt, dBdt]

    def chemical_model(self, y, t, use_herbicide=True, use_pesticide=True):
        """
        使用化学物质的农业生态系统模型

        额外状态变量：
        y[3]: H - 化学物质浓度 (mg/kg)
        """
        C, I, B, H = y

        p = self.params

        # 季节性调整
        S_C = self.seasonality(t, 'crop')
        S_I = self.seasonality(t, 'insect')
        S_B = self.seasonality(t, 'bird')

        # 化学物质的影响参数
        k_herb_crop = 0.001   # 除草剂对作物的影响（正面，除草效应）
        k_pest_insect = 0.5   # 杀虫剂对昆虫的影响（负面）
        k_chem_bird = 0.01    # 化学物质对鸟类的影响（负面，生物富集）
        k_chem_crop = 0.005   # 化学物质对作物的副作用
        degradation = 0.05    # 化学物质日降解率

        # 作物动力学
        herbicide_effect = k_herb_crop * H if use_herbicide else 0
        chem_damage = k_chem_crop * H
        dCdt = (p['r_c'] * C * (1 - C/p['K_c']) * (1 + S_C)
                - p['a_CI'] * C * I
                + herbicide_effect
                - chem_damage)

        # 昆虫动力学
        pesticide_effect = k_pest_insect * H if use_pesticide else 0
        dIdt = (p['r_I'] * I * (1 - I/p['K_I']) * (1 + S_I)
                - p['a_IB'] * I * B
                - p['d_I'] * I
                - pesticide_effect)

        # 鸟类动力学（生物富集效应）
        bioaccumulation = k_chem_bird * H * B
        dBdt = (p['r_B'] * B * (1 - B/p['K_B']) * (1 + S_B)
                - p['d_B'] * B
                + p['e'] * p['a_IB'] * I * B
                - bioaccumulation)

        # 化学物质动力学（周期性投放）
        application_period = 30  # 每30天投放一次
        if t % application_period < 1:  # 投放日
            input_rate = 10  # 单次投放量
        else:
            input_rate = 0

        dHdt = input_rate - degradation * H

        return [dCdt, dIdt, dBdt, dHdt]

    def bat_model(self, y, t, include_bats=True):
        """
        加入蝙蝠的生态系统模型

        额外状态变量：
        y[3] or y[4]: T - 蝙蝠种群数量
        """
        if include_bats:
            C, I, B, T, H = y
        else:
            C, I, B, H = y
            T = 0

        p = self.params

        # 季节性调整
        S_C = self.seasonality(t, 'crop')
        S_I = self.seasonability(t, 'insect')
        S_B = self.seasonability(t, 'bird')
        S_T = self.seasonability(t, 'bat')

        # 蝙蝠相关参数
        a_IT = 0.02       # 蝙蝠对昆虫的捕食系数
        r_T = 0.015       # 蝙蝠增长率
        K_T = 15          # 蝙蝠环境容纳量
        d_T = 0.01        # 蝙蝠死亡率
        p_pollination = 0.005  # 蝙蝠授粉对作物增长的贡献

        # 作物动力学（加入蝙蝠授粉效应）
        dCdt = (p['r_c'] * C * (1 - C/p['K_c']) * (1 + S_C)
                - p['a_CI'] * C * I
                + p_pollination * T * C)

        # 昆虫动力学（蝙蝠捕食）
        dIdt = (p['r_I'] * I * (1 - I/p['K_I']) * (1 + S_I)
                - p['a_IB'] * I * B
                - a_IT * I * T
                - p['d_I'] * I)

        # 鸟类动力学
        dBdt = (p['r_B'] * B * (1 - B/p['K_B']) * (1 + S_B)
                - p['d_B'] * B
                + p['e'] * p['a_IB'] * I * B)

        # 蝙蝠动力学
        if include_bats:
            dTdt = (r_T * T * (1 - T/K_T) * (1 + S_T)
                    - d_T * T
                    + p['e'] * a_IT * I * T)
        else:
            dTdt = 0

        # 化学物质动力学
        k_chem_bat = 0.008   # 化学物质对蝙蝠的影响
        application_period = 30
        if t % application_period < 1:
            input_rate = 10
        else:
            input_rate = 0
        degradation = 0.05
        dHdt = input_rate - degradation * H

        if include_bats:
            return [dCdt, dIdt, dBdt, dTdt, dHdt]
        else:
            return [dCdt, dIdt, dBdt, dHdt]

    def seasonability(self, t, species):
        """季节性函数（修正拼写）"""
        return self.seasonality(t, species)

    def solve(self, model_type='base', t_span=10*365, y0=None, **kwargs):
        """
        求解模型

        参数：
        - model_type: 'base', 'chemical', 'bat'
        - t_span: 模拟时长（天）
        - y0: 初始条件
        """
        t = np.linspace(0, t_span, t_span)

        if y0 is None:
            # 默认初始条件
            y0 = [20, 10, 5]  # [C, I, B]

        if model_type == 'base':
            func = self.base_model
        elif model_type == 'chemical':
            if len(y0) < 4:
                y0 = list(y0) + [0]  # 添加H=0
            func = lambda y, t: self.chemical_model(y, t, **kwargs)
        elif model_type == 'bat':
            if len(y0) < 5:
                y0 = list(y0) + [0, 0]  # 添加T=0, H=0
            func = lambda y, t: self.bat_model(y, t, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        solution = odeint(func, y0, t)

        return t, solution

    def find_equilibrium(self, model_type='base'):
        """求平衡点"""
        if model_type == 'base':
            func = lambda y: self.base_model(y, 0)
            y0 = [50, 20, 10]
        else:
            func = lambda y: self.chemical_model(y, 0, True, True)
            y0 = [50, 20, 10, 5]

        equilibrium = fsolve(func, y0)
        return equilibrium

    def jacobian(self, y, model_type='base'):
        """计算Jacobian矩阵"""
        eps = 1e-6
        n = len(y)

        if model_type == 'base':
            f = lambda x: self.base_model(x, 0)
        elif model_type == 'chemical':
            f = lambda x: self.chemical_model(x, 0, True, True)
        else:
            f = lambda x: self.bat_model(x, 0, True)

        J = np.zeros((n, n))
        f0 = np.array(f(y))

        for i in range(n):
            y_eps = y.copy()
            y_eps[i] += eps
            f_eps = np.array(f(y_eps))
            J[:, i] = (f_eps - f0) / eps

        return J

    def stability_analysis(self, equilibrium, model_type='base'):
        """稳定性分析"""
        J = self.jacobian(equilibrium, model_type)
        eigenvalues = np.linalg.eigvals(J)

        results = {
            'equilibrium': equilibrium,
            'jacobian': J,
            'eigenvalues': eigenvalues,
            'stable': all(np.real(eigenvalues) < 0),
            'max_real_part': np.max(np.real(eigenvalues))
        }

        return results


def simulate_and_plot():
    """运行模拟并绘制结果图"""

    model = AgroEcosystemModel()

    # ===== 场景1：基础农业生态系统 =====
    print("=" * 60)
    print("场景1：基础农业生态系统模拟")
    print("=" * 60)

    y0_base = [20, 10, 5]  # [C, I, B]
    t_base, sol_base = model.solve('base', t_span=5*365, y0=y0_base)

    # 提取结果
    C_base = sol_base[:, 0]
    I_base = sol_base[:, 1]
    B_base = sol_base[:, 2]

    # 转换为年
    t_years = t_base / 365

    # 绘制基础模型结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 子图1：时间序列
    ax1 = axes[0, 0]
    ax1.plot(t_years, C_base, 'g-', label='作物生物量', linewidth=2)
    ax1.plot(t_years, I_base, 'r-', label='昆虫种群', linewidth=2)
    ax1.plot(t_years, B_base, 'b-', label='鸟类种群', linewidth=2)
    ax1.set_xlabel('时间 (年)', fontsize=12)
    ax1.set_ylabel('种群数量/生物量', fontsize=12)
    ax1.set_title('基础农业生态系统时间演化', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2：相图（作物-昆虫）
    ax2 = axes[0, 1]
    ax2.plot(C_base, I_base, 'purple', linewidth=1.5, alpha=0.7)
    ax2.plot(C_base[0], I_base[0], 'go', markersize=10, label='起点')
    ax2.plot(C_base[-1], I_base[-1], 'rs', markersize=10, label='终点')
    ax1.set_xlabel('作物生物量 (kg/m^2)', fontsize=12)
    ax1.set_ylabel('昆虫种群数量', fontsize=12)
    ax2.set_title('作物-昆虫相图', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 子图3：相图（昆虫-鸟类）
    ax3 = axes[1, 0]
    ax3.plot(I_base, B_base, 'orange', linewidth=1.5, alpha=0.7)
    ax3.plot(I_base[0], B_base[0], 'go', markersize=10, label='起点')
    ax3.plot(I_base[-1], B_base[-1], 'rs', markersize=10, label='终点')
    ax3.set_xlabel('昆虫种群数量', fontsize=12)
    ax3.set_ylabel('鸟类种群数量', fontsize=12)
    ax3.set_title('昆虫-鸟类相图', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 子图4：三年期放大图
    ax4 = axes[1, 1]
    mask = t_years <= 3
    ax4.plot(t_years[mask], C_base[mask], 'g-', label='作物', linewidth=2)
    ax4.plot(t_years[mask], I_base[mask], 'r-', label='昆虫', linewidth=2)
    ax4.plot(t_years[mask], B_base[mask], 'b-', label='鸟类', linewidth=2)
    ax4.set_xlabel('时间 (年)', fontsize=12)
    ax4.set_ylabel('种群数量/生物量', fontsize=12)
    ax4.set_title('前三年详细变化', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure1_base_ecosystem.png',
                dpi=300, bbox_inches='tight')
    print("\n图1已保存: figures/figure1_base_ecosystem.png")

    # 计算统计结果
    print("\n基础模型统计结果（5年）:")
    print(f"作物生物量 - 最终值: {C_base[-1]:.2f} kg/m^2, 平均值: {np.mean(C_base):.2f} kg/m^2")
    print(f"昆虫种群 - 最终值: {I_base[-1]:.2f}, 平均值: {np.mean(I_base):.2f}")
    print(f"鸟类种群 - 最终值: {B_base[-1]:.2f}, 平均值: {np.mean(B_base):.2f}")

    # ===== 场景2：使用化学物质 =====
    print("\n" + "=" * 60)
    print("场景2：使用除草剂/杀虫剂的农业生态系统")
    print("=" * 60)

    y0_chem = [20, 10, 5, 0]  # [C, I, B, H]
    t_chem, sol_chem = model.solve('chemical', t_span=5*365, y0=y0_chem,
                                    use_herbicide=True, use_pesticide=True)

    C_chem = sol_chem[:, 0]
    I_chem = sol_chem[:, 1]
    B_chem = sol_chem[:, 2]
    H_chem = sol_chem[:, 3]

    # 绘制化学物质影响图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 子图1：时间序列对比
    ax1 = axes[0, 0]
    ax1.plot(t_years, C_base, 'g--', label='作物(无化学)', linewidth=2, alpha=0.7)
    ax1.plot(t_years, C_chem, 'g-', label='作物(有化学)', linewidth=2)
    ax1.plot(t_years, I_base, 'r--', label='昆虫(无化学)', linewidth=2, alpha=0.7)
    ax1.plot(t_years, I_chem, 'r-', label='昆虫(有化学)', linewidth=2)
    ax1.set_xlabel('时间 (年)', fontsize=12)
    ax1.set_ylabel('种群数量/生物量', fontsize=12)
    ax1.set_title('化学物质对作物和昆虫的影响', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 子图2：化学物质浓度
    ax2 = axes[0, 1]
    ax2.plot(t_years, H_chem, 'm-', linewidth=2)
    ax2.fill_between(t_years, 0, H_chem, alpha=0.3, color='m')
    ax2.set_xlabel('时间 (年)', fontsize=12)
    ax2.set_ylabel('化学物质浓度 (mg/kg)', fontsize=12)
    ax2.set_title('化学物质浓度变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 子图3：鸟类种群对比
    ax3 = axes[1, 0]
    ax3.plot(t_years, B_base, 'b--', label='鸟类(无化学)', linewidth=2, alpha=0.7)
    ax3.plot(t_years, B_chem, 'b-', label='鸟类(有化学)', linewidth=2)
    ax3.set_xlabel('时间 (年)', fontsize=12)
    ax3.set_ylabel('鸟类种群数量', fontsize=12)
    ax3.set_title('化学物质对鸟类的影响（生物富集）', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 子图4：差异分析
    ax4 = axes[1, 1]
    diff_C = C_chem - C_base
    diff_I = I_chem - I_base
    diff_B = B_chem - B_base
    ax4.plot(t_years, diff_C, 'g-', label='Δ作物', linewidth=2)
    ax4.plot(t_years, diff_I, 'r-', label='Δ昆虫', linewidth=2)
    ax4.plot(t_years, diff_B, 'b-', label='Δ鸟类', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('时间 (年)', fontsize=12)
    ax4.set_ylabel('种群差异', fontsize=12)
    ax4.set_title('有无化学物质的种群差异', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure2_chemical_impact.png',
                dpi=300, bbox_inches='tight')
    print("\n图2已保存: figures/figure2_chemical_impact.png")

    # 化学物质影响统计
    print("\n化学物质影响统计（5年）:")
    print(f"作物生物量变化: {diff_C[-1]:.2f} kg/m^2")
    print(f"昆虫种群变化: {diff_I[-1]:.2f}")
    print(f"鸟类种群变化: {diff_B[-1]:.2f}")
    print(f"平均化学物质浓度: {np.mean(H_chem):.2f} mg/kg")

    # ===== 场景3：加入蝙蝠 =====
    print("\n" + "=" * 60)
    print("场景3：加入蝙蝠的生态系统")
    print("=" * 60)

    y0_bat = [20, 10, 5, 0, 0]  # [C, I, B, T, H]
    t_bat, sol_bat = model.solve('bat', t_span=5*365, y0=y0_bat, include_bats=True)

    C_bat = sol_bat[:, 0]
    I_bat = sol_bat[:, 1]
    B_bat = sol_bat[:, 2]
    T_bat = sol_bat[:, 3]

    # 绘制蝙蝠模型结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 子图1：时间序列
    ax1 = axes[0, 0]
    ax1.plot(t_years, C_bat, 'g-', label='作物', linewidth=2)
    ax1.plot(t_years, I_bat, 'r-', label='昆虫', linewidth=2)
    ax1.plot(t_years, B_bat, 'b-', label='鸟类', linewidth=2)
    ax1.plot(t_years, T_bat, 'm-', label='蝙蝠', linewidth=2)
    ax1.set_xlabel('时间 (年)', fontsize=12)
    ax1.set_ylabel('种群数量/生物量', fontsize=12)
    ax1.set_title('引入蝙蝠后的生态系统演化', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2：蝙蝠对昆虫的控制效果
    ax2 = axes[0, 1]
    ax2.plot(t_years, I_chem, 'r--', label='昆虫(无蝙蝠)', linewidth=2)
    ax2.plot(t_years, I_bat, 'r-', label='昆虫(有蝙蝠)', linewidth=2)
    ax2.fill_between(t_years, I_bat, I_chem, where=I_bat < I_chem,
                     alpha=0.3, color='red', label='控制区域')
    ax2.set_xlabel('时间 (年)', fontsize=12)
    ax2.set_ylabel('昆虫种群数量', fontsize=12)
    ax2.set_title('蝙蝠对昆虫的控制效果', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 子图3：作物产量对比
    ax3 = axes[1, 0]
    ax3.plot(t_years, C_chem, 'g--', label='作物(无蝙蝠)', linewidth=2)
    ax3.plot(t_years, C_bat, 'g-', label='作物(有蝙蝠)', linewidth=2)
    ax3.set_xlabel('时间 (年)', fontsize=12)
    ax3.set_ylabel('作物生物量 (kg/m^2)', fontsize=12)
    ax3.set_title('蝙蝠授粉对作物产量的影响', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 子图4：蝙蝠种群动态
    ax4 = axes[1, 1]
    ax4.plot(t_years, T_bat, 'm-', linewidth=2)
    ax4.fill_between(t_years, 0, T_bat, alpha=0.3, color='m')
    ax4.set_xlabel('时间 (年)', fontsize=12)
    ax4.set_ylabel('蝙蝠种群数量', fontsize=12)
    ax4.set_title('蝙蝠种群动态', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/86198/Desktop/E/figures/figure3_bat_introduction.png',
                dpi=300, bbox_inches='tight')
    print("\n图3已保存: figures/figure3_bat_introduction.png")

    # 蝙蝠影响统计
    print("\n蝙蝠引入影响统计（5年）:")
    print(f"昆虫种群减少: {(I_chem[-1] - I_bat[-1]):.2f} ({100*(I_chem[-1] - I_bat[-1])/I_chem[-1]:.1f}%)")
    print(f"作物生物量增加: {(C_bat[-1] - C_chem[-1]):.2f} kg/m^2")
    print(f"蝙蝠最终种群: {T_bat[-1]:.2f}")

    # ===== 结果表格 =====
    print("\n" + "=" * 60)
    print("生成结果表格")
    print("=" * 60)

    results_data = {
        '场景': ['基础生态系统', '使用化学物质', '引入蝙蝠'],
        '作物生物量 (kg/m^2)': [f'{C_base[-1]:.2f}', f'{C_chem[-1]:.2f}', f'{C_bat[-1]:.2f}'],
        '昆虫种群': [f'{I_base[-1]:.2f}', f'{I_chem[-1]:.2f}', f'{I_bat[-1]:.2f}'],
        '鸟类种群': [f'{B_base[-1]:.2f}', f'{B_chem[-1]:.2f}', f'{B_bat[-1]:.2f}'],
        '化学物质浓度 (mg/kg)': ['0', f'{H_chem[-1]:.2f}', '0'],
    }

    if 'T_bat' in locals():
        results_data['蝙蝠种群'] = ['0', '0', f'{T_bat[-1]:.2f}']

    df_results = pd.DataFrame(results_data)
    df_results.to_csv('C:/Users/86198/Desktop/E/data/results.csv', index=False, encoding='utf-8-sig')
    print("\n结果表格已保存: data/results.csv")
    print("\n" + str(df_results))

    return {
        'base': (t_years, sol_base),
        'chemical': (t_years, sol_chem),
        'bat': (t_years, sol_bat)
    }


if __name__ == '__main__':
    results = simulate_and_plot()
    print("\n" + "=" * 60)
    print("模拟完成！所有图表已保存到 figures/ 目录")
    print("=" * 60)
