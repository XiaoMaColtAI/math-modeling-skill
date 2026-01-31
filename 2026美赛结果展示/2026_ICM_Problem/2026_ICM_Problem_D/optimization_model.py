# -*- coding: utf-8 -*-
"""
多目标优化模型模块
Multi-Objective Optimization Module
用于球队阵容优化和利润最大化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2


class RosterOptimizationModel:
    """球队阵容优化模型"""

    def __init__(self, salary_cap=1500000, min_salary=70000, roster_size=12):
        """
        初始化优化模型

        Args:
            salary_cap: 薪资帽（总额）
            min_salary: 最低薪资
            roster_size: 阵容大小
        """
        self.salary_cap = salary_cap
        self.min_salary = min_salary
        self.roster_size = roster_size

        # 位置需求
        self.position_requirements = {
            'PG': (2, 3),   # 最少2个，最多3个
            'SG': (2, 3),
            'SF': (2, 3),
            'PF': (2, 3),
            'C': (2, 3)
        }

    def calculate_team_performance(self, selected_players, weights=None):
        """
        计算球队竞技表现

        Args:
            selected_players: 选中的球员DataFrame
            weights: 各指标的权重

        Returns:
            performance_score: 球队表现得分
        """
        if weights is None:
            weights = {
                'PER': 0.3,
                'WS': 0.25,
                'VORP': 0.2,
                'PPG': 0.1,
                'RPG': 0.05,
                'APG': 0.05,
                'SPG': 0.025,
                'BPG': 0.025
            }

        # 加权平均计算球队表现
        performance = 0
        for indicator, weight in weights.items():
            if indicator in selected_players.columns:
                # 使用加权平均，主力球员权重更高
                if 'Minutes_Per_Game' in selected_players.columns:
                    minutes_weight = selected_players['Minutes_Per_Game'] / \
                                   selected_players['Minutes_Per_Game'].sum()
                    performance += (selected_players[indicator] * minutes_weight).sum() * weight
                else:
                    performance += selected_players[indicator].mean() * weight

        return performance

    def calculate_team_value(self, selected_players):
        """计算球队价值（基于球员综合价值）"""
        if '综合价值得分' in selected_players.columns:
            return selected_players['综合价值得分'].sum()
        # 如果没有综合价值得分，使用PER近似
        return selected_players['PER'].sum()

    def calculate_profit(self, selected_players, team_financial_data, weights=None):
        """
        计算球队利润

        Args:
            selected_players: 选中的球员
            team_financial_data: 球队财务数据
            weights: 权重参数

        Returns:
            profit: 预期利润
        """
        if weights is None:
            weights = {
                'win_rate_impact': 0.5,  # 胜率对收入的影响
                'star_power': 0.3,       # 明星效应
                'efficiency': 0.2        # 性价比
            }

        # 总薪资成本
        total_salary = selected_players['Salary'].sum()

        # 球队表现（简化：使用PER总和）
        team_performance = selected_players['PER'].sum()

        # 明星球员数量
        stars = ((selected_players['PER'] >= 20) &
                (selected_players['All_Star_Appearances'] >= 1)).sum()

        # 基础收入
        base_revenue = team_financial_data.get('Total_Revenue_M', 50) * 1000000

        # 表现带来的收入增量
        performance_bonus = base_revenue * 0.3 * (team_performance / 200)

        # 明星效应带来的收入增量
        star_bonus = base_revenue * 0.1 * stars

        # 总收入
        total_revenue = base_revenue + performance_bonus + star_bonus

        # 其他成本（运营成本等）
        other_costs = team_financial_data.get('Operating_Costs_M', 15) * 1000000

        # 利润
        profit = total_revenue - total_salary - other_costs

        return profit

    def optimize_roster_single_objective(self, available_players, team_financial_data,
                                        objective='profit', lambda_weight=0.5):
        """
        单目标优化：最大化利润或表现

        Args:
            available_players: 可用球员池
            team_financial_data: 球队财务数据
            objective: 优化目标 ('profit' 或 'performance')
            lambda_weight: 利润和表现的权重（用于组合目标）

        Returns:
            optimal_roster: 最优阵容
            result: 优化结果
        """
        n_players = len(available_players)

        # 简化方法：贪心算法选择TOP球员
        # 根据目标函数计算每个球员的性价比

        if objective == 'profit':
            # 利润导向：PER/Salary 越高越好
            available_players = available_players.copy()
            available_players['value_ratio'] = available_players['PER'] / (available_players['Salary'] + 1)
        elif objective == 'performance':
            # 表现导向：直接使用PER
            available_players = available_players.copy()
            available_players['value_ratio'] = available_players['PER']
        else:
            # 组合目标
            available_players = available_players.copy()
            per_normalized = (available_players['PER'] - available_players['PER'].min()) / \
                            (available_players['PER'].max() - available_players['PER'].min() + 1e-10)
            salary_normalized = (available_players['Salary'].max() - available_players['Salary']) / \
                              (available_players['Salary'].max() - available_players['Salary'].min() + 1e-10)
            available_players['value_ratio'] = lambda_weight * salary_normalized + \
                                                (1 - lambda_weight) * per_normalized

        # 按价值排序
        available_players = available_players.sort_values('value_ratio', ascending=False)

        # 贪心选择球员，满足约束
        selected_players = []
        total_salary = 0
        position_counts = {pos: 0 for pos in self.position_requirements.keys()}

        for idx, player in available_players.iterrows():
            # 检查是否已满
            if len(selected_players) >= self.roster_size:
                break

            # 检查薪资约束
            if total_salary + player['Salary'] > self.salary_cap:
                continue

            # 检查位置约束
            pos = player['Position']
            min_pos, max_pos = self.position_requirements.get(pos, (0, 5))
            if position_counts[pos] >= max_pos:
                continue

            # 添加球员
            selected_players.append(player)
            total_salary += player['Salary']
            position_counts[pos] += 1

        # 创建最优阵容DataFrame
        if len(selected_players) > 0:
            optimal_roster = pd.DataFrame(selected_players)

            # 计算结果指标
            profit = self.calculate_profit(optimal_roster, team_financial_data)
            performance = self.calculate_team_performance(optimal_roster)
            total_salary = optimal_roster['Salary'].sum()

            result = {
                'success': True,
                'profit': profit,
                'performance': performance,
                'total_salary': total_salary,
                'n_players': len(optimal_roster)
            }
        else:
            optimal_roster = available_players.head(self.roster_size)
            result = {'success': False}

        return optimal_roster, result

    def optimize_pareto_front(self, available_players, team_financial_data, n_solutions=30):
        """
        求解帕累托前沿（利润 vs 表现）

        Args:
            available_players: 可用球员池
            team_financial_data: 球队财务数据
            n_solutions: 帕累托解的数量

        Returns:
            pareto_solutions: 帕累托解集
        """
        solutions = []

        for lam in np.linspace(0, 1, n_solutions):
            roster, result = self.optimize_roster_single_objective(
                available_players, team_financial_data,
                objective='combined', lambda_weight=lam
            )

            if result.get('success', False):
                solutions.append({
                    'lambda': lam,
                    'roster': roster,
                    'profit': result['profit'],
                    'performance': result['performance']
                })

        # 转换为DataFrame
        if solutions:
            solutions_df = pd.DataFrame([{
                'Lambda': s['lambda'],
                'Profit_M': s['profit'] / 1e6,
                'Performance': s['performance'],
                'Total_Salary_M': s['roster']['Salary'].sum() / 1e6,
                'PER_Sum': s['roster']['PER'].sum(),
                'Players': ', '.join(s['roster']['Name'].tolist())
            } for s in solutions])
        else:
            # 如果没有有效解，返回空DataFrame
            solutions_df = pd.DataFrame(columns=['Lambda', 'Profit_M', 'Performance',
                                               'Total_Salary_M', 'PER_Sum', 'Players'])

        return solutions_df

    def plot_pareto_front(self, pareto_solutions, save_dir='figures'):
        """绘制帕累托前沿"""
        os.makedirs(save_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：帕累托前沿（利润 vs 表现）
        scatter = ax1.scatter(pareto_solutions['Performance'],
                             pareto_solutions['Profit_M'],
                             c=pareto_solutions['Lambda'],
                             cmap='RdYlGn', s=100, alpha=0.7,
                             edgecolors='black', linewidth=1)

        ax1.set_xlabel('球队表现得分 (Performance)', fontsize=12)
        ax1.set_ylabel('利润 (百万美元)', fontsize=12)
        ax1.set_title('帕累托前沿: 利润 vs 表现权衡', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('利润权重 λ', fontsize=10)

        # 标注极端点
        max_profit_idx = pareto_solutions['Profit_M'].idxmax()
        max_perf_idx = pareto_solutions['Performance'].idxmax()

        ax1.annotate('最大利润',
                    xy=(pareto_solutions.loc[max_profit_idx, 'Performance'],
                        pareto_solutions.loc[max_profit_idx, 'Profit_M']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'),
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax1.annotate('最佳表现',
                    xy=(pareto_solutions.loc[max_perf_idx, 'Performance'],
                        pareto_solutions.loc[max_perf_idx, 'Profit_M']),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'),
                    arrowprops=dict(arrowstyle='->', color='blue'))

        # 右图：薪资分布
        ax2.plot(pareto_solutions['Lambda'], pareto_solutions['Total_Salary_M'],
                'o-', linewidth=2, markersize=8, color='#1f77b4', label='总薪资')
        ax2.axhline(y=self.salary_cap / 1e6, color='red', linestyle='--',
                   linewidth=2, label=f'薪资帽 (${self.salary_cap/1e6:.1f}M)')
        ax2.set_xlabel('利润权重 λ', fontsize=12)
        ax2.set_ylabel('总薪资 (百万美元)', fontsize=12)
        ax2.set_title('薪资随权重变化', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure3_pareto_front.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"帕累托前沿图已保存到 {save_dir}/")


class RevenueOptimizationModel:
    """票务收入优化模型"""

    def __init__(self, capacity=18000, base_price=80):
        """
        初始化收入优化模型

        Args:
            capacity: 场馆容量
            base_price: 基础票价
        """
        self.capacity = capacity
        self.base_price = base_price

        # 需求函数参数
        self.price_elasticity = -0.8  # 价格弹性
        self.opponent_factor = 1.2    # 强对手带来的需求增加
        self.weekend_factor = 1.15    # 周末需求增加
        self.marketing_factor = 0.05  # 营销投入效果

    def demand_function(self, price, opponent_strength=1, is_weekend=False, marketing=0):
        """
        需求函数

        Args:
            price: 票价
            opponent_strength: 对手实力（0.5-1.5）
            is_weekend: 是否周末
            marketing: 营销投入（千元）

        Returns:
            demand: 需求量
        """
        # 基础需求
        base_demand = self.capacity * (self.base_price / price) ** abs(self.price_elasticity)

        # 调整因子
        adjustments = 1.0
        adjustments *= opponent_strength
        if is_weekend:
            adjustments *= self.weekend_factor
        adjustments *= (1 + self.marketing_factor * marketing)

        demand = base_demand * adjustments

        return min(demand, self.capacity)

    def calculate_revenue(self, price, opponent_strength=1, is_weekend=False, marketing=0):
        """
        计算票务收入

        Returns:
            revenue: 收入
            costs: 成本（营销成本）
            profit: 利润
        """
        demand = self.demand_function(price, opponent_strength, is_weekend, marketing)
        revenue = price * demand
        marketing_cost = marketing * 1000  # 转换为美元
        profit = revenue - marketing_cost

        return revenue, marketing_cost, profit

    def optimize_single_game_pricing(self, opponent_strength=1, is_weekend=False, max_marketing=50):
        """
        优化单场比赛定价

        Args:
            opponent_strength: 对手实力
            is_weekend: 是否周末
            max_marketing: 最大营销投入（千元）

        Returns:
            optimal_price: 最优价格
            optimal_marketing: 最优营销投入
            results: 结果字典
        """
        best_profit = -np.inf
        best_price = self.base_price
        best_marketing = 0

        # 网格搜索
        prices = np.linspace(30, 200, 100)
        marketing_levels = np.linspace(0, max_marketing, 20)

        for price in prices:
            for marketing in marketing_levels:
                revenue, cost, profit = self.calculate_revenue(
                    price, opponent_strength, is_weekend, marketing
                )
                if profit > best_profit:
                    best_profit = profit
                    best_price = price
                    best_marketing = marketing

        # 计算最优结果
        revenue, cost, profit = self.calculate_revenue(
            best_price, opponent_strength, is_weekend, best_marketing
        )
        attendance = self.demand_function(
            best_price, opponent_strength, is_weekend, best_marketing
        )

        results = {
            'optimal_price': best_price,
            'optimal_marketing': best_marketing,
            'revenue': revenue,
            'cost': cost,
            'profit': profit,
            'attendance': attendance,
            'occupancy_rate': attendance / self.capacity
        }

        return best_price, best_marketing, results

    def plot_pricing_analysis(self, save_dir='figures'):
        """绘制定价分析图（拆分为2个子图）"""
        os.makedirs(save_dir, exist_ok=True)

        prices = np.linspace(30, 200, 100)

        # ========== 图4a：需求曲线与收入曲线 ==========
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：需求曲线
        demands = [self.demand_function(p, 1, False, 0) / 1000 for p in prices]
        ax1.plot(prices, demands, linewidth=2, color='#1f77b4')
        ax1.fill_between(prices, 0, demands, alpha=0.3)
        ax1.set_xlabel('票价 ($)', fontsize=12)
        ax1.set_ylabel('上座人数 (千)', fontsize=12)
        ax1.set_title('(a) 需求曲线', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 右图：收入曲线
        revenues = [self.calculate_revenue(p, 1, False, 0)[0] / 1e6 for p in prices]
        ax2.plot(prices, revenues, linewidth=2, color='#ff7f0e')
        max_rev_idx = np.argmax(revenues)
        ax2.axvline(x=prices[max_rev_idx], color='red', linestyle='--',
                   label=f'最优价格: ${prices[max_rev_idx]:.0f}')
        ax2.set_xlabel('票价 ($)', fontsize=12)
        ax2.set_ylabel('收入 (百万美元)', fontsize=12)
        ax2.set_title('(b) 收入曲线', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure4a_pricing_demand_revenue.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # ========== 图4b：场景定价与弹性分析 ==========
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：不同场景的最优定价
        scenarios = [
            ('弱对手-工作日', 0.7, False),
            ('普通对手-工作日', 1.0, False),
            ('强对手-工作日', 1.3, False),
            ('强对手-周末', 1.3, True)
        ]

        optimal_prices = []
        labels = []
        for label, opp, weekend in scenarios:
            price, _, _ = self.optimize_single_game_pricing(opp, weekend)
            optimal_prices.append(price)
            labels.append(label)

        bars = ax1.bar(labels, optimal_prices, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        ax1.set_ylabel('最优票价 ($)', fontsize=12)
        ax1.set_title('(a) 不同场景的最优定价', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

        # 添加数值标签
        for bar, price in zip(bars, optimal_prices):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'${price:.0f}', ha='center', fontsize=9)

        # 右图：价格弹性分析
        elasticities = [-0.4, -0.6, -0.8, -1.0, -1.2]
        opt_prices = []
        opt_profits = []

        for elast in elasticities:
            self.price_elasticity = elast
            price, _, res = self.optimize_single_game_pricing()
            opt_prices.append(price)
            opt_profits.append(res['profit'] / 1e6)

        ax2.plot(elasticities, opt_prices, 'o-', linewidth=2, markersize=8,
                color='#1f77b4', label='最优价格')
        ax2.set_xlabel('价格弹性', fontsize=12)
        ax2.set_ylabel('最优票价 ($)', fontsize=12)
        ax2.set_title('(b) 价格弹性对最优定价的影响', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure4b_pricing_scenarios_elasticity.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"定价分析图已保存到 {save_dir}/")


def main():
    """主函数：测试优化模型"""
    from data_loader import DataLoader

    # 加载数据
    loader = DataLoader()
    players, teams, financial = loader.load_data()

    # 选择一支球队进行分析
    target_team = 'Las Vegas Aces'
    print(f"分析球队: {target_team}")

    # 获取该球队球员和自由球员池
    team_players = loader.get_team_players(target_team)
    free_agents = loader.get_free_agents(target_team)

    # 合并当前球员和自由球员
    available_pool = pd.concat([team_players, free_agents], ignore_index=True)

    # 添加综合价值得分（如果还没有）
    from player_evaluation import PlayerValuationModel
    evaluator = PlayerValuationModel()
    available_pool = evaluator.evaluate_players(available_pool)

    # 获取该球队的财务数据
    team_financial = financial[financial['Team'] == target_team].iloc[0].to_dict()

    # 创建优化模型
    roster_model = RosterOptimizationModel(salary_cap=1500000, roster_size=12)

    print("\n求解帕累托前沿...")
    pareto_solutions = roster_model.optimize_pareto_front(
        available_pool, team_financial, n_solutions=20
    )

    # 绘制帕累托前沿
    roster_model.plot_pareto_front(pareto_solutions)

    # 找到平衡解
    pareto_solutions['Composite_Score'] = (
        pareto_solutions['Profit_M'] / pareto_solutions['Profit_M'].max() +
        pareto_solutions['Performance'] / pareto_solutions['Performance'].max()
    ) / 2

    balanced_solution = pareto_solutions.loc[pareto_solutions['Composite_Score'].idxmax()]

    print("\n最优平衡方案:")
    print(f"  利润权重: {balanced_solution['Lambda']:.2f}")
    print(f"  预期利润: ${balanced_solution['Profit_M']:.2f}M")
    print(f"  球队表现: {balanced_solution['Performance']:.2f}")
    print(f"  总薪资: ${balanced_solution['Total_Salary_M']:.2f}M")
    print(f"  PER总和: {balanced_solution['PER_Sum']:.2f}")

    # 票务定价优化
    print("\n\n票务定价优化...")
    pricing_model = RevenueOptimizationModel(capacity=18000, base_price=80)
    pricing_model.plot_pricing_analysis()

    # 保存结果
    os.makedirs('results', exist_ok=True)
    pareto_solutions.to_csv('results/pareto_solutions.csv', index=False, encoding='utf-8-sig')

    print("\n优化完成！结果已保存到 results/ 目录")

    return pareto_solutions, balanced_solution


if __name__ == "__main__":
    main()
