# -*- coding: utf-8 -*-
"""
风险分析模块
Risk Analysis Module
使用蒙特卡洛模拟分析伤病等风险
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2


class InjuryRiskModel:
    """伤病风险模型"""

    def __init__(self):
        """初始化伤病风险模型"""
        # 位置基础伤病概率
        self.position_base_prob = {
            'PG': 0.08,
            'SG': 0.09,
            'SF': 0.10,
            'PF': 0.12,
            'C': 0.15
        }

    def calculate_injury_probability(self, player):
        """
        计算球员伤病概率

        Args:
            player: 球员数据（Series或DataFrame行）

        Returns:
            probability: 赛季伤病概率
        """
        base_prob = self.position_base_prob.get(player.get('Position', 'SF'), 0.10)

        # 年龄因子
        age = player.get('Age', 27)
        age_factor = 1 + (age - 27) * 0.05 if age > 27 else 1

        # 历史伤病因子
        injury_count = player.get('Injury_Count', 0)
        history_factor = 1 + injury_count * 0.15

        # 工作量因子（上场时间）
        minutes = player.get('Minutes_Per_Game', 25)
        workload_factor = 1 + (minutes - 25) * 0.01

        # 综合概率
        probability = base_prob * age_factor * history_factor * workload_factor

        return min(probability, 0.8)  # 最多80%

    def calculate_injury_impact(self, player, games_missed):
        """
        计算伤病对球队的影响

        Args:
            player: 球员数据
            games_missed: 错过的比赛数

        Returns:
            impact: 影响程度（0-1）
        """
        # 基于球员价值计算影响
        per = player.get('PER', 15)
        vorp = player.get('VORP', 0)

        # 价值越高的球员，伤病影响越大
        value_impact = (per + vorp * 5) / 50

        # 错过的比赛越多，影响越大
        games_impact = games_missed / 40  # 假设赛季40场

        impact = value_impact * games_impact

        return min(impact, 1)


class MonteCarloSimulator:
    """蒙特卡洛模拟器"""

    def __init__(self, n_simulations=1000, random_seed=42):
        """
        初始化模拟器

        Args:
            n_simulations: 模拟次数
            random_seed: 随机种子
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.injury_model = InjuryRiskModel()

    def simulate_season(self, roster, team_financial_data):
        """
        模拟一个赛季

        Args:
            roster: 球队阵容DataFrame
            team_financial_data: 球队财务数据

        Returns:
            result: 模拟结果字典
        """
        np.random.seed(self.random_seed)

        # 计算球员伤病概率
        injury_probs = roster.apply(self.injury_model.calculate_injury_probability, axis=1)

        # 模拟伤病发生
        injuries_occurred = np.random.random(len(roster)) < injury_probs.values

        # 模拟错过的比赛数
        games_missed = np.zeros(len(roster))
        for i, injured in enumerate(injuries_occurred):
            if injured:
                # 错过5-35场比赛（均匀分布）
                games_missed[i] = np.random.randint(5, 36)

        # 计算伤病影响
        total_impact = 0
        for i, (idx, player) in enumerate(roster.iterrows()):
            impact = self.injury_model.calculate_injury_impact(player, games_missed[i])
            total_impact += impact

        # 计算球队表现
        base_per_sum = roster['PER'].sum()
        actual_performance = base_per_sum * (1 - total_impact * 0.3)

        # 计算收入（表现影响收入）
        base_revenue = team_financial_data.get('Total_Revenue_M', 50) * 1e6
        performance_factor = actual_performance / base_per_sum
        revenue = base_revenue * (0.7 + 0.3 * performance_factor)

        # 计算利润
        total_salary = roster['Salary'].sum()
        other_costs = team_financial_data.get('Operating_Costs_M', 15) * 1e6
        profit = revenue - total_salary - other_costs

        return {
            'performance': actual_performance,
            'revenue': revenue,
            'profit': profit,
            'injuries_count': injuries_occurred.sum(),
            'games_missed_total': games_missed.sum()
        }

    def run_simulation(self, roster, team_financial_data):
        """
        运行多次蒙特卡洛模拟

        Args:
            roster: 球队阵容
            team_financial_data: 财务数据

        Returns:
            results: 模拟结果DataFrame
        """
        results_list = []

        for i in range(self.n_simulations):
            self.random_seed = i + 42
            result = self.simulate_season(roster, team_financial_data)
            result['simulation'] = i
            results_list.append(result)

        results_df = pd.DataFrame(results_list)

        return results_df

    def calculate_risk_metrics(self, results):
        """
        计算风险指标

        Args:
            results: 模拟结果DataFrame

        Returns:
            metrics: 风险指标字典
        """
        profits = results['profit'].values / 1e6  # 转换为百万美元
        performances = results['performance'].values

        # VaR和CVaR（95%置信度）
        var_95 = np.percentile(profits, 5)
        cvar_95 = profits[profits <= var_95].mean()

        # 统计指标
        metrics = {
            'profit_mean': profits.mean(),
            'profit_std': profits.std(),
            'profit_median': np.median(profits),
            'profit_min': profits.min(),
            'profit_max': profits.max(),
            'profit_percentile_5': np.percentile(profits, 5),
            'profit_percentile_25': np.percentile(profits, 25),
            'profit_percentile_75': np.percentile(profits, 75),
            'profit_percentile_95': np.percentile(profits, 95),
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'performance_mean': performances.mean(),
            'performance_std': performances.std(),
            'injury_probability': results['injuries_count'].apply(lambda x: x > 0).mean(),
            'avg_injuries_per_season': results['injuries_count'].mean(),
            'avg_games_missed': results['games_missed_total'].mean()
        }

        return metrics

    def plot_risk_analysis(self, results, save_dir='figures'):
        """绘制风险分析图（拆分为2个子图）"""
        os.makedirs(save_dir, exist_ok=True)

        profits = results['profit'].values / 1e6
        performances = results['performance'].values

        # ========== 图5a：利润分布与累积分布 ==========
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：利润分布直方图
        ax1.hist(profits, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(profits), color='red', linestyle='--',
                   linewidth=2, label=f'均值: ${np.mean(profits):.2f}M')
        ax1.axvline(x=np.percentile(profits, 5), color='orange', linestyle='--',
                   linewidth=2, label=f'5%分位: ${np.percentile(profits, 5):.2f}M')
        ax1.set_xlabel('利润 (百万美元)', fontsize=12)
        ax1.set_ylabel('频数', fontsize=12)
        ax1.set_title('(a) 利润分布', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 右图：累积分布函数
        sorted_profits = np.sort(profits)
        cumulative = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
        ax2.plot(sorted_profits, cumulative, linewidth=2, color='#ff7f0e')
        ax2.fill_between(sorted_profits, 0, cumulative, alpha=0.3)
        ax2.axvline(x=np.percentile(profits, 5), color='red', linestyle='--',
                   linewidth=2, label='VaR (95%)')
        ax2.set_xlabel('利润 (百万美元)', fontsize=12)
        ax2.set_ylabel('累积概率', fontsize=12)
        ax2.set_title('(b) 利润累积分布函数', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure5a_risk_distribution.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # ========== 图5b：表现-利润关系与伤病分析 ==========
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：利润 vs 表现散点图
        scatter = ax1.scatter(performances, profits, alpha=0.5, s=20,
                             c=results['injuries_count'], cmap='YlOrRd')
        ax1.set_xlabel('球队表现 (PER总和)', fontsize=12)
        ax1.set_ylabel('利润 (百万美元)', fontsize=12)
        ax1.set_title('(a) 表现 vs 利润 (颜色=伤病人数)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('伤病球员数', fontsize=10)

        # 右图：伤病次数分布
        injuries_counts = results['injuries_count'].values
        ax2.hist(injuries_counts, bins=range(injuries_counts.min(), injuries_counts.max() + 2),
                color='#d62728', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('伤病球员数', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title('(b) 伤病次数分布', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure5b_risk_performance_injury.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"风险分析图已保存到 {save_dir}/")


class LeagueExpansionModel:
    """联盟扩张影响模型"""

    def __init__(self):
        """初始化联盟扩张模型"""
        # 新球队位置及其市场规模排名
        self.expansion_locations = {
            'Toronto': {'market_rank': 2, 'population_M': 6.2, 'competition': 'High'},
            'Portland': {'market_rank': 18, 'population_M': 2.5, 'competition': 'Medium'},
            'Sacramento': {'market_rank': 20, 'population_M': 2.3, 'competition': 'Low'},
            'San Diego': {'market_rank': 8, 'population_M': 3.3, 'competition': 'High'},
        }

    def analyze_expansion_impact(self, current_teams_data, expansion_location):
        """
        分析联盟扩张的影响

        Args:
            current_teams_data: 当前球队数据
            expansion_location: 扩张位置

        Returns:
            impact_analysis: 影响分析结果
        """
        location_info = self.expansion_locations[expansion_location]

        # 1. 对球员市场的影响
        # 新球队需要12-15名球员，会从自由球员池中抽取
        players_needed = np.random.randint(12, 16)
        talent_reduction = players_needed / len(current_teams_data) * 100

        # 2. 对薪资的影响
        # 新球队会推高薪资（需求增加）
        salary_increase_pct = players_needed * 0.5  # 每个球员约0.5%的涨幅

        # 3. 对媒体收入的影响
        # 新市场可能稀释原有媒体收入
        media_revenue_impact = -1.0 * location_info['market_rank'] / 10  # 市场越大影响越大

        # 4. 对竞争平衡的影响
        # 新球队初期较弱，可能影响联盟竞争平衡
        compet_balance_impact = 'Decrease' if location_info['competition'] == 'Low' else 'Neutral'

        impact_analysis = {
            'expansion_location': expansion_location,
            'market_size_rank': location_info['market_rank'],
            'players_needed': players_needed,
            'talent_reduction_pct': talent_reduction,
            'salary_increase_pct': salary_increase_pct,
            'media_revenue_impact_pct': media_revenue_impact,
            'compet_balance_change': compet_balance_impact
        }

        return impact_analysis

    def compare_expansion_scenarios(self, current_teams_data):
        """比较不同扩张场景的影响"""
        scenarios = []
        for location in self.expansion_locations.keys():
            impact = self.analyze_expansion_impact(current_teams_data, location)
            scenarios.append(impact)

        return pd.DataFrame(scenarios)


def main():
    """主函数：测试风险分析模型"""
    from data_loader import DataLoader
    from player_evaluation import PlayerValuationModel

    # 加载数据
    loader = DataLoader()
    players, teams, financial = loader.load_data()

    # 选择球队并获取阵容
    target_team = 'Las Vegas Aces'
    team_players = loader.get_team_players(target_team)

    # 评价球员
    evaluator = PlayerValuationModel()
    team_players = evaluator.evaluate_players(team_players)

    # 选择TOP 12作为主要阵容
    roster = team_players.head(12)

    team_financial = financial[financial['Team'] == target_team].iloc[0].to_dict()

    # 运行蒙特卡洛模拟
    print(f"对 {target_team} 进行风险分析...")
    simulator = MonteCarloSimulator(n_simulations=1000)
    results = simulator.run_simulation(roster, team_financial)

    # 计算风险指标
    metrics = simulator.calculate_risk_metrics(results)

    print("\n风险分析结果:")
    print(f"  利润均值: ${metrics['profit_mean']:.2f}M")
    print(f"  利润标准差: ${metrics['profit_std']:.2f}M")
    print(f"  VaR (95%): ${metrics['VaR_95']:.2f}M")
    print(f"  CVaR (95%): ${metrics['CVaR_95']:.2f}M")
    print(f"  伤病概率: {metrics['injury_probability']:.1%}")
    print(f"  平均伤病球员数: {metrics['avg_injuries_per_season']:.1f}")

    # 绘制风险分析图
    simulator.plot_risk_analysis(results)

    # 联盟扩张分析
    print("\n\n联盟扩张影响分析:")
    expansion_model = LeagueExpansionModel()
    expansion_scenarios = expansion_model.compare_expansion_scenarios(players)

    print("\n不同扩张位置的影响比较:")
    print(expansion_scenarios.to_string(index=False))

    # 绘制扩张影响图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：球员人才流失率
    ax1.bar(expansion_scenarios['expansion_location'],
           expansion_scenarios['talent_reduction_pct'],
           color='#d62728', alpha=0.7)
    ax1.set_ylabel('人才流失率 (%)', fontsize=12)
    ax1.set_title('不同扩张位置的人才流失影响', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

    # 右图：综合影响对比
    locations = expansion_scenarios['expansion_location']
    x = np.arange(len(locations))
    width = 0.35

    ax2.bar(x - width/2, expansion_scenarios['salary_increase_pct'],
            width, label='薪资增长%', color='#1f77b4')
    ax2.bar(x + width/2, -expansion_scenarios['media_revenue_impact_pct'],
            width, label='媒体收入影响%', color='#ff7f0e')

    ax2.set_ylabel('百分比 (%)', fontsize=12)
    ax2.set_title('薪资与媒体收入影响对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(locations, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.axhline(y=0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig('figures/figure6_expansion_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存结果
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/monte_carlo_results.csv', index=False, encoding='utf-8-sig')
    expansion_scenarios.to_csv('results/expansion_impact.csv', index=False, encoding='utf-8-sig')

    # 创建风险指标报告
    with open('results/risk_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("风险分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"球队: {target_team}\n")
        f.write(f"模拟次数: {len(results)}\n\n")
        f.write("利润指标:\n")
        f.write(f"  均值: ${metrics['profit_mean']:.2f}M\n")
        f.write(f"  标准差: ${metrics['profit_std']:.2f}M\n")
        f.write(f"  最小值: ${metrics['profit_min']:.2f}M\n")
        f.write(f"  最大值: ${metrics['profit_max']:.2f}M\n")
        f.write(f"  中位数: ${metrics['profit_median']:.2f}M\n")
        f.write(f"\n风险指标:\n")
        f.write(f"  VaR (95%): ${metrics['VaR_95']:.2f}M\n")
        f.write(f"  CVaR (95%): ${metrics['CVaR_95']:.2f}M\n")
        f.write(f"\n伤病分析:\n")
        f.write(f"  至少1人受伤概率: {metrics['injury_probability']:.1%}\n")
        f.write(f"  平均伤病球员数: {metrics['avg_injuries_per_season']:.2f}\n")
        f.write(f"  平均错过比赛数: {metrics['avg_games_missed']:.1f}\n")
        f.write(f"\n表现分析:\n")
        f.write(f"  平均表现: {metrics['performance_mean']:.2f}\n")
        f.write(f"  表现标准差: {metrics['performance_std']:.2f}\n")

    print("\n风险分析完成！结果已保存到 results/ 目录")

    return results, metrics


if __name__ == "__main__":
    main()
