# -*- coding: utf-8 -*-
"""
主程序 - 2026 ICM Problem D: Managing Sports for Success
Sports Team Business and Management Model

功能模块:
1. 数据加载 (data_loader.py)
2. 球员价值评价 (player_evaluation.py)
3. 多目标优化 (optimization_model.py)
4. 风险分析 (risk_analysis.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2

# 导入项目模块
from data_loader import DataLoader
from player_evaluation import PlayerValuationModel
from optimization_model import RosterOptimizationModel, RevenueOptimizationModel
from risk_analysis import MonteCarloSimulator, LeagueExpansionModel


class SportsTeamManagementModel:
    """体育球队商业管理综合模型"""

    def __init__(self, target_team='Las Vegas Aces'):
        """
        初始化综合管理模型

        Args:
            target_team: 目标球队名称
        """
        self.target_team = target_team
        self.loader = DataLoader()
        self.evaluator = PlayerValuationModel(alpha=0.6)
        self.roster_optimizer = RosterOptimizationModel(
            salary_cap=1500000,  # WNBA薪资帽约150万美元
            roster_size=12
        )
        self.pricing_optimizer = RevenueOptimizationModel(
            capacity=18000,
            base_price=80
        )

        # 数据
        self.players_data = None
        self.teams_data = None
        self.financial_data = None
        self.team_players = None
        self.team_financial = None

        # 结果
        self.evaluation_results = None
        self.pareto_solutions = None
        self.balanced_solution = None
        self.risk_results = None
        self.risk_metrics = None

    def load_all_data(self):
        """加载所有数据"""
        print("=" * 60)
        print("数据加载中...")
        print("=" * 60)

        self.players_data, self.teams_data, self.financial_data = self.loader.load_data()

        # 获取目标球队的球员和财务数据
        self.team_players = self.loader.get_team_players(self.target_team)
        self.team_financial = self.financial_data[
            self.financial_data['Team'] == self.target_team
        ].iloc[0].to_dict()

        print(f"目标球队: {self.target_team}")
        print(f"当前球员数: {len(self.team_players)}")
        print(f"球队价值: ${self.team_financial['Team_Value_M']:.1f}M")
        print(f"上赛季利润: ${self.team_financial['Profit_M']:.1f}M")

    def evaluate_players(self):
        """评价球员价值"""
        print("\n" + "=" * 60)
        print("球员价值评价 (AHP-熵权-TOPSIS组合模型)...")
        print("=" * 60)

        # 合并当前球队球员和自由球员
        free_agents = self.loader.get_free_agents(self.target_team)
        all_players = pd.concat([self.team_players, free_agents], ignore_index=True)

        # 评价所有球员
        self.evaluation_results = self.evaluator.evaluate_players(all_players)

        # 绘制评价结果图
        self.evaluator.plot_evaluation_results(self.evaluation_results, top_n=15)

        print("\nTOP 15 球员价值排名:")
        display_cols = ['Player_ID', 'Name', 'Team', 'Position', 'Age',
                       'Salary', 'PER', '综合价值得分', '排名']
        print(self.evaluation_results[display_cols].head(15).to_string(index=False))

    def optimize_roster(self):
        """优化球队阵容"""
        print("\n" + "=" * 60)
        print("球队阵容优化 (多目标优化)...")
        print("=" * 60)

        # 获取可用球员池（当前球队 + 自由球员）
        free_agents = self.loader.get_free_agents(self.target_team)
        available_pool = pd.concat([self.team_players, free_agents], ignore_index=True)

        # 添加评价结果
        available_pool = self.evaluator.evaluate_players(available_pool)

        # 求解帕累托前沿
        self.pareto_solutions = self.roster_optimizer.optimize_pareto_front(
            available_pool, self.team_financial, n_solutions=20
        )

        # 找到平衡解
        self.pareto_solutions['Composite_Score'] = (
            self.pareto_solutions['Profit_M'] / self.pareto_solutions['Profit_M'].max() +
            self.pareto_solutions['Performance'] / self.pareto_solutions['Performance'].max()
        ) / 2

        self.balanced_solution = self.pareto_solutions.loc[
            self.pareto_solutions['Composite_Score'].idxmax()
        ]

        # 绘制帕累托前沿图
        self.roster_optimizer.plot_pareto_front(self.pareto_solutions)

        print("\n最优平衡方案:")
        print(f"  利润权重: {self.balanced_solution['Lambda']:.2f}")
        print(f"  预期利润: ${self.balanced_solution['Profit_M']:.2f}M")
        print(f"  球队表现: {self.balanced_solution['Performance']:.2f}")
        print(f"  总薪资: ${self.balanced_solution['Total_Salary_M']:.2f}M")
        print(f"  PER总和: {self.balanced_solution['PER_Sum']:.2f}")

    def analyze_ticket_pricing(self):
        """分析票务定价策略"""
        print("\n" + "=" * 60)
        print("票务定价优化...")
        print("=" * 60)

        # 绘制定价分析图
        self.pricing_optimizer.plot_pricing_analysis()

        # 分析不同场景的最优定价
        scenarios = [
            ('弱对手-工作日', 0.7, False),
            ('普通对手-工作日', 1.0, False),
            ('强对手-工作日', 1.3, False),
            ('强对手-周末', 1.3, True)
        ]

        print("\n不同场景的最优定价策略:")
        print(f"{'场景':<20} {'最优票价':<12} {'预期上座率':<12} {'预期利润':<12}")
        print("-" * 60)
        for label, opp, weekend in scenarios:
            price, marketing, res = self.pricing_optimizer.optimize_single_game_pricing(
                opp, weekend
            )
            print(f"{label:<20} ${price:>10.2f}   {res['occupancy_rate']:>10.1%}   "
                  f"${res['profit']/1e6:>10.2f}M")

    def analyze_risks(self):
        """分析风险（蒙特卡洛模拟）"""
        print("\n" + "=" * 60)
        print("风险分析 (蒙特卡洛模拟)...")
        print("=" * 60)

        # 获取优化后的阵容（TOP 12）
        available_pool = self.evaluator.evaluate_players(
            pd.concat([self.team_players, self.loader.get_free_agents(self.target_team)],
                     ignore_index=True)
        )
        optimized_roster = available_pool.head(12)

        # 运行蒙特卡洛模拟
        simulator = MonteCarloSimulator(n_simulations=1000)
        self.risk_results = simulator.run_simulation(optimized_roster, self.team_financial)
        self.risk_metrics = simulator.calculate_risk_metrics(self.risk_results)

        # 绘制风险分析图
        simulator.plot_risk_analysis(self.risk_results)

        print("\n风险分析结果:")
        print(f"  利润均值: ${self.risk_metrics['profit_mean']:.2f}M")
        print(f"  利润标准差: ${self.risk_metrics['profit_std']:.2f}M")
        print(f"  最小值: ${self.risk_metrics['profit_min']:.2f}M")
        print(f"  最大值: ${self.risk_metrics['profit_max']:.2f}M")
        print(f"  VaR (95%): ${self.risk_metrics['VaR_95']:.2f}M")
        print(f"  CVaR (95%): ${self.risk_metrics['CVaR_95']:.2f}M")
        print(f"  伤病概率: {self.risk_metrics['injury_probability']:.1%}")
        print(f"  平均伤病球员数: {self.risk_metrics['avg_injuries_per_season']:.2f}")

    def analyze_expansion(self):
        """分析联盟扩张影响"""
        print("\n" + "=" * 60)
        print("联盟扩张影响分析...")
        print("=" * 60)

        expansion_model = LeagueExpansionModel()
        expansion_scenarios = expansion_model.compare_expansion_scenarios(self.players_data)

        print("\n不同扩张位置的影响比较:")
        print(expansion_scenarios.to_string(index=False))

        # 绘制扩张影响图
        self._plot_expansion_analysis(expansion_scenarios)

    def _plot_expansion_analysis(self, expansion_scenarios):
        """绘制联盟扩张分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：人才流失率
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

    def generate_summary_report(self):
        """生成综合分析报告"""
        os.makedirs('results', exist_ok=True)

        report = []
        report.append("=" * 70)
        report.append("2026 ICM Problem D: 体育球队商业管理模型")
        report.append("Sports Team Business and Management Model")
        report.append("=" * 70)
        report.append(f"\n目标球队: {self.target_team}\n")

        # 球队现状
        report.append("一、球队现状分析")
        report.append("-" * 70)
        report.append(f"球队价值: ${self.team_financial['Team_Value_M']:.1f}M")
        report.append(f"上赛季收入: ${self.team_financial['Total_Revenue_M']:.1f}M")
        report.append(f"上赛季成本: ${self.team_financial['Total_Costs_M']:.1f}M")
        report.append(f"上赛季利润: ${self.team_financial['Profit_M']:.1f}M")
        report.append(f"当前球员数: {len(self.team_players)}")

        # 球员价值评价
        if self.evaluation_results is not None:
            report.append("\n二、球员价值评价结果")
            report.append("-" * 70)
            top5 = self.evaluation_results.head(5)
            report.append("TOP 5 球员:")
            for _, row in top5.iterrows():
                report.append(f"  {row['Name']:15s} | {row['Position']:3s} | "
                            f"Age: {row['Age']:2d} | Salary: ${row['Salary']:8.0f} | "
                            f"得分: {row['综合价值得分']:.3f}")

        # 阵容优化
        if self.balanced_solution is not None:
            report.append("\n三、阵容优化建议")
            report.append("-" * 70)
            report.append(f"最优平衡方案:")
            report.append(f"  预期利润: ${self.balanced_solution['Profit_M']:.2f}M")
            report.append(f"  球队表现: {self.balanced_solution['Performance']:.2f}")
            report.append(f"  总薪资: ${self.balanced_solution['Total_Salary_M']:.2f}M")

            current_profit = self.team_financial['Profit_M']
            profit_improvement = self.balanced_solution['Profit_M'] - current_profit
            report.append(f"  利润提升: ${profit_improvement:.2f}M ({profit_improvement/current_profit*100:.1f}%)")

        # 风险分析
        if self.risk_metrics is not None:
            report.append("\n四、风险分析结果")
            report.append("-" * 70)
            report.append(f"利润分析:")
            report.append(f"  均值: ${self.risk_metrics['profit_mean']:.2f}M")
            report.append(f"  标准差: ${self.risk_metrics['profit_std']:.2f}M")
            report.append(f"  VaR (95%): ${self.risk_metrics['VaR_95']:.2f}M")
            report.append(f"  CVaR (95%): ${self.risk_metrics['CVaR_95']:.2f}M")
            report.append(f"\n伤病风险:")
            report.append(f"  至少1人受伤概率: {self.risk_metrics['injury_probability']:.1%}")
            report.append(f"  平均伤病球员数: {self.risk_metrics['avg_injuries_per_season']:.2f}")

        # 策略建议
        report.append("\n五、管理策略建议")
        report.append("-" * 70)
        report.append("1. 球员获取策略:")
        report.append("   - 优先考虑综合价值得分高的球员")
        report.append("   - 平衡薪资和表现，追求性价比")
        report.append("   - 适当投资明星球员以提升商业价值")
        report.append("\n2. 票务定价策略:")
        report.append("   - 根据对手实力动态调整票价")
        report.append("   - 周末和强对手比赛可适当提高票价")
        report.append("   - 弱对手比赛可降低票价以提高上座率")
        report.append("\n3. 风险管理策略:")
        report.append("   - 建立伤病应急基金")
        report.append("   - 保持阵容深度以应对伤病")
        report.append("   - 考虑购买相关保险")
        report.append("\n4. 联盟扩张应对:")
        report.append("   - 提前锁定关键球员合同")
        report.append("   - 关注新球队所在市场的薪资影响")

        report.append("\n" + "=" * 70)
        report.append("报告结束")
        report.append("=" * 70)

        # 保存报告
        report_text = "\n".join(report)
        with open('results/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)

    def save_all_results(self):
        """保存所有结果到文件"""
        os.makedirs('results', exist_ok=True)

        # 保存球员评价结果
        if self.evaluation_results is not None:
            self.evaluation_results.to_csv(
                'results/player_evaluation.csv',
                index=False,
                encoding='utf-8-sig'
            )

        # 保存帕累托前沿
        if self.pareto_solutions is not None:
            self.pareto_solutions.to_csv(
                'results/pareto_solutions.csv',
                index=False,
                encoding='utf-8-sig'
            )

        # 保存蒙特卡洛结果
        if self.risk_results is not None:
            self.risk_results.to_csv(
                'results/monte_carlo_simulation.csv',
                index=False,
                encoding='utf-8-sig'
            )

        print("\n所有结果已保存到 results/ 目录")

    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n")
        print("*" * 60)
        print("*" + " " * 58 + "*")
        print("*" + "  2026 ICM Problem D: 体育球队商业管理模型".center(56) + "*")
        print("*" + "  Sports Team Business and Management Model".center(56) + "*")
        print("*" + " " * 58 + "*")
        print("*" * 60)

        # 1. 加载数据
        self.load_all_data()

        # 2. 评价球员
        self.evaluate_players()

        # 3. 优化阵容
        self.optimize_roster()

        # 4. 票务定价分析
        self.analyze_ticket_pricing()

        # 5. 风险分析
        self.analyze_risks()

        # 6. 联盟扩张分析
        self.analyze_expansion()

        # 7. 保存结果
        self.save_all_results()

        # 8. 生成报告
        self.generate_summary_report()

        print("\n" + "*" * 60)
        print("分析完成！")
        print("*" * 60)
        print("\n生成的文件:")
        print("  results/  - 结果数据文件")
        print("  figures/   - 可视化图表")


def main():
    """主函数"""
    # 创建模型实例（可选择不同球队）
    available_teams = ['Las Vegas Aces', 'New York Liberty', 'Seattle Storm',
                      'Connecticut Sun', 'Dallas Wings', 'Chicago Sky']

    print("可用球队:")
    for i, team in enumerate(available_teams, 1):
        print(f"  {i}. {team}")

    # 默认选择 Las Vegas Aces
    target_team = 'Las Vegas Aces'

    print(f"\n选择球队: {target_team}")
    print("如需分析其他球队，请修改 main() 函数中的 target_team 参数\n")

    # 创建并运行模型
    model = SportsTeamManagementModel(target_team=target_team)
    model.run_full_analysis()

    return model


if __name__ == "__main__":
    model = main()
