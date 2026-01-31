# -*- coding: utf-8 -*-
"""
球员价值评价模块
Player Valuation Module
使用AHP-熵权-TOPSIS组合模型评价球员价值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2


class AHPWeightCalculator:
    """AHP层次分析法权重计算"""

    def __init__(self):
        self.ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def calculate_weights(self, comparison_matrix):
        """
        计算AHP权重

        Args:
            comparison_matrix: 判断矩阵 (n x n)

        Returns:
            weights: 权重向量
            lambda_max: 最大特征值
            cr: 一致性比例
        """
        comparison_matrix = np.array(comparison_matrix, dtype=float)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)

        # 找最大特征值及其对应的特征向量
        max_idx = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues[max_idx].real
        weight_vector = eigenvectors[:, max_idx].real

        # 归一化
        weight_vector = weight_vector / weight_vector.sum()

        # 一致性检验
        n = len(comparison_matrix)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        ri = self.ri_dict.get(n, 1.49)
        cr = ci / ri if ri != 0 else 0

        return weight_vector, lambda_max, cr

    def build_criteria_matrix(self):
        """
        构建球员价值评价指标的判断矩阵

        指标层次结构:
        - 目标层: 球员综合价值
        - 准则层: 竞技表现、商业价值、稳定性、潜力、合同价值
        """
        # 准则层判断矩阵
        criteria_matrix = np.array([
            [1,     5,     4,     3,     2],    # 竞技表现
            [1/5,   1,     1/2,   1/3,   1/4],  # 商业价值
            [1/4,   2,     1,     1/2,   1/3],  # 稳定性
            [1/3,   3,     2,     1,     1/2],  # 潜力
            [1/2,   4,     3,     2,     1]     # 合同价值
        ])

        weights, lambda_max, cr = self.calculate_weights(criteria_matrix)

        criteria_names = ['竞技表现', '商业价值', '稳定性', '潜力', '合同价值']

        return criteria_names, weights, lambda_max, cr


class EntropyWeightCalculator:
    """熵权法计算器"""

    def calculate_weights(self, data, directions=None):
        """
        计算熵权法权重

        Args:
            data: 数据矩阵 (样本 x 指标)
            directions: 指标方向列表，1为正向，-1为负向

        Returns:
            weights: 权重向量
            entropy_values: 信息熵值
            utility_values: 信息效用值
        """
        data = np.array(data, dtype=float)
        m, n = data.shape

        if directions is None:
            directions = np.ones(n)
        else:
            directions = np.array(directions)

        # 标准化
        normalized = np.zeros_like(data, dtype=float)
        for j in range(n):
            if directions[j] == 1:  # 正向指标
                min_val = data[:, j].min()
                max_val = data[:, j].max()
                if max_val - min_val != 0:
                    normalized[:, j] = (data[:, j] - min_val) / (max_val - min_val)
                else:
                    normalized[:, j] = 1
            else:  # 负向指标
                min_val = data[:, j].min()
                max_val = data[:, j].max()
                if max_val - min_val != 0:
                    normalized[:, j] = (max_val - data[:, j]) / (max_val - min_val)
                else:
                    normalized[:, j] = 1

        # 坐标平移（避免ln(0)）
        shifted = normalized + 1

        # 计算比重
        p = shifted / shifted.sum(axis=0, keepdims=True)

        # 计算信息熵
        e = np.zeros(n)
        for j in range(n):
            # 避免log(0)
            p_safe = p[:, j]
            p_safe = p_safe[p_safe > 0]
            if len(p_safe) > 0:
                e[j] = -1 / np.log(m) * np.sum(p_safe * np.log(p_safe))

        # 计算权重
        d = 1 - e  # 信息效用值
        weights = d / d.sum()

        return weights, e, d


class TOPSISEvaluator:
    """TOPSIS评价器"""

    def evaluate(self, data, weights, directions=None):
        """
        TOPSIS综合评价

        Args:
            data: 数据矩阵 (样本 x 指标)
            weights: 权重向量
            directions: 指标方向，1为正向，-1为负向

        Returns:
            closeness: 相对贴近度
            d_positive: 到正理想解的距离
            d_negative: 到负理想解的距离
        """
        data = np.array(data, dtype=float)
        weights = np.array(weights)
        m, n = data.shape

        if directions is None:
            directions = np.ones(n)
        else:
            directions = np.array(directions)

        # 向量标准化
        normalized = data / np.sqrt((data ** 2).sum(axis=0))

        # 加权规范化
        weighted = normalized * weights

        # 确定正负理想解
        v_positive = np.zeros(n)
        v_negative = np.zeros(n)

        for j in range(n):
            if directions[j] == 1:  # 正向指标
                v_positive[j] = weighted[:, j].max()
                v_negative[j] = weighted[:, j].min()
            else:  # 负向指标
                v_positive[j] = weighted[:, j].min()
                v_negative[j] = weighted[:, j].max()

        # 计算距离
        d_positive = np.sqrt(((weighted - v_positive) ** 2).sum(axis=1))
        d_negative = np.sqrt(((weighted - v_negative) ** 2).sum(axis=1))

        # 计算相对贴近度
        closeness = d_negative / (d_positive + d_negative + 1e-10)

        return closeness, d_positive, d_negative


class PlayerValuationModel:
    """球员价值评价模型（AHP-熵权-TOPSIS组合）"""

    def __init__(self, alpha=0.6):
        """
        初始化评价模型

        Args:
            alpha: AHP权重占比（1-alpha为熵权法占比）
        """
        self.alpha = alpha
        self.ahp = AHPWeightCalculator()
        self.entropy = EntropyWeightCalculator()
        self.topsis = TOPSISEvaluator()

        # 定义评价指标体系
        self.indicators = {
            '竞技表现': ['PER', 'WS', 'VORP', 'PPG', 'RPG', 'APG', 'TS_pct'],
            '商业价值': ['All_Star_Appearances', 'Social_Media_Followers_K', 'Jersey_Sales_Rank'],
            '稳定性': ['Games_Played', 'Injury_Count', 'Days_Injured'],
            '潜力': ['Age', 'Experience', 'Rookie_of_Year_Votes'],
            '合同价值': ['Salary']
        }

        # 指标方向（1为正向，-1为负向）
        self.directions = {
            'PER': 1, 'WS': 1, 'VORP': 1, 'PPG': 1, 'RPG': 1, 'APG': 1, 'TS_pct': 1,
            'All_Star_Appearances': 1, 'Social_Media_Followers_K': 1, 'Jersey_Sales_Rank': -1,
            'Games_Played': 1, 'Injury_Count': -1, 'Days_Injured': -1,
            'Age': -1, 'Experience': 1, 'Rookie_of_Year_Votes': 1,
            'Salary': -1  # 薪资越低性价比越高
        }

    def evaluate_players(self, players_df):
        """
        评价球员价值

        Args:
            players_df: 球员数据DataFrame

        Returns:
            results: 包含评价结果的DataFrame
        """
        # 1. 计算AHP权重
        criteria_names, ahp_weights, lambda_max, cr = self.ahp.build_criteria_matrix()

        print("AHP权重计算结果:")
        for name, w in zip(criteria_names, ahp_weights):
            print(f"  {name}: {w:.4f}")
        print(f"  最大特征值: {lambda_max:.4f}, 一致性比例: {cr:.4f}")

        # 2. 准备评价数据
        all_indicators = []
        for category in self.indicators.values():
            all_indicators.extend(category)

        # 提取数据并处理缺失值
        evaluation_data = players_df[all_indicators].copy()

        # 对于球衣销售排名，缺失值用最大值（排名最低）填充
        if 'Jersey_Sales_Rank' in evaluation_data.columns:
            evaluation_data['Jersey_Sales_Rank'].fillna(
                evaluation_data['Jersey_Sales_Rank'].max(), inplace=True
            )

        # 处理其他缺失值
        evaluation_data.fillna(evaluation_data.mean(), inplace=True)

        # 3. 计算熵权法权重
        indicator_directions = [self.directions.get(ind, 1) for ind in all_indicators]
        entropy_weights, e_values, d_values = self.entropy.calculate_weights(
            evaluation_data.values, indicator_directions
        )

        print("\n熵权法权重计算结果:")
        for ind, w, e, d in zip(all_indicators, entropy_weights, e_values, d_values):
            print(f"  {ind}: w={w:.4f}, e={e:.4f}, d={d:.4f}")

        # 4. 计算组合权重
        # 为每个指标分配AHP权重
        combined_weights = []
        for i, indicator in enumerate(all_indicators):
            # 确定指标属于哪个准则
            for category_idx, (category, indicators) in enumerate(self.indicators.items()):
                if indicator in indicators:
                    ahp_w = ahp_weights[category_idx] / len(indicators)
                    break
            else:
                ahp_w = 1 / len(all_indicators)

            combined_w = self.alpha * ahp_w + (1 - self.alpha) * entropy_weights[i]
            combined_weights.append(combined_w)

        # 归一化
        combined_weights = np.array(combined_weights)
        combined_weights = combined_weights / combined_weights.sum()

        print("\n组合权重:")
        for ind, w in zip(all_indicators, combined_weights):
            print(f"  {ind}: {w:.4f}")

        # 5. TOPSIS评价
        closeness, d_pos, d_neg = self.topsis.evaluate(
            evaluation_data.values, combined_weights, indicator_directions
        )

        # 6. 构建结果
        results = players_df.copy()
        results['综合价值得分'] = closeness
        results['正理想解距离'] = d_pos
        results['负理想解距离'] = d_neg
        results['排名'] = results['综合价值得分'].rank(ascending=False)

        return results.sort_values('综合价值得分', ascending=False)

    def plot_evaluation_results(self, results, top_n=15, save_dir='figures'):
        """
        绘制评价结果图表

        Args:
            results: 评价结果DataFrame
            top_n: 显示前N名
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        top_players = results.head(top_n)

        # 图1：价值得分排名柱状图
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.RdYlGn_r(top_players['综合价值得分'].values)
        bars = ax.barh(range(len(top_players)), top_players['综合价值得分'].values, color=colors)

        ax.set_yticks(range(len(top_players)))
        ax.set_yticklabels([f"{row['Name']}\n({row['Position']}, {row['Team']})"
                           for _, row in top_players.iterrows()], fontsize=9)
        ax.set_xlabel('综合价值得分', fontsize=12)
        ax.set_title(f'TOP {top_n} 球员价值排名', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.set_xlim(0, 1)

        # 添加数值标签
        for i, (idx, row) in enumerate(top_players.iterrows()):
            ax.text(row['综合价值得分'] + 0.02, i, f"{row['综合价值得分']:.3f}",
                   va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure1_player_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2：维度雷达图（TOP 5球员）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        top5 = results.head(5)

        categories = list(self.indicators.keys())
        n_cat = len(categories)

        # 计算TOP5在各维度的平均得分
        category_scores = {}
        for category in categories:
            indicators = self.indicators[category]
            category_data = top5[indicators].copy()

            # 标准化到0-1
            for ind in indicators:
                if self.directions.get(ind, 1) == 1:
                    category_data[ind] = (category_data[ind] - category_data[ind].min()) / \
                                       (category_data[ind].max() - category_data[ind].min() + 1e-10)
                else:
                    category_data[ind] = (category_data[ind].max() - category_data[ind]) / \
                                       (category_data[ind].max() - category_data[ind].min() + 1e-10)

            category_scores[category] = category_data.mean().mean()

        # 左图：雷达图
        angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
        scores = list(category_scores.values())
        scores += scores[:1]  # 闭合
        angles += angles[:1]

        ax1 = fig.add_subplot(121, polar=True)
        ax1.plot(angles, scores, 'o-', linewidth=2, color='#1f77b4')
        ax1.fill(angles, scores, alpha=0.25, color='#1f77b4')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, fontsize=10)
        ax1.set_ylim(0, 1)
        ax1.set_title('TOP 5 球员各维度平均得分', fontsize=12, pad=20)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 右图：薪资 vs 价值散点图
        ax2.scatter(results['Salary'] / 1000, results['综合价值得分'],
                   alpha=0.6, s=50, c=results['综合价值得分'],
                   cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)

        # 标注TOP 5
        for _, row in top5.iterrows():
            ax2.annotate(row['Name'],
                        (row['Salary'] / 1000, row['综合价值得分']),
                        fontsize=8, alpha=0.8)

        ax2.set_xlabel('薪资 (千元)', fontsize=12)
        ax2.set_ylabel('综合价值得分', fontsize=12)
        ax2.set_title('薪资 vs 价值分析', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure2_value_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n图表已保存到 {save_dir}/ 目录")


def main():
    """主函数：测试球员价值评价模型"""
    from data_loader import DataLoader

    # 加载数据
    loader = DataLoader()
    players, teams, financial = loader.load_data()

    # 创建评价模型
    model = PlayerValuationModel(alpha=0.6)

    # 评价球员
    print("开始球员价值评价...")
    results = model.evaluate_players(players)

    # 显示TOP 20
    print("\nTOP 20 球员价值排名:")
    print(results[['Player_ID', 'Name', 'Team', 'Position', 'Age', 'Salary',
                  '综合价值得分', '排名']].head(20).to_string(index=False))

    # 绘制图表
    model.plot_evaluation_results(results, top_n=15)

    # 保存结果
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/player_evaluation_results.csv',
                  index=False, encoding='utf-8-sig')

    print("\n评价完成！结果已保存到 results/player_evaluation_results.csv")

    return results


if __name__ == "__main__":
    main()
