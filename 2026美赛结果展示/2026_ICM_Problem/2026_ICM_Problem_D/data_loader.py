# -*- coding: utf-8 -*-
"""
数据加载模块
Data Loader Module
用于加载和管理球员数据、球队数据和财务数据
"""

import numpy as np
import pandas as pd
import os

class DataLoader:
    """数据加载类"""

    def __init__(self, data_dir='data'):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.players_data = None
        self.teams_data = None
        self.financial_data = None

    def load_sample_data(self):
        """
        加载示例数据（WNBA球队）

        Returns:
            players_df: 球员数据DataFrame
            teams_df: 球队数据DataFrame
            financial_df: 财务数据DataFrame
        """
        # 创建示例球员数据（基于WNBA真实数据结构）
        np.random.seed(42)

        # WNBA球员位置
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        teams = ['Las Vegas Aces', 'New York Liberty', 'Seattle Storm',
                 'Connecticut Sun', 'Dallas Wings', 'Chicago Sky',
                 'Washington Mystics', 'Phoenix Mercury', 'Los Angeles Sparks',
                 'Atlanta Dream', 'Minnesota Lynx', 'Indiana Fever']

        n_players = 150

        players_data = {
            'Player_ID': [f'P{i:03d}' for i in range(1, n_players + 1)],
            'Name': [f'Player {i}' for i in range(1, n_players + 1)],
            'Team': np.random.choice(teams, n_players),
            'Position': np.random.choice(positions, n_players),
            'Age': np.random.randint(21, 36, n_players),
            'Experience': np.random.randint(0, 15, n_players),
            'Salary': np.random.uniform(100000, 250000, n_players),

            # 竞技表现指标
            'PER': np.random.uniform(10, 28, n_players),  # Player Efficiency Rating
            'WS': np.random.uniform(0.5, 8, n_players),    # Win Shares
            'VORP': np.random.uniform(-1, 4, n_players),   # Value Over Replacement Player
            'PPG': np.random.uniform(3, 25, n_players),    # Points Per Game
            'RPG': np.random.uniform(1, 12, n_players),    # Rebounds Per Game
            'APG': np.random.uniform(0.5, 8, n_players),   # Assists Per Game
            'SPG': np.random.uniform(0.3, 2.5, n_players), # Steals Per Game
            'BPG': np.random.uniform(0.1, 2, n_players),   # Blocks Per Game
            'TS_pct': np.random.uniform(45, 65, n_players), # True Shooting %
            'USG_pct': np.random.uniform(12, 30, n_players),# Usage %

            # 稳定性指标
            'Games_Played': np.random.randint(10, 40, n_players),
            'Games_Started': np.random.randint(0, 40, n_players),
            'Minutes_Per_Game': np.random.uniform(10, 35, n_players),
            'Injury_Count': np.random.randint(0, 5, n_players),
            'Days_Injured': np.random.randint(0, 50, n_players),

            # 商业价值指标
            'All_Star_Appearances': np.random.randint(0, 8, n_players),
            'Social_Media_Followers_K': np.random.uniform(10, 2000, n_players),
            'Jersey_Sales_Rank': np.random.randint(1, 151, n_players),

            # 潜力指标
            'Draft_Pick': np.random.choice(list(range(1, 37)) + [None], n_players),
            'Rookie_of_Year_Votes': np.random.randint(0, 100, n_players),
            'Most_Improved_Votes': np.random.randint(0, 50, n_players),
        }

        self.players_data = pd.DataFrame(players_data)

        # 创建球队数据
        teams_data = {
            'Team': teams,
            'Market_Size_Rank': [1, 2, 10, 15, 8, 3, 4, 12, 5, 11, 7, 9],
            'City_Population_M': [2.3, 8.4, 0.75, 1.3, 1.3, 2.7, 6.9, 1.7, 3.9, 0.5, 0.43, 0.88],
            'Arena_Capacity': [12000, 18000, 10000, 8000, 8000, 10000, 20000, 14000, 17000, 8000, 9000, 18000],
            'Win_Rate_Pct': np.random.uniform(40, 70, 12),
            'Playoff_Appearances_Last_5_Years': np.random.randint(0, 5, 12),
            'Championships_Last_5_Years': np.random.randint(0, 2, 12),
            'Ticket_Price_Avg': np.random.uniform(30, 150, 12),
            'Attendance_Per_Game': np.random.uniform(5000, 18000, 12),
        }

        self.teams_data = pd.DataFrame(teams_data)

        # 创建财务数据
        financial_data = {
            'Team': teams,
            'Revenue_Ticket_M': np.random.uniform(5, 50, 12),
            'Revenue_Media_M': np.random.uniform(10, 30, 12),
            'Revenue_Merchandise_M': np.random.uniform(2, 15, 12),
            'Revenue_Sponsorship_M': np.random.uniform(3, 20, 12),
            'Total_Revenue_M': np.random.uniform(20, 100, 12),
            'Player_Salaries_M': np.random.uniform(10, 40, 12),
            'Operating_Costs_M': np.random.uniform(5, 25, 12),
            'Total_Costs_M': np.random.uniform(15, 65, 12),
            'Profit_M': np.random.uniform(-5, 35, 12),
            'Team_Value_M': np.random.uniform(50, 200, 12),
        }

        self.financial_data = pd.DataFrame(financial_data)

        # 保存到文件
        self._save_data()

        return self.players_data, self.teams_data, self.financial_data

    def _save_data(self):
        """保存数据到文件"""
        os.makedirs(self.data_dir, exist_ok=True)

        self.players_data.to_csv(os.path.join(self.data_dir, 'players_data.csv'),
                                  index=False, encoding='utf-8-sig')
        self.teams_data.to_csv(os.path.join(self.data_dir, 'teams_data.csv'),
                                 index=False, encoding='utf-8-sig')
        self.financial_data.to_csv(os.path.join(self.data_dir, 'financial_data.csv'),
                                    index=False, encoding='utf-8-sig')

        print("数据已保存到 data/ 目录")

    def load_data(self):
        """
        从文件加载数据

        Returns:
            players_df: 球员数据DataFrame
            teams_df: 球队数据DataFrame
            financial_df: 财务数据DataFrame
        """
        players_path = os.path.join(self.data_dir, 'players_data.csv')
        teams_path = os.path.join(self.data_dir, 'teams_data.csv')
        financial_path = os.path.join(self.data_dir, 'financial_data.csv')

        if os.path.exists(players_path):
            self.players_data = pd.read_csv(players_path, encoding='utf-8-sig')
            self.teams_data = pd.read_csv(teams_path, encoding='utf-8-sig')
            self.financial_data = pd.read_csv(financial_path, encoding='utf-8-sig')
            print("数据已从文件加载")
        else:
            print("数据文件不存在，生成示例数据...")
            return self.load_sample_data()

        return self.players_data, self.teams_data, self.financial_data

    def get_team_players(self, team_name):
        """获取指定球队的球员数据"""
        if self.players_data is None:
            self.load_data()
        return self.players_data[self.players_data['Team'] == team_name].copy()

    def get_free_agents(self, team_name=None):
        """
        获取自由球员数据
        Args:
            team_name: 排除的球队（用于模拟该球队的自由球员签约）
        """
        if self.players_data is None:
            self.load_data()

        # 模拟自由球员：经验>=3年且合同即将到期的球员
        free_agents = self.players_data[
            (self.players_data['Experience'] >= 3) &
            (np.random.random(len(self.players_data)) < 0.3)  # 30%概率成为自由球员
        ].copy()

        if team_name:
            free_agents = free_agents[free_agents['Team'] != team_name]

        return free_agents


if __name__ == "__main__":
    # 测试数据加载
    loader = DataLoader()
    players, teams, financial = loader.load_data()

    print(f"\n球员数据形状: {players.shape}")
    print(f"球队数据形状: {teams.shape}")
    print(f"财务数据形状: {financial.shape}")

    print("\n球员数据预览:")
    print(players.head())

    print("\n球队数据预览:")
    print(teams.head())
