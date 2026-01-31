"""
被动式太阳能遮阳优化 - 主程序（改进版）
Passive Solar Shading Optimization - Main Program (Enhanced)

2026 ICM Problem E Solution
Enhanced visualization and improved model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle, Circle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from solar_position import SolarPositionCalculator
from shading_geometry import OverhangShading, LouverShading, VerticalShading, VegetativeShading
from thermal_simulation import BuildingThermalModel, generate_synthetic_weather_data

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2


class EnhancedPassiveSolarOptimizer:
    """增强的被动式太阳能遮阳优化器"""

    def __init__(self, university='sungrove'):
        """初始化优化器"""
        self.university = university

        if university == 'sungrove':
            self.name = "Sungrove University"
            self.latitude = 20
            self.climate_type = "Warm"
            self.building_params = {
                'length': 60, 'width': 24, 'height': 3, 'floors': 2,
                'wwr_south': 0.45, 'wwr_other': 0.30,
                'u_wall': 0.5, 'u_window': 2.8,
                'thermal_mass_per_area': 200, 'internal_load': 20
            }
            self.setpoint_cooling = 24
            self.setpoint_heating = 20
            self.base_cooling = 50000
            self.base_heating = 30000
        else:
            self.name = "Borealis University"
            self.latitude = 60
            self.climate_type = "Cold"
            self.building_params = {
                'length': 60, 'width': 24, 'height': 3, 'floors': 2,
                'wwr_south': 0.50, 'wwr_other': 0.25,
                'u_wall': 0.3, 'u_window': 1.8,
                'thermal_mass_per_area': 250, 'internal_load': 15
            }
            self.setpoint_cooling = 25
            self.setpoint_heating = 21
            self.base_cooling = 20000
            self.base_heating = 80000

        self.solar_calc = SolarPositionCalculator(latitude=self.latitude, longitude=0, timezone=0)
        self.building = BuildingThermalModel(self.building_params)

    def calculate_hourly_shading_coefficients(self, overhang_depth, louver_angle=None):
        """计算全年每小时遮阳系数"""
        data = []

        for day in range(1, 366):
            for hour in range(24):
                pos = self.solar_calc.calculate_solar_position(day, hour)
                if pos['altitude'] > 0:
                    overhang = OverhangShading(depth=overhang_depth)

                    sc_south = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], 180)
                    sc_east = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], 90)
                    sc_west = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], 270)

                    data.append({
                        'day': day,
                        'hour': hour,
                        'month': (day - 1) // 30 + 1,
                        'altitude': pos['altitude'],
                        'sc_south': sc_south,
                        'sc_east': sc_east,
                        'sc_west': sc_west
                    })

        return pd.DataFrame(data)

    def calculate_energy_consumption(self, overhang_depth):
        """计算能耗（改进模型）"""
        # 计算关键时间的遮阳系数
        # 夏季（6月21日正午）
        summer_pos = self.solar_calc.calculate_solar_position(172, 12)
        overhang = OverhangShading(depth=overhang_depth)
        summer_sc = overhang.calculate_shading_coefficient(summer_pos['altitude'], summer_pos['azimuth'], 180)

        # 冬季（12月21日正午）
        winter_pos = self.solar_calc.calculate_solar_position(356, 12)
        winter_sc = overhang.calculate_shading_coefficient(winter_pos['altitude'], winter_pos['azimuth'], 180)

        # 改进的能耗模型
        # 制冷负荷：与夏季遮阳系数相关
        cooling_load = self.base_cooling * (1 - 0.7 * (1 - summer_sc))

        # 供暖负荷：与冬季遮阳系数相关
        heating_load = self.base_heating * (1 + 0.5 * (1 - winter_sc))

        total_energy = cooling_load + heating_load

        return {
            'cooling_load': cooling_load,
            'heating_load': heating_load,
            'total_energy': total_energy,
            'summer_sc': summer_sc,
            'winter_sc': winter_sc
        }

    def analyze_comprehensive_designs(self):
        """全面分析设计方案"""
        designs = []

        # 测试不同的悬挑深度
        for depth in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            energy = self.calculate_energy_consumption(depth)
            designs.append({
                'name': f'Overhang {depth}m',
                'type': 'overhang',
                'overhang_depth': depth,
                'louver_angle': 0,
                'summer_shading': energy['summer_sc'],
                'winter_shading': energy['winter_sc'],
                'cooling_load': energy['cooling_load'],
                'heating_load': energy['heating_load'],
                'total_energy': energy['total_energy']
            })

        # 测试百叶遮阳
        for angle in [30, 45, 60]:
            energy = self.calculate_energy_consumption(1.5)
            # 百叶额外的遮阳效果
            louver_bonus = 0.15 * angle / 90
            energy['summer_shading'] *= (1 - louver_bonus)
            energy['cooling_load'] = self.base_cooling * (1 - 0.7 * (1 - energy['summer_shading']))
            energy['total_energy'] = energy['cooling_load'] + energy['heating_load']

            designs.append({
                'name': f'Overhang 1.5m + Louver {angle}deg',
                'type': 'combined',
                'overhang_depth': 1.5,
                'louver_angle': angle,
                'summer_shading': energy['summer_shading'],
                'winter_shading': energy['winter_shading'],
                'cooling_load': energy['cooling_load'],
                'heating_load': energy['heating_load'],
                'total_energy': energy['total_energy']
            })

        return pd.DataFrame(designs)

    def create_comprehensive_visualizations(self, shading_df):
        """创建全面的可视化图表"""
        import os
        os.makedirs('figures', exist_ok=True)

        print(f"\n{'='*60}")
        print(f"生成增强可视化图表...")
        print(f"{'='*60}")

        # 图1: 太阳轨迹图和季节变化
        self._plot_solar_trajectories()

        # 图2: 遮阳系数季节变化
        self._plot_seasonal_shading_coefficients()

        # 图3: 能耗对比分析
        self._plot_energy_comparison(shading_df)

        # 图4: Pareto前沿分析
        self._plot_pareto_front(shading_df)

        # 图5: 不同朝向遮阳效果
        self._plot_orientation_effects()

        # 图6: 遮阳几何示意图
        self._plot_shading_geometry_diagrams()

        # 图7: 月度能耗分布
        self._plot_monthly_energy_distribution()

        # 图8: 全年遮阳系数热力图
        self._plot_annual_shading_heatmap()

        print(f"\n已生成 8 张图表，保存到 figures/ 目录")

    def _plot_solar_trajectories(self):
        """太阳轨迹图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：太阳轨迹
        for day_name, day_idx in [('Winter Solstice', 356), ('Summer Solstice', 172), ('Equinox', 80)]:
            altitudes, azimuths = [], []
            for hour in range(6, 19):
                pos = self.solar_calc.calculate_solar_position(day_idx, hour)
                altitudes.append(pos['altitude'])
                azimuths.append(pos['azimuth_from_south'])

            ax1.plot(azimuths, altitudes, 'o-', label=day_name, linewidth=2, markersize=5)

        ax1.set_xlabel('Azimuth Angle from South (degrees)', fontsize=12)
        ax1.set_ylabel('Solar Altitude (degrees)', fontsize=12)
        ax1.set_title(f'Solar Path Diagram - {self.name}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(0, 90)

        # 右图：正午太阳高度角季节变化
        days = np.arange(1, 366)
        noon_altitudes = []
        for day in days:
            pos = self.solar_calc.calculate_solar_position(day, 12)
            noon_altitudes.append(pos['altitude'])

        ax2.plot(days, noon_altitudes, 'b-', linewidth=2)
        ax2.fill_between(days, 0, noon_altitudes, alpha=0.3)
        ax2.axhline(y=90, color='r', linestyle='--', label='Zenith', linewidth=2)
        ax2.axhline(y=90-self.latitude, color='g', linestyle='--', label='Summer Solstice Noon', linewidth=2)
        ax2.axhline(y=90-self.latitude-23.45, color='orange', linestyle='--', label='Winter Solstice Noon', linewidth=2)

        ax2.set_xlabel('Day of Year', fontsize=12)
        ax2.set_ylabel('Noon Solar Altitude (degrees)', fontsize=12)
        ax2.set_title('Noon Solar Altitude Throughout Year', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(1, 365)

        plt.tight_layout()
        plt.savefig('figures/01_solar_position.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/01_solar_position.png")
        plt.close()

    def _plot_seasonal_shading_coefficients(self):
        """遮阳系数季节变化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 不同悬挑深度的遮阳系数变化
        depths = [0.5, 1.0, 1.5, 2.0, 2.5]
        months = np.arange(1, 13)

        for depth in depths:
            monthly_sc = []
            for month in months:
                day = month * 30
                pos = self.solar_calc.calculate_solar_position(day, 12)
                overhang = OverhangShading(depth=depth)
                sc = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], 180)
                monthly_sc.append(sc)

            ax1.plot(months, monthly_sc, 'o-', label=f'{depth}m', linewidth=2, markersize=6)

        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Shading Coefficient at Noon', fontsize=12)
        ax1.set_title('Seasonal Variation of Shading Coefficient', fontsize=14, fontweight='bold')
        ax1.legend(title='Overhang Depth', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xticks(months)
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # 每日遮阳系数（1.5m悬挑）
        sc_data = self.calculate_hourly_shading_coefficients(1.5)
        daily_avg = sc_data.groupby('day')['sc_south'].mean()

        ax2.plot(daily_avg.index, daily_avg.values, 'b-', linewidth=1.5, alpha=0.7)
        ax2.fill_between(daily_avg.index, 0, daily_avg.values, alpha=0.3)

        # 标记关键日期
        key_dates = {80: 'Spring Equinox', 172: 'Summer Solstice', 266: 'Fall Equinox', 356: 'Winter Solstice'}
        for day, name in key_dates.items():
            if day <= len(daily_avg):
                ax2.axvline(x=day, color='r', linestyle='--', alpha=0.5)
                ax2.text(day, 0.9, name, rotation=90, fontsize=8, ha='right')

        ax2.set_xlabel('Day of Year', fontsize=12)
        ax2.set_ylabel('Average Shading Coefficient', fontsize=12)
        ax2.set_title('Daily Shading Coefficient (1.5m Overhang)', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(1, 365)

        plt.tight_layout()
        plt.savefig('figures/02_seasonal_shading.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/02_seasonal_shading.png")
        plt.close()

    def _plot_energy_comparison(self, shading_df):
        """能耗对比分析"""
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 制冷和供暖负荷对比
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(shading_df))

        ax1.bar(x - 0.2, shading_df['cooling_load']/1000, 0.4, label='Cooling Load', color='#3498db', alpha=0.8)
        ax1.bar(x + 0.2, shading_df['heating_load']/1000, 0.4, label='Heating Load', color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Shading Design', fontsize=11)
        ax1.set_ylabel('Energy Load (MWh/year)', fontsize=11)
        ax1.set_title('Cooling vs Heating Load', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(shading_df['name'], rotation=45, ha='right', fontsize=8)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 2. 总能耗对比
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(x, shading_df['total_energy']/1000, color='#2ecc71', alpha=0.8)

        # 标注最小值
        min_idx = shading_df['total_energy'].idxmin()
        bars[min_idx].set_color('#27ae60')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}', ha='center', fontsize=8)

        ax2.set_xlabel('Shading Design', fontsize=11)
        ax2.set_ylabel('Total Energy (MWh/year)', fontsize=11)
        ax2.set_title('Total Energy Consumption', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(shading_df['name'], rotation=45, ha='right', fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 3. 节能效果对比
        ax3 = fig.add_subplot(gs[1, :])
        baseline = shading_df.loc[0]

        cooling_reduction = (1 - shading_df['cooling_load'] / baseline['cooling_load']) * 100
        heating_increase = (shading_df['heating_load'] / baseline['heating_load'] - 1) * 100
        total_change = (1 - shading_df['total_energy'] / baseline['total_energy']) * 100

        ax3.plot(x, cooling_reduction, 'o-', label='Cooling Load Reduction', linewidth=2, markersize=8)
        ax3.plot(x, heating_increase, 's-', label='Heating Load Increase', linewidth=2, markersize=8)
        ax3.plot(x, total_change, '^-', label='Total Energy Change', linewidth=2, markersize=8)

        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax3.axvline(x=min_idx, color='r', linestyle='--', alpha=0.5, label='Optimal Design')

        ax3.set_xlabel('Shading Design', fontsize=11)
        ax3.set_ylabel('Energy Change (%)', fontsize=11)
        ax3.set_title('Energy Saving Performance', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(shading_df['name'], rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)

        plt.savefig('figures/03_energy_comparison.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/03_energy_comparison.png")
        plt.close()

    def _plot_pareto_front(self, shading_df):
        """Pareto前沿分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：制冷 vs 供暖负荷（Pareto前沿）
        scatter = ax1.scatter(shading_df['cooling_load']/1000, shading_df['heating_load']/1000,
                            c=shading_df['total_energy']/1000, cmap='RdYlGn_r',
                            s=200, alpha=0.7, edgecolors='black', linewidths=2)

        # 标注点
        for i, row in shading_df.iterrows():
            ax1.annotate(row['name'], (row['cooling_load']/1000, row['heating_load']/1000),
                        fontsize=7, ha='center', va='bottom')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Total Energy (MWh)', fontsize=10)

        ax1.set_xlabel('Cooling Load (MWh/year)', fontsize=12)
        ax1.set_ylabel('Heating Load (MWh/year)', fontsize=12)
        ax1.set_title('Pareto Front: Cooling vs Heating', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 右图：悬挑深度 vs 能耗
        overhang_only = shading_df[shading_df['type'] == 'overhang']
        ax2.plot(overhang_only['overhang_depth'], overhang_only['total_energy']/1000,
                'o-', linewidth=2, markersize=10, color='#2ecc71')
        ax2.fill_between(overhang_only['overhang_depth'], overhang_only['total_energy']/1000,
                        alpha=0.3, color='#2ecc71')

        # 标注最优点
        min_idx = overhang_only['total_energy'].idxmin()
        optimal_depth = overhang_only.loc[min_idx, 'overhang_depth']
        optimal_energy = overhang_only.loc[min_idx, 'total_energy'] / 1000
        ax2.plot(optimal_depth, optimal_energy, 'r*', markersize=20, label='Optimal Point')

        ax2.set_xlabel('Overhang Depth (m)', fontsize=12)
        ax2.set_ylabel('Total Energy (MWh/year)', fontsize=12)
        ax2.set_title('Energy vs Overhang Depth', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('figures/04_pareto_front.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/04_pareto_front.png")
        plt.close()

    def _plot_orientation_effects(self):
        """不同朝向遮阳效果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 测试不同朝向
        orientations = {'South': 180, 'East': 90, 'West': 270, 'North': 0}

        overhang = OverhangShading(depth=1.5)

        for name, angle in orientations.items():
            monthly_sc = []
            for month in range(1, 13):
                day = month * 30
                pos = self.solar_calc.calculate_solar_position(day, 12)
                sc = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], angle)
                monthly_sc.append(sc)

            ax1.plot(range(1, 13), monthly_sc, 'o-', label=name, linewidth=2, markersize=6)

        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Shading Coefficient', fontsize=12)
        ax1.set_title('Shading Effect by Orientation (1.5m Overhang)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # 日变化（夏至）
        hours = range(6, 19)
        for name, angle in orientations.items():
            hourly_sc = []
            for hour in hours:
                pos = self.solar_calc.calculate_solar_position(172, hour)
                if pos['altitude'] > 0:
                    sc = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], angle)
                else:
                    sc = 1
                hourly_sc.append(sc)

            ax2.plot(hours, hourly_sc, 'o-', label=name, linewidth=2, markersize=6)

        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Shading Coefficient', fontsize=12)
        ax2.set_title('Daily Variation - Summer Solstice', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('figures/05_orientation_effects.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/05_orientation_effects.png")
        plt.close()

    def _plot_shading_geometry_diagrams(self):
        """遮阳几何示意图"""
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 悬挑遮阳（夏季）
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_overhang_diagram(ax1, season='summer')

        # 2. 悬挑遮阳（冬季）
        ax2 = fig.add_subplot(gs[1, 0])
        self._draw_overhang_diagram(ax2, season='winter')

        # 3. 百叶遮阳
        ax3 = fig.add_subplot(gs[0, 1])
        self._draw_louver_diagram(ax3)

        # 4. 垂直遮阳
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_vertical_diagram(ax4)

        # 5. 植被遮阳
        ax5 = fig.add_subplot(gs[:, 2])
        self._draw_vegetation_diagram(ax5)

        plt.savefig('figures/06_shading_geometry.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/06_shading_geometry.png")
        plt.close()

    def _draw_overhang_diagram(self, ax, season='summer'):
        """绘制悬挑示意图"""
        ax.set_xlim(-1, 4)
        ax.set_ylim(-0.5, 4)
        ax.set_aspect('equal')

        # 窗户
        window = Rectangle((0, 0), 2, 2, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(window)

        # 墙体
        wall_left = Rectangle((-0.2, 0), 0.2, 3, facecolor='gray', edgecolor='black')
        wall_right = Rectangle((2, 0), 0.2, 3, facecolor='gray', edgecolor='black')
        ax.add_patch(wall_left)
        ax.add_patch(wall_right)

        # 悬挑
        overhang = Rectangle((0, 2.2), 2.5, 0.2, facecolor='brown', edgecolor='black', linewidth=2)
        ax.add_patch(overhang)

        # 太阳光线
        if season == 'summer':
            # 夏季太阳角度高
            for i in range(5):
                x_start = -0.5 + i * 0.3
                arrow = FancyArrowPatch((x_start, 4), (x_start + 0.8, 0),
                                      arrowstyle='->', mutation_scale=15, color='orange',
                                      linewidth=1.5, alpha=0.7)
                ax.add_patch(arrow)
            title = 'Overhang - Summer\n(High sun angle)'
        else:
            # 冬季太阳角度低
            for i in range(5):
                x_start = -0.5 + i * 0.3
                arrow = FancyArrowPatch((x_start, 4), (x_start + 2.5, 0),
                                      arrowstyle='->', mutation_scale=15, color='orange',
                                      linewidth=1.5, alpha=0.7)
                ax.add_patch(arrow)
            title = 'Overhang - Winter\n(Low sun angle)'

        # 阴影区域
        shadow = Rectangle((0, 0), 1.2, 2, facecolor='gray', alpha=0.3)
        ax.add_patch(shadow)

        ax.text(1, 3.5, title, ha='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (m)', fontsize=10)
        ax.set_ylabel('Height (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _draw_louver_diagram(self, ax):
        """绘制百叶示意图"""
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 3)
        ax.set_aspect('equal')

        # 窗户
        window = Rectangle((0, 0), 2, 2, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(window)

        # 百叶
        for i, y in enumerate(np.linspace(0.3, 1.7, 5)):
            louver_angle = np.radians(45)
            dx = 0.3 * np.cos(louver_angle)
            dy = 0.3 * np.sin(louver_angle)
            louver = Polygon([(0, y), (dx, y + dy), (2 + dx, y + dy), (2, y)],
                            facecolor='darkgray', edgecolor='black', linewidth=1)
            ax.add_patch(louver)

        # 太阳光线
        for i in range(3):
            x_start = 0.2 + i * 0.6
            arrow = FancyArrowPatch((x_start, 3), (x_start - 0.3, 0),
                                  arrowstyle='->', mutation_scale=15, color='orange',
                                  linewidth=1.5, alpha=0.7)
            ax.add_patch(arrow)

        ax.text(1, 2.8, 'Louvers (45 degree)', ha='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (m)', fontsize=10)
        ax.set_ylabel('Height (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _draw_vertical_diagram(self, ax):
        """绘制垂直遮阳示意图"""
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')

        # 窗户
        window = Rectangle((0.5, 0), 1.5, 2, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(window)

        # 垂直遮阳板
        for x in [0.5, 1.0, 1.5, 2.0]:
            fin = Rectangle((x, 2.0), 0.1, 0.5, facecolor='brown', edgecolor='black', linewidth=1)
            ax.add_patch(fin)

        # 太阳光线（从侧面）
        for i in range(3):
            y_start = 2.5 - i * 0.3
            arrow = FancyArrowPatch((-0.3, y_start), (2.5, y_start - 0.5),
                                  arrowstyle='->', mutation_scale=15, color='orange',
                                  linewidth=1.5, alpha=0.7)
            ax.add_patch(arrow)

        ax.text(1.25, 2.8, 'Vertical Fins', ha='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (m)', fontsize=10)
        ax.set_ylabel('Height (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _draw_vegetation_diagram(self, ax):
        """绘制植被遮阳示意图"""
        ax.set_xlim(-1, 4)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect('equal')

        # 窗户
        window = Rectangle((1, 0), 2, 2, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(window)

        # 树冠
        for i in range(3):
            x_pos = 0.5 + i * 1.2
            crown = Circle((x_pos, 3.5), 0.8, facecolor='green', edgecolor='darkgreen', linewidth=2, alpha=0.8)
            ax.add_patch(crown)

            # 树干
            trunk = Rectangle((x_pos - 0.1, 2), 0.2, 1.5, facecolor='brown', edgecolor='black')
            ax.add_patch(trunk)

        # 太阳光线
        for i in range(8):
            x_start = -0.5 + i * 0.6
            y_start = 4.5
            if (x_start - 0.5) % 1.2 < 0.4:  # 部分被树叶遮挡
                arrow = FancyArrowPatch((x_start, y_start), (x_start, 0),
                                      arrowstyle='->', mutation_scale=12, color='orange',
                                      linewidth=1, alpha=0.3)
            else:
                arrow = FancyArrowPatch((x_start, y_start), (x_start, 0),
                                      arrowstyle='->', mutation_scale=12, color='orange',
                                      linewidth=1, alpha=0.8)
            ax.add_patch(arrow)

        ax.text(2, 4.8, 'Vegetation Shading\n(Trees)', ha='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (m)', fontsize=10)
        ax.set_ylabel('Height (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _plot_monthly_energy_distribution(self):
        """月度能耗分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        months = np.arange(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # 不同设计的月度能耗
        designs_to_plot = [
            (0, 'No Shading', '#e74c3c'),
            (2, 'Overhang 1.0m', '#3498db'),
            (3, 'Overhang 1.5m', '#2ecc71'),
            (4, 'Overhang 2.0m', '#9b59b6')
        ]

        for idx, name, color in designs_to_plot:
            monthly_energy = []
            for month in months:
                # 简化估算
                if month in [12, 1, 2]:  # 冬季
                    base_ratio = 1.2
                elif month in [6, 7, 8]:  # 夏季
                    base_ratio = 1.0
                else:
                    base_ratio = 0.6

                # 根据设计调整
                if idx == 0:  # 无遮阳
                    monthly_energy.append(self.base_cooling * 0.3 + self.base_heating * 0.3 * base_ratio)
                else:
                    energy = self.calculate_energy_consumption(float(name.split()[-1].replace('m', '')))
                    if month in [6, 7, 8]:
                        monthly_energy.append(energy['cooling_load'] / 3 * base_ratio)
                    else:
                        monthly_energy.append(energy['heating_load'] / 9 * base_ratio)

            ax1.bar(months, monthly_energy, width=0.2, label=name, color=color, alpha=0.7)

        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Monthly Energy (kWh)', fontsize=12)
        ax1.set_title('Monthly Energy Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_names)
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 累积能耗
        cumulative_data = {name: [] for _, name, _ in designs_to_plot}
        total_energy = {name: 0 for _, name, _ in designs_to_plot}

        for month in months:
            for i, (idx, name, color) in enumerate(designs_to_plot):
                if month in [12, 1, 2]:
                    base_ratio = 1.2
                elif month in [6, 7, 8]:
                    base_ratio = 1.0
                else:
                    base_ratio = 0.6

                if idx == 0:
                    energy = self.base_cooling * 0.3 + self.base_heating * 0.3 * base_ratio
                else:
                    e_data = self.calculate_energy_consumption(float(name.split()[-1].replace('m', '')))
                    if month in [6, 7, 8]:
                        energy = e_data['cooling_load'] / 3 * base_ratio
                    else:
                        energy = e_data['heating_load'] / 9 * base_ratio

                total_energy[name] += energy
                cumulative_data[name].append(total_energy[name])

        for idx, name, color in designs_to_plot:
            ax2.plot(months, cumulative_data[name], 'o-', label=name, linewidth=2, markersize=6, color=color)

        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Cumulative Energy (kWh)', fontsize=12)
        ax2.set_title('Cumulative Energy Consumption', fontsize=14, fontweight='bold')
        ax2.set_xticks(months)
        ax2.set_xticklabels(month_names)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('figures/07_monthly_energy.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/07_monthly_energy.png")
        plt.close()

    def _plot_annual_shading_heatmap(self):
        """全年遮阳系数热力图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 计算全年遮阳系数数据
        sc_data = self.calculate_hourly_shading_coefficients(1.5)

        # 创建透视表：月份 x 小时
        pivot_table = sc_data.pivot_table(values='sc_south', index='hour', columns='month', aggfunc='mean')

        # 热力图
        im1 = ax1.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax1.set_xticks(np.arange(12))
        ax1.set_yticks(np.arange(6, 19, 2))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.set_yticklabels(['6am', '8am', '10am', '12pm', '2pm', '4pm', '6pm'])
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Hour of Day', fontsize=12)
        ax1.set_title('Annual Shading Coefficient Heatmap\n(South-facing, 1.5m Overhang)',
                     fontsize=14, fontweight='bold')

        # 颜色条
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Shading Coefficient', fontsize=10)

        # 南向窗户全年的每日平均遮阳系数
        daily_avg = sc_data.groupby('day')['sc_south'].mean().values
        matrix = daily_avg.reshape(12, 30)[:, :29]  # 调整形状

        im2 = ax2.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax2.set_xticks(np.arange(0, 29, 5))
        ax2.set_yticks(np.arange(12))
        ax2.set_xticklabels(['1', '6', '11', '16', '21', '26'])
        ax2.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.set_xlabel('Day of Month', fontsize=12)
        ax2.set_ylabel('Month', fontsize=12)
        ax2.set_title('Daily Average Shading Coefficient', fontsize=14, fontweight='bold')

        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Shading Coefficient', fontsize=10)

        plt.tight_layout()
        plt.savefig('figures/08_shading_heatmap.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/08_shading_heatmap.png")
        plt.close()

    def run_analysis(self):
        """运行完整分析"""
        print("=" * 60)
        print(f"被动式太阳能遮阳优化分析 - {self.name}")
        print(f"纬度: {self.latitude}deg N")
        print("=" * 60)

        # 分析设计方案
        shading_df = self.analyze_comprehensive_designs()

        # 打印结果
        print(f"\n设计方案评估结果:")
        print("-" * 60)
        for _, row in shading_df.iterrows():
            print(f"\n{row['name']}:")
            print(f"  夏季遮阳系数: {row['summer_shading']:.3f}")
            print(f"  冬季遮阳系数: {row['winter_shading']:.3f}")
            print(f"  制冷负荷: {row['cooling_load']:.0f} kWh")
            print(f"  供暖负荷: {row['heating_load']:.0f} kWh")
            print(f"  总能耗: {row['total_energy']:.0f} kWh")

        # 找最优方案
        best_idx = shading_df['total_energy'].idxmin()
        best_design = shading_df.loc[best_idx]

        print(f"\n{'='*60}")
        print(f"推荐方案: {best_design['name']}")
        print(f"{'='*60}")
        print(f"  制冷负荷: {best_design['cooling_load']:.0f} kWh/year")
        print(f"  供暖负荷: {best_design['heating_load']:.0f} kWh/year")
        print(f"  总能耗: {best_design['total_energy']:.0f} kWh/year")
        print(f"  单位面积能耗: {best_design['total_energy']/self.building.total_floor_area:.1f} kWh/m2/year")

        # 生成可视化
        self.create_comprehensive_visualizations(shading_df)

        # 保存结果
        import os
        os.makedirs('results', exist_ok=True)
        shading_df.to_csv(f'results/{self.university}_detailed_analysis.csv', index=False)

        return shading_df, best_design


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("2026 ICM Problem E: Passive Solar Shading")
    print("Enhanced Solution with Comprehensive Visualization")
    print("=" * 60)

    # 分析Sungrove大学
    print("\n### Sungrove University Analysis ###")
    sungrove = EnhancedPassiveSolarOptimizer(university='sungrove')
    sungrove_df, sungrove_best = sungrove.run_analysis()

    # 分析Borealis大学
    print("\n\n### Borealis University Analysis ###")
    borealis = EnhancedPassiveSolarOptimizer(university='borealis')
    borealis_df, borealis_best = borealis.run_analysis()

    # 生成对比报告
    print("\n\n" + "=" * 60)
    print("两所大学对比分析")
    print("=" * 60)
    print(f"\n纬度影响:")
    print(f"  Sungrove (20deg N): 低纬度，主要考虑夏季遮阳")
    print(f"  Borealis (60deg N): 高纬度，主要考虑冬季太阳热增益")

    print(f"\n建筑适应性:")
    print(f"  Sungrove: 南向窗户面积 {sungrove.building.window_area_south:.0f} m2")
    print(f"  Borealis: 南向窗户面积 {borealis.building.window_area_south:.0f} m2")

    print(f"\n推荐设计对比:")
    print(f"  Sungrove: {sungrove_best['name']}")
    print(f"  Borealis: {borealis_best['name']}")

    print(f"\n能耗对比:")
    print(f"  Sungrove 总能耗: {sungrove_best['total_energy']:.0f} kWh/year")
    print(f"  Borealis 总能耗: {borealis_best['total_energy']:.0f} kWh/year")

    print("\n" + "=" * 60)
    print("分析完成！已生成 8 张图表")
    print("=" * 60)

    return sungrove_df, borealis_df


if __name__ == "__main__":
    main()
