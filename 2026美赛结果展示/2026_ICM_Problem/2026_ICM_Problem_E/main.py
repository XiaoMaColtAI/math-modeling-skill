"""
被动式太阳能遮阳优化 - 主程序
Passive Solar Shading Optimization - Main Program

2026 ICM Problem E Solution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from matplotlib.patches import Polygon, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from solar_position import SolarPositionCalculator
from shading_geometry import OverhangShading, LouverShading, calculate_window_solar_gain
from thermal_simulation import BuildingThermalModel, generate_synthetic_weather_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2


class PassiveSolarShadingOptimizer:
    """被动式太阳能遮阳优化器"""

    def __init__(self, university='sungrove'):
        """
        初始化优化器

        Parameters:
        -----------
        university : str
            'sungrove' (低纬度) 或 'borealis' (高纬度)
        """
        self.university = university

        # 大学参数配置
        if university == 'sungrove':
            self.name = "Sungrove University"
            self.latitude = 20  # 低纬度
            self.climate_type = "Warm"
            self.building_params = {
                'length': 60,
                'width': 24,
                'height': 3,
                'floors': 2,
                'wwr_south': 0.45,
                'wwr_other': 0.30,
                'u_wall': 0.5,
                'u_window': 2.8,
                'thermal_mass_per_area': 200,
                'internal_load': 20
            }
            self.setpoint_cooling = 24
            self.setpoint_heating = 20

        else:  # borealis
            self.name = "Borealis University"
            self.latitude = 60  # 高纬度
            self.climate_type = "Cold"
            self.building_params = {
                'length': 60,
                'width': 24,
                'height': 3,
                'floors': 2,
                'wwr_south': 0.50,  # 高纬度增加南向窗户
                'wwr_other': 0.25,
                'u_wall': 0.3,  # 更好的保温
                'u_window': 1.8,  # 三层玻璃
                'thermal_mass_per_area': 250,  # 更大的热质量
                'internal_load': 15
            }
            self.setpoint_cooling = 25
            self.setpoint_heating = 21

        # 创建太阳位置计算器
        self.solar_calc = SolarPositionCalculator(
            latitude=self.latitude,
            longitude=0,
            timezone=0
        )

        # 创建建筑模型
        self.building = BuildingThermalModel(self.building_params)

    def analyze_solar_position(self):
        """分析太阳位置"""
        print(f"\n{'='*60}")
        print(f"{self.name} - 太阳位置分析")
        print(f"{'='*60}")

        # 关键日期
        key_dates = {
            'Winter Solstice': 356,
            'Summer Solstice': 172,
            'Spring Equinox': 80,
            'Autumn Equinox': 266
        }

        results = {}
        for name, day in key_dates.items():
            sun_times = self.solar_calc.calculate_sunrise_sunset(day)
            noon_pos = self.solar_calc.calculate_solar_position(day, 12)

            results[name] = {
                'day': day,
                'sunrise': sun_times['sunrise'],
                'sunset': sun_times['sunset'],
                'daylight': sun_times['daylight'],
                'noon_altitude': noon_pos['altitude']
            }

            print(f"\n{name} (第{day}天):")
            print(f"  日出: {sun_times['sunrise']:.2f}时, 日落: {sun_times['sunset']:.2f}时")
            print(f"  白昼: {sun_times['daylight']:.2f}小时")
            print(f"  正午高度角: {noon_pos['altitude']:.1f}°")

        return results

    def evaluate_shading_designs(self):
        """评估不同遮阳设计方案"""
        print(f"\n{'='*60}")
        print(f"{self.name} - 遮阳设计方案评估")
        print(f"{'='*60}")

        # 测试方案
        designs = [
            {'name': '无遮阳', 'overhang': 0, 'louver': None},
            {'name': '悬挑1.0m', 'overhang': 1.0, 'louver': None},
            {'name': '悬挑1.5m', 'overhang': 1.5, 'louver': None},
            {'name': '悬挑2.0m', 'overhang': 2.0, 'louver': None},
            {'name': '悬挑1.5m+百叶30°', 'overhang': 1.5, 'louver': 30},
            {'name': '悬挑1.5m+百叶45°', 'overhang': 1.5, 'louver': 45},
            {'name': '悬挑2.0m+百叶45°', 'overhang': 2.0, 'louver': 45},
        ]

        results = []

        for design in designs:
            # 创建遮阳设施
            overhang = OverhangShading(depth=design['overhang'])

            # 计算关键时间的遮阳系数
            summer_sc = overhang.calculate_shading_coefficient(90, 180, 180)  # 夏至正午
            winter_sc = overhang.calculate_shading_coefficient(
                90 - 46, 180, 180
            ) if self.latitude == 20 else overhang.calculate_shading_coefficient(
                90 - (90 - 23.45 - 30), 180, 180
            )

            # 估算能耗（简化模型）
            base_cooling = 50000 if self.university == 'sungrove' else 20000
            base_heating = 30000 if self.university == 'sungrove' else 80000

            cooling_load = base_cooling * (1 - 0.6 * (1 - summer_sc))
            heating_load = base_heating * (1 + 0.4 * (1 - winter_sc))

            total_energy = cooling_load + heating_load

            results.append({
                'design': design['name'],
                'overhang_depth': design['overhang'],
                'louver_angle': design.get('louver', 0),
                'summer_shading': summer_sc,
                'winter_shading': winter_sc,
                'cooling_load': cooling_load,
                'heating_load': heating_load,
                'total_energy': total_energy,
                'cooling_reduction': (1 - cooling_load / base_cooling) * 100,
                'heating_increase': (heating_load / base_heating - 1) * 100
            })

            print(f"\n{design['name']}:")
            print(f"  夏季遮阳系数: {summer_sc:.3f}")
            print(f"  冬季遮阳系数: {winter_sc:.3f}")
            print(f"  制冷负荷: {cooling_load:.0f} kWh (减少 {-results[-1]['cooling_reduction']:.1f}%)")
            print(f"  供暖负荷: {heating_load:.0f} kWh (增加 {results[-1]['heating_increase']:.1f}%)")
            print(f"  总能耗: {total_energy:.0f} kWh")

        return pd.DataFrame(results)

    def calculate_energy_consumption_optimized(self, overhang_depth):
        """优化能耗计算方法"""
        # 夏季（6月21日正午）遮阳系数
        summer_pos = self.solar_calc.calculate_solar_position(172, 12)
        overhang = OverhangShading(depth=overhang_depth)
        summer_sc = overhang.calculate_shading_coefficient(summer_pos['altitude'], summer_pos['azimuth'], 180)

        # 冬季（12月21日正午）遮阳系数
        winter_pos = self.solar_calc.calculate_solar_position(356, 12)
        winter_sc = overhang.calculate_shading_coefficient(winter_pos['altitude'], winter_pos['azimuth'], 180)

        # 改进的能耗模型
        base_cooling = 50000 if self.university == 'sungrove' else 20000
        base_heating = 30000 if self.university == 'sungrove' else 80000

        cooling_load = base_cooling * (1 - 0.7 * (1 - summer_sc))
        heating_load = base_heating * (1 + 0.5 * (1 - winter_sc))

        return {'cooling': cooling_load, 'heating': heating_load}

    def create_visualizations(self, solar_results, shading_results):
        """创建可视化图表"""
        print(f"\n{'='*60}")
        print(f"生成可视化图表...")
        print(f"{'='*60}")

        import os
        os.makedirs('figures', exist_ok=True)

        # 图1: 太阳轨迹图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：全年太阳轨迹
        days = np.arange(1, 366)
        hours = np.arange(6, 19, 1)

        for day_name, day_idx in [('Winter Solstice', 356), ('Summer Solstice', 172),
                                   ('Equinox', 80)]:
            altitudes = []
            azimuths = []
            for hour in hours:
                pos = self.solar_calc.calculate_solar_position(day_idx, hour)
                altitudes.append(pos['altitude'])
                azimuths.append(pos['azimuth_from_south'])

            ax1.plot(azimuths, altitudes, 'o-', label=day_name, linewidth=2, markersize=4)

        ax1.set_xlabel('Azimuth Angle from South (degrees)', fontsize=12)
        ax1.set_ylabel('Solar Altitude (degrees)', fontsize=12)
        ax1.set_title(f'Solar Path Diagram - {self.name}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(0, 90)

        # 右图：正午太阳高度角季节变化
        noon_altitudes = []
        for day in range(1, 366):
            pos = self.solar_calc.calculate_solar_position(day, 12)
            noon_altitudes.append(pos['altitude'])

        ax2.plot(days, noon_altitudes, 'b-', linewidth=2)
        ax2.fill_between(days, 0, noon_altitudes, alpha=0.3)
        ax2.axhline(y=90, color='r', linestyle='--', label='Zenith')
        ax2.set_xlabel('Day of Year', fontsize=12)
        ax2.set_ylabel('Noon Solar Altitude (degrees)', fontsize=12)
        ax2.set_title('Noon Solar Altitude Throughout Year', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig('figures/solar_position.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/solar_position.png")
        plt.close()

        # 图2: 遮阳效果对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        designs = shading_results['design']
        x = np.arange(len(designs))

        # 制冷和供暖负荷对比
        width = 0.35
        ax1.bar(x - width/2, shading_results['cooling_load']/1000, width,
                label='Cooling Load', color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, shading_results['heating_load']/1000, width,
                label='Heating Load', color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Shading Design', fontsize=12)
        ax1.set_ylabel('Energy Load (MWh/year)', fontsize=12)
        ax1.set_title('Cooling vs Heating Load by Design', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(designs, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 总能耗对比
        bars = ax2.bar(x, shading_results['total_energy']/1000, color='#2ecc71', alpha=0.8)
        ax2.set_xlabel('Shading Design', fontsize=12)
        ax2.set_ylabel('Total Energy Consumption (MWh/year)', fontsize=12)
        ax2.set_title('Total Energy Consumption by Design', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(designs, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 标注最小值
        min_idx = shading_results['total_energy'].idxmin()
        bars[min_idx].set_color('#27ae60')
        ax2.text(min_idx, bars[min_idx].get_height() + 1,
                f'Min: {shading_results.loc[min_idx, "total_energy"]/1000:.1f}',
                ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('figures/energy_comparison.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/energy_comparison.png")
        plt.close()

        # 图3: 遮阳几何示意图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 悬挑遮阳示意图
        ax1.set_xlim(-1, 4)
        ax1.set_ylim(-0.5, 3.5)
        ax1.set_aspect('equal')

        # 窗户
        window = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)], closed=True,
                        facecolor='lightblue', edgecolor='black', linewidth=2, label='Window')
        ax1.add_patch(window)

        # 墙体
        wall_left = Polygon([(-0.2, 0), (-0.2, 2.5), (0, 2.5), (0, 0)], closed=True,
                          facecolor='gray', edgecolor='black')
        wall_right = Polygon([(2, 0), (2, 2.5), (2.2, 2.5), (2.2, 0)], closed=True,
                           facecolor='gray', edgecolor='black')
        ax1.add_patch(wall_left)
        ax1.add_patch(wall_right)

        # 悬挑
        overhang = Polygon([(0, 2.2), (2.5, 2.2), (2.5, 2.0), (0, 2.0)], closed=True,
                          facecolor='brown', edgecolor='black', linewidth=2, label='Overhang')
        ax1.add_patch(overhang)

        # 太阳光线（夏季）
        for i in range(5):
            x_start = -0.5 + i * 0.3
            arrow = FancyArrowPatch((x_start, 3.5), (x_start + 1.2, 0.5),
                                  arrowstyle='->', mutation_scale=20, color='orange',
                                  linewidth=1.5, alpha=0.7)
            ax1.add_patch(arrow)

        ax1.text(1, 3, "Summer Sun", ha='center', fontsize=11, fontweight='bold', color='orange')
        ax1.text(1.25, 2.1, "D=1.5m", ha='center', fontsize=10, fontweight='bold', color='white')
        ax1.set_xlabel('Distance (m)', fontsize=12)
        ax1.set_ylabel('Height (m)', fontsize=12)
        ax1.set_title('Overhang Shading Design', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 百叶遮阳示意图
        ax2.set_xlim(-0.5, 2.5)
        ax2.set_ylim(-0.5, 3.5)
        ax2.set_aspect('equal')

        # 窗户
        window = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)], closed=True,
                        facecolor='lightblue', edgecolor='black', linewidth=2)
        ax2.add_patch(window)

        # 百叶
        for i, y in enumerate(np.linspace(0.3, 1.7, 5)):
            louver_angle = np.radians(45)
            dx = 0.3 * np.cos(louver_angle)
            dy = 0.3 * np.sin(louver_angle)
            louver = Polygon([(0, y), (dx, y + dy), (2 + dx, y + dy), (2, y)],
                            closed=True, facecolor='darkgray', edgecolor='black', linewidth=1)
            ax2.add_patch(louver)

        # 太阳光线
        for i in range(3):
            x_start = 0.2 + i * 0.6
            arrow = FancyArrowPatch((x_start, 3), (x_start - 0.5, 0),
                                  arrowstyle='->', mutation_scale=20, color='orange',
                                  linewidth=1.5, alpha=0.7)
            ax2.add_patch(arrow)

        ax2.text(1, 2.8, "Louvers (45°)", ha='center', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Distance (m)', fontsize=12)
        ax2.set_ylabel('Height (m)', fontsize=12)
        ax2.set_title('Louver Shading Design', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig('figures/shading_geometry.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/shading_geometry.png")
        plt.close()

        # 图4: 遮阳系数季节变化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 不同悬挑深度的遮阳系数季节变化
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
        ax1.set_title('Seasonal Shading Coefficient Variation', fontsize=14, fontweight='bold')
        ax1.legend(title='Overhang Depth', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xticks(months)
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # 日变化（夏至）
        hours = np.arange(6, 19)
        for depth in [0.5, 1.0, 1.5, 2.0]:
            hourly_sc = []
            for hour in hours:
                pos = self.solar_calc.calculate_solar_position(172, hour)
                if pos['altitude'] > 0:
                    overhang = OverhangShading(depth=depth)
                    sc = overhang.calculate_shading_coefficient(pos['altitude'], pos['azimuth'], 180)
                else:
                    sc = 1
                hourly_sc.append(sc)

            ax2.plot(hours, hourly_sc, 'o-', label=f'{depth}m', linewidth=2, markersize=6)

        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Shading Coefficient', fontsize=12)
        ax2.set_title('Daily Shading Coefficient - Summer Solstice', fontsize=14, fontweight='bold')
        ax2.legend(title='Overhang Depth', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('figures/04_shading_coefficient_variation.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/04_shading_coefficient_variation.png")
        plt.close()

        # 图5: Pareto前沿分析
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 制冷 vs 供暖负荷散点图
        scatter = ax1.scatter(shading_results['cooling_load']/1000, shading_results['heating_load']/1000,
                            c=shading_results['total_energy']/1000, cmap='RdYlGn_r',
                            s=200, alpha=0.7, edgecolors='black', linewidths=2)

        # 标注点
        for i, row in shading_results.iterrows():
            ax1.annotate(row['design'], (row['cooling_load']/1000, row['heating_load']/1000),
                        fontsize=7, ha='center', va='bottom')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Total Energy (MWh)', fontsize=10)

        ax1.set_xlabel('Cooling Load (MWh/year)', fontsize=12)
        ax1.set_ylabel('Heating Load (MWh/year)', fontsize=12)
        ax1.set_title('Pareto Front: Cooling vs Heating', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 悬挑深度 vs 总能耗
        overhang_only = shading_results[shading_results['louver_angle'] == 0]
        if len(overhang_only) > 1:
            ax2.plot(overhang_only['overhang_depth'], overhang_only['total_energy']/1000,
                    'o-', linewidth=2, markersize=10, color='#2ecc71')
            ax2.fill_between(overhang_only['overhang_depth'], overhang_only['total_energy']/1000,
                            alpha=0.3, color='#2ecc71')

            min_idx = overhang_only['total_energy'].idxmin()
            optimal_depth = overhang_only.loc[min_idx, 'overhang_depth']
            optimal_energy = overhang_only.loc[min_idx, 'total_energy'] / 1000
            ax2.plot(optimal_depth, optimal_energy, 'r*', markersize=20, label='Optimal')

        ax2.set_xlabel('Overhang Depth (m)', fontsize=12)
        ax2.set_ylabel('Total Energy (MWh/year)', fontsize=12)
        ax2.set_title('Energy vs Overhang Depth', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('figures/05_pareto_front.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/05_pareto_front.png")
        plt.close()

        # 图6: 不同朝向遮阳效果
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
        ax1.set_title('Shading by Orientation (1.5m Overhang)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # 夏至日变化
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
        plt.savefig('figures/06_orientation_effects.png', dpi=300, bbox_inches='tight')
        print("  保存: figures/06_orientation_effects.png")
        plt.close()

        # 图7: 月度能耗分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        months = np.arange(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # 选择代表性设计
        key_designs = [
            (0, 'No Shading', '#e74c3c'),
            (2, 'Overhang 1.0m', '#3498db'),
            (3, 'Overhang 1.5m', '#2ecc71'),
            (4, 'Overhang 2.0m', '#9b59b6')
        ]

        for idx, name, color in key_designs:
            monthly_energy = []
            for month in months:
                if month in [12, 1, 2]:  # 冬季
                    base_ratio = 1.2
                elif month in [6, 7, 8]:  # 夏季
                    base_ratio = 1.0
                else:
                    base_ratio = 0.6

                if idx == 0:
                    energy = 50000 * 0.3 + 30000 * 0.3 * base_ratio
                else:
                    depth = shading_results.loc[idx, 'overhang_depth']
                    e = self.calculate_energy_consumption_optimized(depth)
                    if month in [6, 7, 8]:
                        energy = e['cooling'] / 3 * base_ratio
                    else:
                        energy = e['heating'] / 9 * base_ratio

                monthly_energy.append(energy)

            ax1.bar(months, monthly_energy, width=0.2, label=name, color=color, alpha=0.7)

        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Monthly Energy (kWh)', fontsize=12)
        ax1.set_title('Monthly Energy Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_names)
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 累积能耗
        for idx, name, color in key_designs:
            cumulative = []
            total = 0
            for month in months:
                if month in [12, 1, 2]:
                    base_ratio = 1.2
                elif month in [6, 7, 8]:
                    base_ratio = 1.0
                else:
                    base_ratio = 0.6

                if idx == 0:
                    energy = 50000 * 0.3 + 30000 * 0.3 * base_ratio
                else:
                    depth = shading_results.loc[idx, 'overhang_depth']
                    e = self.calculate_energy_consumption_optimized(depth)
                    if month in [6, 7, 8]:
                        energy = e['cooling'] / 3 * base_ratio
                    else:
                        energy = e['heating'] / 9 * base_ratio

                total += energy
                cumulative.append(total)

            ax2.plot(months, cumulative, 'o-', label=name, linewidth=2, markersize=6, color=color)

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

        print(f"\n已生成 7 张图表，保存到 figures/ 目录")

    def generate_summary_report(self, solar_results, shading_results):
        """生成汇总报告"""
        print(f"\n{'='*60}")
        print(f"{self.name} - 设计方案汇总报告")
        print(f"{'='*60}")

        # 找最优方案
        best_idx = shading_results['total_energy'].idxmin()
        best_design = shading_results.loc[best_idx]

        print(f"\n推荐方案:")
        print(f"  设计类型: {best_design['design']}")
        print(f"  悬挑深度: {best_design['overhang_depth']:.2f} m")
        print(f"  百叶角度: {best_design['louver_angle']:.0f}°" if best_design['louver_angle'] > 0 else "  百叶角度: 无")

        print(f"\n能耗性能:")
        print(f"  制冷负荷: {best_design['cooling_load']:.0f} kWh/year")
        print(f"  供暖负荷: {best_design['heating_load']:.0f} kWh/year")
        print(f"  总能耗: {best_design['total_energy']:.0f} kWh/year")
        print(f"  单位面积能耗: {best_design['total_energy']/self.building.total_floor_area:.1f} kWh/m2/year")

        print(f"\n节能效果:")
        print(f"  制冷负荷减少: {best_design['cooling_reduction']:.1f}%%")
        print(f"  供暖负荷变化: {best_design['heating_increase']:+.1f}%%")

        # 与无遮阳对比
        baseline = shading_results.loc[0]
        energy_savings = (baseline['total_energy'] - best_design['total_energy']) / baseline['total_energy'] * 100
        print(f"  相比无遮阳总能耗减少: {energy_savings:.1f}%%")

        return best_design

    def run_full_analysis(self):
        """运行完整分析"""
        print("=" * 60)
        print(f"被动式太阳能遮阳优化分析")
        print(f"目标: {self.name} (纬度: {self.latitude}°N)")
        print("=" * 60)

        # 步骤1: 太阳位置分析
        solar_results = self.analyze_solar_position()

        # 步骤2: 遮阳方案评估
        shading_results = self.evaluate_shading_designs()

        # 步骤3: 生成可视化
        self.create_visualizations(solar_results, shading_results)

        # 步骤4: 生成报告
        best_design = self.generate_summary_report(solar_results, shading_results)

        # 保存结果
        import os
        os.makedirs('results', exist_ok=True)

        shading_results.to_csv(f'results/{self.university}_shading_evaluation.csv', index=False)

        return {
            'solar_results': solar_results,
            'shading_results': shading_results,
            'best_design': best_design
        }


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("2026 ICM Problem E: Passive Solar Shading")
    print("被动式太阳能遮阳优化解决方案")
    print("=" * 60)

    # 分析Sungrove大学
    print("\n### 分析 Sungrove University (低纬度温暖地区) ###")
    sungrove_optimizer = PassiveSolarShadingOptimizer(university='sungrove')
    sungrove_results = sungrove_optimizer.run_full_analysis()

    # 分析Borealis大学
    print("\n\n### 分析 Borealis University (高纬度寒冷地区) ###")
    borealis_optimizer = PassiveSolarShadingOptimizer(university='borealis')
    borealis_results = borealis_optimizer.run_full_analysis()

    # 生成对比报告
    print("\n\n" + "=" * 60)
    print("两所大学对比分析")
    print("=" * 60)

    print(f"\n纬度影响:")
    print(f"  Sungrove ({sungrove_optimizer.latitude}deg N): 低纬度，主要考虑夏季遮阳")
    print(f"  Borealis ({borealis_optimizer.latitude}deg N): 高纬度，主要考虑冬季太阳热增益")

    print(f"\n建筑适应性:")
    print(f"  Sungrove: 南向窗户面积 {sungrove_optimizer.building.window_area_south:.0f} m2")
    print(f"  Borealis: 南向窗户面积 {borealis_optimizer.building.window_area_south:.0f} m2")

    print(f"\n推荐设计对比:")
    s_best = sungrove_results['best_design']
    b_best = borealis_results['best_design']

    print(f"  Sungrove: {s_best['design']}")
    print(f"  Borealis: {b_best['design']}")

    print(f"\n能耗对比:")
    print(f"  Sungrove 总能耗: {s_best['total_energy']:.0f} kWh/year")
    print(f"  Borealis 总能耗: {b_best['total_energy']:.0f} kWh/year")

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
