"""
遮阳几何计算模块 (Shading Geometry Calculator)
计算不同类型遮阳设施的遮阳系数和阴影效果
"""

import numpy as np
import pandas as pd
from math import tan, cos, sin, radians, degrees, atan, sqrt


class ShadingDevice:
    """遮阳设施基类"""

    def __init__(self, params):
        """
        初始化遮阳设施

        Parameters:
        -----------
        params : dict
            遮阳参数
        """
        self.params = params

    def calculate_shading_coefficient(self, solar_altitude, solar_azimuth, window_orientation):
        """
        计算遮阳系数

        Parameters:
        -----------
        solar_altitude : float
            太阳高度角（度）
        solar_azimuth : float
            太阳方位角（度，相对于正北）
        window_orientation : float
            窗户朝向（度，相对于正北）

        Returns:
        --------
        float
            遮阳系数（0-1），0为完全遮挡，1为无遮挡
        """
        raise NotImplementedError("子类必须实现此方法")


class OverhangShading(ShadingDevice):
    """水平悬挑遮阳"""

    def __init__(self, depth, height_above_window=0.2, width=None):
        """
        初始化水平悬挑遮阳

        Parameters:
        -----------
        depth : float
            悬挑深度（m）
        height_above_window : float
            悬挑距离窗户上方的距离（m）
        width : float
            悬挑宽度（m），None表示与窗户同宽
        """
        params = {
            'type': 'overhang',
            'depth': depth,
            'height_above_window': height_above_window,
            'width': width
        }
        super().__init__(params)

    def calculate_shading_coefficient(self, solar_altitude, solar_azimuth, window_orientation):
        """
        计算水平悬挑的遮阳系数

        原理：计算悬挑投影在窗户上的阴影面积
        """
        if solar_altitude <= 0:
            return 0  # 无太阳直射

        # 计算太阳相对于窗户法线的角度
        # 窗户朝向转换为弧度（南向为180度）
        window_azimuth_rad = radians(window_orientation)
        solar_azimuth_rad = radians(solar_azimuth)

        # 太阳相对于窗户法线的水平夹角
        relative_azimuth = solar_azimuth_rad - window_azimuth_rad

        # 计算阴影深度
        # 阴影深度 = 悬挑深度 / tan(太阳高度角)
        shadow_depth = self.params['depth'] / tan(radians(solar_altitude))

        # 考虑太阳方位角的影响
        # 当太阳不在窗户正前方时，阴影深度变短
        shadow_depth = shadow_depth * cos(relative_azimuth)

        # 如果阴影深度为负或很小，说明太阳从下方照射（不可能）或阴影很短
        shadow_depth = max(0, shadow_depth)

        # 假设标准窗户高度为2m
        window_height = 2.0

        # 计算遮阳系数（阴影覆盖窗户的比例）
        shading_ratio = min(1.0, shadow_depth / window_height)

        # 遮阳系数 = 1 - 遮挡比例
        # 当shading_ratio = 1时，窗户完全被遮挡，遮阳系数为0
        # 当shading_ratio = 0时，窗户无遮挡，遮阳系数为1
        shading_coefficient = 1 - shading_ratio

        return max(0, min(1, shading_coefficient))


class LouverShading(ShadingDevice):
    """百叶遮阳"""

    def __init__(self, angle, spacing=0.3, width=0.15, reflectivity=0.5):
        """
        初始化百叶遮阳

        Parameters:
        -----------
        angle : float
            百叶倾斜角度（度），0为水平，正值向上倾斜
        spacing : float
            百叶间距（m）
        width : float
            百叶板宽度（m）
        reflectivity : float
            百叶反射率（0-1）
        """
        params = {
            'type': 'louver',
            'angle': angle,
            'spacing': spacing,
            'width': width,
            'reflectivity': reflectivity
        }
        super().__init__(params)

    def calculate_shading_coefficient(self, solar_altitude, solar_azimuth, window_orientation):
        """
        计算百叶遮阳的遮阳系数

        原理：基于百叶角度和太阳位置计算透过率
        """
        if solar_altitude <= 0:
            return 0

        # 百叶角度转换为弧度
        louver_angle = radians(self.params['angle'])

        # 太阳高度角与百叶角度的差值
        angle_diff = radians(solar_altitude) - louver_angle

        # 如果太阳从百叶上方照射
        if angle_diff > 0:
            # 计算光线穿过百叶的路径
            # 简化模型：基于角度差和间距
            transmission = min(1.0, tan(angle_diff) * self.params['spacing'] / self.params['width'])
            transmission = max(0, min(1, transmission))
        else:
            # 太阳被百叶完全遮挡
            transmission = 0

        # 考虑反射率的影响
        # 部分光线会被百叶反射进入室内
        reflected = transmission * self.params['reflectivity'] * 0.5

        # 总透过率 = 直射透过 + 反射
        total_transmission = transmission + reflected

        return max(0, min(1, total_transmission))


class VerticalShading(ShadingDevice):
    """垂直遮阳板（鳍形遮阳）"""

    def __init__(self, depth, spacing=1.0):
        """
        初始化垂直遮阳

        Parameters:
        -----------
        depth : float
            垂直板深度（m）
        spacing : float
            垂直板间距（m）
        """
        params = {
            'type': 'vertical',
            'depth': depth,
            'spacing': spacing
        }
        super().__init__(params)

    def calculate_shading_coefficient(self, solar_altitude, solar_azimuth, window_orientation):
        """
        计算垂直遮阳的遮阳系数
        """
        if solar_altitude <= 0:
            return 0

        # 太阳相对于窗户法线的水平夹角
        relative_azimuth = radians(solar_azimuth - window_orientation)

        # 计算阴影宽度
        shadow_width = self.params['depth'] * tan(abs(relative_azimuth))

        # 计算遮挡比例
        shading_ratio = min(1.0, shadow_width / self.params['spacing'])

        return 1 - shading_ratio


class VegetativeShading(ShadingDevice):
    """植被遮阳"""

    def __init__(self, density=0.7, height=3.0, distance_from_window=1.0):
        """
        初始化植被遮阳

        Parameters:
        -----------
        density : float
            植被密度（0-1），1表示完全茂密
        height : float
            植被高度（m）
        distance_from_window : float
            植被距离窗户的距离（m）
        """
        params = {
            'type': 'vegetation',
            'density': density,
            'height': height,
            'distance': distance_from_window
        }
        super().__init__(params)

    def calculate_shading_coefficient(self, solar_altitude, solar_azimuth, window_orientation):
        """
        计算植被遮阳的遮阳系数

        简化模型：基于植被密度和太阳位置
        """
        if solar_altitude <= 0:
            return 0

        # 计算阴影深度
        shadow_depth = self.params['height'] / tan(radians(solar_altitude))

        # 如果阴影能到达窗户
        if shadow_depth > self.params['distance']:
            # 基于植被密度的遮阳效果
            # 季节变化（简化：夏季茂密，冬季稀疏）
            seasonal_factor = 0.8 + 0.2 * sin(radians(solar_altitude))
            effective_density = self.params['density'] * seasonal_factor

            return 1 - effective_density
        else:
            return 1


class CombinedShading(ShadingDevice):
    """组合遮阳系统"""

    def __init__(self, shading_devices):
        """
        初始化组合遮阳系统

        Parameters:
        -----------
        shading_devices : list
            遮阳设施列表
        """
        params = {'type': 'combined', 'devices': shading_devices}
        super().__init__(params)

    def calculate_shading_coefficient(self, solar_altitude, solar_azimuth, window_orientation):
        """
        计算组合遮阳的遮阳系数

        原理：组合遮阳的遮挡效果叠加
        """
        # 总遮挡率 = 1 - 各个设施未遮挡率的乘积
        unshaded_fraction = 1.0

        for device in self.params['devices']:
            sc = device.calculate_shading_coefficient(
                solar_altitude, solar_azimuth, window_orientation
            )
            unshaded_fraction *= sc

        return unshaded_fraction


def calculate_window_solar_gain(solar_irradiance, window_area, SHGC=0.7, shading_coefficient=1.0):
    """
    计算通过窗户的太阳热增益

    Parameters:
    -----------
    solar_irradiance : float
        太阳辐射强度（W/m²）
    window_area : float
        窗户面积（m²）
    SHGC : float
        太阳热增益系数（0-1）
    shading_coefficient : float
        遮阳系数（0-1）

    Returns:
    --------
    float
        太阳热增益（W）
    """
    return solar_irradiance * window_area * SHGC * shading_coefficient


def main():
    """测试遮阳几何计算"""
    print("=" * 60)
    print("遮阳几何计算模块测试")
    print("=" * 60)

    # 创建不同类型的遮阳设施
    overhang = OverhangShading(depth=1.5, height_above_window=0.2)
    louver = LouverShading(angle=45, spacing=0.3)
    vertical = VerticalShading(depth=0.5, spacing=1.0)
    vegetation = VegetativeShading(density=0.7)

    print("\n" + "=" * 60)
    print("测试不同遮阳类型在正午的遮阳系数")
    print("=" * 60)

    # 测试场景：南向窗户，正午时刻
    window_orientation = 180  # 南向
    test_cases = [
        ("夏至正午", 90, 180),    # 高度角90度，方位角180度（正南）
        ("春秋分正午", 70, 180),  # 高度角70度
        ("冬至正午", 47, 180),    # 高度角47度（纬度20度时的冬至）
    ]

    for name, altitude, azimuth in test_cases:
        print(f"\n{name} (太阳高度角: {altitude}°):")
        print("-" * 40)

        sc_overhang = overhang.calculate_shading_coefficient(altitude, azimuth, window_orientation)
        sc_louver = louver.calculate_shading_coefficient(altitude, azimuth, window_orientation)
        sc_vertical = vertical.calculate_shading_coefficient(altitude, azimuth, window_orientation)
        sc_vegetation = vegetation.calculate_shading_coefficient(altitude, azimuth, window_orientation)

        print(f"悬挑遮阳 (1.5m): 遮阳系数 = {sc_overhang:.3f} (遮挡 {(1-sc_overhang)*100:.1f}%)")
        print(f"百叶遮阳 (45°): 遮阳系数 = {sc_louver:.3f} (遮挡 {(1-sc_louver)*100:.1f}%)")
        print(f"垂直遮阳 (0.5m): 遮阳系数 = {sc_vertical:.3f} (遮挡 {(1-sc_vertical)*100:.1f}%)")
        print(f"植被遮阳 (密度0.7): 遮阳系数 = {sc_vegetation:.3f} (遮挡 {(1-sc_vegetation)*100:.1f}%)")

    # 计算全年遮阳系数曲线
    print("\n" + "=" * 60)
    print("生成全年遮阳系数数据...")
    print("=" * 60)

    # 导入太阳位置计算模块
    import sys
    sys.path.append('.')
    from solar_position import SolarPositionCalculator

    # 创建计算器
    solar_calc = SolarPositionCalculator(latitude=20, longitude=0, timezone=0)

    # 生成全年数据（每小时）
    data = []
    for day in range(1, 366):
        # 获取日出日落时间
        sun_times = solar_calc.calculate_sunrise_sunset(day)
        if sun_times['sunrise'] is None:
            continue

        for hour in range(0, 24):
            if sun_times['sunrise'] <= hour <= sun_times['sunset']:
                # 计算太阳位置
                pos = solar_calc.calculate_solar_position(day, hour)
                if pos['altitude'] > 0:
                    # 计算辐射
                    irradiance = solar_calc.calculate_solar_irradiance_clear_sky(
                        pos['altitude'], day
                    )

                    # 计算不同遮阳的系数（南向窗户）
                    sc_overhang = overhang.calculate_shading_coefficient(
                        pos['altitude'], pos['azimuth'], 180
                    )
                    sc_louver = louver.calculate_shading_coefficient(
                        pos['altitude'], pos['azimuth'], 180
                    )

                    data.append({
                        'day_of_year': day,
                        'hour': hour,
                        'altitude': pos['altitude'],
                        'azimuth': pos['azimuth'],
                        'irradiance': irradiance['total'],
                        'sc_overhang': sc_overhang,
                        'sc_louver': sc_louver
                    })

    df = pd.DataFrame(data)

    # 保存数据
    import os
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}/shading_coefficient_yearly.csv', index=False)
    print(f"全年遮阳系数数据已保存到 {output_dir}/shading_coefficient_yearly.csv")
    print(f"数据点数: {len(df)}")

    # 统计分析
    print("\n" + "=" * 60)
    print("遮阳效果统计分析")
    print("=" * 60)

    # 夏季（6-8月）遮阳效果
    summer_data = df[(df['day_of_year'] >= 152) & (df['day_of_year'] <= 243)]
    winter_data = df[(df['day_of_year'] >= 335) | (df['day_of_year'] <= 59)]

    summer_irradiance = summer_data['irradiance'].sum()
    winter_irradiance = winter_data['irradiance'].sum()

    summer_overhang = (summer_data['irradiance'] * summer_data['sc_overhang']).sum()
    winter_overhang = (winter_data['irradiance'] * winter_data['sc_overhang']).sum()

    summer_louver = (summer_data['irradiance'] * summer_data['sc_louver']).sum()
    winter_louver = (winter_data['irradiance'] * winter_data['sc_louver']).sum()

    print(f"\n夏季总辐射: {summer_irradiance:.0f} W·h/m²")
    print(f"  - 悬挑遮阳后: {summer_overhang:.0f} W·h/m² (减少 {(1-summer_overhang/summer_irradiance)*100:.1f}%)")
    print(f"  - 百叶遮阳后: {summer_louver:.0f} W·h/m² (减少 {(1-summer_louver/summer_irradiance)*100:.1f}%)")

    print(f"\n冬季总辐射: {winter_irradiance:.0f} W·h/m²")
    print(f"  - 悬挑遮阳后: {winter_overhang:.0f} W·h/m² (减少 {(1-winter_overhang/winter_irradiance)*100:.1f}%)")
    print(f"  - 百叶遮阳后: {winter_louver:.0f} W·h/m² (减少 {(1-winter_louver/winter_irradiance)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
