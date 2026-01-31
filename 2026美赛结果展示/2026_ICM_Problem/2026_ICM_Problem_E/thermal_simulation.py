"""
热传递仿真模块 (Thermal Simulation Module)
建筑热传递动态仿真，考虑热质量效应
使用RC网络模型（电阻-电容电路类比）
"""

import numpy as np
import pandas as pd
from math import exp
from typing import Dict, List, Tuple


class ThermalZone:
    """建筑热区域模型"""

    def __init__(self, params):
        """
        初始化热区域

        Parameters:
        -----------
        params : dict
            热区域参数
            - area: 面积 (m²)
            - height: 高度 (m)
            - u_value: 传热系数 (W/m²·K)
            - thermal_mass: 热质量 (kJ/K)
            - window_area: 窗户面积 (m²)
            - window_shgc: 窗户太阳热增益系数
            - internal_load: 内部热负荷 (W/m²)
        """
        self.params = params

        # 计算体积
        self.volume = params['area'] * params['height']

        # 空气热容量（简化）
        self.air_heat_capacity = 1.2 * 1005 * self.volume  # J/K

        # 有效热容量（空气 + 建筑材料）
        self.effective_thermal_mass = (
            params['thermal_mass'] * 1000 + self.air_heat_capacity
        )  # J/K

        # 热阻
        self.thermal_resistance = 1 / (params['u_value'] * params['area'])  # K/W

        # 时间常数（RC）
        self.time_constant = self.thermal_resistance * self.effective_thermal_mass  # 秒

    def calculate_temperature_response(self, T_outdoor, T_indoor_initial,
                                       Q_solar, Q_internal, Q_hvac,
                                       time_step=3600):
        """
        计算温度响应（一阶RC模型）

        Parameters:
        -----------
        T_outdoor : float
            室外温度 (°C)
        T_indoor_initial : float
            初始室内温度 (°C)
        Q_solar : float
            太阳热增益 (W)
        Q_internal : float
            内部热增益 (W)
        Q_hvac : float
            HVAC供热量 (W)，正值为供热，负值为制冷
        time_step : float
            时间步长 (秒)

        Returns:
        --------
        float
            时间步长后的室内温度 (°C)
        """
        # 总热输入
        Q_total = Q_solar + Q_internal + Q_hvac

        # 稳态温度
        T_steady = T_outdoor + Q_total * self.thermal_resistance

        # 温度衰减因子
        lambda_factor = exp(-time_step / self.time_constant)

        # 新温度 = T_steady + lambda * (T_initial - T_steady)
        T_new = T_steady + lambda_factor * (T_indoor_initial - T_steady)

        return T_new


class BuildingThermalModel:
    """建筑热模型"""

    def __init__(self, building_params):
        """
        初始化建筑热模型

        Parameters:
        -----------
        building_params : dict
            建筑参数
            - length: 长度 (m)
            - width: 宽度 (m)
            - height: 层高 (m)
            - floors: 楼层数
            - wwr_south: 南面窗墙比
            - wwr_other: 其他面窗墙比
            - u_wall: 墙体传热系数 (W/m²·K)
            - u_window: 窗户传热系数 (W/m²·K)
            - thermal_mass_per_area: 热质量 (kJ/m²·K)
            - internal_load: 内部负荷 (W/m²)
        """
        self.params = building_params

        # 计算建筑围护结构面积
        self.floor_area = building_params['length'] * building_params['width']
        self.total_floor_area = self.floor_area * building_params['floors']
        self.perimeter = 2 * (building_params['length'] + building_params['width'])
        self.wall_height = building_params['height'] * building_params['floors']

        # 各朝向墙体面积
        self.wall_area_south = building_params['length'] * self.wall_height
        self.wall_area_north = building_params['length'] * self.wall_height
        self.wall_area_east = building_params['width'] * self.wall_height
        self.wall_area_west = building_params['width'] * self.wall_height
        self.total_wall_area = self.wall_area_south + self.wall_area_north + \
                              self.wall_area_east + self.wall_area_west

        # 窗户面积
        self.window_area_south = self.wall_area_south * building_params['wwr_south']
        self.window_area_north = self.wall_area_north * building_params['wwr_other']
        self.window_area_east = self.wall_area_east * building_params['wwr_other']
        self.window_area_west = self.wall_area_west * building_params['wwr_other']
        self.total_window_area = (self.window_area_south + self.window_area_north +
                                 self.window_area_east + self.window_area_west)

        # 墙体面积（扣除窗户）
        self.wall_area_south_solid = self.wall_area_south - self.window_area_south
        self.wall_area_north_solid = self.wall_area_north - self.window_area_north
        self.wall_area_east_solid = self.wall_area_east - self.window_area_east
        self.wall_area_west_solid = self.wall_area_west - self.window_area_west

        # 总热质量
        self.total_thermal_mass = (
            self.total_floor_area * building_params['thermal_mass_per_area'] * 1000
        )  # J/K

        # 初始化热区域（简化：整个建筑为一个区域）
        zone_params = {
            'area': self.total_floor_area,
            'height': building_params['height'] * building_params['floors'],
            'u_value': self._calculate_overall_u_value(),
            'thermal_mass': self.total_thermal_mass / 1000,  # kJ/K
            'window_area': self.total_window_area,
            'window_shgc': 0.7,  # 双层玻璃的SHGC
            'internal_load': building_params['internal_load'] * self.total_floor_area
        }

        self.zone = ThermalZone(zone_params)

        # 初始室内温度
        self.T_indoor = 20.0  # 初始20°C

    def _calculate_overall_u_value(self):
        """计算整体传热系数"""
        # 墙体传热
        q_wall = (
            (self.wall_area_south_solid + self.wall_area_north_solid) *
            self.params['u_wall']
        )
        # 窗户传热
        q_window = self.total_window_area * self.params['u_window']

        total_area = self.total_wall_area + self.total_window_area
        overall_u = (q_wall + q_window) / total_area

        return overall_u

    def simulate_hour(self, weather_data, solar_gains, shading_schedule=None,
                     setpoint_cooling=24, setpoint_heating=20, cop_cooling=3.0,
                     cop_heating=0.9):
        """
        模拟一个小时的建筑热响应

        Parameters:
        -----------
        weather_data : dict
            天气数据
            - temperature: 室外温度 (°C)
        solar_gains : dict
            太阳热增益 (W)
            - south, north, east, west
        shading_schedule : dict or None
            遮阳系数 (0-1)
            - south, north, east, west
        setpoint_cooling : float
            制冷设定温度 (°C)
        setpoint_heating : float
            供暖设定温度 (°C)
        cop_cooling : float
            制冷能效比
        cop_heating : float
            供暖能效比

        Returns:
        --------
        dict
            模拟结果
            - indoor_temp: 室内温度 (°C)
            - cooling_load: 制冷负荷 (kWh)
            - heating_load: 供暖负荷 (kWh)
            - energy_cooling: 制冷能耗 (kWh)
            - energy_heating: 供暖能耗 (kWh)
        """
        # 应用遮阳系数
        if shading_schedule is None:
            shading_schedule = {'south': 1, 'north': 1, 'east': 1, 'west': 1}

        # 计算总太阳热增益
        total_solar_gain = (
            solar_gains.get('south', 0) * shading_schedule.get('south', 1) +
            solar_gains.get('north', 0) * shading_schedule.get('north', 1) +
            solar_gains.get('east', 0) * shading_schedule.get('east', 1) +
            solar_gains.get('west', 0) * shading_schedule.get('west', 1)
        )

        # 计算温度响应（无HVAC时）
        T_outdoor = weather_data['temperature']
        T_indoor_free = self.zone.calculate_temperature_response(
            T_outdoor=T_outdoor,
            T_indoor_initial=self.T_indoor,
            Q_solar=total_solar_gain,
            Q_internal=self.zone.params['internal_load'],
            Q_hvac=0,
            time_step=3600
        )

        # 确定HVAC需求
        cooling_load = 0
        heating_load = 0
        Q_hvac = 0

        if T_indoor_free > setpoint_cooling:
            # 需要制冷
            # 计算移除的热量
            delta_T = T_indoor_free - setpoint_cooling
            cooling_load = delta_T / self.zone.thermal_resistance  # W
            Q_hvac = -cooling_load  # 移除热量

            # 重新计算带制冷的温度
            T_indoor_new = self.zone.calculate_temperature_response(
                T_outdoor=T_outdoor,
                T_indoor_initial=self.T_indoor,
                Q_solar=total_solar_gain,
                Q_internal=self.zone.params['internal_load'],
                Q_hvac=Q_hvac,
                time_step=3600
            )

        elif T_indoor_free < setpoint_heating:
            # 需要供暖
            delta_T = setpoint_heating - T_indoor_free
            heating_load = delta_T / self.zone.thermal_resistance  # W
            Q_hvac = heating_load  # 添加热量

            # 重新计算带供暖的温度
            T_indoor_new = self.zone.calculate_temperature_response(
                T_outdoor=T_outdoor,
                T_indoor_initial=self.T_indoor,
                Q_solar=total_solar_gain,
                Q_internal=self.zone.params['internal_load'],
                Q_hvac=Q_hvac,
                time_step=3600
            )

        else:
            # 舒适范围内，无需HVAC
            T_indoor_new = T_indoor_free

        # 计算能耗（转换为kWh）
        energy_cooling = cooling_load / 1000 / cop_cooling  # kWh
        energy_heating = heating_load / 1000 / cop_heating  # kWh

        # 更新室内温度
        self.T_indoor = T_indoor_new

        return {
            'indoor_temp': T_indoor_new,
            'cooling_load': cooling_load / 1000,  # kW
            'heating_load': heating_load / 1000,  # kW
            'energy_cooling': energy_cooling,  # kWh
            'energy_heating': energy_heating,  # kWh
            'solar_gain': total_solar_gain / 1000  # kW
        }


def generate_synthetic_weather_data(latitude, days=365):
    """
    生成合成气象数据（用于测试）

    Parameters:
    -----------
    latitude : float
        纬度
    days : int
        天数

    Returns:
    --------
    pandas.DataFrame
        气象数据
    """
    data = []

    for day in range(1, days + 1):
        # 季节变化（简化模型）
        seasonal_temp = 20 - 15 * cos(2 * np.pi * (day - 15) / 365)

        # 纬度影响
        latitude_effect = (30 - latitude) / 30 * 5

        for hour in range(24):
            # 日变化
            daily_variation = 5 * sin(2 * np.pi * (hour - 9) / 24)

            # 随机波动
            random_variation = np.random.normal(0, 2)

            temperature = seasonal_temp + latitude_effect + daily_variation + random_variation

            # 太阳辐射（白天才有）
            if 6 <= hour <= 18:
                solar_factor = sin(np.pi * (hour - 6) / 12)
                irradiance = 800 * solar_factor * (1 + 0.1 * np.random.randn())
                irradiance = max(0, irradiance)
            else:
                irradiance = 0

            data.append({
                'day_of_year': day,
                'hour': hour,
                'temperature': temperature,
                'irradiance': irradiance
            })

    return pd.DataFrame(data)


def calculate_solar_gains_by_orientation(solar_position_data, irradiance_data,
                                       window_areas, window_shgc=0.7):
    """
    根据朝向计算太阳热增益

    Parameters:
    -----------
    solar_position_data : dict
        太阳位置数据
    irradiance_data : dict
        辐射数据
    window_areas : dict
        各朝向窗户面积
    window_shgc : float
        窗户太阳热增益系数

    Returns:
    --------
    dict
        各朝向太阳热增益 (W)
    """
    gains = {}

    for orientation in ['south', 'east', 'north', 'west']:
        orientation_angle = {'south': 180, 'east': 90, 'north': 0, 'west': 270}[orientation]

        # 计算入射角
        solar_altitude = solar_position_data['altitude']
        solar_azimuth = solar_position_data['azimuth']

        # 相对方位角
        relative_azimuth = abs(solar_azimuth - orientation_angle)
        if relative_azimuth > 180:
            relative_azimuth = 360 - relative_azimuth

        # 入射角
        if solar_altitude > 0:
            incident_angle = degrees(acos(
                max(0, min(1,
                   sin(radians(solar_altitude)) * cos(radians(relative_azimuth))
                ))
            ))
        else:
            incident_angle = 90

        # 投影面积（简化）
        projection_factor = max(0, cos(radians(incident_angle)))

        # 太阳热增益
        gains[orientation] = (
            irradiance_data['total'] *
            window_areas[orientation] *
            window_shgc *
            projection_factor
        )

    return gains


def main():
    """测试热传递仿真"""
    print("=" * 60)
    print("热传递仿真模块测试")
    print("=" * 60)

    # 建筑参数（Sungrove大学 Academic Hall North）
    building_params = {
        'length': 60,
        'width': 24,
        'height': 3,
        'floors': 2,
        'wwr_south': 0.45,
        'wwr_other': 0.30,
        'u_wall': 0.5,  # W/m²·K（保温墙体）
        'u_window': 2.8,  # W/m²·K（双层玻璃）
        'thermal_mass_per_area': 200,  # kJ/m²·K（混凝土）
        'internal_load': 20  # W/m²（教室/办公室）
    }

    # 创建建筑模型
    building = BuildingThermalModel(building_params)

    print("\n建筑信息:")
    print("-" * 40)
    print(f"建筑面积: {building.total_floor_area:.0f} m²")
    print(f"窗户总面积: {building.total_window_area:.0f} m²")
    print(f"南向窗户: {building.window_area_south:.0f} m²")
    print(f"总热质量: {building.total_thermal_mass/1e6:.2f} MJ/K")
    print(f"时间常数: {building.zone.time_constant/3600:.1f} 小时")

    # 生成测试气象数据
    weather_data = generate_synthetic_weather_data(latitude=20, days=365)

    # 模拟全年
    print("\n" + "=" * 60)
    print("开始全年仿真...")
    print("=" * 60)

    results = []
    T_indoor = 20  # 初始温度

    for day in range(1, 366):
        for hour in range(24):
            idx = (day - 1) * 24 + hour
            if idx >= len(weather_data):
                break

            # 当前天气
            current_weather = {
                'temperature': weather_data.loc[idx, 'temperature']
            }

            # 估算太阳热增益
            if 6 <= hour <= 18 and weather_data.loc[idx, 'irradiance'] > 0:
                solar_gains = {
                    'south': weather_data.loc[idx, 'irradiance'] *
                            building.window_area_south * 0.7,
                    'north': weather_data.loc[idx, 'irradiance'] *
                            building.window_area_north * 0.3,
                    'east': weather_data.loc[idx, 'irradiance'] *
                            building.window_area_east * 0.5,
                    'west': weather_data.loc[idx, 'irradiance'] *
                            building.window_area_west * 0.5
                }
            else:
                solar_gains = {'south': 0, 'north': 0, 'east': 0, 'west': 0}

            # 模拟
            result = building.simulate_hour(
                weather_data=current_weather,
                solar_gains=solar_gains,
                setpoint_cooling=24,
                setpoint_heating=20
            )

            results.append({
                'day_of_year': day,
                'hour': hour,
                'outdoor_temp': current_weather['temperature'],
                'indoor_temp': result['indoor_temp'],
                'cooling_load': result['cooling_load'],
                'heating_load': result['heating_load'],
                'energy_cooling': result['energy_cooling'],
                'energy_heating': result['energy_heating'],
                'solar_gain': result['solar_gain']
            })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 统计结果
    print("\n" + "=" * 60)
    print("全年能耗统计")
    print("=" * 60)

    annual_cooling = results_df['energy_cooling'].sum()
    annual_heating = results_df['energy_heating'].sum()
    annual_total = annual_cooling + annual_heating

    summer_months = [6, 7, 8]
    winter_months = [12, 1, 2]

    summer_data = results_df[results_df['day_of_year'].apply(
        lambda d: (d - 1) // 30 + 1 in summer_months
    )]
    winter_data = results_df[results_df['day_of_year'].apply(
        lambda d: (d - 1) // 30 + 1 in [12, 1, 2]
    )]

    summer_cooling = summer_data['energy_cooling'].sum()
    winter_heating = winter_data['energy_heating'].sum()

    print(f"\n全年制冷能耗: {annual_cooling:.0f} kWh")
    print(f"全年供暖能耗: {annual_heating:.0f} kWh")
    print(f"全年总能耗: {annual_total:.0f} kWh")
    print(f"\n夏季制冷能耗: {summer_cooling:.0f} kWh")
    print(f"冬季供暖能耗: {winter_heating:.0f} kWh")
    print(f"\n单位面积能耗: {annual_total/building.total_floor_area:.1f} kWh/m²")

    # 保存结果
    import os
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(f'{output_dir}/thermal_simulation_results.csv', index=False)
    print(f"\n结果已保存到 {output_dir}/thermal_simulation_results.csv")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
