"""
太阳位置计算模块 (Solar Position Calculator)
计算全年任意时刻的太阳位置（高度角、方位角）
基于Reda & Andreas (2004)的太阳位置算法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import sin, cos, tan, asin, acos, atan2, radians, degrees


class SolarPositionCalculator:
    """太阳位置计算器"""

    def __init__(self, latitude, longitude, timezone, year=2024):
        """
        初始化太阳位置计算器

        Parameters:
        -----------
        latitude : float
            纬度（度），北纬为正
        longitude : float
            经度（度），东经为正
        timezone : float
            时区（小时），东八区为8
        year : int
            年份
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.year = year

        # 常数
        self.solar_constant = 1367  # W/m²
        self.axial_tilt = 23.44  # 地轴倾角（度）

    def calculate_day_of_year(self, date):
        """计算年积日（1-365）"""
        beginning_of_year = datetime(date.year, 1, 1)
        delta = date - beginning_of_year
        return delta.days + 1

    def calculate_solar_declination(self, day_of_year):
        """
        计算太阳赤纬角

        Parameters:
        -----------
        day_of_year : int
            年积日（1-365）

        Returns:
        --------
        float
            太阳赤纬角（度）
        """
        # Cooper方程
        declination = 23.45 * sin(radians(360 * (284 + day_of_year) / 365))
        return declination

    def calculate_equation_of_time(self, day_of_year):
        """
        计算时间方程（真太阳时与平太阳时的差值）

        Parameters:
        -----------
        day_of_year : int
            年积日（1-365）

        Returns:
        --------
        float
            时间方程（分钟）
        """
        b = 360 * (day_of_year - 81) / 365
        eot = 9.87 * sin(radians(2 * b)) - 7.53 * cos(radians(b)) - 1.5 * sin(radians(b))
        return eot

    def calculate_hour_angle(self, solar_time):
        """
        计算时角

        Parameters:
        -----------
        solar_time : float
            真太阳时（小时，0-24）

        Returns:
        --------
        float
            时角（度），正午为0度，上午为负，下午为正
        """
        return 15 * (solar_time - 12)

    def calculate_solar_time(self, clock_time, day_of_year):
        """
        计算真太阳时

        Parameters:
        -----------
        clock_time : float
            钟表时间（小时，0-24）
        day_of_year : int
            年积日

        Returns:
        --------
        float
            真太阳时（小时）
        """
        # 时间方程修正
        eot = self.calculate_equation_of_time(day_of_year)

        # 经度修正（每度4分钟）
        longitude_correction = 4 * (self.longitude - 15 * self.timezone)

        # 真太阳时 = 钟表时间 + 时间方程修正 + 经度修正
        solar_time = clock_time + eot / 60 + longitude_correction / 60

        # 归一化到0-24范围
        solar_time = solar_time % 24

        return solar_time

    def calculate_solar_position(self, day_of_year, hour):
        """
        计算太阳位置（高度角和方位角）

        Parameters:
        -----------
        day_of_year : int
            年积日（1-365）
        hour : float
            钟表时间（小时，0-24）

        Returns:
        --------
        dict
            {'altitude': 太阳高度角（度）, 'azimuth': 太阳方位角（度, 南向为0）}
        """
        # 计算真太阳时
        solar_time = self.calculate_solar_time(hour, day_of_year)

        # 计算太阳赤纬角
        declination = self.calculate_solar_declination(day_of_year)

        # 计算时角
        hour_angle = self.calculate_hour_angle(solar_time)

        # 纬度和赤纬角转换为弧度
        lat_rad = radians(self.latitude)
        dec_rad = radians(declination)
        ha_rad = radians(hour_angle)

        # 计算太阳高度角
        sin_altitude = (sin(lat_rad) * sin(dec_rad) +
                        cos(lat_rad) * cos(dec_rad) * cos(ha_rad))
        altitude = degrees(asin(sin_altitude))

        # 计算太阳方位角（相对于正北）
        # 先计算相对于正南的方位角
        cos_azimuth = ((sin(dec_rad) * cos(lat_rad) -
                        cos(dec_rad) * sin(lat_rad) * cos(ha_rad)) /
                       cos(radians(altitude)))

        # 处理数值误差
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth_from_south = degrees(acos(cos_azimuth))

        # 根据时角判断方位（上午偏东，下午偏西）
        if hour_angle > 0:  # 下午
            azimuth_from_south = -azimuth_from_south

        # 转换为相对于正北的方位角
        azimuth_from_north = (azimuth_from_south + 180) % 360

        return {
            'altitude': max(0, altitude),  # 高度角不能为负
            'azimuth': azimuth_from_north,  # 相对于正北
            'azimuth_from_south': azimuth_from_south,  # 相对于正南
            'hour_angle': hour_angle,
            'declination': declination
        }

    def calculate_sunrise_sunset(self, day_of_year):
        """
        计算日出日落时间

        Parameters:
        -----------
        day_of_year : int
            年积日

        Returns:
        --------
        dict
            {'sunrise': 日出时间（小时）, 'sunset': 日落时间（小时）, 'daylight': 白昼时长（小时）}
        """
        declination = self.calculate_solar_declination(day_of_year)
        lat_rad = radians(self.latitude)
        dec_rad = radians(declination)

        # 计算日出日落的时角
        cos_hour_angle = -tan(lat_rad) * tan(dec_rad)

        # 极地情况处理
        if cos_hour_angle > 1:
            # 极夜
            return {'sunrise': None, 'sunset': None, 'daylight': 0}
        elif cos_hour_angle < -1:
            # 极昼
            return {'sunrise': 0, 'sunset': 24, 'daylight': 24}

        hour_angle_rad = acos(cos_hour_angle)
        hour_angle = degrees(hour_angle_rad)

        # 日出日落时间（真太阳时）
        sunrise_solar = 12 - hour_angle / 15
        sunset_solar = 12 + hour_angle / 15

        # 转换为钟表时间
        eot = self.calculate_equation_of_time(day_of_year)
        longitude_correction = 4 * (self.longitude - 15 * self.timezone)

        sunrise = sunrise_solar - eot / 60 - longitude_correction / 60
        sunset = sunset_solar - eot / 60 - longitude_correction / 60

        return {
            'sunrise': sunrise % 24,
            'sunset': sunset % 24,
            'daylight': (sunset - sunrise) % 24
        }

    def calculate_yearly_solar_position(self, time_step=1):
        """
        计算全年太阳位置数据

        Parameters:
        -----------
        time_step : float
            时间步长（小时），默认1小时

        Returns:
        --------
        pandas.DataFrame
            包含全年太阳位置的DataFrame
        """
        data = []

        # 遍历全年每一天
        for day in range(1, 366):
            date = datetime(self.year, 1, 1) + timedelta(days=day-1)

            # 日出日落时间
            sun_times = self.calculate_sunrise_sunset(day)

            if sun_times['sunrise'] is None:  # 极夜
                continue

            # 白昼时间内每隔time_step小时计算一次
            for hour in np.arange(0, 24, time_step):
                # 只计算白昼时间
                if sun_times['sunrise'] <= hour <= sun_times['sunset']:
                    position = self.calculate_solar_position(day, hour)
                    data.append({
                        'day_of_year': day,
                        'date': date.strftime('%Y-%m-%d'),
                        'hour': hour,
                        'altitude': position['altitude'],
                        'azimuth': position['azimuth'],
                        'azimuth_from_south': position['azimuth_from_south'],
                        'declination': position['declination'],
                        'hour_angle': position['hour_angle']
                    })

        df = pd.DataFrame(data)
        return df

    def calculate_solar_irradiance_clear_sky(self, altitude, day_of_year):
        """
        计算晴空太阳辐射（ASHRAE模型）

        Parameters:
        -----------
        altitude : float
            太阳高度角（度）
        day_of_year : int
            年积日

        Returns:
        --------
        dict
            {'direct': 直射辐射, 'diffuse': 散射辐射, 'total': 总辐射} (W/m²)
        """
        if altitude <= 0:
            return {'direct': 0, 'diffuse': 0, 'total': 0}

        # 地外辐射
        extraterrestrial = self.solar_constant * (
            1 + 0.033 * cos(radians(360 * day_of_year / 365))
        )

        # 大气透明度（简化模型）
        air_mass = 1 / sin(radians(altitude))
        tau = 0.7 ** (air_mass ** 0.678)  # 大气透过率

        # 直射辐射
        direct = extraterrestrial * tau * sin(radians(altitude))

        # 散射辐射（经验公式）
        diffuse = 0.1 * extraterrestrial * sin(radians(altitude))

        # 总辐射
        total = direct + diffuse

        return {
            'direct': max(0, direct),
            'diffuse': max(0, diffuse),
            'total': max(0, total)
        }


def main():
    """测试太阳位置计算"""
    print("=" * 60)
    print("太阳位置计算模块测试")
    print("=" * 60)

    # 创建计算器实例（Sungrove大学：低纬度，约20°N）
    sungrove = SolarPositionCalculator(latitude=20, longitude=0, timezone=0)

    # 测试关键日期
    key_dates = {
        'Winter Solstice': 356,  # 约12月21日
        'Summer Solstice': 172,  # 约6月21日
        'Spring Equinox': 80,    # 约3月21日
        'Autumn Equinox': 266    # 约9月23日
    }

    print(f"\n{'='*60}")
    print(f"Sungrove University (纬度: {sungrove.latitude}°N)")
    print(f"{'='*60}")

    for name, day in key_dates.items():
        print(f"\n{name} (第{day}天):")
        print("-" * 40)

        # 日出日落
        sun_times = sungrove.calculate_sunrise_sunset(day)
        print(f"日出时间: {sun_times['sunrise']:.2f} 时")
        print(f"日落时间: {sun_times['sunset']:.2f} 时")
        print(f"白昼时长: {sun_times['daylight']:.2f} 小时")

        # 正午太阳位置
        noon_pos = sungrove.calculate_solar_position(day, 12)
        print(f"正午太阳高度角: {noon_pos['altitude']:.2f}°")

        # 正午辐射
        irradiance = sungrove.calculate_solar_irradiance_clear_sky(
            noon_pos['altitude'], day
        )
        print(f"正午总辐射: {irradiance['total']:.1f} W/m²")

    # 创建Borealis大学计算器（高纬度，约60°N）
    borealis = SolarPositionCalculator(latitude=60, longitude=0, timezone=0)

    print(f"\n{'='*60}")
    print(f"Borealis University (纬度: {borealis.latitude}°N)")
    print(f"{'='*60}")

    for name, day in key_dates.items():
        print(f"\n{name} (第{day}天):")
        print("-" * 40)

        sun_times = borealis.calculate_sunrise_sunset(day)
        if sun_times['sunrise'] is not None:
            print(f"日出时间: {sun_times['sunrise']:.2f} 时")
            print(f"日落时间: {sun_times['sunset']:.2f} 时")
            print(f"白昼时长: {sun_times['daylight']:.2f} 小时")

            noon_pos = borealis.calculate_solar_position(day, 12)
            print(f"正午太阳高度角: {noon_pos['altitude']:.2f}°")

            irradiance = borealis.calculate_solar_irradiance_clear_sky(
                noon_pos['altitude'], day
            )
            print(f"正午总辐射: {irradiance['total']:.1f} W/m²")
        else:
            print("极夜现象（全天无日照）")

    # 生成全年太阳位置数据并保存
    print(f"\n{'='*60}")
    print("生成全年太阳位置数据...")
    print(f"{'='*60}")

    yearly_data_sungrove = sungrove.calculate_yearly_solar_position(time_step=1)
    print(f"Sungrove全年数据点数: {len(yearly_data_sungrove)}")

    yearly_data_borealis = borealis.calculate_yearly_solar_position(time_step=1)
    print(f"Borealis全年数据点数: {len(yearly_data_borealis)}")

    # 保存到文件
    output_dir = 'results'
    import os
    os.makedirs(output_dir, exist_ok=True)

    yearly_data_sungrove.to_csv(f'{output_dir}/solar_position_sungrove.csv', index=False)
    yearly_data_borealis.to_csv(f'{output_dir}/solar_position_borealis.csv', index=False)

    print(f"\n数据已保存到 {output_dir}/ 目录")

    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
