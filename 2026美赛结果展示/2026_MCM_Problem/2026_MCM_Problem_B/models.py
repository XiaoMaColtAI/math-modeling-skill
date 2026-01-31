# -*- coding: utf-8 -*-
"""
2026 MCM Problem B - 月球殖民地运输优化模型
核心模型实现
"""

import numpy as np
from scipy.optimize import minimize, linprog
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """场景结果数据类"""
    name: str
    completion_time: float           # 年
    total_cost: float                # 美元
    elevator_usage: float            # 公吨
    rocket_usage: float              # 公吨
    carbon_emissions: float          # 吨CO2
    success_rate: float              # 成功率
    cost_per_ton: float              # 单位成本
    annual_breakdown: List[Dict]     # 年度分解
    cost_breakdown: Dict[str, float] = field(default_factory=dict)  # 成本分解
    validation_status: bool = True   # 验证状态
    warnings: List[str] = field(default_factory=list)  # 警告信息


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_capacity(annual_capacity: float, total_demand: float,
                         max_years: float = 100) -> Tuple[bool, List[str]]:
        """验证运量是否合理"""
        warnings_list = []
        is_valid = True

        estimated_years = total_demand / annual_capacity

        if estimated_years > max_years:
            is_valid = False
            warnings_list.append(f"估计完成时间{estimated_years:.1f}年超过最大可接受时间{max_years}年")

        if annual_capacity < 1000:
            warnings_list.append(f"年运量{annual_capacity:.0f}公吨可能过低")

        return is_valid, warnings_list

    @staticmethod
    def validate_cost(cost: float, material: float,
                     min_cost: float = 100, max_cost: float = 10000) -> Tuple[bool, List[str]]:
        """验证成本是否合理"""
        warnings_list = []
        is_valid = True

        unit_cost = cost / material

        if unit_cost < min_cost:
            warnings_list.append(f"单位成本${unit_cost:.0f}/吨低于合理下限${min_cost}/吨")

        if unit_cost > max_cost:
            warnings_list.append(f"单位成本${unit_cost:.0f}/吨高于合理上限${max_cost}/吨")

        return is_valid, warnings_list


class SpaceElevatorModel:
    """太空电梯运输模型"""

    def __init__(self, config):
        self.config = config
        self.elevator_config = config['ELEVATOR_CONFIG']
        self.project_config = config['PROJECT_CONFIG']
        self.validator = DataValidator()

    def calculate_annual_capacity(self, year: int = 0) -> float:
        """计算指定年份的年运量（考虑建设阶段）"""
        construction_years = self.elevator_config.get('construction_years', 0)

        if year < construction_years:
            # 建设期间运量为0
            return 0.0
        elif year == construction_years:
            # 建成当年部分运量
            return (self.elevator_config['capacity_per_port'] *
                   self.elevator_config['num_ports'] *
                   self.elevator_config['availability'] * 0.5)
        else:
            # 完全运行
            return (self.elevator_config['capacity_per_port'] *
                   self.elevator_config['num_ports'] *
                   self.elevator_config['availability'])

    def calculate_scenario(self, total_material: float = None) -> ScenarioResult:
        """计算仅使用太空电梯的场景"""
        if total_material is None:
            total_material = self.project_config['total_material']

        logger.info(f"计算太空电梯方案，总材料需求: {total_material:,.0f} 公吨")

        construction_years = self.elevator_config.get('construction_years', 0)
        annual_capacity = self.calculate_annual_capacity(construction_years + 1)

        # 验证运量
        is_valid, warnings_list = self.validator.validate_capacity(
            annual_capacity, total_material, 100
        )

        # 计算完成时间（考虑建设期）
        if annual_capacity > 0:
            operational_years_needed = np.ceil(total_material / annual_capacity)
            completion_time = construction_years + operational_years_needed
        else:
            completion_time = float('inf')
            warnings_list.append("太空电梯年运量为0，无法完成运输")

        # 计算总成本
        construction_cost = self.elevator_config['construction_cost']
        transport_cost = total_material * self.elevator_config['unit_cost']

        # 年维护成本（仅运营期间）
        maintenance_cost = 0
        if completion_time < float('inf'):
            operational_years = completion_time - construction_years
            for year in range(int(operational_years)):
                maintenance_cost += construction_cost * self.elevator_config['maintenance_cost_rate']

        total_cost = construction_cost + transport_cost + maintenance_cost

        # 碳排放
        carbon_emissions = total_material * self.elevator_config['carbon_per_ton']

        # 成本分解
        cost_breakdown = {
            'construction': construction_cost,
            'transport': transport_cost,
            'maintenance': maintenance_cost,
        }

        # 年度分解
        annual_breakdown = []
        remaining = total_material

        for year in range(int(completion_time)):
            year_capacity = self.calculate_annual_capacity(year)
            shipped = min(year_capacity, remaining)
            remaining -= shipped

            annual_breakdown.append({
                'year': self.project_config['start_year'] + year,
                'elevator_shipment': shipped,
                'rocket_shipment': 0,
                'cumulative': total_material - remaining,
                'is_construction': year < construction_years
            })

        result = ScenarioResult(
            name="Elevator Only",
            completion_time=completion_time if completion_time < float('inf') else 999,
            total_cost=total_cost,
            elevator_usage=total_material,
            rocket_usage=0,
            carbon_emissions=carbon_emissions,
            success_rate=self.elevator_config['availability'],
            cost_per_ton=total_cost / total_material,
            annual_breakdown=annual_breakdown,
            cost_breakdown=cost_breakdown,
            validation_status=is_valid,
            warnings=warnings_list
        )

        logger.info(f"太空电梯方案: {completion_time:.0f}年, ${total_cost/1e9:.1f}B")
        return result


class RocketModel:
    """火箭运输模型"""

    def __init__(self, config):
        self.config = config
        self.rocket_config = config['ROCKET_CONFIG']
        self.launch_sites = config['LAUNCH_SITES']
        self.project_config = config['PROJECT_CONFIG']
        self.validator = DataValidator()

    def calculate_annual_capacity(self, year: int = 0) -> float:
        """计算指定年份的年运量（考虑技术进步）"""
        total_annual_launches = sum(site['launches_per_year']
                                   for site in self.launch_sites)

        # 考虑发射频率年增长
        growth_rate = self.rocket_config.get('frequency_growth_rate', 1.0)
        adjusted_launches = total_annual_launches * (growth_rate ** year)

        annual_capacity = (adjusted_launches *
                          self.rocket_config['payload_per_launch'] *
                          self.rocket_config['success_rate'])

        return annual_capacity

    def get_effective_unit_cost(self, year: int = 0) -> float:
        """获取考虑技术进步后的有效单位成本"""
        avg_cost_multiplier = np.mean([site['cost_multiplier']
                                      for site in self.launch_sites])

        base_cost = self.rocket_config['unit_cost'] * avg_cost_multiplier

        # 考虑成本年下降
        degradation_rate = self.rocket_config.get('cost_degradation_rate', 1.0)
        effective_cost = base_cost * (degradation_rate ** year)

        return effective_cost

    def calculate_scenario(self, total_material: float = None,
                         use_all_sites: bool = True) -> ScenarioResult:
        """计算仅使用火箭的场景"""
        if total_material is None:
            total_material = self.project_config['total_material']

        logger.info(f"计算火箭方案，总材料需求: {total_material:,.0f} 公吨")

        # 计算考虑技术进步的运量
        annual_capacities = []
        cumulative_capacity = 0
        year = 0

        while cumulative_capacity < total_material and year < 200:
            year_capacity = self.calculate_annual_capacity(year)
            annual_capacities.append(year_capacity)
            cumulative_capacity += year_capacity
            year += 1

        completion_time = year

        # 验证运量
        is_valid, warnings_list = self.validator.validate_capacity(
            sum(annual_capacities) / len(annual_capacities) if annual_capacities else 0,
            total_material, 100
        )

        # 计算总成本（考虑成本逐年下降）
        total_cost = 0
        remaining = total_material

        for year in range(completion_time):
            year_capacity = annual_capacities[year] if year < len(annual_capacities) else 0
            shipped = min(year_capacity, remaining)
            remaining -= shipped

            year_cost = shipped * self.get_effective_unit_cost(year)
            total_cost += year_cost

        # 碳排放
        total_launches = total_material / self.rocket_config['payload_per_launch']
        carbon_emissions = total_launches * self.rocket_config['carbon_per_launch']

        # 成本分解
        cost_breakdown = {
            'launch_operations': total_cost,
            'average_annual': total_cost / completion_time if completion_time > 0 else 0,
        }

        # 年度分解
        annual_breakdown = []
        remaining = total_material

        for year in range(completion_time):
            year_capacity = annual_capacities[year] if year < len(annual_capacities) else 0
            shipped = min(year_capacity, remaining)
            remaining -= shipped

            annual_breakdown.append({
                'year': self.project_config['start_year'] + year,
                'elevator_shipment': 0,
                'rocket_shipment': shipped,
                'cumulative': total_material - remaining,
                'launches': shipped / self.rocket_config['payload_per_launch'],
                'unit_cost': self.get_effective_unit_cost(year)
            })

        result = ScenarioResult(
            name="Rocket Only",
            completion_time=completion_time,
            total_cost=total_cost,
            elevator_usage=0,
            rocket_usage=total_material,
            carbon_emissions=carbon_emissions,
            success_rate=self.rocket_config['success_rate'],
            cost_per_ton=total_cost / total_material,
            annual_breakdown=annual_breakdown,
            cost_breakdown=cost_breakdown,
            validation_status=is_valid,
            warnings=warnings_list
        )

        logger.info(f"火箭方案: {completion_time:.0f}年, ${total_cost/1e9:.1f}B")
        return result


class HybridModel:
    """混合运输模型"""

    def __init__(self, config):
        self.config = config
        self.elevator_model = SpaceElevatorModel(config)
        self.rocket_model = RocketModel(config)
        self.project_config = config['PROJECT_CONFIG']
        self.validator = DataValidator()

    def optimize_mix(self, total_material: float = None,
                    target_time: float = 25) -> ScenarioResult:
        """优化混合方案，在目标时间内完成运输"""
        if total_material is None:
            total_material = self.project_config['total_material']

        logger.info(f"计算混合方案，目标时间: {target_time}年")

        elevator_config = self.config['ELEVATOR_CONFIG']
        rocket_config = self.config['ROCKET_CONFIG']
        construction_years = elevator_config.get('construction_years', 0)

        # 逐年优化分配
        annual_breakdown = []
        remaining = total_material
        total_cost = 0
        total_elevator = 0
        total_rocket = 0
        total_carbon = 0

        for year in range(int(target_time) + construction_years):
            # 当年可用运量
            elevator_capacity = self.elevator_model.calculate_annual_capacity(year)
            rocket_capacity = self.rocket_model.calculate_annual_capacity(year)

            if remaining <= 0:
                break

            # 优化：优先使用成本较低的运输方式
            elevator_cost = elevator_config['unit_cost']
            rocket_cost = self.rocket_model.get_effective_unit_cost(year)

            # 动态分配
            if elevator_cost < rocket_cost and year >= construction_years:
                # 优先使用电梯
                elevator_use = min(elevator_capacity, remaining)
                rocket_use = 0

                if remaining - elevator_use > 0:
                    # 剩余用火箭
                    rocket_use = min(rocket_capacity, remaining - elevator_use)
            else:
                # 优先使用火箭
                rocket_use = min(rocket_capacity, remaining)
                elevator_use = 0

                if year >= construction_years and remaining - rocket_use > 0:
                    elevator_use = min(elevator_capacity, remaining - rocket_use)

            # 确保不超过剩余需求
            total_shipped = elevator_use + rocket_use
            if total_shipped > remaining:
                if elevator_use > 0:
                    elevator_use = remaining * (elevator_use / total_shipped)
                if rocket_use > 0:
                    rocket_use = remaining * (rocket_use / total_shipped)

            # 计算当年成本
            year_cost = (elevator_use * elevator_cost +
                        rocket_use * rocket_config['unit_cost'])

            # 第一年加上电梯建设成本分摊
            if year == 0:
                year_cost += elevator_config['construction_cost'] / target_time * 0.5

            total_cost += year_cost
            total_elevator += elevator_use
            total_rocket += rocket_use

            # 碳排放
            total_carbon += (elevator_use * elevator_config['carbon_per_ton'] +
                           (rocket_use / rocket_config['payload_per_launch']) *
                           rocket_config['carbon_per_launch'])

            annual_breakdown.append({
                'year': self.project_config['start_year'] + year,
                'elevator_shipment': elevator_use,
                'rocket_shipment': rocket_use,
                'cumulative': total_material - remaining,
                'cost': year_cost,
                'is_construction': year < construction_years
            })

            remaining -= (elevator_use + rocket_use)

            # 完成运输
            if remaining <= 0:
                break

        completion_time = len(annual_breakdown)

        # 计算综合成功率
        if total_elevator + total_rocket > 0:
            success_rate = ((elevator_config['availability'] * total_elevator +
                           rocket_config['success_rate'] * total_rocket) /
                          (total_elevator + total_rocket))
        else:
            success_rate = 0.95

        # 成本分解
        cost_breakdown = {
            'elevator_transport': total_elevator * elevator_config['unit_cost'],
            'rocket_transport': total_rocket * rocket_config['unit_cost'],
            'construction': elevator_config['construction_cost'] * 0.5,
        }

        # 验证
        is_valid = remaining <= 0
        warnings_list = []
        if remaining > 0:
            warnings_list.append(f"未完成运输，剩余{remaining:,.0f}公吨")

        result = ScenarioResult(
            name="Hybrid",
            completion_time=completion_time,
            total_cost=total_cost,
            elevator_usage=total_elevator,
            rocket_usage=total_rocket,
            carbon_emissions=total_carbon,
            success_rate=success_rate,
            cost_per_ton=total_cost / total_material,
            annual_breakdown=annual_breakdown,
            cost_breakdown=cost_breakdown,
            validation_status=is_valid,
            warnings=warnings_list
        )

        logger.info(f"混合方案: {completion_time}年, ${total_cost/1e9:.1f}B")
        logger.info(f"  电梯: {total_elevator/1e6:.2f}M吨, 火箭: {total_rocket/1e6:.2f}M吨")

        return result


class SensitivityAnalysisModel:
    """敏感性分析模型"""

    def __init__(self, config):
        self.config = config
        self.hybrid_model = HybridModel(config)

    def monte_carlo_simulation(self, n_runs: int = 10000) -> Dict:
        """蒙特卡洛模拟"""
        logger.info(f"开始蒙特卡洛模拟，{n_runs}次运行")

        results = {
            'completion_time': [],
            'total_cost': [],
            'carbon_emissions': [],
            'elevator_usage': [],
            'rocket_usage': []
        }

        for i in range(n_runs):
            if (i + 1) % 1000 == 0:
                logger.info(f"  进度: {i+1}/{n_runs}")

            # 随机扰动参数
            perturbed_config = self._perturb_config()
            model = HybridModel(perturbed_config)
            result = model.optimize_mix()

            results['completion_time'].append(result.completion_time)
            results['total_cost'].append(result.total_cost)
            results['carbon_emissions'].append(result.carbon_emissions)
            results['elevator_usage'].append(result.elevator_usage)
            results['rocket_usage'].append(result.rocket_usage)

        # 计算统计量
        summary = {}
        for key, values in results.items():
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'percentile_5': float(np.percentile(values, 5)),
                'percentile_25': float(np.percentile(values, 25)),
                'percentile_75': float(np.percentile(values, 75)),
                'percentile_95': float(np.percentile(values, 95)),
                'values': values
            }

        logger.info("蒙特卡洛模拟完成")
        return summary

    def _perturb_config(self) -> Dict:
        """扰动配置参数"""
        config = self.config.copy()
        config = {k: v.copy() if isinstance(v, dict) else v
                 for k, v in config.items()}

        # 扰动电梯参数 - 使用更合理的分布
        elevator_avail = np.random.beta(95, 5)
        config['ELEVATOR_CONFIG']['availability'] = np.clip(elevator_avail, 0.85, 0.99)

        # 扰动火箭参数
        rocket_success = np.random.beta(98, 2)
        config['ROCKET_CONFIG']['success_rate'] = np.clip(rocket_success, 0.95, 0.999)

        # 扰动成本参数
        cost_multiplier = np.random.lognormal(0, 0.1)  # 对数正态分布
        config['ROCKET_CONFIG']['unit_cost'] *= np.clip(cost_multiplier, 0.8, 1.2)

        # 扰动运量参数
        capacity_multiplier = np.random.normal(1.0, 0.05)
        config['ELEVATOR_CONFIG']['capacity_per_port'] *= np.clip(capacity_multiplier, 0.9, 1.1)

        return config


class WaterDemandModel:
    """水资源需求模型"""

    def __init__(self, config):
        self.config = config
        self.water_config = config['WATER_CONFIG']
        self.project_config = config['PROJECT_CONFIG']

    def calculate_annual_demand(self) -> Dict:
        """计算年用水量"""
        population = self.project_config['target_population']
        daily_per_capita = self.water_config['per_capita_daily']
        recycling_rate = self.water_config['recycling_rate']
        reserve_factor = self.water_config['reserve_factor']

        # 计算日用水量（升）
        daily_demand_liters = population * daily_per_capita

        # 考虑循环利用后的实际需求
        net_daily_demand = daily_demand_liters * (1 - recycling_rate)

        # 年需求（升）
        annual_demand_liters = net_daily_demand * self.water_config['days_per_year']

        # 加上储备
        annual_with_reserve = annual_demand_liters * reserve_factor

        # 转换为吨
        annual_demand_tons = annual_with_reserve / 1000 * self.water_config['water_density']

        # 计算运输成本（分别计算电梯和火箭）
        elevator_cost = (annual_demand_tons *
                        self.config['ELEVATOR_CONFIG']['unit_cost'])
        rocket_cost = (annual_demand_tons *
                      self.config['ROCKET_CONFIG']['unit_cost'])

        # 计算需要的发射/运输次数
        num_rocket_launches = annual_demand_tons / self.config['ROCKET_CONFIG']['payload_per_launch']
        elevator_years_needed = annual_demand_tons / (
            self.config['ELEVATOR_CONFIG']['capacity_per_port'] *
            self.config['ELEVATOR_CONFIG']['num_ports'] *
            self.config['ELEVATOR_CONFIG']['availability']
        )

        return {
            'population': population,
            'daily_per_capita_liters': daily_per_capita,
            'daily_demand_liters': daily_demand_liters,
            'daily_demand_m3': daily_demand_liters / 1000,
            'recycling_rate': recycling_rate,
            'net_daily_demand_liters': net_daily_demand,
            'annual_demand_liters': annual_demand_liters,
            'annual_demand_m3': annual_demand_liters / 1000,
            'annual_with_reserve_liters': annual_with_reserve,
            'annual_demand_tons': annual_demand_tons,
            'reserve_factor': reserve_factor,
            'transport_cost_elevator_usd': elevator_cost,
            'transport_cost_rocket_usd': rocket_cost,
            'cost_savings_ratio': (rocket_cost - elevator_cost) / rocket_cost,
            'num_rocket_launches': num_rocket_launches,
            'elevator_years_needed': elevator_years_needed,
            'recommended_split': {
                'elevator_tons': annual_demand_tons * 0.9,  # 90%用电梯
                'rocket_tons': annual_demand_tons * 0.1,    # 10%用火箭
            }
        }


class EnvironmentalImpactModel:
    """环境影响评估模型"""

    def __init__(self, config):
        self.config = config

    def calculate_impact(self, scenario: ScenarioResult) -> Dict:
        """计算环境影响"""
        # 碳排放
        carbon_emissions = scenario.carbon_emissions

        # 碳成本
        carbon_cost = carbon_emissions * self.config['ENVIRONMENT_CONFIG']['carbon_price']

        # 化学污染指数（基于火箭使用量）
        chemical_pollution = (scenario.rocket_usage / 1e6 *
                            self.config['ENVIRONMENT_CONFIG']['chemical_pollution_factor'])

        # 臭氧层影响指数
        ozone_impact = (scenario.rocket_usage / 1e6 *
                       self.config['ENVIRONMENT_CONFIG'].get('ozone_depletion_factor', 0.5))

        # 综合环境指数
        environmental_index = np.sqrt(
            (carbon_emissions / 1e6) ** 2 +
            chemical_pollution ** 2 +
            ozone_impact ** 2
        )

        # 环境评分（0-100，越高越好）
        environmental_score = max(0, 100 - environmental_index * 10)

        return {
            'carbon_emissions_tons': carbon_emissions,
            'carbon_cost_usd': carbon_cost,
            'chemical_pollution_index': chemical_pollution,
            'ozone_impact_index': ozone_impact,
            'environmental_impact_index': environmental_index,
            'environmental_score': environmental_score,
            'relative_impact': environmental_index / 10.0  # 相对于基准
        }


class ModelValidator:
    """模型验证器"""

    @staticmethod
    def validate_scenario_result(result: ScenarioResult) -> Dict:
        """验证场景结果的合理性"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }

        # 检查完成时间
        if result.completion_time > 100:
            validation['warnings'].append(
                f"完成时间{result.completion_time:.0f}年超过100年，可能不合理"
            )

        # 检查成本
        if result.cost_per_ton < 100:
            validation['warnings'].append(
                f"单位成本${result.cost_per_ton:.0f}/吨过低，可能不合理"
            )

        # 检查运量
        total_shipped = result.elevator_usage + result.rocket_usage
        if abs(total_shipped - 100_000_000) / 100_000_000 > 0.01:
            validation['errors'].append(
                f"总运量{total_shipped:,.0f}吨与需求100,000,000吨不匹配"
            )
            validation['is_valid'] = False

        # 检查成功率
        if result.success_rate < 0.8:
            validation['warnings'].append(
                f"成功率{result.success_rate:.1%}较低"
            )

        validation['warnings'].extend(result.warnings)

        return validation
