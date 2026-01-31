# -*- coding: utf-8 -*-
"""
2026 MCM Problem B - 月球殖民地运输优化模型
配置文件
"""

# 项目基础参数
PROJECT_CONFIG = {
    'total_material': 100_000_000,  # 建筑材料总需求（公吨）
    'target_population': 100_000,     # 目标人口（人）
    'start_year': 2050,               # 开始年份
    'target_completion_year': 2070,   # 目标完成年份（20年内）
}

# 太空电梯参数（基于题目数据）
ELEVATOR_CONFIG = {
    # 题目：每个银河港口每年可运输179,000公吨
    'capacity_per_port': 179_000,     # 单港口年运量（公吨）
    'num_ports': 3,                   # 银河港口数量
    'unit_cost': 1000,                # 单位运输成本（$/吨）- 电力驱动成本较低
    'availability': 0.95,             # 可用率（考虑系索摆动、故障等）
    'construction_cost': 100_000_000_000,  # 建设成本（1000亿美元）
    'carbon_per_ton': 0.1,            # 碳排放（吨CO2/吨）- 主要是电力生产
    'maintenance_cost_rate': 0.02,    # 年维护成本率（建设成本的2%）
    'construction_years': 10,         # 建设周期（年）
}

# 火箭参数（基于题目和合理假设）
ROCKET_CONFIG = {
    # 题目：到2050年火箭可运载100-150公吨到月球
    'payload_per_launch': 125,        # 单次载重（公吨）- 取中间值
    'num_launch_sites': 10,           # 发射场数量
    # 题目说10个发射场，假设到2050年技术进步，每个场可发射50次/年
    'launches_per_site_per_year': 50, # 每个发射场年发射次数（技术进步提升）
    'unit_cost': 5000,                # 单位运输成本（$/吨）
    'success_rate': 0.98,             # 成功率（2%失败率）
    'carbon_per_launch': 50,          # 每次发射碳排放（吨CO2）
    'cost_degradation_rate': 0.95,    # 成本年下降率（技术进步）
    'frequency_growth_rate': 1.05,    # 发射频率年增长率
}

# 发射场详细信息（基于题目列出的10个地点）
LAUNCH_SITES = [
    {'name': 'Alaska', 'country': 'USA', 'launches_per_year': 30, 'cost_multiplier': 1.2},
    {'name': 'California', 'country': 'USA', 'launches_per_year': 100, 'cost_multiplier': 1.0},
    {'name': 'Texas', 'country': 'USA', 'launches_per_year': 80, 'cost_multiplier': 1.0},
    {'name': 'Florida', 'country': 'USA', 'launches_per_year': 100, 'cost_multiplier': 0.9},
    {'name': 'Virginia', 'country': 'USA', 'launches_per_year': 25, 'cost_multiplier': 1.1},
    {'name': 'Kazakhstan', 'country': 'Kazakhstan', 'launches_per_year': 30, 'cost_multiplier': 0.8},
    {'name': 'French Guiana', 'country': 'France', 'launches_per_year': 50, 'cost_multiplier': 0.9},
    {'name': 'Satish Dhawan', 'country': 'India', 'launches_per_year': 40, 'cost_multiplier': 0.7},
    {'name': 'Taiyuan', 'country': 'China', 'launches_per_year': 50, 'cost_multiplier': 0.7},
    {'name': 'Mahia Peninsula', 'country': 'New Zealand', 'launches_per_year': 20, 'cost_multiplier': 1.3},
]

# 水资源参数（基于生命维持系统研究）
WATER_CONFIG = {
    'per_capita_daily': 50,           # 人均日用水量（升）
    'recycling_rate': 0.85,           # 水循环利用率（ISS水平）
    'reserve_factor': 1.2,            # 储备系数（20%安全储备）
    'water_density': 1.0,             # 水的密度（吨/立方米）
    'days_per_year': 365,
}

# 环境影响参数
ENVIRONMENT_CONFIG = {
    'carbon_price': 50,               # 碳价格（$/吨CO2）
    'chemical_pollution_factor': 1.0,  # 化学污染因子
    'ozone_depletion_factor': 0.5,    # 臭氧层破坏因子
}

# 模拟参数
SIMULATION_CONFIG = {
    'monte_carlo_runs': 10000,        # 蒙特卡洛模拟次数
    'sobol_samples': 1000,            # Sobol敏感性分析样本数
    'random_seed': 42,
    'parallel_processes': 4,          # 并行进程数
}

# 多目标优化参数
MULTIOBJECTIVE_CONFIG = {
    'population_size': 100,
    'generations': 200,
    'crossover_prob': 0.9,
    'mutation_prob': 0.1,
}

# 验证和检查参数
VALIDATION_CONFIG = {
    'enable_sanity_checks': True,     # 启用合理性检查
    'max_completion_time': 100,       # 最大可接受完成时间（年）
    'min_annual_capacity': 100_000,   # 最小年运量检查（公吨）
    'cost_tolerance': 0.2,            # 成本偏差容忍度（20%）
}

# 输出配置
OUTPUT_CONFIG = {
    'results_dir': './results',
    'figures_dir': './figures',
    'data_dir': './data',
    'figure_dpi': 300,
    'figure_format': 'png',
    'save_intermediate': True,        # 保存中间结果
    'verbose': True,                  # 详细输出
}
