"""
遮阳多目标优化模块 (Shading Multi-objective Optimization)
使用NSGA-II算法优化遮阳设计参数
平衡夏季遮阳（减少制冷负荷）和冬季太阳热增益（减少供暖负荷）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random


class ShadingDesignProblem:
    """遮阳设计优化问题"""

    def __init__(self, building_model, weather_data, solar_position_data):
        """
        初始化优化问题

        Parameters:
        -----------
        building_model : BuildingThermalModel
            建筑热模型
        weather_data : pandas.DataFrame
            气象数据
        solar_position_data : pandas.DataFrame
            太阳位置数据
        """
        self.building = building_model
        self.weather_data = weather_data
        self.solar_data = solar_position_data

        # 定义决策变量边界
        self.bounds = {
            'overhang_depth': (0.5, 3.0),  # m
            'louver_angle': (0, 90),  # degrees
            'louver_spacing': (0.2, 0.5),  # m
            'vertical_depth': (0.2, 1.0),  # m
            'vegetation_density': (0, 1.0)  # 0-1
        }

        # 定义季节
        self.summer_days = range(152, 243)  # 6-8月
        self.winter_days = list(range(335, 366)) + list(range(1, 59))  # 12, 1-2月

    def decode_chromosome(self, chromosome):
        """
        解码染色体为设计参数

        Parameters:
        -----------
        chromosome : array
            编码的设计参数

        Returns:
        --------
        dict
            设计参数
        """
        return {
            'overhang_depth': chromosome[0],
            'louver_angle': chromosome[1],
            'louver_spacing': chromosome[2],
            'vertical_depth': chromosome[3],
            'vegetation_density': chromosome[4]
        }

    def calculate_shading_coefficients(self, design_params):
        """
        根据设计参数计算遮阳系数

        Parameters:
        -----------
        design_params : dict
            设计参数

        Returns:
        --------
        dict
            各时刻各朝向的遮阳系数
        """
        # 导入遮阳几何模块
        import sys
        sys.path.append('.')
        from shading_geometry import OverhangShading, LouverShading

        # 创建遮阳设施
        overhang = OverhangShading(depth=design_params['overhang_depth'])
        louver = LouverShading(
            angle=design_params['louver_angle'],
            spacing=design_params['louver_spacing']
        )

        # 计算遮阳系数时间序列
        shading_schedules = []

        for _, row in self.solar_data.iterrows():
            altitude = row['altitude']
            azimuth = row['azimuth']

            # 南向窗户
            sc_south_overhang = overhang.calculate_shading_coefficient(
                altitude, azimuth, 180
            )
            sc_south_louver = louver.calculate_shading_coefficient(
                altitude, azimuth, 180
            )
            # 组合效果（相乘）
            sc_south = sc_south_overhang * sc_south_louver

            # 其他朝向（简化处理）
            sc_east = overhang.calculate_shading_coefficient(altitude, azimuth, 90)
            sc_west = overhang.calculate_shading_coefficient(altitude, azimuth, 270)
            sc_north = 1.0  # 北向通常不需要遮阳

            shading_schedules.append({
                'day_of_year': row['day_of_year'],
                'hour': row['hour'],
                'south': sc_south,
                'east': sc_east,
                'west': sc_west,
                'north': sc_north
            })

        return pd.DataFrame(shading_schedules)

    def evaluate_design(self, design_params):
        """
        评估设计方案

        Parameters:
        -----------
        design_params : dict
            设计参数

        Returns:
        --------
        dict
            评估结果
        """
        # 计算遮阳系数
        shading_df = self.calculate_shading_coefficients(design_params)

        # 模拟能耗（简化版，使用度日法）
        summer_data = self.weather_data[
            self.weather_data['day_of_year'].isin(self.summer_days)
        ]
        winter_data = self.weather_data[
            self.weather_data['day_of_year'].isin(self.winter_days)
        ]

        # 夏季平均遮阳系数
        summer_shading = shading_df[
            shading_df['day_of_year'].isin(self.summer_days)
        ]['south'].mean()

        # 冬季平均遮阳系数
        winter_shading = shading_df[
            shading_df['day_of_year'].isin(self.winter_days)
        ]['south'].mean()

        # 估算制冷负荷（简化）
        # 制冷负荷与太阳热增益成正比
        base_cooling_load = 50000  # kWh（基准）
        cooling_load = base_cooling_load * (1 - 0.7 * (1 - summer_shading))

        # 估算供暖负荷
        # 供暖负荷与太阳热增益成反比
        base_heating_load = 30000  # kWh（基准）
        heating_load = base_heating_load * (1 + 0.5 * (1 - winter_shading))

        # 总能耗
        total_energy = cooling_load + heating_load

        return {
            'cooling_load': cooling_load,
            'heating_load': heating_load,
            'total_energy': total_energy,
            'summer_shading': summer_shading,
            'winter_shading': winter_shading
        }


class NSGA2Optimizer:
    """NSGA-II多目标优化算法"""

    def __init__(self, problem, population_size=50, max_generations=100):
        """
        初始化优化器

        Parameters:
        -----------
        problem : ShadingDesignProblem
            优化问题
        population_size : int
            种群大小
        max_generations : int
            最大迭代次数
        """
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations

        # 决策变量数量
        self.n_variables = len(problem.bounds)

        # 变量边界
        self.bounds = list(problem.bounds.values())

    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for i in range(self.n_variables):
                value = random.uniform(self.bounds[i][0], self.bounds[i][1])
                chromosome.append(value)
            population.append(chromosome)
        return population

    def evaluate_population(self, population):
        """评估种群"""
        objectives = []
        for chromosome in population:
            design_params = self.problem.decode_chromosome(chromosome)
            result = self.problem.evaluate_design(design_params)

            # 目标函数：
            # f1: 最小化夏季制冷负荷
            # f2: 最小化冬季供暖负荷（等价于最大化冬季太阳热增益）
            objectives.append([result['cooling_load'], result['heating_load']])

        return objectives

    def fast_non_dominated_sort(self, population, objectives):
        """快速非支配排序"""
        fronts = [[]]
        S = [[] for _ in range(len(population))]
        n = [0] * len(population)
        rank = [0] * len(population)

        for p in range(len(population)):
            S[p] = []
            n[p] = 0
            for q in range(len(population)):
                if self.dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    n[p] += 1

            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts[:-1]

    def dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and \
               any(o1 < o2 for o1, o2 in zip(obj1, obj2))

    def calculate_crowding_distance(self, front, objectives):
        """计算拥挤度距离"""
        if len(front) <= 2:
            return [float('inf')] * len(front)

        distance = [0] * len(front)

        for m in range(len(objectives[0])):
            # 按目标函数m排序
            sorted_indices = sorted(
                range(len(front)),
                key=lambda i: objectives[front[i]][m]
            )

            # 边界点设为无穷大
            distance[sorted_indices[0]] = float('inf')
            distance[sorted_indices[-1]] = float('inf')

            # 计算中间点的拥挤度
            obj_min = objectives[front[sorted_indices[0]]][m]
            obj_max = objectives[front[sorted_indices[-1]]][m]

            if obj_max - obj_min > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distance[sorted_indices[i]] += (
                        objectives[front[sorted_indices[i + 1]]][m] -
                        objectives[front[sorted_indices[i - 1]]][m]
                    ) / (obj_max - obj_min)

        return distance

    def selection(self, population, fronts, objectives):
        """选择操作（锦标赛选择）"""
        selected = []
        for _ in range(len(population)):
            # 随机选择两个个体
            i, j = random.sample(range(len(population)), 2)

            # 比较排序等级
            if fronts[i] < fronts[j]:
                selected.append(population[i][:])
            elif fronts[i] > fronts[j]:
                selected.append(population[j][:])
            else:
                # 等级相同时选择拥挤度大的
                selected.append(population[i][:] if random.random() < 0.5 else population[j][:])

        return selected

    def crossover(self, parent1, parent2):
        """交叉操作（模拟二进制交叉）"""
        child1, child2 = parent1[:], parent2[:]
        eta_c = 20  # 交叉分布指数

        for i in range(len(parent1)):
            if random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    beta = 1.0 + (2.0 * min(
                        parent1[i] - self.bounds[i][0],
                        self.bounds[i][1] - parent1[i]
                    ) / abs(parent2[i] - parent1[i]) if parent2[i] != parent1[i] else 1)

                    alpha = 2.0 - beta ** (-(eta_c + 1.0))

                    u = random.random()
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    child1[i] = 0.5 * ((parent1[i] + parent2[i]) - betaq * abs(parent2[i] - parent1[i]))
                    child2[i] = 0.5 * ((parent1[i] + parent2[i]) + betaq * abs(parent2[i] - parent1[i]))

                # 边界检查
                child1[i] = max(self.bounds[i][0], min(self.bounds[i][1], child1[i]))
                child2[i] = max(self.bounds[i][0], min(self.bounds[i][1], child2[i]))

        return child1, child2

    def mutate(self, chromosome):
        """变异操作（多项式变异）"""
        eta_m = 20  # 变异分布指数

        for i in range(len(chromosome)):
            if random.random() <= 1.0 / len(chromosome):
                delta1 = (chromosome[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0])
                delta2 = (self.bounds[i][1] - chromosome[i]) / (self.bounds[i][1] - self.bounds[i][0])

                rnd = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rnd < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                chromosome[i] = chromosome[i] + deltaq * (self.bounds[i][1] - self.bounds[i][0])
                chromosome[i] = max(self.bounds[i][0], min(self.bounds[i][1], chromosome[i]))

        return chromosome

    def optimize(self):
        """执行优化"""
        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        objectives = self.evaluate_population(population)

        # 非支配排序
        fronts = self.fast_non_dominated_sort(population, objectives)

        # 计算拥挤度
        for front in fronts:
            distance = self.calculate_crowding_distance(front, objectives)

        # 主循环
        for generation in range(self.max_generations):
            # 选择
            selected = self.selection(population, [0] * len(population), objectives)

            # 交叉变异生成后代
            offspring = []
            while len(offspring) < self.population_size:
                i, j = random.sample(range(len(selected)), 2)
                child1, child2 = self.crossover(selected[i], selected[j])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.append(child1)
                if len(offspring) < self.population_size:
                    offspring.append(child2)

            # 合并种群
            combined = population + offspring
            combined_objectives = self.evaluate_population(combined)

            # 非支配排序
            fronts = self.fast_non_dominated_sort(combined, combined_objectives)

            # 选择新种群
            population = []
            objectives = []
            for front in fronts:
                if len(population) + len(front) <= self.population_size:
                    population.extend([combined[i][:] for i in front])
                    objectives.extend([combined_objectives[i] for i in front])
                else:
                    # 按拥挤度选择
                    distance = self.calculate_crowding_distance(front, combined_objectives)
                    sorted_indices = sorted(front, key=lambda i: distance[i], reverse=True)
                    remaining = self.population_size - len(population)
                    for i in range(remaining):
                        population.append(combined[sorted_indices[i]][:])
                        objectives.append(combined_objectives[sorted_indices[i]])
                    break

            if generation % 10 == 0:
                print(f"Generation {generation}: Best solutions in front 0: {len(fronts[0])}")

        # 返回Pareto前沿
        pareto_front = fronts[0]
        pareto_solutions = [combined[i] for i in pareto_front]
        pareto_objectives = [combined_objectives[i] for i in pareto_front]

        return pareto_solutions, pareto_objectives


def main():
    """测试多目标优化"""
    print("=" * 60)
    print("遮阳多目标优化模块测试")
    print("=" * 60)

    # 简化测试（不依赖完整数据）
    print("\n创建简化的优化问题...")

    class SimpleProblem:
        """简化的测试问题"""

        def __init__(self):
            self.bounds = {
                'overhang_depth': (0.5, 2.5),
                'louver_angle': (0, 90)
            }

        def decode_chromosome(self, chromosome):
            return {
                'overhang_depth': chromosome[0],
                'louver_angle': chromosome[1]
            }

        def evaluate(self, chromosome):
            """简化的目标函数"""
            d = chromosome[0]  # overhang depth
            a = chromosome[1]  # louver angle

            # 目标1: 夏季制冷负荷（与遮阳效果相关）
            # 悬挑越深、角度越大，制冷负荷越小
            cooling = 50000 * (1 - 0.6 * d / 2.5 - 0.2 * a / 90)

            # 目标2: 冬季供暖负荷（与冬季太阳热增益相关）
            # 悬挑越深、角度越大，冬季太阳热增益越少，供暖负荷越大
            heating = 30000 * (1 + 0.4 * d / 2.5 + 0.1 * a / 90)

            return cooling, heating

    # 创建优化器
    problem = SimpleProblem()

    # 生成一些随机解来展示Pareto前沿
    print("\n生成随机解并筛选Pareto最优解...")
    solutions = []
    objectives = []

    for _ in range(200):
        chromosome = [
            random.uniform(0.5, 2.5),
            random.uniform(0, 90)
        ]
        obj = problem.evaluate(chromosome)
        solutions.append(chromosome)
        objectives.append(obj)

    # 简单的非支配排序
    pareto_indices = []
    for i, obj1 in enumerate(objectives):
        is_dominated = False
        for j, obj2 in enumerate(objectives):
            if i != j:
                if obj2[0] <= obj1[0] and obj2[1] <= obj1[1] and \
                   (obj2[0] < obj1[0] or obj2[1] < obj1[1]):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)

    print(f"\n找到 {len(pareto_indices)} 个Pareto最优解")

    # 保存结果
    import os
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Pareto前沿数据
    pareto_df = pd.DataFrame([
        {
            'overhang_depth': solutions[i][0],
            'louver_angle': solutions[i][1],
            'cooling_load': objectives[i][0],
            'heating_load': objectives[i][1],
            'total_energy': objectives[i][0] + objectives[i][1]
        }
        for i in pareto_indices
    ])

    pareto_df.to_csv(f'{output_dir}/pareto_front.csv', index=False)

    print("\nPareto前沿部分解:")
    print("-" * 60)
    print(pareto_df.head(10).to_string(index=False))

    # 找折衷解（总能耗最小）
    best_idx = pareto_df['total_energy'].idxmin()
    best_solution = pareto_df.loc[best_idx]

    print("\n" + "=" * 60)
    print("推荐方案（总能耗最小）:")
    print("=" * 60)
    print(f"悬挑深度: {best_solution['overhang_depth']:.2f} m")
    print(f"百叶角度: {best_solution['louver_angle']:.1f}°")
    print(f"制冷负荷: {best_solution['cooling_load']:.0f} kWh")
    print(f"供暖负荷: {best_solution['heating_load']:.0f} kWh")
    print(f"总能耗: {best_solution['total_energy']:.0f} kWh")

    print(f"\nPareto前沿数据已保存到 {output_dir}/pareto_front.csv")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
