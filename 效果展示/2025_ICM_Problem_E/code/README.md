# 2025 ICM Problem E: Making Room for Agriculture - 项目说明

## 项目概述

本项目为2025年ICM数学建模竞赛E题"Making Room for Agriculture"的完整解决方案，主要研究森林转变为农业用地后的生态系统演化过程。

## 目录结构

```
E/
├── 2025_ICM_Problem_E.md          # 题目文件
├── 2025_ICM_Problem_E.pdf         # 题目PDF
├── code/                          # 代码目录
│   ├── Problem1_生态系统基础模型.py
│   ├── Problem2_物种重新出现模型.py
│   ├── Problem3_稳定性分析与有机农业评价.py
│   ├── main.py                    # 主程序
│   └── README.md                  # 本文件
├── figures/                       # 生成的图表
├── data/                          # 结果数据
├── output/                        # 分析文档
│   ├── 题目分析报告.md
│   └── 术语表格.md
└── 论文.md                        # 最终论文
```

## 模型说明

### 1. 基础农业生态系统模型 (Problem1)

**模型类型**: 改进的Lotka-Volterra生态系统动力学模型

**状态变量**:
- C: 作物生物量 (kg/m²)
- I: 昆虫种群数量
- B: 鸟类种群数量
- H: 化学物质浓度 (mg/kg)
- T: 蝙蝠种群数量

**主要方程**:
```
dC/dt = r_c·C·(1 - C/K_c) - a_CI·C·I
dI/dt = r_I·I·(1 - I/K_I) - a_IB·I·B - d_I·I
dB/dt = r_B·B·(1 - B/K_B) - d_B·B + e·a_IB·I·B
```

**功能**:
- 模拟农业生态系统中物种间的相互作用
- 分析除草剂/杀虫剂的影响
- 研究蝙蝠引入的效果

### 2. 物种重新出现模型 (Problem2)

**模型类型**: 栖息地成熟度驱动下的物种迁入模型

**状态变量**:
- M: 栖息地成熟度 [0,1]
- Bee: 蜜蜂种群数量（授粉者）
- Spider: 蜘蛛种群数量（害虫捕食者）

**主要方程**:
```
dM/dt = α·M·(1 - M)
im(M) = im_max / (1 + exp(-k(M - M_crit)))
```

**功能**:
- 模拟边缘栖息地成熟过程
- 研究蜜蜂（授粉者）和蜘蛛（害虫捕食者）的重新出现
- 分析协同效应

### 3. 稳定性分析与有机农业评价模型 (Problem3)

**稳定性分析方法**:
- Jacobian矩阵特征值分析
- 局部稳定性判定
- 恢复时间计算
- 鲁棒性指标

**有机农业评价方法**:
- AHP（层次分析法）确定权重
- TOPSIS（逼近理想解排序法）综合评价

**评价指标**:
- 害虫控制能力
- 作物健康状况
- 植物繁殖能力
- 生物多样性
- 长期可持续性
- 成本效益
- 土壤健康
- 生态平衡度

## 运行说明

### 环境要求

```bash
Python 3.8+
numpy
scipy
matplotlib
pandas
```

### 安装依赖

```bash
pip install numpy scipy matplotlib pandas
```

### 运行方式

#### 方式1：运行主程序（推荐）

```bash
cd C:\Users\86198\Desktop\E\code
python main.py
```

这将依次运行所有模型并生成所有图表和数据。

#### 方式2：单独运行各模块

```bash
# 问题1：基础生态系统
python Problem1_生态系统基础模型.py

# 问题2：物种重新出现
python Problem2_物种重新出现模型.py

# 问题3：稳定性分析与有机农业评价
python Problem3_稳定性分析与有机农业评价.py
```

## 参数说明

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| r_c | 作物增长率 | 0.05 /day |
| K_c | 作物环境容纳量 | 100 kg/m² |
| r_I | 昆虫增长率 | 0.08 /day |
| K_I | 昆虫环境容纳量 | 50 |
| d_I | 昆虫死亡率 | 0.03 /day |
| r_B | 鸟类增长率 | 0.02 /day |
| K_B | 鸟类环境容纳量 | 20 |
| d_B | 鸟类死亡率 | 0.015 /day |
| e | 能量转化效率 | 0.1 |

### 相互作用系数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| a_CI | 昆虫对作物的影响系数 | 0.02 |
| a_IB | 鸟类对昆虫的捕食系数 | 0.015 |

### 化学物质参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| k_pest_insect | 杀虫剂对昆虫的影响 | 0.5 |
| k_chem_bird | 化学物质对鸟类的影响 | 0.01 |
| k_chem_crop | 化学物质对作物的副作用 | 0.005 |
| degradation | 化学物质降解率 | 0.05 /day |

## 输出文件

### 图表 (figures/)

| 文件名 | 说明 |
|--------|------|
| figure0_food_web.png | 食物网结构图 |
| figure1_base_ecosystem.png | 基础生态系统演化 |
| figure2_chemical_impact.png | 化学物质影响分析 |
| figure3_bat_introduction.png | 引入蝙蝠效果 |
| figure4_species_reintroduction.png | 物种重新出现模拟 |
| figure5_habitat_maturity_analysis.png | 栖息地成熟度分析 |
| figure6_stability_analysis.png | 稳定性分析 |
| figure7_organic_farming_evaluation.png | 有机农业评价 |
| figure8_summary_comparison.png | 综合对比分析 |

### 数据 (data/)

| 文件名 | 说明 |
|--------|------|
| results.csv | 基础模型结果 |
| organic_evaluation.csv | 有机农业评价结果 |
| summary_report.csv | 汇总报告 |

## 主要发现

1. **化学物质的双重影响**:
   - 短期提高作物产量（除草、杀虫效果）
   - 长期降低生态系统稳定性（生物富集效应）

2. **蝙蝠的生态价值**:
   - 有效控制害虫种群
   - 授粉服务提高作物产量
   - 提高生态系统整体稳定性

3. **物种协同效应**:
   - 蜜蜂（授粉）+ 蜘蛛（捕食）> 单独效应之和
   - 有机农业 + 蝙蝠 + 蚯蚓 = 最佳综合方案

4. **有机农业评价**:
   - 虽然短期产量略低
   - 但长期可持续性显著优于传统农业
   - 成本效益需考虑长期生态服务价值

## 创新点

1. **多层次耦合模型**: 将食物网、季节性、人类活动耦合到统一框架
2. **动态栖息地成熟度模型**: 将栖息地质量变化纳入物种迁入模拟
3. **蝙蝠-蚯蚓协同模型**: 分析不同生态位有益物种的互补作用
4. **多维度有机农业评价**: 结合生态、经济、社会三方面综合评估

## 作者

本项目使用Python实现，基于2025年ICM数学建模竞赛E题要求。

## 参考文献

1. Lotka, A. J. (1925). Elements of Physical Biology.
2. Volterra, V. (1926). Variations and fluctuations of the number of individuals.
3. McCann, K. S. (2000). The diversity-stability debate. Nature.
4. Kremen, C., & Miles, A. (2012). Ecosystem services in biologically diversified versus conventional farming systems.
5. Tilman, D., et al. (2014). Biodiversity and ecosystem functioning.

## 联系方式

如有问题或建议，请通过项目主页联系。
