# 2026 MCM Problem A: Smartphone Battery Drain Modeling

## 项目概述

本项目针对2026年MCM（数学建模竞赛）A题，建立了基于连续时间微分方程的智能手机电池荷电状态(SOC)预测模型。模型能够预测在不同使用场景下的电池续航时间(Time-to-Empty)，并进行敏感性分析和不确定性量化。

**版本说明**: 本项目具有以下特性：
- 增强的数值精度控制（rtol=1e-8, atol=1e-10）
- 完整的物理约束验证
- 改进的分段温度模型
- 蒙特卡洛不确定性估计（95%置信区间）
- 能量守恒验证机制

## 核心模型

### 控制方程

```
dSOC/dt = -P_total(t) / (Q_eff * V_nom * 3600)
```

其中：
- `SOC(t)`: 荷电状态 (0-1)
- `P_total(t)`: 总功率消耗
- `Q_eff`: 有效电池容量 (考虑老化)
- `V_nom`: 标称电压

### 功率消耗分解

```text
P_total = P_base + P_screen + P_cpu + P_network + P_gps + P_background + P_thermal
```

各组件功率模型：

- **基础功耗**: P_base × η_T(T)，其中 η_T 为温度影响因子
- **屏幕功耗**: P_screen_base × (0.3 + 0.7 × (1 + α_B × brightness))
- **CPU功耗**: P_cpu_idle + (P_cpu_max - P_cpu_idle) × load^γ
- **网络功耗**: 根据连接类型 (WiFi/4G/5G)
- **GPS功耗**: P_gps (当激活时)
- **温度影响**: 分段函数模型
  - 低温区 (<0°C): 线性增长
  - 正常区 (0-35°C): 二次函数
  - 高温区 (>35°C): 指数增长

## 使用场景

模型支持以下典型使用场景：

| 场景 | 描述 | 平均功耗 | TTE | 95%置信区间 |
|------|------|---------|-----|-------------|
| Idle | 空闲模式 | 0.76 W | 68.0 h | (59.3, 73.0) h |
| Web Browsing | 网页浏览 | 1.92 W | 26.8 h | (24.1, 28.3) h |
| Video Playback | 视频播放 | 2.04 W | 25.3 h | (23.2, 28.1) h |
| Gaming | 游戏场景 | 5.34 W | 9.7 h | (9.3, 10.3) h |
| Navigation | GPS导航 | 3.44 W | 15.0 h | (14.5, 15.9) h |
| Video Call | 视频通话 | 3.65 W | 14.1 h | (12.2, 15.8) h |

## 环境要求

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib

## 安装依赖

```bash
pip install numpy pandas scipy matplotlib
```

## 运行方法

```bash
python battery_model_solver.py
```

## 文件结构

```
A/
├── battery_model_solver.py      # 主求解程序
├── data/                         # 数据文件夹
├── results/                      # 结果输出文件夹
│   ├── tte_summary.csv           # 场景TTE汇总
│   ├── detailed_results.txt      # 详细结果说明
│   └── model_parameters.json     # 模型参数
├── figures/                      # 可视化图表文件夹
│   ├── figure1_soc_curves.png    # SOC曲线对比
│   ├── figure2_scenario_comparison.png  # 场景对比图
│   ├── figure3_temperature_effect.png   # 温度影响曲线
│   └── figure4_sensitivity_heatmap.png  # 敏感性热力图
├── 题目分析报告.md               # 建模分析文档
├── 术语表格.md                   # 术语对照表
└── README.md                     # 本文档
```

## 主要结果

### 1. 各场景续航时间

- **最长续航**: Idle模式，约68小时
- **最短续航**: Gaming场景，约9.7小时
- **典型使用**: Web浏览约26.8小时

### 2. 不确定性分析

- **Idle场景**: 不确定性最大（±11.5%），因功耗低受扰动影响大
- **Gaming场景**: 不确定性最小（±6.3%），因高功耗主导
- **平均不确定度**: 约±10%

### 3. 温度影响

- **低温(-20°C)**: TTE减少约30%
- **高温(45°C)**: TTE减少约15%
- **最佳温度**: 约25°C

### 4. 敏感性分析

参数对TTE的影响（Web Browsing场景基准）：

1. **屏幕功耗** (P_screen_base): -7.4% 到 +8.6%
2. **CPU最大功耗** (P_cpu_max): -6.4% 到 +7.3%
3. **WiFi功耗** (P_wifi): -2.3% 到 +2.4%
4. **GPS功耗** (P_gps): 影响较小（仅在导航场景显著）

## 模型特点

### 优点

1. **物理基础**: 基于电化学原理的等效电路模型
2. **连续时间**: 使用微分方程描述动态变化
3. **多因素耦合**: 综合考虑屏幕、CPU、网络、温度等因素
4. **可扩展性**: 易于添加新的功耗组件
5. **数值精确**: 使用Runge-Kutta方法求解ODE，精度达1e-8
6. **不确定性量化**: 通过蒙特卡洛模拟提供95%置信区间
7. **物理约束**: 自动验证并强制执行物理边界条件

### 局限性

1. **参数依赖**: 需要准确测量设备特定参数
2. **简化假设**: 某些复杂效应被简化处理
3. **个体差异**: 不同设备参数差异较大

## 实用建议

基于模型分析，对用户的建议：

1. **降低屏幕亮度**: 可延长续航10-15%
2. **减少高CPU负载活动**: 游戏功耗是浏览的2.7倍
3. **使用WiFi代替4G/5G**: 节省约75%的网络功耗
4. **注意温度**: 避免极端温度环境
5. **关闭GPS**: 不使用时关闭GPS可节省约0.5W

## 技术细节

### ODE求解方法

使用scipy.integrate.solve_ivp的RK45方法（Runge-Kutta 4/5阶自适应步长）

- 相对容差: 1e-8
- 绝对容差: 1e-10
- 最大步长: 3600秒（1小时）
- 事件检测: SOC降至截止值自动终止

### 不确定性量化

采用蒙特卡洛方法进行参数扰动分析：

- 参数扰动范围: ±10%
- 采样次数: 10-100次（可配置）
- 置信水平: 95%
- 输出: 置信区间 + 标准差

### 参数估计

模型参数可从以下来源获取：
- 设备制造商规格
- 实测功率数据
- 学术文献

### 验证方法

- 与实测数据对比
- 极限情况测试
- 跨设备验证

## 参考文献

1. Tremblay, O., & Dessaint, L. A. (2009). Experimental validation of a battery dynamic model for electric vehicle applications.
2. Carroll, A., & Heiser, G. (2010). An analysis of power consumption in a smartphone.
3. Jaguemont, J., et al. (2016). A comprehensive review of lithium-ion battery used in hybrid and electric vehicles at cold temperatures.

## 作者

数学建模 Skill-Math Modeling Skill

## 许可

本项目仅用于学术研究和数学建模竞赛。
