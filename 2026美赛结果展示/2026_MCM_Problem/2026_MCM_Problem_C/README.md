# 2026年MCM问题C：数据与星光 - 完整解决方案

## 作者信息

**作者**: 数学建模 Skill-Math Modeling Skill

## 项目概述

本项目为2026年MCM问题C提供了完整的数学建模解决方案，分析了电视节目《与星共舞》(Dancing with the Stars, DWTS)的投票系统。

### 问题摘要

DWTS节目将专家评委打分与粉丝投票相结合，决定每周的淘汰结果。节目历史上使用过两种主要的投票合并方法：

1. **基于排名的方法**（第1-2季，第28-34季）：将评委分数和粉丝投票都转换为排名后相加
2. **基于百分比的方法**（第3-27季）：将评委分数和粉丝投票都转换为百分比后相加

本解决方案：
- 开发了估计未知粉丝投票的数学模型
- 比较了两种投票合并方法在所有赛季中的表现
- 分析了评委分数与粉丝投票差异显著的争议案例
- 评估了专业舞者和名人特征对比赛结果的影响
- 提出了新的、更公平的投票系统设计

---

## 使用的数学模型

| 模型 | 用途 | 实现方式 |
|------|------|----------|
| **逆优化** | 粉丝投票估计 | 基于约束的可行域搜索 |
| **排名相关性分析** | 投票方法比较 | Spearman相关系数 |
| **线性回归** | 影响因素分析 | scikit-learn |
| **蒙特卡洛模拟** | 不确定性量化 | numpy.random |
| **加权自适应系统** | 新投票系统 | 自定义实现 |

---

## 文件结构

```
F:\2026_MCM_Problem\C\
├── 2026_MCM_Problem_C_Data.csv       # 原始数据文件（只读）
├── 2026_MCM_Problem_C.md             # 问题陈述
├── MCM_Problem_C_Solution.py         # 主要求解代码
├── 题目分析报告.md                     # 建模分析文档
├── 术语表格.md                        # 术语对照表
├── 论文.md                           # 完整学术论文
├── README.md                         # 本文件
│
├── results/                           # 生成的结果文件
│   ├── analysis_report.txt            # 综合分析报告
│   ├── voting_method_comparison.csv   # 投票方法比较结果
│   ├── controversial_cases_analysis.csv # 争议案例分析
│   ├── age_impact_analysis.csv        # 年龄影响分析
│   ├── industry_impact_analysis.csv   # 行业影响分析
│   └── dancer_impact_analysis.csv     # 舞者影响分析
│
└── figures/                           # 生成的可视化图表（SCI/Nature风格）
    ├── figure1_age_vs_placement.png
    ├── figure2_industry_performance.png
    ├── figure3_top_dancers.png
    ├── figure4_score_by_placement.png
    ├── figure5_controversial_cases.png
    └── figure6_season_comparison.png
```

---

## 环境要求

### Python版本
- Python 3.8 或更高版本

### 依赖包

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

| 包名 | 版本要求 | 用途 |
|---------|---------|---------|
| numpy | >=1.20.0 | 数值计算 |
| pandas | >=1.3.0 | 数据处理 |
| matplotlib | >=3.3.0 | 可视化 |
| seaborn | >=0.11.0 | 统计可视化 |
| scipy | >=1.7.0 | 统计分析 |
| scikit-learn | >=0.24.0 | 机器学习 |

---

## 运行代码

### 快速开始

```bash
# 进入项目目录
cd F:\2026_MCM_Problem\C

# 运行完整解决方案
python MCM_Problem_C_Solution.py
```

### 执行流程

代码按以下顺序执行：

1. **数据加载**：加载并预处理DWTS数据（421名选手，34个赛季）
2. **投票方法比较**：比较基于排名与基于百分比的投票方法
3. **争议案例分析**：分析Jerry Rice、Billy Ray Cyrus、Bristol Palin、Bobby Bones
4. **影响因素分析**：评估年龄、行业和专业舞者的影响
5. **新系统提案**：设计并论证新的投票系统
6. **可视化生成**：创建所有6张SCI/Nature风格的图表
7. **报告生成**：输出综合分析报告

---

## 主要结果

### 1. 评委-排名相关性

分析显示评委分数与最终排名之间存在**极强的相关性（0.979）**，证实了评委分数对比赛结果的高度预测性。

### 2. 争议案例分析

| 赛季 | 名人 | 排名 | 平均分数 | 赛季平均 | 分析 |
|------|------|------|----------|----------|------|
| 2 | Jerry Rice | 第2名 | 16.38 | 10.96 | 分数高于赛季平均 |
| 4 | Billy Ray Cyrus | 第5名 | 13.82 | 13.97 | 分数略低于赛季平均 |
| 11 | Bristol Palin | 第3名 | 20.83 | 13.45 | 分数显著高于赛季平均 |
| 27 | Bobby Bones | 第1名 | 18.32 | 13.80 | 分数高于赛季平均 |

**关键发现**：这些"争议"案例的平均分数实际上都达到或超过了赛季平均水平，这表明粉丝投票可能合理地反映了评委分数未能完全捕捉的其他表演维度。

### 3. 影响因素

**年龄影响**：正相关（相关系数0.433）——年龄较大的选手倾向于获得更好的最终排名

**表现最佳的行业**：
1. Conservationist（自然保护主义者）——平均排名1.00
2. Musician（音乐家）——平均排名2.00
3. Social media personality（社交媒体名人）——平均排名2.00

**顶级专业舞者**：
1. Derek Hough（平均排名2.94，17个赛季）
2. Julianne Hough（平均排名4.20，5个赛季）
3. Daniella Karagach（平均排名4.60，5个赛季）

### 4. 推荐的新系统

**加权自适应投票系统**：
- **初期（第1-4周）**：w1=0.6（评委），w2=0.3（粉丝），w3=0.1（进步）
- **中期（第5-8周）**：w1=0.4，w2=0.4，w3=0.2
- **后期（第9周+）**：w1=0.3，w2=0.5，w3=0.2

**附加特性**：
- 进步奖励：奖励持续改进
- 一致性奖励：奖励稳定表现
- 保留底部两位评委裁决作为安全阀

---

## 可视化

所有图表均采用SCI/Nature出版风格生成（300 DPI）：

1. **figure1_age_vs_placement.png**：年龄与排名关系的散点图和箱线图
2. **figure2_industry_performance.png**：行业表现对比和参赛人数
3. **figure3_top_dancers.png**：前15名专业舞者的平均搭档排名
4. **figure4_score_by_placement.png**：不同最终排名的分数分布（箱线图和小提琴图）
5. **figure5_controversial_cases.png**：争议案例的视觉分析
6. **figure6_season_comparison.png**：赛季级别的对比（参赛人数和获胜者分数）

---

## 对DWTS制作方的建议

### 1. 采用基于百分比的投票方法
- 与评委分数的相关性更强
- 在数学上更加一致和公平
- 对持续高分的选手更公平

### 2. 实施加权自适应系统
- 平衡评委专业性和粉丝参与度
- 鼓励整个比赛过程中的持续改进
- 适应不同的比赛阶段

### 3. 增加进步奖励
- 减少争议结果的可能性
- 奖励持续改进
- 使节目更加精彩

### 4. 保留评委淘汰权
- 在接近的情况下保留底部两位评委淘汰机制
- 为极端的粉丝投票模式增加安全阀
- 保护评审团的可信度

---

## 代码结构

### 主类：`DWTSSolver`

```python
class DWTSSolver:
    def __init__(self, data_path)          # 初始化并加载数据
    def load_data(self, data_path)         # 加载CSV数据
    def preprocess_data(self)              # 清理和准备数据
    def estimate_fan_votes_rank_method()   # 基于排名的估计
    def estimate_fan_votes_percent_method() # 基于百分比的估计
    def compare_voting_methods(self)       # 比较投票方法
    def analyze_controversial_cases(self)  # 分析争议案例
    def analyze_impact_factors(self)       # 分析影响因素
    def propose_new_system(self)           # 提出新系统
    def create_visualizations(self)        # 生成所有图表
    def generate_report(self)              # 生成报告
```

### 可视化方法

- `_plot_age_vs_placement()`
- `_plot_industry_performance()`
- `_plot_top_dancers()`
- `_plot_score_by_placement()`
- `_plot_controversial_cases()`
- `_plot_season_comparison()`

---

## 输出

### 控制台输出

程序输出：
- 数据加载进度
- 各组件的分析进度
- 包含关键发现的分析报告
- 生成文件的最终摘要

### CSV输出文件

位于`results/`目录：
- `voting_method_comparison.csv` —— 逐季比较
- `controversial_cases_analysis.csv` —— 详细案例分析
- `age_impact_analysis.csv` —— 年龄组统计
- `industry_impact_analysis.csv` —— 行业表现
- `dancer_impact_analysis.csv` —— 专业舞者排名
- `analysis_report.txt` —— 完整文本报告

### PNG输出文件

位于`figures/`目录：
- 6张高分辨率（300 DPI）图表
- SCI/Nature出版风格
- 色盲友好的调色板
- 清晰的标签和图例

---

## 作者贡献

本解决方案代表了以下三个阶段的协作努力：

**作者**: 数学建模 Skill-Math Modeling Skill

1. **建模分析阶段**：
   - 问题分解
   - 模型选择和论证
   - 数学公式化
   - 文档编写

2. **代码实现阶段**：
   - 数据预处理
   - 模型实现
   - 可视化
   - 结果生成

3. **论文撰写阶段**：
   - 学术论文结构
   - 结果解释
   - 建议
   - 最终展示

---

## 引用

如果使用本解决方案，请引用：

```
数学建模 Skill-Math Modeling Skill. (2026). Complete Solution for 2026 MCM Problem C:
Data With The Stars. Mathematical Contest in Modeling.
```

---

## 许可证

本解决方案仅供2026年数学建模竞赛教育目的使用。

---

## 联系方式

如有疑问或问题，请联系 数学建模 Skill-Math Modeling Skill 作者。

---

**版本**: 1.0
**日期**: 2026-01-31
**状态**: 已完成
