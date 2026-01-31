"""
2026 ICM Problem F: Gen-AI Impact on Career Education
数学模型实现代码

包含以下模型：
1. GM(1,1)灰色预测模型 - 就业需求预测
2. AHP-熵权法组合赋权 - 指标权重确定
3. TOPSIS评价模型 - 职业可替代性评价
4. 系统动力学模型 - 教育策略仿真

作者: Math Modeling Team
日期: 2026-01-31
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.linalg import inv
import warnings
import os
warnings.filterwarnings('ignore')

# Create directories if not exist
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 第一部分：GM(1,1)灰色预测模型
# ============================================

class GM11Model:
    """
    GM(1,1)灰色预测模型
    用于小样本时间序列预测
    """
    
    def __init__(self):
        self.a = None  # 发展系数
        self.b = None  # 灰色作用量
        self.x0 = None  # 原始序列
        self.x1 = None  # 累加序列
        
    def fit(self, data):
        """
        拟合GM(1,1)模型
        
        Parameters:
        -----------
        data : array-like
            原始数据序列（非负）
        """
        self.x0 = np.array(data, dtype=float)
        n = len(self.x0)
        
        # 一次累加生成 (AGO)
        self.x1 = np.cumsum(self.x0)
        
        # 构造矩阵B和Y
        B = np.zeros((n-1, 2))
        Y = self.x0[1:].reshape(-1, 1)
        
        for i in range(n-1):
            B[i, 0] = -0.5 * (self.x1[i] + self.x1[i+1])
            B[i, 1] = 1
        
        # 最小二乘估计参数
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        self.a = params[0, 0]
        self.b = params[1, 0]
        
        return self
    
    def predict(self, steps=1):
        """
        预测未来值
        
        Parameters:
        -----------
        steps : int
            预测步数
            
        Returns:
        --------
        predictions : array
            预测值序列（包含历史拟合值和未来预测值）
        """
        n = len(self.x0)
        
        # 计算拟合值
        predictions = np.zeros(n + steps)
        
        for k in range(n + steps):
            if k == 0:
                predictions[k] = self.x0[0]
            else:
                # GM(1,1)预测公式
                x1_pred = (self.x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
                x1_pred_prev = (self.x0[0] - self.b/self.a) * np.exp(-self.a * (k-1)) + self.b/self.a
                predictions[k] = x1_pred - x1_pred_prev
        
        return predictions
    
    def fitted_values(self):
        """返回历史数据的拟合值"""
        return self.predict(steps=0)[:len(self.x0)]
    
    def evaluate(self):
        """
        模型检验
        
        Returns:
        --------
        metrics : dict
            包含各项检验指标的字典
        """
        fitted = self.fitted_values()
        residuals = self.x0 - fitted
        
        # 残差标准差
        s2 = np.std(residuals, ddof=1)
        # 原始数据标准差
        s1 = np.std(self.x0, ddof=1)
        
        # 后验差比值
        C = s2 / s1
        
        # 小误差概率
        error = np.abs(residuals)
        P = np.sum(error < 0.6745 * s1) / len(error)
        
        # 平均相对误差
        mape = np.mean(np.abs(residuals / self.x0)) * 100
        
        return {
            'C': C,  # 后验差比值
            'P': P,  # 小误差概率
            'MAPE': mape,  # 平均相对误差百分比
            'a': self.a,  # 发展系数
            'b': self.b   # 灰色作用量
        }
    
    def print_evaluation(self):
        """打印模型评估结果"""
        metrics = self.evaluate()
        print("=" * 50)
        print("GM(1,1)模型评估结果")
        print("=" * 50)
        print(f"发展系数 a: {metrics['a']:.6f}")
        print(f"灰色作用量 b: {metrics['b']:.6f}")
        print(f"后验差比值 C: {metrics['C']:.4f}")
        print(f"小误差概率 P: {metrics['P']:.4f}")
        print(f"平均相对误差 MAPE: {metrics['MAPE']:.2f}%")
        
        # 模型精度等级判断
        if metrics['C'] < 0.35 and metrics['P'] > 0.95:
            grade = "优秀"
        elif metrics['C'] < 0.5 and metrics['P'] > 0.8:
            grade = "合格"
        elif metrics['C'] < 0.65 and metrics['P'] > 0.7:
            grade = "勉强合格"
        else:
            grade = "不合格"
        
        print(f"模型精度等级: {grade}")
        print("=" * 50)


# ============================================
# 第二部分：AHP层次分析法
# ============================================

class AHPMethod:
    """
    层次分析法 (Analytic Hierarchy Process)
    用于确定指标主观权重
    """
    
    # 随机一致性指标RI
    RI_TABLE = {
        1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    def __init__(self):
        self.weights = None
        self.lambda_max = None
        self.CI = None
        self.CR = None
        
    def fit(self, judgment_matrix):
        """
        计算AHP权重
        
        Parameters:
        -----------
        judgment_matrix : array-like
            判断矩阵（n×n）
            
        Returns:
        --------
        self : AHPMethod
        """
        A = np.array(judgment_matrix, dtype=float)
        n = A.shape[0]
        
        # 特征根法计算权重
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # 找最大特征值及其对应的特征向量
        max_idx = np.argmax(eigenvalues.real)
        self.lambda_max = eigenvalues[max_idx].real
        weight_vector = eigenvectors[:, max_idx].real
        
        # 归一化
        self.weights = weight_vector / weight_vector.sum()
        
        # 一致性检验
        self.CI = (self.lambda_max - n) / (n - 1)
        self.CR = self.CI / self.RI_TABLE.get(n, 1.49)
        
        return self
    
    def check_consistency(self):
        """检查一致性"""
        if self.CR is None:
            raise ValueError("请先调用fit方法")
        return self.CR < 0.1
    
    def print_results(self):
        """打印AHP结果"""
        print("=" * 50)
        print("AHP层次分析法结果")
        print("=" * 50)
        print(f"最大特征值 λ_max: {self.lambda_max:.4f}")
        print(f"一致性指标 CI: {self.CI:.4f}")
        print(f"一致性比例 CR: {self.CR:.4f}")
        print(f"一致性检验: {'通过' if self.check_consistency() else '未通过'}")
        print("\n指标权重:")
        for i, w in enumerate(self.weights):
            print(f"  指标{i+1}: {w:.4f}")
        print("=" * 50)


# ============================================
# 第三部分：熵权法
# ============================================

class EntropyWeightMethod:
    """
    熵权法 (Entropy Weight Method)
    用于确定指标客观权重
    """
    
    def __init__(self):
        self.weights = None
        self.entropy = None
        self.diversity = None
        
    def fit(self, data, directions=None):
        """
        计算熵权
        
        Parameters:
        -----------
        data : array-like
            数据矩阵 (样本×指标)
        directions : array-like, optional
            指标方向，1为正向，-1为负向，默认为全正向
            
        Returns:
        --------
        self : EntropyWeightMethod
        """
        X = np.array(data, dtype=float)
        m, n = X.shape
        
        if directions is None:
            directions = np.ones(n)
        else:
            directions = np.array(directions)
        
        # 数据标准化
        normalized = np.zeros_like(X)
        for j in range(n):
            col = X[:, j]
            if directions[j] == 1:  # 正向指标
                min_val, max_val = col.min(), col.max()
                if max_val - min_val != 0:
                    normalized[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    normalized[:, j] = 1
            else:  # 负向指标
                min_val, max_val = col.min(), col.max()
                if max_val - min_val != 0:
                    normalized[:, j] = (max_val - col) / (max_val - min_val)
                else:
                    normalized[:, j] = 1
        
        # 坐标平移（避免ln(0)）
        shifted = normalized + 1e-10
        
        # 计算比重
        p = shifted / shifted.sum(axis=0)
        
        # 计算信息熵
        self.entropy = np.zeros(n)
        for j in range(n):
            self.entropy[j] = -1 / np.log(m) * np.sum(p[:, j] * np.log(p[:, j]))
        
        # 计算信息效用值和权重
        self.diversity = 1 - self.entropy
        self.weights = self.diversity / self.diversity.sum()
        
        return self
    
    def print_results(self):
        """打印熵权法结果"""
        print("=" * 50)
        print("熵权法结果")
        print("=" * 50)
        print("\n信息熵:")
        for i, e in enumerate(self.entropy):
            print(f"  指标{i+1}: {e:.4f}")
        print("\n信息效用值:")
        for i, d in enumerate(self.diversity):
            print(f"  指标{i+1}: {d:.4f}")
        print("\n指标权重:")
        for i, w in enumerate(self.weights):
            print(f"  指标{i+1}: {w:.4f}")
        print("=" * 50)


# ============================================
# 第四部分：博弈论组合赋权
# ============================================

class GameTheoryCombination:
    """
    博弈论组合赋权方法
    综合多种赋权方法的结果
    """
    
    def __init__(self):
        self.combined_weights = None
        self.alpha = None
        
    def fit(self, weight_list):
        """
        博弈论组合赋权
        
        Parameters:
        -----------
        weight_list : list of arrays
            多种方法得到的权重列表
            
        Returns:
        --------
        self : GameTheoryCombination
        """
        # 构建权重矩阵
        W = np.column_stack(weight_list)
        n_methods = W.shape[1]
        
        # 构建优化问题：最小化偏差
        def objective(alpha):
            w_combined = W @ alpha
            diff = W - w_combined.reshape(-1, 1)
            return np.sum(diff ** 2)
        
        # 约束条件
        constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1}
        bounds = [(0, None) for _ in range(n_methods)]
        initial_guess = np.ones(n_methods) / n_methods
        
        # 求解优化问题
        res = minimize(objective, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)
        
        self.alpha = res.x
        self.combined_weights = W @ self.alpha
        
        return self
    
    def print_results(self, method_names=None):
        """打印组合赋权结果"""
        print("=" * 50)
        print("博弈论组合赋权结果")
        print("=" * 50)
        print("\n组合系数:")
        if method_names is None:
            method_names = [f"方法{i+1}" for i in range(len(self.alpha))]
        for name, a in zip(method_names, self.alpha):
            print(f"  {name}: {a:.4f}")
        print("\n组合权重:")
        for i, w in enumerate(self.combined_weights):
            print(f"  指标{i+1}: {w:.4f}")
        print("=" * 50)


# ============================================
# 第五部分：TOPSIS评价方法
# ============================================

class TOPSISMethod:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    优劣解距离法
    """
    
    def __init__(self):
        self.closeness = None
        self.d_positive = None
        self.d_negative = None
        self.v_positive = None
        self.v_negative = None
        
    def evaluate(self, data, weights, directions=None):
        """
        TOPSIS评价
        
        Parameters:
        -----------
        data : array-like
            数据矩阵 (样本×指标)
        weights : array-like
            指标权重
        directions : array-like, optional
            指标方向，1为正向，-1为负向
            
        Returns:
        --------
        self : TOPSISMethod
        """
        X = np.array(data, dtype=float)
        w = np.array(weights)
        m, n = X.shape
        
        if directions is None:
            directions = np.ones(n)
        else:
            directions = np.array(directions)
        
        # 向量标准化
        normalized = X / np.sqrt((X ** 2).sum(axis=0))
        
        # 加权规范化
        weighted = normalized * w
        
        # 确定正负理想解
        self.v_positive = np.zeros(n)
        self.v_negative = np.zeros(n)
        
        for j in range(n):
            if directions[j] == 1:  # 正向指标
                self.v_positive[j] = weighted[:, j].max()
                self.v_negative[j] = weighted[:, j].min()
            else:  # 负向指标
                self.v_positive[j] = weighted[:, j].min()
                self.v_negative[j] = weighted[:, j].max()
        
        # 计算距离
        self.d_positive = np.sqrt(((weighted - self.v_positive) ** 2).sum(axis=1))
        self.d_negative = np.sqrt(((weighted - self.v_negative) ** 2).sum(axis=1))
        
        # 计算相对贴近度
        self.closeness = self.d_negative / (self.d_positive + self.d_negative + 1e-10)
        
        return self
    
    def get_ranking(self):
        """获取排序结果"""
        if self.closeness is None:
            raise ValueError("请先调用evaluate方法")
        return np.argsort(-self.closeness) + 1  # 从1开始计数
    
    def print_results(self, alternatives=None):
        """打印TOPSIS结果"""
        print("=" * 50)
        print("TOPSIS评价结果")
        print("=" * 50)
        
        if alternatives is None:
            alternatives = [f"方案{i+1}" for i in range(len(self.closeness))]
        
        print("\n评价结果:")
        print(f"{'方案':<15} {'贴近度':<12} {'排名':<8}")
        print("-" * 40)
        
        ranking = self.get_ranking()
        for i, alt in enumerate(alternatives):
            rank = np.where(ranking == i + 1)[0][0] + 1
            print(f"{alt:<15} {self.closeness[i]:<12.4f} {rank:<8}")
        
        print("=" * 50)


# ============================================
# 第六部分：系统动力学模型
# ============================================

class EducationSystemDynamics:
    """
    教育系统动力学模型
    模拟教育-就业系统的动态演化
    """
    
    def __init__(self):
        self.history = {
            'enrollment': [],
            'students': [],
            'graduates': [],
            'employed': [],
            'employment_rate': []
        }
        
    def simulate(self, initial_students, years, params):
        """
        模拟教育系统演化
        
        Parameters:
        -----------
        initial_students : int
            初始在校生人数
        years : int
            模拟年数
        params : dict
            模型参数
            - target_enrollment_rate: 目标招生率
            - graduation_rate: 毕业率
            - base_employment_rate: 基础就业率
            - ai_impact_factor: Gen-AI影响因子
            - adaptation_rate: 教育适应率
            
        Returns:
        --------
        history : dict
            模拟历史数据
        """
        students = initial_students
        
        for year in range(years):
            # 计算Gen-AI影响（随时间递增）
            ai_penetration = min(0.1 + year * 0.08, 0.8)  # Gen-AI渗透率
            ai_impact = ai_penetration * params['ai_impact_factor']
            
            # 教育适应效果（随时间递减）
            adaptation = min(year * params['adaptation_rate'], 0.5)
            
            # 净就业率影响
            net_impact = ai_impact - adaptation
            employment_rate = params['base_employment_rate'] - net_impact
            employment_rate = max(0.3, min(0.95, employment_rate))  # 边界限制
            
            # 根据就业率调整招生
            if employment_rate > 0.8:
                enrollment_rate = params['target_enrollment_rate'] * 1.1
            elif employment_rate < 0.5:
                enrollment_rate = params['target_enrollment_rate'] * 0.8
            else:
                enrollment_rate = params['target_enrollment_rate']
            
            # 计算各变量
            enrollment = int(students * enrollment_rate)
            graduates = int(students * params['graduation_rate'])
            employed = int(graduates * employment_rate)
            
            # 更新在校生人数
            students = students + enrollment - graduates
            
            # 记录历史
            self.history['enrollment'].append(enrollment)
            self.history['students'].append(students)
            self.history['graduates'].append(graduates)
            self.history['employed'].append(employed)
            self.history['employment_rate'].append(employment_rate)
        
        return self.history


# ============================================
# 第七部分：主程序 - 2026 ICM Problem F 求解
# ============================================

def main():
    """
    主程序：求解2026 ICM Problem F
    """
    print("\n" + "="*70)
    print(" "*15 + "2026 ICM Problem F")
    print(" "*5 + "Gen-AI Impact on Career Education Analysis")
    print("="*70 + "\n")
    
    # ----------------------------------------
    # 1. Employment Demand Prediction (GM(1,1) Model)
    # ----------------------------------------
    print("[Part 1] Employment Demand Prediction - GM(1,1) Grey Prediction Model\n")
    
    # Historical employment data (in thousands, based on BLS data simulation)
    # Software Engineer
    software_employment = np.array([1200, 1280, 1360, 1450, 1550, 1650, 1750, 1850, 1950, 2050])
    # Electrician
    electrician_employment = np.array([650, 660, 670, 685, 700, 715, 730, 745, 760, 775])
    # Graphic Designer
    designer_employment = np.array([280, 285, 290, 295, 300, 305, 310, 315, 320, 325])
    
    years_hist = np.arange(2015, 2025)
    years_pred = np.arange(2025, 2031)
    
    # Store prediction results
    predictions = {}
    
    for career_name, data in [
        ("Software Engineer", software_employment),
        ("Electrician", electrician_employment),
        ("Graphic Designer", designer_employment)
    ]:
        print(f"\n--- {career_name} ---")
        
        # Build GM(1,1) model
        gm_model = GM11Model()
        gm_model.fit(data)
        gm_model.print_evaluation()
        
        # Predict next 6 years
        pred = gm_model.predict(steps=6)
        predictions[career_name] = {
            'historical': data,
            'fitted': gm_model.fitted_values(),
            'predicted': pred[-6:],
            'all': pred
        }
        
        print(f"2025-2030 Employment Demand Prediction (thousands): {pred[-6:].astype(int)}")
    
    # ----------------------------------------
    # 2. AI Replaceability Evaluation (AHP-Entropy-TOPSIS)
    # ----------------------------------------
    print("\n\n" + "="*70)
    print("[Part 2] AI Replaceability Evaluation - AHP-Entropy-TOPSIS Model\n")
    
    # Evaluation indicator data (3 careers × 9 indicators)
    # Indicators: Repetition, Creativity, Interaction, Complexity, Update Speed,
    #             Experience, Current Substitution, Technical Feasibility, Economic Feasibility
    # Direction: 1 = positive (higher = more replaceable), -1 = negative (higher = less replaceable)
    
    evaluation_data = np.array([
        # Software Engineer
        [7, 6, 4, 8, 9, 5, 6, 8, 7],  # High skill update speed, high technical feasibility
        # Electrician
        [5, 3, 7, 6, 4, 9, 2, 3, 4],  # High experience importance, low technical feasibility
        # Graphic Designer
        [6, 8, 5, 5, 7, 4, 7, 7, 6],  # High creativity, high current substitution
    ])
    
    indicators = [
        "Repetition", "Creativity", "Interaction",
        "Complexity", "Update Speed", "Experience",
        "Current Substitution", "Technical Feasibility", "Economic Feasibility"
    ]
    directions = [1, -1, -1, -1, 1, -1, 1, 1, 1]  # Indicator directions
    careers = ["Software Engineer", "Electrician", "Graphic Designer"]
    
    # AHP for subjective weights
    # Judgment matrix (based on expert experience)
    judgment_matrix = np.array([
        [1, 1/2, 1/2, 1/3, 2, 1/3, 2, 2, 2],
        [2, 1, 1, 1/2, 3, 1/2, 3, 3, 3],
        [2, 1, 1, 1/2, 3, 1/2, 3, 3, 3],
        [3, 2, 2, 1, 4, 1, 4, 4, 4],
        [1/2, 1/3, 1/3, 1/4, 1, 1/4, 2, 2, 2],
        [3, 2, 2, 1, 4, 1, 4, 4, 4],
        [1/2, 1/3, 1/3, 1/4, 1/2, 1/4, 1, 1, 1],
        [1/2, 1/3, 1/3, 1/4, 1/2, 1/4, 1, 1, 1],
        [1/2, 1/3, 1/3, 1/4, 1/2, 1/4, 1, 1, 1]
    ])
    
    print("--- AHP Analytic Hierarchy Process ---")
    ahp = AHPMethod()
    ahp.fit(judgment_matrix)
    ahp.print_results()
    
    # Entropy method for objective weights
    print("\n--- Entropy Weight Method ---")
    ewm = EntropyWeightMethod()
    ewm.fit(evaluation_data, directions=directions)
    ewm.print_results()
    
    # Game theory combination weighting
    print("\n--- Game Theory Combination Weighting ---")
    gtc = GameTheoryCombination()
    gtc.fit([ahp.weights, ewm.weights])
    gtc.print_results(method_names=["AHP Subjective Weights", "Entropy Objective Weights"])
    
    # TOPSIS评价
    print("\n--- TOPSIS评价 ---")
    topsis = TOPSISMethod()
    topsis.evaluate(evaluation_data, gtc.combined_weights, directions=directions)
    topsis.print_results(alternatives=careers)
    
    # ----------------------------------------
    # 3. Education Strategy Simulation (System Dynamics)
    # ----------------------------------------
    print("\n\n" + "="*70)
    print("[Part 3] Education Strategy Simulation - System Dynamics Model\n")
    
    # 不同职业的仿真参数
    career_params = {
        "Software Engineer": {
            'target_enrollment_rate': 0.25,
            'graduation_rate': 0.20,
            'base_employment_rate': 0.85,
            'ai_impact_factor': 0.35,  # High impact
            'adaptation_rate': 0.08    # High adaptation
        },
        "Electrician": {
            'target_enrollment_rate': 0.20,
            'graduation_rate': 0.18,
            'base_employment_rate': 0.90,
            'ai_impact_factor': 0.10,  # Low impact
            'adaptation_rate': 0.03    # Low adaptation
        },
        "Graphic Designer": {
            'target_enrollment_rate': 0.22,
            'graduation_rate': 0.19,
            'base_employment_rate': 0.75,
            'ai_impact_factor': 0.45,  # Very high impact
            'adaptation_rate': 0.06    # Medium adaptation
        }
    }
    
    simulation_results = {}
    
    for career, params in career_params.items():
        print(f"\n--- {career} Education Strategy Simulation ---")
        
        sd_model = EducationSystemDynamics()
        history = sd_model.simulate(
            initial_students=1000,
            years=10,
            params=params
        )
        
        simulation_results[career] = history
        
        print(f"Initial Students: 1000")
        print(f"Students after 10 years: {history['students'][-1]}")
        print(f"Average Employment Rate: {np.mean(history['employment_rate']):.2%}")
        print(f"Final Employment Rate: {history['employment_rate'][-1]:.2%}")
    
    # ----------------------------------------
    # 4. 综合建议
    # ----------------------------------------
    print("\n\n" + "="*70)
    print("【第四部分】综合建议\n")
    
    print("="*70)
    print("Education Strategy Recommendations Based on Model Analysis")
    print("="*70)
    
    print("\n[1. Enrollment Scale Recommendations]")
    print("-" * 50)
    
    for i, career in enumerate(careers):
        closeness = topsis.closeness[i]
        final_employment = simulation_results[career]['employment_rate'][-1]
        
        if closeness > 0.6:
            risk_level = "High Risk"
            suggestion = "Moderately reduce enrollment scale, focus on improving education quality"
        elif closeness > 0.4:
            risk_level = "Medium Risk"
            suggestion = "Maintain current enrollment scale, strengthen curriculum reform"
        else:
            risk_level = "Low Risk"
            suggestion = "Can appropriately expand enrollment scale"
        
        print(f"\n{career}:")
        print(f"  Replaceability Score: {closeness:.4f} ({risk_level})")
        print(f"  Predicted Employment Rate: {final_employment:.2%}")
        print(f"  Recommendation: {suggestion}")
    
    print("\n\n[2. Curriculum Recommendations]")
    print("-" * 50)
    
    curriculum_suggestions = {
        "Software Engineer": [
            "Strengthen training on AI-assisted programming tools",
            "Increase proportion of system architecture design courses",
            "Offer AI ethics and responsibility courses",
            "Enhance human-AI collaboration skills"
        ],
        "Electrician": [
            "Introduce intelligent diagnostic system operation training",
            "Strengthen renewable energy technology courses",
            "Enhance digital skills training",
            "Maintain traditional hands-on skills training"
        ],
        "Graphic Designer": [
            "Teach AI image generation tool usage",
            "Strengthen brand strategy and narrative abilities",
            "Add cross-media design courses",
            "Develop creative thinking that AI cannot replace"
        ]
    }
    
    for career, suggestions in curriculum_suggestions.items():
        print(f"\n{career}:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    print("\n\n[3. Other Success Factors]")
    print("-" * 50)
    print("""
In addition to employability, educational institutions should also focus on:

1. Student Satisfaction: Learning experience, course quality, teacher-student interaction
2. Lifelong Learning Ability: Developing self-directed learning and adaptability to new technologies
3. Social Contribution: Positive impact of graduates on society
4. Innovation Capability: Developing original thinking and problem-solving abilities
5. Ethical Awareness: Professional ethics and social responsibility in the AI era
6. Interdisciplinary Competence: Comprehensive literacy that breaks disciplinary boundaries
7. Mental Health: Psychological resilience to cope with technological changes
    """)
    
    # ----------------------------------------
    # 5. 可视化
    # ----------------------------------------
    print("\n\n" + "="*70)
    print("[Part 5] Generating Visualization Charts\n")
    
    create_visualizations(predictions, topsis, simulation_results, careers, years_hist, years_pred)
    
    print("\nVisualization charts saved!")
    print("="*70)
    
    # ----------------------------------------
    # 6. Save results to CSV and text files
    # ----------------------------------------
    print("\n" + "="*70)
    print("[Part 6] Saving Results to Files")
    print("="*70)
    
    # Save employment prediction results
    employment_results = []
    for career_name, data in predictions.items():
        for i, year in enumerate(years_pred):
            employment_results.append({
                'Career': career_name,
                'Year': year,
                'Predicted_Employment': data['predicted'][i]
            })
    
    df_employment = pd.DataFrame(employment_results)
    df_employment.to_csv('results/employment_prediction.csv', index=False)
    print("  - results/employment_prediction.csv saved")
    
    # Save TOPSIS evaluation results
    evaluation_results = []
    for i, career in enumerate(careers):
        evaluation_results.append({
            'Career': career,
            'Closeness_Coefficient': topsis.closeness[i],
            'Risk_Level': 'High' if topsis.closeness[i] > 0.6 else ('Medium' if topsis.closeness[i] > 0.4 else 'Low')
        })
    
    df_evaluation = pd.DataFrame(evaluation_results)
    df_evaluation.to_csv('results/topsis_evaluation.csv', index=False)
    print("  - results/topsis_evaluation.csv saved")
    
    # Save system dynamics simulation results
    simulation_results_df = []
    for career, history in simulation_results.items():
        for year_idx, year in enumerate(range(2025, 2035)):
            simulation_results_df.append({
                'Career': career,
                'Year': year,
                'Students': history['students'][year_idx],
                'Graduates': history['graduates'][year_idx],
                'Employed': history['employed'][year_idx],
                'Employment_Rate': history['employment_rate'][year_idx]
            })
    
    df_simulation = pd.DataFrame(simulation_results_df)
    df_simulation.to_csv('results/system_dynamics_simulation.csv', index=False)
    print("  - results/system_dynamics_simulation.csv saved")
    
    # Save comprehensive results to text file
    with open('results/analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("2026 ICM Problem F: Gen-AI Impact on Career Education\n")
        f.write("Analysis Results\n")
        f.write("="*70 + "\n\n")
        
        f.write("[Part 1] Employment Demand Prediction (2025-2030)\n")
        f.write("-"*70 + "\n")
        for career_name, data in predictions.items():
            f.write(f"\n{career_name}:\n")
            for i, year in enumerate(years_pred):
                f.write(f"  {year}: {data['predicted'][i]:.0f} thousand\n")
        
        f.write("\n\n[Part 2] AI Replaceability Evaluation (TOPSIS)\n")
        f.write("-"*70 + "\n")
        for i, career in enumerate(careers):
            f.write(f"\n{career}:\n")
            f.write(f"  Closeness Coefficient: {topsis.closeness[i]:.4f}\n")
            f.write(f"  Risk Level: {'High' if topsis.closeness[i] > 0.6 else ('Medium' if topsis.closeness[i] > 0.4 else 'Low')}\n")
        
        f.write("\n\n[Part 3] Education Strategy Simulation (10-year forecast)\n")
        f.write("-"*70 + "\n")
        for career, history in simulation_results.items():
            f.write(f"\n{career}:\n")
            f.write(f"  Initial Students: 1000\n")
            f.write(f"  Final Students: {history['students'][-1]}\n")
            f.write(f"  Average Employment Rate: {np.mean(history['employment_rate']):.2%}\n")
            f.write(f"  Final Employment Rate: {history['employment_rate'][-1]:.2%}\n")
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("End of Results\n")
        f.write("="*70 + "\n")
    
    print("  - results/analysis_results.txt saved")
    print("\nAll results saved successfully!")
    print("="*70)


def create_visualizations(predictions, topsis, simulation_results, careers, years_hist, years_pred):
    """
    创建可视化图表
    """
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 图1：就业需求预测
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Employment Demand Prediction (2015-2030)', fontsize=16, fontweight='bold')
    
    career_keys = list(predictions.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (career_key, color) in enumerate(zip(career_keys, colors)):
        ax = axes[idx // 2, idx % 2]
        data = predictions[career_key]
        
        # 历史数据
        ax.plot(years_hist, data['historical'], 'o-', color=color, 
                label='Historical Data', linewidth=2, markersize=6)
        
        # 拟合值
        all_years = np.arange(2015, 2031)
        ax.plot(all_years, data['all'], '--', color=color, 
                label='GM(1,1) Fitted', linewidth=1.5, alpha=0.7)
        
        # 预测值
        ax.plot(years_pred, data['predicted'], 's-', color=color, 
                label='Prediction', linewidth=2, markersize=6, markerfacecolor='white')
        
        # 分隔线
        ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.5)
        ax.text(2024.7, ax.get_ylim()[1]*0.95, 'Prediction →', 
                fontsize=9, color='gray', style='italic')
        
        career_name_en = career_key.split('(')[0].strip()
        ax.set_title(career_name_en, fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Employment (Thousands)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 第四个图：三个职业对比
    ax = axes[1, 1]
    for career_key, color in zip(career_keys, colors):
        data = predictions[career_key]
        career_name_en = career_key.split('(')[0].strip()
        ax.plot(all_years, data['all'], '-', color=color, 
                label=career_name_en, linewidth=2)
    
    ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Comparison of Three Careers', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Employment (Thousands)', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/employment_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图2：可替代性评价雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    indicators = ["Repetition", "Creativity", "Interaction", 
                  "Complexity", "Update Speed", "Experience",
                  "Current AI Sub", "Tech Feasibility", "Econ Feasibility"]
    
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    evaluation_data = np.array([
        [7, 6, 4, 8, 9, 5, 6, 8, 7],  # 软件工程师
        [5, 3, 7, 6, 4, 9, 2, 3, 4],  # 电工
        [6, 8, 5, 5, 7, 4, 7, 7, 6],  # 平面设计师
    ])
    
    for i, (career, color) in enumerate(zip(careers, colors)):
        values = evaluation_data[i].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=career, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_title('Replaceability Evaluation Radar Chart', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/replaceability_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图3：TOPSIS贴近度对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(careers))
    bars = ax.bar(x_pos, topsis.closeness, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, val in zip(bars, topsis.closeness):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加风险等级线
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Medium Risk Threshold')
    
    ax.set_xlabel('Career', fontsize=12)
    ax.set_ylabel('Closeness Coefficient', fontsize=12)
    ax.set_title('TOPSIS Evaluation: AI Replaceability', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(careers, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/topsis_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图4：系统动力学仿真结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Education System Dynamics Simulation (2025-2034)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('students', 'Student Enrollment', 'Number of Students'),
        ('graduates', 'Annual Graduates', 'Number of Graduates'),
        ('employed', 'Employed Graduates', 'Number of Employed'),
        ('employment_rate', 'Employment Rate', 'Employment Rate')
    ]
    
    sim_years = np.arange(2025, 2035)
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for career, color in zip(careers, colors):
            values = simulation_results[career][metric]
            if metric == 'employment_rate':
                values = np.array(values) * 100  # 转换为百分比
            ax.plot(sim_years, values, 'o-', color=color, label=career, 
                   linewidth=2, markersize=5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(ylabel + (' (%)' if metric == 'employment_rate' else ''), fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/system_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图5：权重分布图（AHP-熵权法组合权重）
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    indicators_full = ["Repetition", "Creativity", "Interaction", 
                       "Complexity", "Update Speed", "Experience",
                       "Current AI Sub", "Tech Feasibility", "Econ Feasibility"]
    
    # AHP权重
    ahp_weights = [0.0835, 0.1373, 0.1373, 0.2222, 0.0630, 0.2222, 0.0448, 0.0448, 0.0448]
    ax1.barh(indicators_full, ahp_weights, color='#3498db', alpha=0.8)
    ax1.set_xlabel('Weight', fontsize=10)
    ax1.set_title('AHP Subjective Weights', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 熵权法权重
    entropy_weights = [0.1181, 0.1279, 0.1088, 0.1088, 0.1117, 0.1052, 0.1052, 0.1052, 0.1088]
    ax2.barh(indicators_full, entropy_weights, color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Weight', fontsize=10)
    ax2.set_title('Entropy Objective Weights', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 组合权重
    combined_weights = [0.1008, 0.1326, 0.1230, 0.1655, 0.0874, 0.1637, 0.0750, 0.0750, 0.0768]
    ax3.barh(indicators_full, combined_weights, color='#2ecc71', alpha=0.8)
    ax3.set_xlabel('Weight', fontsize=10)
    ax3.set_title('Combined Weights (Game Theory)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/weight_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图6：增长率对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算年增长率
    growth_rates = {}
    for career_key in career_keys:
        data = predictions[career_key]
        rates = []
        for i in range(1, len(data['all'])):
            rate = (data['all'][i] - data['all'][i-1]) / data['all'][i-1] * 100
            rates.append(rate)
        growth_rates[career_key] = rates
    
    years_growth = np.arange(2016, 2031)
    for career_key, color in zip(career_keys, colors):
        ax.plot(years_growth, growth_rates[career_key], 'o-', 
                color=color, label=career_key, linewidth=2, markersize=5)
    
    ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Growth Rate (%)', fontsize=12)
    ax.set_title('Employment Growth Rate Comparison (2016-2030)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/growth_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图7：评价指标热力图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    evaluation_data = np.array([
        [7, 6, 4, 8, 9, 5, 6, 8, 7],  # Software Engineer
        [5, 3, 7, 6, 4, 9, 2, 3, 4],  # Electrician
        [6, 8, 5, 5, 7, 4, 7, 7, 6],  # Graphic Designer
    ])
    
    im = ax.imshow(evaluation_data, cmap='RdYlBu_r', aspect='auto', vmin=1, vmax=10)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(indicators_full)))
    ax.set_yticks(np.arange(len(careers)))
    ax.set_xticklabels(indicators_full, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(careers, fontsize=10)
    
    # 添加数值标签
    for i in range(len(careers)):
        for j in range(len(indicators_full)):
            text = ax.text(j, i, evaluation_data[i, j],
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Evaluation Indicators Heatmap (Score: 1-10)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/evaluation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图8：2025 vs 2030就业人数对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    employment_2025 = []
    employment_2030 = []
    for career_key in career_keys:
        data = predictions[career_key]
        employment_2025.append(data['predicted'][0])
        employment_2030.append(data['predicted'][5])
    
    x = np.arange(len(career_keys))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, employment_2025, width, label='2025', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, employment_2030, width, label='2030', color='#e74c3c', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Career', fontsize=12)
    ax.set_ylabel('Employment (Thousands)', fontsize=12)
    ax.set_title('Employment Comparison: 2025 vs 2030', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(career_keys, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/employment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图9：AI影响与适应效果对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ai_impact_factors = [0.35, 0.10, 0.45]
    adaptation_rates = [0.08, 0.03, 0.06]
    
    x = np.arange(len(careers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ai_impact_factors, width, label='AI Impact Factor', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, adaptation_rates, width, label='Adaptation Rate', color='#2ecc71', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Career', fontsize=12)
    ax.set_ylabel('Factor Value', fontsize=12)
    ax.set_title('AI Impact Factor vs Adaptation Rate by Career', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(careers, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/ai_impact_adaptation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  - figures/employment_prediction.png: Employment Demand Prediction")
    print("  - figures/replaceability_radar.png: Replaceability Evaluation Radar Chart")
    print("  - figures/topsis_evaluation.png: TOPSIS Evaluation Results")
    print("  - figures/system_dynamics.png: System Dynamics Simulation")
    print("  - figures/weight_distribution.png: Weight Distribution (AHP-Entropy)")
    print("  - figures/growth_rate_comparison.png: Growth Rate Comparison")
    print("  - figures/evaluation_heatmap.png: Evaluation Indicators Heatmap")
    print("  - figures/employment_comparison.png: Employment Comparison 2025 vs 2030")
    print("  - figures/ai_impact_adaptation.png: AI Impact vs Adaptation Effect")


if __name__ == "__main__":
    main()
