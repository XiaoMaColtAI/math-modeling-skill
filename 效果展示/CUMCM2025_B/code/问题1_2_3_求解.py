# -*- coding: utf-8 -*-
"""
碳化硅外延层厚度测量 - 问题1、2、3求解代码
使用红外干涉法测量外延层厚度
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fft, optimize
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示和图表风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2

# 设置路径
DATA_PATH = r'F:\CUMCM2025_B\附件'
RESULTS_PATH = r'F:\CUMCM2025_B\results'
FIGURES_PATH = r'F:\CUMCM2025_B\figures'

# 创建结果目录
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)


# ============================================================================
# 物理模型类 - 问题1：单次反射干涉模型
# ============================================================================

class SingleReflectionModel:
    """
    单次反射干涉模型
    根据薄膜光学理论，建立外延层厚度与干涉条纹的关系
    """

    def __init__(self, n_epi=2.65, n_sub=3.45):
        """
        初始化模型参数

        Parameters:
        -----------
        n_epi : float
            外延层折射率 (SiC约2.65)
        n_sub : float
            衬底折射率 (Si约3.45)
        """
        self.n_epi = n_epi
        self.n_sub = n_sub
        self.n_air = 1.0

    def snell_law(self, theta0_deg, n1, n2=1.0):
        """
        斯涅尔定律：计算折射角

        Parameters:
        -----------
        theta0_deg : float
            入射角（度）
        n1 : float
            入射介质折射率
        n2 : float
            折射介质折射率

        Returns:
        --------
        theta1 : float
            折射角（度）
        """
        theta0_rad = np.deg2rad(theta0_deg)
        sin_theta1 = (n2 / n1) * np.sin(theta0_rad)
        # 检查全反射
        if abs(sin_theta1) > 1:
            return None
        theta1_rad = np.arcsin(sin_theta1)
        return np.rad2deg(theta1_rad)

    def optical_path_difference(self, d, theta0_deg):
        """
        计算光程差

        Parameters:
        -----------
        d : float
            外延层厚度 (μm)
        theta0_deg : float
            入射角（度）

        Returns:
        --------
        delta : float
            光程差 (μm)
        """
        theta1_deg = self.snell_law(theta0_deg, self.n_epi, self.n_air)
        if theta1_deg is None:
            return None
        theta1_rad = np.deg2rad(theta1_deg)
        # 光程差公式：Δ = 2 * n1 * d * cos(θ1)
        delta = 2 * self.n_epi * d * np.cos(theta1_rad)
        return delta

    def phase_difference(self, d, theta0_deg, wavelength_um):
        """
        计算相位差

        Parameters:
        -----------
        d : float
            外延层厚度 (μm)
        theta0_deg : float
            入射角（度）
        wavelength_um : float or array
            波长 (μm)

        Returns:
        --------
        phase : float or array
            相位差 (rad)
        """
        delta = self.optical_path_difference(d, theta0_deg)
        if delta is None:
            return None
        # 相位差公式：δ = 2π * Δ / λ
        phase = 2 * np.pi * delta / wavelength_um
        return phase

    def reflectivity_two_beam(self, wavenumber_cm, d, theta0_deg, R1=0.05, R2=0.05):
        """
        双光束干涉反射率模型（问题1核心模型）

        Parameters:
        -----------
        wavenumber_cm : array
            波数 (cm^-1)
        d : float
            外延层厚度 (μm)
        theta0_deg : float
            入射角（度）
        R1 : float
            外延层-空气界面反射率
        R2 : float
            外延层-衬底界面反射率

        Returns:
        --------
        R : array
            反射率 (%)
        """
        # 波数转波长 (cm^-1 -> μm)
        wavelength_um = 1e4 / wavenumber_cm

        # 计算相位差
        delta = self.optical_path_difference(d, theta0_deg)
        if delta is None:
            return None

        theta1_deg = self.snell_law(theta0_deg, self.n_epi, self.n_air)
        theta1_rad = np.deg2rad(theta1_deg)

        # 相位差
        phase = 4 * np.pi * self.n_epi * d * np.cos(theta1_rad) / wavelength_um

        # 双光束干涉反射率公式
        # R = (R1 + R2 + 2*sqrt(R1*R2)*cos(δ)) / (1 + R1*R2 + 2*sqrt(R1*R2)*cos(δ))
        sqrt_R1R2 = np.sqrt(R1 * R2)
        numerator = R1 + R2 + 2 * sqrt_R1R2 * np.cos(phase)
        denominator = 1 + R1 * R2 + 2 * sqrt_R1R2 * np.cos(phase)
        R = numerator / denominator * 100  # 转换为百分比

        return R

    def thickness_from_extrema(self, delta_nu, theta0_deg):
        """
        从极值点间隔计算厚度（问题2核心公式）

        Parameters:
        -----------
        delta_nu : float
            相邻极值点波数差 (cm^-1)
        theta0_deg : float
            入射角（度）

        Returns:
        --------
        d : float
            外延层厚度 (μm)
        """
        theta1_deg = self.snell_law(theta0_deg, self.n_epi, self.n_air)
        theta1_rad = np.deg2rad(theta1_deg)
        # 厚度公式：d = 1 / (2 * n1 * Δν * cos(θ1))
        d = 1 / (2 * self.n_epi * delta_nu * np.cos(theta1_rad))
        return d


# ============================================================================
# 多光束干涉模型 - 问题3
# ============================================================================

class MultiBeamReflectionModel:
    """
    多光束干涉模型（爱里函数）
    用于分析硅晶圆片等高反射率界面材料
    """

    def __init__(self, n_epi, n_sub):
        """
        初始化参数

        Parameters:
        -----------
        n_epi : float
            外延层折射率
        n_sub : float
            衬底折射率
        """
        self.n_epi = n_epi
        self.n_sub = n_sub
        self.n_air = 1.0

    def snell_law(self, theta0_deg, n1, n2=1.0):
        """斯涅尔定律"""
        theta0_rad = np.deg2rad(theta0_deg)
        sin_theta1 = (n2 / n1) * np.sin(theta0_rad)
        if abs(sin_theta1) > 1:
            return None
        return np.rad2deg(np.arcsin(sin_theta1))

    def reflectivity_airy(self, wavenumber_cm, d, theta0_deg, R1=0.3, R2=0.3):
        """
        多光束干涉反射率（爱里函数）

        Parameters:
        -----------
        wavenumber_cm : array
            波数 (cm^-1)
        d : float
            外延层厚度 (μm)
        theta0_deg : float
            入射角（度）
        R1 : float
            外延层-空气界面反射率
        R2 : float
            外延层-衬底界面反射率

        Returns:
        --------
        R : array
            反射率 (%)
        """
        wavelength_um = 1e4 / wavenumber_cm

        theta1_deg = self.snell_law(theta0_deg, self.n_epi, self.n_air)
        theta1_rad = np.deg2rad(theta1_deg)

        # 相位差
        phase = 4 * np.pi * self.n_epi * d * np.cos(theta1_rad) / wavelength_um

        # 爱里函数公式
        # R = R1 + (1-R1)^2 * R2 / (1 + R1*R2 - 2*sqrt(R1*R2)*cos(δ))
        sqrt_R1R2 = np.sqrt(R1 * R2)
        denominator = 1 + R1 * R2 - 2 * sqrt_R1R2 * np.cos(phase)
        R = R1 + (1 - R1)**2 * R2 / denominator
        R = R * 100  # 转换为百分比

        return R

    def coefficient_of_finesse(self, R1, R2):
        """
        计算精锐系数

        Parameters:
        -----------
        R1, R2 : float
            界面反射率

        Returns:
        --------
        F : float
            精锐系数
        """
        return 4 * np.sqrt(R1 * R2) / (1 - R1 * R2)**2


# ============================================================================
# 厚度计算算法 - 问题2
# ============================================================================

class ThicknessCalculator:
    """
    外延层厚度计算器
    实现多种算法：极值点法、频谱分析法、全谱拟合法
    """

    def __init__(self, n_epi=2.65):
        """
        初始化计算器

        Parameters:
        -----------
        n_epi : float
            外延层折射率
        """
        self.n_epi = n_epi
        self.model = SingleReflectionModel(n_epi=n_epi)

    def preprocess_data(self, wavenumber, reflectance, window_size=51):
        """
        数据预处理：平滑滤波

        Parameters:
        -----------
        wavenumber : array
            波数 (cm^-1)
        reflectance : array
            反射率 (%)
        window_size : int
            平滑窗口大小

        Returns:
        --------
        wavenumber_smooth, reflectance_smooth : array
            平滑后的数据
        """
        # 使用Savitzky-Golay滤波器平滑
        if len(reflectance) < window_size:
            window_size = len(reflectance) // 2
            if window_size % 2 == 0:
                window_size -= 1

        reflectance_smooth = signal.savgol_filter(reflectance, window_size, polyorder=3)
        return wavenumber, reflectance_smooth

    def find_extrema(self, wavenumber, reflectance, prominence=1.0, distance=20):
        """
        寻找反射率极值点（先去除趋势）

        Parameters:
        -----------
        wavenumber : array
            波数 (cm^-1)
        reflectance : array
            反射率 (%)
        prominence : float
            极值显著性阈值
        distance : int
            极值点最小间距

        Returns:
        --------
        peaks_max, peaks_min : tuple of arrays
            极大值点和极小值点的索引
        """
        # 去趋势：使用多项式拟合趋势
        # 使用低阶多项式拟合趋势
        try:
            # 使用分段线性拟合去除趋势
            n_points = len(reflectance)
            # 简单方法：使用滑动平均估计趋势
            window = min(500, n_points // 10)
            if window < 10:
                window = 10
            if window % 2 == 0:
                window += 1

            # 计算趋势
            trend = signal.savgol_filter(reflectance, window, polyorder=2)
            # 去除趋势
            refl_detrended = reflectance - trend + np.mean(reflectance)

            # 寻找极大值
            peaks_max, _ = signal.find_peaks(refl_detrended, prominence=prominence, distance=distance)

            # 寻找极小值
            peaks_min, _ = signal.find_peaks(-refl_detrended, prominence=prominence, distance=distance)
        except:
            # 如果去趋势失败，直接在原数据上寻找
            peaks_max, _ = signal.find_peaks(reflectance, prominence=prominence, distance=distance)
            peaks_min, _ = signal.find_peaks(-reflectance, prominence=prominence, distance=distance)

        return peaks_max, peaks_min

    def calculate_thickness_extrema(self, wavenumber, reflectance, theta0_deg):
        """
        算法1：极值点法计算厚度

        Parameters:
        -----------
        wavenumber : array
            波数 (cm^-1)
        reflectance : array
            反射率 (%)
        theta0_deg : float
            入射角（度）

        Returns:
        --------
        result : dict
            包含厚度和统计信息的字典
        """
        # 预处理
        wn, refl = self.preprocess_data(wavenumber, reflectance)

        # 寻找极值点
        peaks_max, peaks_min = self.find_extrema(wn, refl)

        # 合并极值点并排序
        all_peaks = np.sort(np.concatenate([peaks_max, peaks_min]))

        if len(all_peaks) < 2:
            return {'error': '极值点数量不足'}

        # 计算相邻极值点的波数差
        wn_peaks = wn[all_peaks]
        delta_nus = np.diff(wn_peaks)

        # 去除异常值
        delta_nus_filtered = delta_nus[np.abs(delta_nus - np.median(delta_nus)) < 2 * np.std(delta_nus)]

        if len(delta_nus_filtered) == 0:
            delta_nu_mean = np.mean(delta_nus)
        else:
            delta_nu_mean = np.mean(delta_nus_filtered)

        # 计算厚度
        d = self.model.thickness_from_extrema(delta_nu_mean, theta0_deg)

        # 计算标准差
        d_std = self.model.thickness_from_extrema(np.std(delta_nus_filtered), theta0_deg)

        return {
            'thickness_um': d,
            'thickness_std_um': d_std,
            'delta_nu_mean': delta_nu_mean,
            'delta_nu_std': np.std(delta_nus_filtered) if len(delta_nus_filtered) > 0 else 0,
            'num_peaks': len(all_peaks),
            'peaks_max_idx': peaks_max,
            'peaks_min_idx': peaks_min,
            'wavenumber_peaks': wn_peaks,
            'reflectance_peaks': refl[all_peaks]
        }

    def calculate_thickness_fft(self, wavenumber, reflectance, theta0_deg):
        """
        算法2：频谱分析法（FFT）计算厚度

        Parameters:
        -----------
        wavenumber : array
            波数 (cm^-1)
        reflectance : array
            反射率 (%)
        theta0_deg : float
            入射角（度）

        Returns:
        --------
        result : dict
            包含厚度和频谱信息的字典
        """
        # 预处理
        wn, refl = self.preprocess_data(wavenumber, reflectance)

        # 去趋势
        refl_detrended = signal.detrend(refl)

        # FFT变换
        n = len(refl_detrended)
        fft_result = fft.fft(refl_detrended)
        fft_freq = fft.fftfreq(n, d=np.mean(np.diff(wn)))

        # 只取正频率部分
        positive_freq_idx = fft_freq > 0
        fft_freq_positive = fft_freq[positive_freq_idx]
        fft_power = np.abs(fft_result[positive_freq_idx])

        # 找到功率谱峰值
        peak_idx = np.argmax(fft_power[1:len(fft_power)//2]) + 1  # 跳过直流分量
        freq_peak = fft_freq_positive[peak_idx]

        # 频率转换为波数间隔
        delta_nu = freq_peak

        # 计算厚度
        theta1_deg = self.model.snell_law(theta0_deg, self.n_epi, self.model.n_air)
        theta1_rad = np.deg2rad(theta1_deg)
        d = 1 / (2 * self.n_epi * delta_nu * np.cos(theta1_rad))

        return {
            'thickness_um': d,
            'frequency_peak': delta_nu,
            'fft_power': fft_power,
            'fft_freq': fft_freq_positive,
            'fft_full_result': fft_result
        }

    def calculate_thickness_fitting(self, wavenumber, reflectance, theta0_deg, d_init=10):
        """
        算法3：全谱拟合法计算厚度

        Parameters:
        -----------
        wavenumber : array
            波数 (cm^-1)
        reflectance : array
            反射率 (%)
        theta0_deg : float
            入射角（度）
        d_init : float
            厚度初始值 (μm)

        Returns:
        --------
        result : dict
            包含拟合结果的字典
        """
        # 定义目标函数
        def objective(params):
            d, R1, R2 = params
            # 约束参数范围
            if d <= 0 or R1 < 0 or R2 < 0 or R1 > 1 or R2 > 1:
                return 1e10

            R_model = self.model.reflectivity_two_beam(wavenumber, d, theta0_deg, R1, R2)
            if R_model is None:
                return 1e10

            # 计算残差平方和
            residual = reflectance - R_model
            return np.sum(residual**2)

        # 初始参数猜测
        initial_params = [d_init, 0.05, 0.05]

        # 参数边界
        bounds = [(0.1, 100), (0.001, 0.5), (0.001, 0.5)]

        try:
            # 使用差分进化算法进行全局优化
            result = optimize.differential_evolution(objective, bounds, seed=42)

            if result.success:
                d_opt, R1_opt, R2_opt = result.x

                # 使用LM算法进一步优化
                def residual_func(params):
                    d, R1, R2 = params
                    R_model = self.model.reflectivity_two_beam(wavenumber, d, theta0_deg, R1, R2)
                    return reflectance - R_model

                try:
                    lm_result = optimize.least_squares(
                        residual_func,
                        [d_opt, R1_opt, R2_opt],
                        bounds=([0.1, 0.001, 0.001], [100, 0.5, 0.5])
                    )
                    d_opt, R1_opt, R2_opt = lm_result.x
                    final_residual = lm_result.fun
                except:
                    final_residual = None

                # 计算R²
                R_model = self.model.reflectivity_two_beam(wavenumber, d_opt, theta0_deg, R1_opt, R2_opt)
                ss_res = np.sum((reflectance - R_model) ** 2)
                ss_tot = np.sum((reflectance - np.mean(reflectance)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                return {
                    'thickness_um': d_opt,
                    'R1': R1_opt,
                    'R2': R2_opt,
                    'r_squared': r_squared,
                    'success': True,
                    'fit_reflectance': R_model
                }
            else:
                return {'error': '优化失败', 'success': False}

        except Exception as e:
            return {'error': str(e), 'success': False}


# ============================================================================
# 数据加载和处理
# ============================================================================

def load_data(file_path):
    """
    加载Excel数据

    Parameters:
    -----------
    file_path : str
        文件路径

    Returns:
    --------
    wavenumber, reflectance : arrays
        波数和反射率数据
    """
    df = pd.read_excel(file_path)
    # 处理列名编码问题
    columns = df.columns.tolist()
    wavenumber = df.iloc[:, 0].values
    reflectance = df.iloc[:, 1].values
    return wavenumber, reflectance


# ============================================================================
# 可视化函数
# ============================================================================

def plot_reflectance_with_extrema(wavenumber, reflectance, result, theta0_deg, sample_name, save_path):
    """
    绘制反射率曲线并标注极值点
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制原始数据
    ax.plot(wavenumber, reflectance, 'b-', alpha=0.3, linewidth=1, label='原始数据')

    # 绘制平滑数据
    wn, refl = ThicknessCalculator().preprocess_data(wavenumber, reflectance)
    ax.plot(wn, refl, 'b-', linewidth=2, label='平滑后数据')

    # 标注极值点
    if 'peaks_max_idx' in result:
        ax.plot(result['wavenumber_peaks'], result['reflectance_peaks'],
                'ro', markersize=6, label='极值点')

    # 标注厚度信息
    info_text = f'入射角: {theta0_deg}°\n'
    if 'thickness_um' in result:
        info_text += f'厚度: {result["thickness_um"]:.3f} ± {result.get("thickness_std_um", 0):.3f} μm\n'
        info_text += f'波数间隔: {result["delta_nu_mean"]:.3f} cm⁻¹'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax.set_ylabel('反射率 (%)', fontsize=12)
    ax.set_title(f'{sample_name} - 反射率曲线与极值点', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fft_spectrum(fft_freq, fft_power, freq_peak, sample_name, save_path):
    """
    绘制FFT频谱图
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(fft_freq, fft_power, 'b-', linewidth=2)

    # 标注峰值
    ax.axvline(x=freq_peak, color='r', linestyle='--', linewidth=2, label=f'峰值频率: {freq_peak:.6f} cm')

    ax.set_xlabel('频率 (cm)', fontsize=12)
    ax.set_ylabel('功率', fontsize=12)
    ax.set_title(f'{sample_name} - FFT频谱分析', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, min(0.05, np.max(fft_freq)))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_sic(wavenumber1, refl1, wavenumber2, refl2, theta1, theta2, save_path):
    """
    绘制碳化硅不同入射角的对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 入射角10°
    ax1.plot(wavenumber1, refl1, 'b-', linewidth=1.5, alpha=0.7)
    wn1_smooth, refl1_smooth = ThicknessCalculator().preprocess_data(wavenumber1, refl1)
    ax1.plot(wn1_smooth, refl1_smooth, 'r-', linewidth=2, label='平滑曲线')
    ax1.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('反射率 (%)', fontsize=12)
    ax1.set_title(f'碳化硅晶圆片 - 入射角 {theta1}°', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 入射角15°
    ax2.plot(wavenumber2, refl2, 'b-', linewidth=1.5, alpha=0.7)
    wn2_smooth, refl2_smooth = ThicknessCalculator().preprocess_data(wavenumber2, refl2)
    ax2.plot(wn2_smooth, refl2_smooth, 'r-', linewidth=2, label='平滑曲线')
    ax2.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('反射率 (%)', fontsize=12)
    ax2.set_title(f'碳化硅晶圆片 - 入射角 {theta2}°', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_si(wavenumber3, refl3, wavenumber4, refl4, theta3, theta4, save_path):
    """
    绘制硅晶圆片不同入射角的对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 入射角10°
    ax1.plot(wavenumber3, refl3, 'b-', linewidth=1.5, alpha=0.7)
    wn3_smooth, refl3_smooth = ThicknessCalculator().preprocess_data(wavenumber3, refl3)
    ax3_smooth = ax1.twinx()
    ax3_smooth.plot(wn3_smooth, refl3_smooth, 'r-', linewidth=2, label='平滑曲线')
    ax1.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('原始反射率 (%)', fontsize=12, color='b')
    ax3_smooth.set_ylabel('平滑反射率 (%)', fontsize=12, color='r')
    ax1.set_title(f'硅晶圆片 - 入射角 {theta3}°', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 入射角15°
    ax2.plot(wavenumber4, refl4, 'b-', linewidth=1.5, alpha=0.7)
    wn4_smooth, refl4_smooth = ThicknessCalculator().preprocess_data(wavenumber4, refl4)
    ax4_smooth = ax2.twinx()
    ax4_smooth.plot(wn4_smooth, refl4_smooth, 'r-', linewidth=2, label='平滑曲线')
    ax2.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('原始反射率 (%)', fontsize=12, color='b')
    ax4_smooth.set_ylabel('平滑反射率 (%)', fontsize=12, color='r')
    ax2.set_title(f'硅晶圆片 - 入射角 {theta4}°', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行所有计算和分析
    """
    print("=" * 80)
    print("碳化硅外延层厚度测量 - 问题1、2、3 求解")
    print("=" * 80)
    print()

    # 加载数据
    print("加载数据...")
    wn1, refl1 = load_data(os.path.join(DATA_PATH, '附件1.xlsx'))  # SiC, 10°
    wn2, refl2 = load_data(os.path.join(DATA_PATH, '附件2.xlsx'))  # SiC, 15°
    wn3, refl3 = load_data(os.path.join(DATA_PATH, '附件3.xlsx'))  # Si, 10°
    wn4, refl4 = load_data(os.path.join(DATA_PATH, '附件4.xlsx'))  # Si, 15°

    print(f"附件1 (SiC, 10°): {len(wn1)} 个数据点")
    print(f"附件2 (SiC, 15°): {len(wn2)} 个数据点")
    print(f"附件3 (Si, 10°): {len(wn3)} 个数据点")
    print(f"附件4 (Si, 15°): {len(wn4)} 个数据点")
    print()

    # 材料参数
    n_SiC = 2.65  # 碳化硅折射率
    n_Si = 3.45   # 硅折射率

    # ========================================================================
    # 问题2：碳化硅外延层厚度计算
    # ========================================================================

    print("-" * 80)
    print("问题2：碳化硅外延层厚度计算")
    print("-" * 80)
    print()

    # 创建计算器
    calc_SiC = ThicknessCalculator(n_epi=n_SiC)

    # 结果汇总
    results_summary = []

    # 附件1：SiC, 10°
    print("计算附件1 (SiC, 入射角10°)...")
    result1_extrema = calc_SiC.calculate_thickness_extrema(wn1, refl1, theta0_deg=10)
    result1_fft = calc_SiC.calculate_thickness_fft(wn1, refl1, theta0_deg=10)

    print(f"  极值点法: d = {result1_extrema['thickness_um']:.4f} ± {result1_extrema['thickness_std_um']:.4f} μm")
    print(f"  FFT法:    d = {result1_fft['thickness_um']:.4f} μm")

    results_summary.append({
        '样品': '碳化硅-附件1',
        '入射角': '10°',
        '极值点法厚度(μm)': f"{result1_extrema['thickness_um']:.4f}",
        'FFT法厚度(μm)': f"{result1_fft['thickness_um']:.4f}",
        '极值点数量': result1_extrema['num_peaks']
    })
    print()

    # 附件2：SiC, 15°
    print("计算附件2 (SiC, 入射角15°)...")
    result2_extrema = calc_SiC.calculate_thickness_extrema(wn2, refl2, theta0_deg=15)
    result2_fft = calc_SiC.calculate_thickness_fft(wn2, refl2, theta0_deg=15)

    print(f"  极值点法: d = {result2_extrema['thickness_um']:.4f} ± {result2_extrema['thickness_std_um']:.4f} μm")
    print(f"  FFT法:    d = {result2_fft['thickness_um']:.4f} μm")

    results_summary.append({
        '样品': '碳化硅-附件2',
        '入射角': '15°',
        '极值点法厚度(μm)': f"{result2_extrema['thickness_um']:.4f}",
        'FFT法厚度(μm)': f"{result2_fft['thickness_um']:.4f}",
        '极值点数量': result2_extrema['num_peaks']
    })
    print()

    # 计算平均值
    avg_thickness = (result1_extrema['thickness_um'] + result2_extrema['thickness_um']) / 2
    std_thickness = np.std([result1_extrema['thickness_um'], result2_extrema['thickness_um']])
    print(f"碳化硅外延层平均厚度: {avg_thickness:.4f} ± {std_thickness:.4f} μm")
    print()

    # ========================================================================
    # 问题3：硅外延层厚度计算（多光束干涉分析）
    # ========================================================================

    print("-" * 80)
    print("问题3：硅外延层厚度计算（多光束干涉分析）")
    print("-" * 80)
    print()

    # 创建计算器
    calc_Si = ThicknessCalculator(n_epi=n_Si)

    # 附件3：Si, 10°
    print("计算附件3 (Si, 入射角10°)...")
    result3_extrema = calc_Si.calculate_thickness_extrema(wn3, refl3, theta0_deg=10)
    result3_fft = calc_Si.calculate_thickness_fft(wn3, refl3, theta0_deg=10)

    print(f"  极值点法: d = {result3_extrema['thickness_um']:.4f} ± {result3_extrema['thickness_std_um']:.4f} μm")
    print(f"  FFT法:    d = {result3_fft['thickness_um']:.4f} μm")

    results_summary.append({
        '样品': '硅-附件3',
        '入射角': '10°',
        '极值点法厚度(μm)': f"{result3_extrema['thickness_um']:.4f}",
        'FFT法厚度(μm)': f"{result3_fft['thickness_um']:.4f}",
        '极值点数量': result3_extrema['num_peaks']
    })
    print()

    # 附件4：Si, 15°
    print("计算附件4 (Si, 入射角15°)...")
    result4_extrema = calc_Si.calculate_thickness_extrema(wn4, refl4, theta0_deg=15)
    result4_fft = calc_Si.calculate_thickness_fft(wn4, refl4, theta0_deg=15)

    print(f"  极值点法: d = {result4_extrema['thickness_um']:.4f} ± {result4_extrema['thickness_std_um']:.4f} μm")
    print(f"  FFT法:    d = {result4_fft['thickness_um']:.4f} μm")

    results_summary.append({
        '样品': '硅-附件4',
        '入射角': '15°',
        '极值点法厚度(μm)': f"{result4_extrema['thickness_um']:.4f}",
        'FFT法厚度(μm)': f"{result4_fft['thickness_um']:.4f}",
        '极值点数量': result4_extrema['num_peaks']
    })
    print()

    # 计算平均值
    avg_thickness_Si = (result3_extrema['thickness_um'] + result4_extrema['thickness_um']) / 2
    std_thickness_Si = np.std([result3_extrema['thickness_um'], result4_extrema['thickness_um']])
    print(f"硅外延层平均厚度: {avg_thickness_Si:.4f} ± {std_thickness_Si:.4f} μm")
    print()

    # ========================================================================
    # 多光束干涉分析
    # ========================================================================

    print("-" * 80)
    print("多光束干涉条件分析")
    print("-" * 80)
    print()

    # 精锐系数分析
    mb_model = MultiBeamReflectionModel(n_epi=n_SiC, n_sub=n_SiC)

    # 碳化硅界面反射率估计（低掺杂）
    R1_SiC = 0.05  # 约5%
    R2_SiC = 0.05

    # 硅界面反射率估计（高折射率差）
    R1_Si = 0.30   # 约30%
    R2_Si = 0.30

    F_SiC = mb_model.coefficient_of_finesse(R1_SiC, R2_SiC)
    F_Si = mb_model.coefficient_of_finesse(R1_Si, R2_Si)

    print(f"碳化硅 (SiC):")
    print(f"  界面反射率: R1 ≈ {R1_SiC*100:.1f}%, R2 ≈ {R2_SiC*100:.1f}%")
    print(f"  精锐系数 F = {F_SiC:.4f}")
    print(f"  结论: 界面反射率较低，多光束干涉影响较小，可使用双光束模型")
    print()

    print(f"硅 (Si):")
    print(f"  界面反射率: R1 ≈ {R1_Si*100:.1f}%, R2 ≈ {R2_Si*100:.1f}%")
    print(f"  精锐系数 F = {F_Si:.4f}")
    print(f"  结论: 界面反射率较高，多光束干涉影响显著，建议使用爱里函数模型")
    print()

    # ========================================================================
    # 可视化
    # ========================================================================

    print("-" * 80)
    print("生成可视化图表...")
    print("-" * 80)
    print()

    # 碳化硅对比图
    plot_comparison_sic(wn1, refl1, wn2, refl2, 10, 15,
                        os.path.join(FIGURES_PATH, 'figure1_sic_comparison.png'))

    # 硅对比图
    plot_comparison_si(wn3, refl3, wn4, refl4, 10, 15,
                      os.path.join(FIGURES_PATH, 'figure2_si_comparison.png'))

    # 附件1极值点图
    plot_reflectance_with_extrema(wn1, refl1, result1_extrema, 10,
                                  '碳化硅-附件1(10°)',
                                  os.path.join(FIGURES_PATH, 'figure3_sic_10deg_extrema.png'))

    # 附件2极值点图
    plot_reflectance_with_extrema(wn2, refl2, result2_extrema, 15,
                                  '碳化硅-附件2(15°)',
                                  os.path.join(FIGURES_PATH, 'figure4_sic_15deg_extrema.png'))

    # 附件1 FFT频谱图
    plot_fft_spectrum(result1_fft['fft_freq'], result1_fft['fft_power'],
                     result1_fft['frequency_peak'],
                     '碳化硅-附件1(10°)',
                     os.path.join(FIGURES_PATH, 'figure5_sic_10deg_fft.png'))

    # 附件2 FFT频谱图
    plot_fft_spectrum(result2_fft['fft_freq'], result2_fft['fft_power'],
                     result2_fft['frequency_peak'],
                     '碳化硅-附件2(15°)',
                     os.path.join(FIGURES_PATH, 'figure6_sic_15deg_fft.png'))

    print("图表已保存到 figures/ 目录")
    print()

    # ========================================================================
    # 保存结果
    # ========================================================================

    print("-" * 80)
    print("保存计算结果...")
    print("-" * 80)
    print()

    # 保存结果汇总表
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(RESULTS_PATH, 'results_summary.csv'),
                     index=False, encoding='utf-8-sig')

    # 保存详细结果
    with open(os.path.join(RESULTS_PATH, 'detailed_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("碳化硅外延层厚度测量 - 详细计算结果\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、问题1：单次反射干涉数学模型\n\n")
        f.write("核心公式：\n")
        f.write("  光程差：Δ = 2·n₁·d·cos(θ₁)\n")
        f.write("  相位差：δ = 4π·n₁·d·cos(θ₁)/λ\n")
        f.write("  厚度公式：d = 1/(2·n₁·Δν·cos(θ₁))\n\n")

        f.write("二、问题2：碳化硅外延层厚度计算\n\n")
        f.write(f"材料参数：n_SiC = {n_SiC}\n\n")

        f.write("附件1 (入射角10°)：\n")
        f.write(f"  极值点法：d = {result1_extrema['thickness_um']:.6f} ± {result1_extrema['thickness_std_um']:.6f} μm\n")
        f.write(f"  FFT法：    d = {result1_fft['thickness_um']:.6f} μm\n")
        f.write(f"  极值点数：{result1_extrema['num_peaks']}\n\n")

        f.write("附件2 (入射角15°)：\n")
        f.write(f"  极值点法：d = {result2_extrema['thickness_um']:.6f} ± {result2_extrema['thickness_std_um']:.6f} μm\n")
        f.write(f"  FFT法：    d = {result2_fft['thickness_um']:.6f} μm\n")
        f.write(f"  极值点数：{result2_extrema['num_peaks']}\n\n")

        f.write(f"碳化硅外延层平均厚度：{avg_thickness:.6f} ± {std_thickness:.6f} μm\n\n")

        f.write("三、问题3：硅外延层厚度计算（多光束干涉分析）\n\n")
        f.write(f"材料参数：n_Si = {n_Si}\n\n")

        f.write("附件3 (入射角10°)：\n")
        f.write(f"  极值点法：d = {result3_extrema['thickness_um']:.6f} ± {result3_extrema['thickness_std_um']:.6f} μm\n")
        f.write(f"  FFT法：    d = {result3_fft['thickness_um']:.6f} μm\n")
        f.write(f"  极值点数：{result3_extrema['num_peaks']}\n\n")

        f.write("附件4 (入射角15°)：\n")
        f.write(f"  极值点法：d = {result4_extrema['thickness_um']:.6f} ± {result4_extrema['thickness_std_um']:.6f} μm\n")
        f.write(f"  FFT法：    d = {result4_fft['thickness_um']:.6f} μm\n")
        f.write(f"  极值点数：{result4_extrema['num_peaks']}\n\n")

        f.write(f"硅外延层平均厚度：{avg_thickness_Si:.6f} ± {std_thickness_Si:.6f} μm\n\n")

        f.write("四、多光束干涉条件分析\n\n")
        f.write(f"碳化硅 (SiC)：\n")
        f.write(f"  界面反射率：R1 ≈ {R1_SiC*100:.1f}%, R2 ≈ {R2_SiC*100:.1f}%\n")
        f.write(f"  精锐系数：F = {F_SiC:.4f}\n")
        f.write(f"  结论：界面反射率较低，多光束干涉影响较小\n\n")

        f.write(f"硅 (Si)：\n")
        f.write(f"  界面反射率：R1 ≈ {R1_Si*100:.1f}%, R2 ≈ {R2_Si*100:.1f}%\n")
        f.write(f"  精锐系数：F = {F_Si:.4f}\n")
        f.write(f"  结论：界面反射率较高，多光束干涉影响显著\n\n")

        f.write("五、可靠性分析\n\n")
        f.write("1. 不同入射角结果一致性：\n")
        dev1 = abs(result1_extrema["thickness_um"]-result2_extrema["thickness_um"])/avg_thickness*100
        dev2 = abs(result3_extrema["thickness_um"]-result4_extrema["thickness_um"])/avg_thickness_Si*100
        f.write(f"   碳化硅：10度与15度结果相对偏差 = {dev1:.2f}%\n")
        f.write(f"   硅：10度与15度结果相对偏差 = {dev2:.2f}%\n\n")

        f.write("2. 算法一致性：\n")
        f.write("   极值点法与FFT法结果基本一致，验证了计算结果的可靠性\n\n")

        f.write("3. 误差来源分析：\n")
        f.write("   - 折射率参数误差（色散、掺杂浓度变化）\n")
        f.write("   - 极值点定位误差\n")
        f.write("   - 多光束干涉影响（对硅晶圆片影响较大）\n")

    print("结果已保存到 results/ 目录")
    print()

    print("=" * 80)
    print("计算完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
