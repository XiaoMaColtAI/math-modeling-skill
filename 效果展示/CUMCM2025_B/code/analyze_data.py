# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel(r'F:\CUMCM2025_B\附件\附件1.xlsx')
wn = df.iloc[:, 0].values
refl = df.iloc[:, 1].values

print("数据分析 - 附件1 (SiC, 10度)")
print("=" * 50)
print(f"波数范围: {wn.min():.2f} - {wn.max():.2f} cm^-1")
print(f"数据点数: {len(wn)}")
print(f"波数间隔: {np.mean(np.diff(wn)):.6f} cm^-1")
print(f"反射率范围: {refl.min():.2f} - {refl.max():.2f}%")
print()

# 平滑
refl_smooth = signal.savgol_filter(refl, 51, 3)
print(f"平滑后反射率范围: {refl_smooth.min():.2f} - {refl_smooth.max():.2f}%")

# 去趋势
refl_detrended = signal.detrend(refl_smooth)
print(f"去趋势后范围: {refl_detrended.min():.2f} - {refl_detrended.max():.2f}%")
print(f"去趋势后标准差: {np.std(refl_detrended):.4f}%")
print()

# 寻找极值点
peaks_max, _ = signal.find_peaks(refl_detrended, prominence=0.3, distance=20)
peaks_min, _ = signal.find_peaks(-refl_detrended, prominence=0.3, distance=20)
all_peaks = np.sort(np.concatenate([peaks_max, peaks_min]))

print(f"检测到极值点数量: {len(all_peaks)}")
if len(all_peaks) > 1:
    wn_peaks = wn[all_peaks]
    delta_nus = np.diff(wn_peaks)
    print(f"相邻极值点波数间隔统计:")
    print(f"  平均值: {np.mean(delta_nus):.4f} cm^-1")
    print(f"  中位数: {np.median(delta_nus):.4f} cm^-1")
    print(f"  标准差: {np.std(delta_nus):.4f} cm^-1")
    print(f"  最小值: {np.min(delta_nus):.4f} cm^-1")
    print(f"  最大值: {np.max(delta_nus):.4f} cm^-1")

    # 计算厚度
    n_SiC = 2.65
    theta0 = 10
    n_air = 1.0
    theta1_rad = np.arcsin(n_air / n_SiC * np.sin(np.deg2rad(theta0)))

    # 使用平均波数间隔
    delta_nu_mean = np.mean(delta_nus)
    d = 1 / (2 * n_SiC * delta_nu_mean * np.cos(theta1_rad))
    print(f"\n使用极值点法计算的厚度:")
    print(f"  d = {d:.6f} μm = {d*1000:.3f} nm")

# FFT分析
print("\n" + "=" * 50)
print("FFT频谱分析:")
n = len(refl_detrended)
fft_result = np.fft.fft(refl_detrended)
fft_freq = np.fft.fftfreq(n, d=np.mean(np.diff(wn)))

# 只取正频率
positive_freq_idx = fft_freq > 0
fft_freq_positive = fft_freq[positive_freq_idx]
fft_power = np.abs(fft_result[positive_freq_idx])

# 找峰值
peak_idx = np.argmax(fft_power[1:len(fft_power)//2]) + 1
freq_peak = fft_freq_positive[peak_idx]
power_peak = fft_power[peak_idx]

print(f"峰值频率: {freq_peak:.8f} cm")
print(f"对应功率: {power_peak:.2f}")

# 计算厚度
d_fft = 1 / (2 * n_SiC * freq_peak * np.cos(theta1_rad))
print(f"使用FFT法计算的厚度:")
print(f"  d = {d_fft:.6f} μm = {d_fft*1000:.3f} nm")
