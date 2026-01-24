import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def period_to_freq(period_days):
    return 1.0 / np.array(period_days, dtype=float)

def bandpass_filter(x, fs, lowcut, highcut, order=3):
    nyq = 0.5 * fs
    # avoid invalid normalized frequencies
    low = None if lowcut is None else lowcut / nyq
    high = None if highcut is None else highcut / nyq
    if low is None and high is None:
        return x
    
    # 增加异常处理：如果由于频率极低导致滤波器不稳定，返回全零或原始信号
    # 使用 try-except 在外层处理或者检查输入范围
    if high is not None and high <= 0: return np.zeros_like(x)
    
    if low is None:
        b, a = butter(order, high, btype='low')
    elif high is None:
        b, a = butter(order, low, btype='high')
    else:
        if high <= low:
            # empty band
            return np.zeros_like(x)
        b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)

# 修改：增加了 pad_factor 参数用于平滑频谱图
def compute_band_spectrum(x, fs, pad_factor=1):
    N = len(x)
    # 补零长度：原长度乘以因子，提升频域可视化的密度
    N_fft = int(N * pad_factor)
    
    X = np.fft.fft(x - np.mean(x), n=N_fft)
    freqs = np.fft.fftfreq(N_fft, d=1.0/fs)
    
    pos = freqs > 0
    # 注意：归一化系数仍除以原始信号长度 N，而不是 FFT 长度
    return freqs[pos], (2.0 / N) * np.abs(X[pos])

def analyze_and_plot(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'], dayfirst=False)
    if 'close' not in df.columns:
        raise RuntimeError("CSV must contain 'close' column")
    price = df['close'].astype(float).values
    fs = 1.0  # 1 sample per trading day
    N_original = len(price)
    f_res_limit = 1.0 / N_original

    # 修改：针对股票交易日调整周期定义 (一年约250交易日)
    # 如果想看自然日周期，需要换算。这里假设 250 ~= 1年
    bands = [
        ("短期 (5-20天)", 5, 20),           # 对应周线/月线级别
        ("中期 (20-120天)", 20, 120),       # 对应季线/半年线
        ("长期 (120-250天)", 120, 250),     # 对应年线
        ("超长 (1年-5年)", 250, 250*5),     # 对应康波等长周期
    ]

    t = np.arange(len(price))
    fig, axes = plt.subplots(len(bands), 2, figsize=(14, 3 * len(bands)))
    
    for i, (name, Tmin, Tmax) in enumerate(bands):
        f_high = 1.0 / Tmin
        f_low = 1.0 / Tmax

        # 滤波
        try:
            # 降低阶数可以减少低频滤波时的边界效应与振铃
            filtered = bandpass_filter(price, fs=fs, lowcut=f_low, highcut=f_high, order=2)
        except Exception as e:
            print(f"Filter failed for {name}: {e}, falling back to spectral cut.")
            # 简单的频域硬截断作为回退
            freqs_raw, _ = compute_band_spectrum(price, fs, pad_factor=1)
            # ... (这部分简化处理，主要关注后续绘图)
            filtered = np.zeros_like(price) # 若失败暂置零

        # --- 绘图 ---
        # 1. 时域图
        ax_time = axes[i, 0]
        ax_time.plot(t, price, color='gray', alpha=0.3, label='原始价格')
        ax_time.plot(t, filtered, color='C1', linewidth=1.5, label=f'分量 {name}')
        ax_time.set_title(f"{name} 时域波动 (周期 {Tmin}-{Tmax} 交易日)")
        ax_time.set_ylabel("价格波动")
        ax_time.legend(loc='upper left')
        ax_time.grid(True, alpha=0.3)

        # 2. 频域图
        ax_freq = axes[i, 1]
        
        # 关键修改：设置 pad_factor=10，这会让频域数据点多10倍，曲线变平滑
        freqs_pos, amp_pos = compute_band_spectrum(filtered, fs, pad_factor=20)

        # 仅保留显示范围内的频段数据
        band_mask = (freqs_pos >= f_low * 0.9) & (freqs_pos <= f_high * 1.1)
        
        # 绘图主体
        ax_freq.plot(freqs_pos, amp_pos, color='C2')
        
        # 增加：标记频率分辨率极限（1/N）
        ax_freq.axvline(f_res_limit, color='red', linestyle='--', alpha=0.6, label=f'分辨率极限 (1/{N_original})')
        
        # 某种程度上，为了美观，填充颜色
        ax_freq.fill_between(freqs_pos, 0, amp_pos, color='C2', alpha=0.1)

        # 智能设置坐标轴范围和刻度
        # 找出当前频段内的最大幅值，避免 Y 轴被整体最大值撑大而显得当前频段很平
        if band_mask.any():
            local_max = amp_pos[band_mask].max()
            ax_freq.set_ylim(0, local_max * 1.2)
        
        ax_freq.set_xlim(f_low * 0.8, f_high * 1.2)
        ax_freq.set_title(f"{name} 频谱 (平滑优化后)")
        ax_freq.set_xlabel("频率 (1/交易日)")
        ax_freq.grid(True, alpha=0.3)

        # 辅助刻度：在 X 轴上方标记对应的“天数”
        def freqs2days(x):
            with np.errstate(divide='ignore', invalid='ignore'):
                return 1.0 / x
        
        secax = ax_freq.secondary_xaxis('top', functions=(freqs2days, freqs2days))
        secax.set_xlabel("周期 (交易日)")

        # 你可以随意在此数组增加数值，它们会被准确放置
        valid_ticks = [250*5, 365, 250*2, 250, 120, 60, 20, 10, 5] 

        # 动态捕捉主轴刻度，确保一一对应
        ax_freq.xaxis.set_major_locator(plt.MaxNLocator(6))
        primary_xticks = ax_freq.get_xticks()
        # 将主轴频率转换为天数并加入显示列表
        auto_days = [1.0/f for f in primary_xticks if f > 0]
        
        # 汇总你指定的和自动生成的刻度，并过滤在当前视图范围内的
        all_display_days = [t for t in (valid_ticks + auto_days) if Tmin*0.8 <= t <= Tmax*1.2]
        
        secax.set_ticks(all_display_days)
        secax.set_xticklabels([f"{t:.1f}" if t < 100 else f"{int(t)}" for t in all_display_days])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = "频域分析/MacroTrends_Data_Download_NVDA.csv"  # 相对路径（工作目录为项目根）
    # csv_path = "频域分析/百年道琼斯指数.csv"
    # csv_path = "频域分析/01年至今A股指数.csv"
    
    analyze_and_plot(csv_path)
    
    