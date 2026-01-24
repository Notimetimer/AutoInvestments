import numpy as np
from math import pi, sin, cos, tan
import matplotlib.pyplot as plt
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Y=[]
cost=[]
precent=[]
income=[]

tlist=[]

# 投资参数
v=300  # 每月投入金额数

amount=0
last_cost=0
phi = 0.9 # 自回归系数
mu = 10 # 均值
std = 0.1 # 标准差

increase_rate = (1 + 0.05) ** (1/365) - 1 # 年化增长率 5%

# 信号生成，时间单位为天
# sampling_rate = 1
# t_list = np.linspace(0, 365*2, sampling_rate, endpoint=False)

# 使用步长为1天，生成 0..729（不含 730）
t_list = np.arange(0, 365*3, 1)

# 如果想包含 730，则用：
# t_list = np.arange(0, 365*2 + 1, 1)
# 波动周期
T1 = 20
T2 = 10
T3 = 365
# 波动幅度
A0 = mu
A1 = 1
A2 = 1
A3 = 1

# 采样周期
sampling_rate = 1

# # 预先生成数据
# signal = A0 + \
#         A1 * np.sin(2 * np.pi / T1 * t_list) + \
#         A2 * np.sin(2 * np.pi / T2 * t_list) + \
#         A3 * np.sin(2 * np.pi / T3 * t_list)

signal = []

y = mu
for i in range(len(t_list)):
    y = mu + phi*(y-mu) + np.random.randn()*std # 自回归噪声，均值+负反馈项+白噪声项
    mu *= (1+increase_rate)
    signal.append(y)

signal = np.array(signal)

# 定投仓库1 
cost1 = 0
asset1 = 0  # 资产
share1 = 0  # 股份
STORE1 = [(0, cost1, asset1, share1)]
T_invest1 = 14

# 定投仓库2
cost2 = 0
asset2 = 0
share2 = 0
STORE2 = [(0, cost2, asset2, share2)]
T_invest2 = 30

for i, t in enumerate(t_list):
    # “实时”股价
    price = signal[i]
        
    # 仓库1 14天定投一次
    if int(round(t) % T_invest1) == 0:
        # 成本累计
        cost1 += v *T_invest1/30
        # 股权累计
        share1 += v *T_invest1/30 / price
        # 资产记录
        asset1 = share1 * price - cost1
        STORE1.append((t, cost1, asset1, share1))

    # 仓库2 30天定投一次
    if int(round(t) % T_invest2) == 0:
        # 成本累计
        cost2 += v *T_invest2/30
        # 股权累计
        share2 += v *T_invest2/30 / price
        # 资产记录
        asset2 = share2 * price - cost2
        STORE2.append((t, cost2, asset2, share2))

# 数据格式转换
TIMES1 = np.array([s[0] for s in STORE1])
COST1 = np.array([s[1] for s in STORE1])
ASSETS1 = np.array([s[2] for s in STORE1])
SHARE1 = np.array([s[3] for s in STORE1])

TIMES2 = np.array([s[0] for s in STORE2])
COST2 = np.array([s[1] for s in STORE2])
ASSETS2 = np.array([s[2] for s in STORE2])
SHARE2 = np.array([s[3] for s in STORE2])

# 价格曲线
plt.subplot(411)
plt.plot(t_list, signal)
plt.title("价格曲线")
plt.grid()

# assets 对比
plt.subplot(412)
plt.plot(TIMES1, ASSETS1, 'r', label='14天定投资产')
plt.plot(TIMES2, ASSETS2, 'b', label='30天定投资产')
plt.legend()
plt.grid()

# share 对比
plt.subplot(413)
plt.plot(TIMES1, SHARE1, 'r', label='14天定投持股')
plt.plot(TIMES2, SHARE2, 'b', label='30天定投持股')
plt.legend()
plt.tight_layout()
plt.grid()

# 试验性举措，用傅里叶变换处理信号
from scipy.fftpack import fft
# FFT变换及处理
fft_result = np.fft.fft(signal)
N = len(signal)
freqs = np.fft.fftfreq(N, d=1/sampling_rate)

# 双边幅值（归一化）
ampl = np.abs(fft_result) / N

# 取非负频率（单边谱），并对除直流和奈奎斯特以外的分量乘 2
pos_mask = freqs >= 0
freqs_pos = freqs[pos_mask]
ampl_pos = ampl[pos_mask].copy()
if N % 2 == 0:
    ampl_pos[1:-1] *= 2
else:
    ampl_pos[1:] *= 2

plt.subplot(414)
plt.semilogy(freqs_pos, ampl_pos)
plt.title("单边振幅谱（对数纵坐标）")
plt.xlabel("频率 (Hz)")
plt.ylabel("振幅 (log scale)")
plt.tight_layout()
plt.grid()

plt.show()
