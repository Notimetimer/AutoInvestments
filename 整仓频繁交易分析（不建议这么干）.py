'''
优化目标：
指数滑动平均后的盈利率

背景信息：
长时股价波动曲线
定投频率和数额

优化自变量：
1. 止盈比例
2. 触发止盈时的卖出比例

求解算法：穷举法（似乎也不需要很高的精度）
'''

import numpy as np
from math import pi, sin, cos, tan
import matplotlib.pyplot as plt
import pandas as pd  # <-- 新增：用于从 CSV 读取价格序列
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Y=[]
cost=[]
precent=[]
income=[]

tlist=[]

# 投资参数
v=100  # 每月投入金额数

amount=0
last_cost=0
phi = 0.9 # 自回归系数
mu = 10 # 均值
std = 3 # 标准差

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
'正弦信号'
# # 预先生成数据
# signal = A0 + \
#         A1 * np.sin(2 * np.pi / T1 * t_list) + \
#         A2 * np.sin(2 * np.pi / T2 * t_list) + \
#         A3 * np.sin(2 * np.pi / T3 * t_list)

'自回归噪声信号'
# signal = []
# y = mu
# for i in range(len(t_list)):
#     y = mu + phi*(y-mu) + np.random.randn()*std # 自回归噪声，均值+负反馈项+白噪声项
#     y = max(y, 1e-3)
#     mu *= (1+increase_rate)
#     signal.append(y)
# signal = np.array(signal)


'改进版自回归噪声信号'
signal = []

# --- 重新设置平滑参数 ---
mu = 100.0                 # 提高初始均值基数
y = mu
phi = 0.98                 # 提高自回归系数，越接近1价格越平滑（惯性大）
std = 1.5                  # 缩小随机扰动的绝对振幅
increase_rate = (1 + 0.05) ** (1/365) - 1 

# --- 退市设定 ---
delist_threshold = 20.0    # 设定退市红线（比如跌到20块直接清零）
is_delisted = False        # 退市状态标记

for i in range(len(t_list)):
    if is_delisted:
        # 一旦退市，后续所有时间点价格均为0
        signal.append(0.0)
        continue
        
    # 自回归噪声：均值 + 负反馈项 + 白噪声
    y = mu + phi * (y - mu) + np.random.randn() * std 
    
    # 退市判定
    if y < delist_threshold:
        y = 0.0
        is_delisted = True
    
    mu *= (1 + increase_rate)
    signal.append(y)

signal = np.array(signal)

'读取历史数据'
# # 删除自回归噪声部分，改为从 CSV 读取价格序列
# # csv_path 请根据你的项目结构替换为实际文件路径

# # csv_path = r"MacroTrends_Data_Download_NVDA.csv"
# # csv_path = r"百年道琼斯指数.csv"
# csv_path = r"01年至今A股指数.csv"

# df = pd.read_csv(csv_path, parse_dates=['date'], dayfirst=False)
# # 优先使用 'close' 列，若无则使用第一个数值列
# if 'close' in df.columns:
#     signal = df['close'].astype(float).values
# else:
#     numeric_cols = df.select_dtypes(include=[float, int]).columns
#     if len(numeric_cols) == 0:
#         raise RuntimeError("CSV must contain a numeric price column (e.g. 'close')")
#     signal = df[numeric_cols[0]].astype(float).values
# # '“倒霉蛋”测试：高位进场会发生什么？'
# # # 从最大值位置开始切片
# signal = signal[signal.argmax():]

# 数据段截取
# 以样本点数量重建时间轴（单位：天），采样率仍假定为 1/day
t_list = np.arange(0, len(signal), 1)


# 预计算移动平均线 (MA7, MA30, MA60)
# min_periods=1 保证最开始几天也有值，但早期不构成真实的长均线
price_series = pd.Series(signal)
ma7 = price_series.rolling(window=7, min_periods=1).mean().values
ma30 = price_series.rolling(window=30, min_periods=1).mean().values
ma60 = price_series.rolling(window=60, min_periods=1).mean().values

# ----------------- 仿真逻辑函数 -----------------
def Full_Position_Simulation(signal, take_profit_rate, stop_loss_rate, cool_down_days=14): # 30
    """
    整仓仿真系统（显式处理退市逻辑版）：
    1. 初始资金 1500，第0天无条件满仓买入。
    2. 到达止盈或止损比例后，全仓卖出并触发 30 天冷却期。
    3. 冷却期满后，若满足 MA7 > MA30 > MA60 且标的未退市，则再次全仓买入。
    4. 显式拦截：若持仓标的突发退市（价格为0），资产直接清零并终止后续交易。
    """
    cash = 1500.0       # 初始本金
    shares = 0.0        # 持有份额
    cost_price = 0.0    # 持仓成本线
    cooldown = 0        # 冷却倒计时（天）
    
    ASSETS = []
    
    # # 1. 最开始的买入 (第0天无条件全仓)
    # # 增加防御：如果第0天就已经是退市状态(0.0)，则无法买入
    # if signal[0] > 0:
    #     shares = cash / signal[0]
    #     cash = 0.0
    #     cost_price = signal[0]
    # else:
    #     shares = 0.0
    #     cash = 0.0  # 初始即退市，资产直接清零
    #     cost_price = 0.0

    for i, price in enumerate(signal):
        # 显式拦截：如果价格为 0（代表标的已退市）
        if price == 0.0:
            cash = 0.0
            shares = 0.0
            ASSETS.append(0.0)  # 资产清零
            continue            # 直接跳过后续的持仓/空仓状态机判断
            
        # 记录每日总资产状态（现金 + 股票现值）
        current_asset = cash + shares * price
        ASSETS.append(current_asset)
        
        # 状态机处理
        if shares > 0: 
            # 状态A：持仓中 -> 判断是否触发止盈或止损
            # 此时 price > 0 且 cost_price > 0 (因为买入时做过限制)，分母绝对安全
            return_rate = (price - cost_price) / cost_price
            
            if return_rate >= take_profit_rate or return_rate <= -stop_loss_rate:
                # 触发条件，全仓卖出
                cash = shares * price
                shares = 0.0
                cooldown = cool_down_days  # 卖出后触发30天冷却
                
        else: 
            # 状态B：空仓中 -> 检查冷却及买入条件
            if cooldown > 0:
                cooldown -= 1  # 冷却倒计时
            else:
                # 冷却期结束，检查均线多头排列条件: MA7 > MA30 > MA60
                if ma7[i] > ma30[i] and ma30[i] > ma60[i]:
                    # 满足条件，且当前价格有效(安全防御)，全仓买入
                    shares = cash / price
                    cash = 0.0
                    cost_price = price

    # 最终结果：使用期末总净值作为评估标准
    final_reward = ASSETS[-1]
    return t_list, np.array(ASSETS), final_reward

if __name__=='__main__':
    # ----------------- 参数穷举 (网格搜索) -----------------
    # 缩小步长以提高网格精度
    TAKE_PROFIT_RATE_range = np.arange(0.1, 1.51, 0.1)  # 止盈 10% ~ 150%
    STOP_LOSS_RATE_range = np.arange(0.05, 0.51, 0.05)  # 止损 5% ~ 50%

    results = []
    print("正在执行网格搜索，请稍候...")
    for tp in TAKE_PROFIT_RATE_range:
        for sl in STOP_LOSS_RATE_range:
            _, _, reward = Full_Position_Simulation(signal, tp, sl)
            results.append((tp, sl, reward))

    results = np.array(results)
    
    # 获取使最终资金最大化的最优参数组
    best_idx = np.argmax(results[:, 2])
    best_tp, best_sl, best_reward = results[best_idx]
    
    print("\n--- 优化结果 ---")
    print(f"最佳止盈比例: {best_tp*100:.1f}%")
    print(f"最佳止损比例: {best_sl*100:.1f}%")
    print(f"最终总资产: {best_reward:.2f} (初始本金 1500)")
    print("----------------\n")

    # ----------------- 可视化部分 -----------------
    # 1. 绘制 3D 收益平面图
    fig1 = plt.figure(figsize=(9, 6))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_trisurf(results[:, 0], results[:, 1], results[:, 2], cmap='viridis', alpha=0.8)
    ax.scatter([best_tp], [best_sl], [best_reward], color='r', s=80, label='最优参数点')
    ax.set_xlabel('止盈比例 (TP)')
    ax.set_ylabel('止损比例 (SL)')
    ax.set_zlabel('期末总资产')
    ax.set_title('止盈/止损参数与最终净资产收益关系面')
    ax.legend()

    # 2. 用最优参数执行一次最终回测以获取曲线
    TIMES, ASSETS, _ = Full_Position_Simulation(signal, best_tp, best_sl)

    fig2 = plt.figure(figsize=(10, 8))
    
    # 价格与均线对比图
    plt.subplot(211)
    plt.plot(t_list, signal, color='black', label='股价', alpha=0.7)
    plt.plot(t_list, ma7, label='MA7', linewidth=1)
    plt.plot(t_list, ma30, label='MA30', linewidth=1)
    plt.plot(t_list, ma60, label='MA60', linewidth=1)
    plt.title("标的价格与均线系统")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 资产净值变化图
    plt.subplot(212)
    plt.plot(TIMES, ASSETS, 'b', label=f'策略净资产 (最优止盈:{best_tp:.2f}, 止损:{best_sl:.2f})')
    plt.axhline(y=1500, color='r', linestyle='--', label='初始本金线(1500)')
    plt.title("账户总资产变化")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()