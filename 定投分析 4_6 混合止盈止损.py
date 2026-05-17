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
v1=300  # 每月投入金额数
v2=300  # 带止盈定投

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
signal = []
y = mu
for i in range(len(t_list)):
    y = mu + phi*(y-mu) + np.random.randn()*std # 自回归噪声，均值+负反馈项+白噪声项
    y = max(y, 1e-3)
    mu *= (1+increase_rate)
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

# '“倒霉蛋”测试：高位进场会发生什么？'
# # 从最大值位置开始切片
# signal = signal[signal.argmax():]

# 数据段截取


# 以样本点数量重建时间轴（单位：天），采样率仍假定为 1/day
t_list = np.arange(0, len(signal), 1)

# 采样频率
sampling_rate = 1

# 定投仓库1 (单定投)
cost1 = 0.0
asset1 = 0.0  # 资产
share1 = 0.0  # 股份
STORE1 = [(0.0, cost1, asset1, share1)]
T_invest1 = 30  # 统一为30天

# 定投仓库2
cost2 = 0.0
asset2 = 0.0
share2 = 0.0
STORE2 = [(0.0, cost2, asset2, share2)]
T_invest2 = 30  # 统一为30天
cost2_in_system = 0.0
realized_profit2 = 0.0  # 累计已实现的净利润

# 止盈规则
TAKE_PROFIT_RATE = 0.50  # 累计收益率阈值（保持原值，如需改为0.20请告诉我）
SELL_RATIO_PROFIT = 0.2 # 卖出比例（保持原值）
# 止损规则
STOP_LOSS_RATE = 0.2  # 止损阈值
SELL_RATIO_LOSS = 0.1        # 卖出比例，不能超过1的

current_max_day_end = 0
for i, t in enumerate(t_list):
    # “实时”股价
    price = signal[i]
    # 累积最高日收盘价(4%止损是给场内中短线用的，几周到几个月就完了)
    current_max_day_end = max(price, current_max_day_end)

    # 仓库1 30天定投一次（保持不变）
    if int(round(t) % T_invest1) == 0:
        # 成本累计
        cost1 += v1 * T_invest1 / 30
        # 股权累计
        share1 += v1 * T_invest1 / 30 / price
        # 资产记录
        asset1 = share1 * price - cost1
        STORE1.append((t, cost1, asset1, share1))

    # 仓库2 30天定投一次
    # 每次只执行一次止盈/止损
    sold = 0
    # 每周检查一次止盈止损 (7天)
    if int(round(t) % 7) == 0:
        # 执行止盈（基于系统内持仓的累计收益率）
        if cost2_in_system > 0 and share2 > 0:
            cum_return = (share2 * price - cost2_in_system) / cost2_in_system
            if cum_return >= TAKE_PROFIT_RATE:
                sold = 1
                sell_shares = share2 * SELL_RATIO_PROFIT
                proceeds_gross = sell_shares * price
                cost_of_this_sell = cost2_in_system * SELL_RATIO_PROFIT
                profit_of_this_sell = proceeds_gross - cost_of_this_sell
                
                # 更新已实现净利润与系统内持仓/成本
                realized_profit2 += profit_of_this_sell
                share2 -= sell_shares
                cost2_in_system -= cost_of_this_sell
                # print(f"Day {t}: 触发止盈, 卖出 {sell_shares:.2f} 份")

        # 执行止损（基于当前最高日收盘价）
        if cost2_in_system > 0 and share2 > 0 and (not sold):
            if price <= current_max_day_end * (1 - STOP_LOSS_RATE):
                # 触发止损：卖出部分股份并持有现金
                sell_shares = share2 * SELL_RATIO_LOSS
                proceeds_gross = sell_shares * price
                cost_of_this_sell = cost2_in_system * SELL_RATIO_LOSS
                profit_of_this_sell = proceeds_gross - cost_of_this_sell
                
                # 更新已实现净利润与系统内持仓/成本
                realized_profit2 += profit_of_this_sell
                share2 -= sell_shares
                cost2_in_system -= cost_of_this_sell
                
                # 重置最高价计数
                current_max_day_end = price 
                # print(f"Day {t}: 触发止损, 卖出 {sell_shares:.2f} 份")
        
        # 每周记录一次仓库2状态 (为了更准确地反映止损后的收益率)
        asset2 = share2 * price - cost2_in_system + realized_profit2
        STORE2.append((t, cost2, asset2, share2)) # cost2, cost2_in_system

    # 仓库2定投
    if int(round(t) % T_invest2) == 0:
        # 本次定投（买入）
        invest = v2 * T_invest2 / 30        # 瞬时投资金额
        cost2 += invest                    # 总投入记录
        cost2_in_system += invest          # 系统内成本基数
        share2 += invest / price

        # 记录定投后的状态
        asset2 = share2 * price - cost2_in_system + realized_profit2
        STORE2.append((t, cost2, asset2, share2)) # cost2, cost2_in_system
        

# 数据格式转换
TIMES1 = np.array([s[0] for s in STORE1])
COST1 = np.array([s[1] for s in STORE1])
ASSETS1 = np.array([s[2] for s in STORE1])
SHARE1 = np.array([s[3] for s in STORE1])

TIMES2 = np.array([s[0] for s in STORE2])
COST2 = np.array([s[1] for s in STORE2])
ASSETS2 = np.array([s[2] for s in STORE2])
SHARE2 = np.array([s[3] for s in STORE2])

# 绘图
plt.figure(figsize=(10, 12))

# 价格曲线
plt.subplot(411)
plt.plot(t_list, signal, label="价格")
plt.title("价格曲线")
plt.legend()
plt.grid()

# 净资产 (收益额) 对比
plt.subplot(412)
plt.plot(TIMES1, ASSETS1, 'r', label='单定投')
plt.plot(TIMES2, ASSETS2, 'b', label='混合止盈止损')
plt.title("净资产 (累计收益额)")
plt.legend()
plt.grid()

# 净资产/成本比 (收益率) 对比
plt.subplot(413)
plt.plot(TIMES1, ASSETS1/(COST1+1e-8), 'r', label='单定投')
plt.plot(TIMES2, ASSETS2/(COST2+1e-8), 'b', label='混合止盈止损')
plt.title("收益率 (收益/总投入)")
plt.legend()
plt.grid()

# share 对比
plt.subplot(414)
plt.plot(TIMES1, SHARE1, 'r', label='单定投')
plt.plot(TIMES2, SHARE2, 'b', label='混合止盈止损')
plt.title("持有份额")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
