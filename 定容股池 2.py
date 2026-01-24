from math import pi, sin, cos, tan
import numpy as np
import matplotlib.pyplot as plt

Y=[]
precent=[]
income=[]

tlist=[]

v0 = 648
T= 3.1  # 股价波动周期(月)
dt=T/5.1 # 交易周期（月）
t=0
period = 240 # 交易时长(月)
phi = 0.7 # 自回归系数
mu = 10 # 均值
std = 1 # 标准差

amount=0
income_so_far=0
income_t=0

y=mu
for i in range(round(period/dt)):
    t=t+dt
    # y=10*(1.2+0.1*cos(2*pi/T * t)+0*t) # 股单价函数，正弦函数简化
    y = mu + phi*(y-mu) + np.random.randn()*std # 自回归噪声，均值+负反馈项+白噪声项
    if i==0:
        income_t=0 # 排除额外动作之后是整v0存入，整v0提出，所以不算入成本
        funds_precent=v0
    else:
        # 现价
        funds_precent=y*amount
        # 高于v0掏出，低于v0补入
        income_t=funds_precent-v0
    income_so_far += income_t # 累积收入/损失小计
    amount=v0/y # 每次买卖维持固定数额后股票持有数
    
    tlist.append(t)
    Y.append(y)
    precent.append(funds_precent)
    income.append(income_so_far)

plt.subplot(211)
plt.plot(tlist, Y)
plt.title('price-t')
plt.subplot(212)
plt.title('price-t vs income-t')
plt.plot(tlist, Y)
plt.plot(tlist, income)
plt.show()