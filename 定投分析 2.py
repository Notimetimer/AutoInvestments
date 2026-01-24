from math import pi, sin, cos, tan
import numpy as np
import matplotlib.pyplot as plt

dt=1
t=0

Y=[]
cost=[]
precent=[]
income=[]

tlist=[]

v=500  # 每期投入金额数
T=2.3*dt # 股价波动周期是交易周期的多少倍
amount=0
last_cost=0
phi = 0.7 # 自回归系数
mu = 10 # 均值
std = 1 # 标准差

y=mu
for i in range(240):
    t=t+dt
    # y=10*(1.2+0.1*cos(2*pi/T * t)) # 股单价函数，用正弦函数简化
    y = mu + phi*(y-mu) + np.random.randn()*std # 自回归噪声，均值+负反馈项+白噪声项
    cost.append(last_cost+v)
    
    amount+=v/y # 数量是金额/股单价
    total = amount*y
    tlist.append(t)
    Y.append(y)
    precent.append(total)
    income.append(total-cost[-1])
    
    last_cost=cost[-1]
plt.subplot(211)
plt.plot(tlist, Y)
plt.subplot(212)
plt.plot(tlist, Y)
plt.plot(tlist, income)
plt.show()

# plt.plot(tlist, cost)
# plt.plot(tlist, precent)