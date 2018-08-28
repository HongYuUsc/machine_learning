# -*- coding: UTF-8 -*-
#该文件的代码根据buy-and-sell模型计算回报率

import numpy as np
import pandas as pd
buy_signal = 0
sell_signal = 0

#wlstm1的回报率
finance_data = np.array(pd.read_excel("wlstm1_000988.xlsx"))
data = finance_data[:,0]
a = finance_data[:,1]
rate = 0

for i in range(len(a)-1):
    if a[i+1]>data[i]:
        buy_signal = 1
        rate += (data[i+1] - data[i] -0.002*(data[i]+data[i+1]))/data[i]
    else:
        sell_signal = 1
        rate += (data[i] - data[i+1] -0.002*(data[i]+data[i+1]))/data[i]

print(rate)

#wlstm2的回报率
finance_data = np.array(pd.read_excel("wlstm2_000988.xlsx"))
data = finance_data[:,0]
a = finance_data[:,1]
rate = 0

for i in range(len(a)-1):
    if a[i+1]>data[i]:
        buy_signal = 1
        rate += (data[i+1] - data[i] -0.002*(data[i]+data[i+1]))/data[i]
    else:
        sell_signal = 1
        rate += (data[i] - data[i+1] -0.002*(data[i]+data[i+1]))/data[i]

print(rate)

#raf的回报率
finance_data = np.array(pd.read_excel("raf_000988.xlsx"))
data = finance_data[:,0]
a = finance_data[:,1]
rate = 0

for i in range(len(a)-1):
    if a[i+1]>data[i]:
        buy_signal = 1
        rate += (data[i+1] - data[i] -0.002*(data[i]+data[i+1]))/data[i]
    else:
        sell_signal = 1
        rate += (data[i] - data[i+1] -0.002*(data[i]+data[i+1]))/data[i]

print(rate)
            
            
            
            
            
            