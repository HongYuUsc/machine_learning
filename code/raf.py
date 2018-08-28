# -*- coding: UTF-8 -*-
#该文件的代码为RAF模型
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy  as np

#数据归一化
def MaxMinNormalization(x):
    for i in range(x.shape[1]-1):
       Max = np.max(x[:,i+1])
       Min = np.min(x[:,i+1])
       for j in range(x.shape[0]):
           x[j,i+1] = (x[j,i+1] - Min) / (Max - Min)
    return x

#Mape指标	
def Mape(pred,y):
    Sum = 0
    for i in range(len(pred)):
        Sum = Sum + abs((y[i] - pred[i])/y[i])
    mape = Sum/len(pred)
    return mape

#预测
def predict_data(x_test,y_test,Max,Min):
    pred = regr.predict(x_test)
    for i in range(len(pred)):
        pred[i] = (Max - Min)*pred[i] + Min
        y_test[i] = (Max - Min)*y_test[i] + Min
    pred = np.reshape(pred,(len(pred),1))
    y_test = np.reshape(y_test,(len(y_test),1))
    d = np.hstack((y_test,pred))
    pred_excel = pd.DataFrame(d,columns=["actual","predict"])
    pred_excel.to_excel("raf_000988.xlsx", sheet_name = "S&P500 Index Data")
    return [pred,y_test]

#TRAIN_SIZE = 484
#START_INDEX = 0
#VALIDATION_SIZE = 543
#WHOLE_SIZE = 782
START_INDEX = 0
TRAIN_SIZE = 2640
VALIDATION_SIZE = 2701
WHOLE_SIZE = 2880
BATCH_SIZE = 60
EPOCH = 5000

finance_data = pd.read_csv("000988.csv")
#finance_data = pd.read_excel("RawData.xlsx",sheet_name = "Nifty 50 index Data")
data = np.array(finance_data)
close_max = np.max(data[:,2])
close_min = np.min(data[:,2])
data = MaxMinNormalization(data)
data_train = data[:,1:]
x_train = data_train[START_INDEX:START_INDEX+TRAIN_SIZE,:]
y_train = data[START_INDEX+1:START_INDEX+TRAIN_SIZE+1,2]
x_validation = data_train[START_INDEX+TRAIN_SIZE:START_INDEX+VALIDATION_SIZE,:]
y_validation = data[START_INDEX+TRAIN_SIZE+1:START_INDEX+VALIDATION_SIZE+1,2]
x_test = data_train[START_INDEX+VALIDATION_SIZE:START_INDEX+WHOLE_SIZE+1,:]
y_true = data[START_INDEX+VALIDATION_SIZE+1:START_INDEX+WHOLE_SIZE+2,2]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)


regr = RandomForestRegressor(n_estimators=1000, max_depth=20)#构造随机森林，1000颗回归树，最大深度20
regr.fit(x_train,y_train)#训练
pred,y_true = predict_data(x_test,y_true,close_max,close_min)
print(Mape(pred,y_true))








