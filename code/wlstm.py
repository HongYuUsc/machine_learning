# -*- coding: UTF-8 -*-
#该文件的代码为WLSTM模型
#根据本文的注释适当调整代码可得到WLSTM1和WLSTM2模型

import numpy as np
import pandas as pd
import pywt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#无偏风险估计阈值
def threshhold(array): 
    sort1 = np.sort(np.square(np.absolute(array)))
    n = len(sort1)
    risk = []
    for k in range(n):
        Sum = 0
        for i in range(k):
            Sum += sort1[i]
        risk.append((n - 2*k + Sum + (n - (k+1))*sort1[n - (k+1)])/n)
    k = risk.index(min(risk))
    thresh = math.sqrt(sort1[k])
    return thresh

#小波变换
def wavelet(data):
    thresh = math.sqrt(2*math.log10(len(data))) #固定阈值处理，WLSTM2
    wavelet_result = []
    for i in range(data.shape[1]):
        if i >= 1:
            data_test = data[:,i]
            coeffs = pywt.wavedec(data_test,"haar",level = 2) #二级小波变换
            thresh1 = threshhold(coeffs[-1]) #无偏风险估计阈值
            thresh2 = threshhold(coeffs[-2]) #无偏风险估计阈值
            #coeffs[-2] = pywt.threshold(coeffs[-2], thresh, 'soft')
            #coeffs[-1] = pywt.threshold(coeffs[-1], thresh, 'soft')
            coeffs[-2] = pywt.threshold(coeffs[-2], thresh2, 'soft')
            coeffs[-1] = pywt.threshold(coeffs[-1], thresh1, 'soft') #无偏风险估计阈值处理，WLSTM1
            #coeffs[-2] = np.zeros_like(coeffs[-2])
            #coeffs[-1] = np.zeros_like(coeffs[-1])
            #x = np.arange(len(data_test))
            result = pywt.waverec(coeffs,"haar") #小波重构
            wavelet_result.append(result[:len(data)])
        else:
            result = np.zeros((len(data),))
            wavelet_result.append(result)

    wavelet_result = np.array(wavelet_result)
    data = np.transpose(wavelet_result)

    #第二次小波变换
    wavelet_result2 = []
    for j in range(data.shape[1]):
        if i >= 1:
            data_test = data[:,j]
            coeffs = pywt.wavedec(data_test,"haar",level = 2)
            thresh1 = threshhold(coeffs[-1])
            thresh2 = threshhold(coeffs[-2])
            #coeffs[-2] = pywt.threshold(coeffs[-2], thresh, 'soft')
            #coeffs[-1] = pywt.threshold(coeffs[-1], thresh, 'soft')
            coeffs[-2] = pywt.threshold(coeffs[-2], thresh2, 'soft')
            coeffs[-1] = pywt.threshold(coeffs[-1], thresh1, 'soft')
            #coeffs[-2] = np.zeros_like(coeffs[-2])
            #coeffs[-1] = np.zeros_like(coeffs[-1])
            result = pywt.waverec(coeffs,"haar")
            wavelet_result2.append(result[:len(result)])
        else:
            result = np.zeros((len(data),))
            wavelet_result2.append(result)

    wavelet_result2 = np.transpose(np.array(wavelet_result2))
    return wavelet_result2

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

#读入金融数据
def load_data(filename, sheetname, seq_len,start_index, train_size, validation_size,whole_size):
    #finance_data = pd.read_excel(filename, sheet_name = sheetname)
    finance_data = pd.read_csv(filename)
    finance_data = np.array(finance_data)
    close_max = np.max(finance_data[:,2])
    close_min = np.min(finance_data[:,2])
    finance_data = MaxMinNormalization(finance_data)
    data = wavelet(finance_data) #对数据进行小波去噪
    
    train_result = []
    for index in range(train_size - seq_len):
        train_result.append(data[start_index+index: start_index+index + seq_len,1:])
    x_train = np.array(train_result)
    y_train = finance_data[start_index+seq_len:start_index+train_size,2]
    
    validation_result = []
    for index1 in range(train_size - (seq_len+1),validation_size - seq_len):
        validation_result.append(data[start_index+index1: start_index+index1 + seq_len,1:])
    x_validation = np.array(validation_result)
    y_validation = finance_data[start_index+train_size-1:start_index+validation_size,2]
    
    test_result = []
    for index2 in range(validation_size - (seq_len+1),whole_size - seq_len):
        test_result.append(data[start_index+index2:start_index+index2 + seq_len,1:])
    x_test = np.array(test_result)
    y_test = finance_data[start_index+validation_size-1:start_index+whole_size,2]
    
    return [x_train,y_train,x_validation,y_validation,x_test,y_test,close_min,close_max]

#预测
def predict_data(x_test,y_test,Max,Min,BATCH_SIZE):
    pred = model.predict(x_test,batch_size = BATCH_SIZE)
    for i in range(len(pred)):
        pred[i] = (Max - Min)*pred[i] + Min
        #y_test[i] = (Max - Min)*y_test[i] + Min
    y_test = np.reshape(y_test,(len(y_test),1))
    d = np.hstack((y_test,pred))
    pred_excel = pd.DataFrame(d,columns=["actual","predict"])
    pred_excel.to_excel("wlstm1_000988.xlsx", sheet_name = "S&P500 Index Predict")
    return [pred,y_test] 

BATCH_START = 0
TIME_STEPS = 4
BATCH_SIZE = 60
INPUT_SIZE = 19
OUTPUT_SIZE = 1
CELL_SIZE = 8 #LSTM细胞输出维度
EPOCH = 200
LR = 0.05 #学习率
#TRAIN_SIZE = 484
#VALIDATION_SIZE = 543
#WHOLE_SIZE = 782
START_INDEX = 0
TRAIN_SIZE = 2644
VALIDATION_SIZE = 2703
WHOLE_SIZE = 2882

x_train, y_train,x_validation,y_validation, x_test, y_test, close_min,close_max = load_data("000988.csv","Nifty 50 index Data",4,
                                                                       START_INDEX,TRAIN_SIZE,VALIDATION_SIZE,WHOLE_SIZE)



#wlstm model
model = Sequential()
model.add(LSTM(
          batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True,
))
model.add(LSTM(
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True
))
model.add(LSTM(
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True
))
model.add(LSTM(
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True
))
model.add(LSTM(
          output_dim=CELL_SIZE,
          #stateful = True
))
model.add(Dense(OUTPUT_SIZE,activation = "sigmoid"))#模型输出

adam = Adam(LR)#使用Adam优化函数
model.compile(optimizer=adam,
loss='mse',)#loss函数采用均方误差

print('Training ------------')
history = model.fit(x_train, y_train,
          batch_size = BATCH_SIZE,
          epochs = EPOCH,
          validation_data = (x_validation,y_validation))

finance_data = pd.read_csv("000004.csv")
#finance_data = pd.read_excel("RawData.xlsx",sheet_name = "Nifty 50 index Data")
data = np.array(finance_data)
y_test = data[START_INDEX+VALIDATION_SIZE-1:START_INDEX+WHOLE_SIZE,2]

predict, y_true = predict_data(x_test,y_test,close_max,close_min,BATCH_SIZE)

print(Mape(predict, y_true))
plt.plot(history.history['loss'],label = "train")
plt.plot(history.history['val_loss'],label = "validation")
plt.legend()
plt.show()
















