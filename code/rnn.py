# -*- coding: UTF-8 -*-
#该文件的代码为RNN模型
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#数据归一化
def MaxMinNormalization(x):
    for i in range(x.shape[1]-2):
       Max = np.max(x[:,i+2])
       Min = np.min(x[:,i+2])
       for j in range(x.shape[0]):
           x[j,i+2] = (x[j,i+2] - Min) / (Max - Min)
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
    finance_data = pd.read_excel(filename, sheet_name = sheetname)
    data = np.array(finance_data)
    close_max = np.max(data[:,2])
    close_min = np.min(data[:,2])
    data = MaxMinNormalization(data)
    
    train_result = []
    for index in range(train_size - seq_len):
        train_result.append(data[start_index+index: start_index+index + seq_len,2:])
    x_train = np.array(train_result)
    y_train = data[start_index+seq_len:start_index+train_size,2]
    
    validation_result = []
    for index1 in range(train_size - (seq_len+1),validation_size - seq_len):
        validation_result.append(data[start_index+index1: start_index+index1 + seq_len,2:])
    x_validation = np.array(validation_result)
    y_validation = data[start_index+train_size-1:start_index+validation_size,2]
    
    test_result = []
    for index2 in range(validation_size - (seq_len+1),whole_size - seq_len):
        test_result.append(data[start_index+index2:start_index+index2 + seq_len,2:])
    x_test = np.array(test_result)
    y_test = data[start_index+validation_size-1:start_index+whole_size,2]
    
    
    return [x_train,y_train,x_validation,y_validation,x_test,y_test,close_min,close_max]
	
#预测
def predict_data(x_test,y_test,Max,Min,BATCH_SIZE):
    pred = model.predict(x_test,batch_size = BATCH_SIZE)
    for i in range(len(pred)):
        pred[i] = (Max - Min)*pred[i] + Min
        y_test[i] = (Max - Min)*y_test[i] + Min
    y_test = np.reshape(y_test,(len(y_test),1))
    d = np.hstack((y_test,pred))
    pred_excel = pd.DataFrame(d,columns=["actual","predict"])
    pred_excel.to_excel("rnn.xlsx", sheet_name = "S&P500 Predict")
    return [pred,y_test] 

BATCH_START = 0
TIME_STEPS = 4
BATCH_SIZE = 60
INPUT_SIZE = 19
OUTPUT_SIZE = 1
CELL_SIZE = 8 #RNN细胞输出维度
EPOCH = 300
LR = 0.05 #学习率
TRAIN_SIZE = 484
START_INDEX = 746
VALIDATION_SIZE = 543
WHOLE_SIZE = 782

x_train, y_train,x_validation,y_validation, x_test, y_test, close_min,close_max = load_data("RawData.xlsx","Nifty 50 index Data",4,
                                                                       START_INDEX,TRAIN_SIZE,VALIDATION_SIZE,WHOLE_SIZE)



#RNN model
model = Sequential()
model.add(SimpleRNN(
          batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True,
))
model.add(SimpleRNN(
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True
))
model.add(SimpleRNN(
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True
))
model.add(SimpleRNN(
          output_dim=CELL_SIZE,
          return_sequences = True,
          #stateful = True
))
model.add(SimpleRNN(
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
#捕获训练误差历史数据

predict, y_true = predict_data(x_test,y_test,close_max,close_min,BATCH_SIZE)

print(Mape(predict, y_true))
plt.plot(history.history['loss'],label = "train")
plt.plot(history.history['val_loss'],label = "validation")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("RNN")
plt.legend(loc = "upper right")
plt.show()

















