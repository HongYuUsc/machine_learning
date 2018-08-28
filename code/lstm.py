import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def MaxMinNormalization(x):
    for i in range(x.shape[1]-2):
       Max = np.max(x[:,i+2])
       Min = np.min(x[:,i+2])
       for j in range(x.shape[0]):
           x[j,i+2] = (x[j,i+2] - Min) / (Max - Min)
    return x


def Mape(pred,y):
    Sum = 0
    for i in range(len(pred)):
        Sum = Sum + abs((y[i] - pred[i])/y[i])
    mape = Sum/len(pred)
    return mape


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


def predict_data(x_test,y_test,Max,Min,BATCH_SIZE):
    pred = model.predict(x_test,batch_size = BATCH_SIZE)
    for i in range(len(pred)):
        pred[i] = (Max - Min)*pred[i] + Min
        y_test[i] = (Max - Min)*y_test[i] + Min
    y_test = np.reshape(y_test,(len(y_test),1))
    d = np.hstack((y_test,pred))
    pred_excel = pd.DataFrame(d,columns=["actual","predict"])
    pred_excel.to_excel("lstm.xlsx", sheet_name = "S&P500 Predict")
    return [pred,y_test] 

BATCH_START = 0
TIME_STEPS = 4
BATCH_SIZE = 60
INPUT_SIZE = 19
OUTPUT_SIZE = 1
CELL_SIZE = 8
EPOCH = 200
LR = 0.05
TRAIN_SIZE = 484
START_INDEX = 0
VALIDATION_SIZE = 543
WHOLE_SIZE = 782


x_train, y_train,x_validation,y_validation, x_test, y_test, close_min,close_max = load_data("RawData.xlsx","Nifty 50 index Data",TIME_STEPS,
                                                                       START_INDEX,TRAIN_SIZE,VALIDATION_SIZE,WHOLE_SIZE)


#lstm model
model = Sequential()
model.add(LSTM(
          batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
          units=CELL_SIZE,
          return_sequences=True,
          #stateful = True,
))
#model.add(LSTM(
 #         output_dim=CELL_SIZE,
          #return_sequences = True,
          #stateful = True
#))
#model.add(LSTM(
 #         output_dim=CELL_SIZE,
          #return_sequences = True,
          #stateful = True
#))
#model.add(LSTM(
 #         output_dim=CELL_SIZE,
          #return_sequences = True,
          #stateful = True
#))
#model.add(LSTM(
 #         output_dim=CELL_SIZE,
          #stateful = True
#))
model.add(Dense(OUTPUT_SIZE,activation = "sigmoid"))

adam = Adam(LR)
model.compile(optimizer=adam,
loss='mse',)

print('Training ------------')
history = model.fit(x_train, y_train,
          batch_size = BATCH_SIZE,
          epochs = EPOCH,
          validation_data = (x_validation,y_validation))


predict, y_true = predict_data(x_test,y_test,close_max,close_min,BATCH_SIZE)


print(Mape(predict, y_true))
plt.plot(history.history['loss'],label = "train")
plt.plot(history.history['val_loss'],label = "validation")
plt.legend()
plt.show()
















