
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_wlstm1 = np.array(pd.read_excel("wlstm1_000988.xlsx"))
data_wlstm2 = np.array(pd.read_excel("wlstm2_000988.xlsx"))
#data_rnn = np.array(pd.read_excel("rnn.xlsx"))
data_raf = np.array(pd.read_excel("raf_000988.xlsx"))


data_actual = data_wlstm1[:,0]
data_wlstm1 = data_wlstm1[:,1]
data_raf = data_raf[:,1]
data_wlstm2 = data_wlstm2[:,1]

x = np.arange(len(data_actual))


plt.plot(x,data_actual,'black',label = "Actual Data")
plt.plot(x,data_raf,'darkgoldenrod',label = "RAF")
#plt.plot(x,data_rnn,'g',label = "RNN")
#plt.plot(x,data_lstm,'b',label = "LSTM")
plt.plot(x,data_wlstm1,'r',label = "WLSTM1")
plt.plot(x,data_wlstm2,'g',label = "WLSTM2")
plt.xlabel("Trading Day")
plt.ylabel("Close Price")
plt.title("000988")
plt.legend(loc = "upper left")
plt.show()