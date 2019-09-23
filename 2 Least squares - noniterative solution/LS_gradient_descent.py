## -------- 2018-9-22 ------------
## from Haitao Zhao
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
import datetime
from sklearn import preprocessing
import numpy as np

style.use('ggplot')

## read in the stock data from "robinhood"
# df = web.DataReader('AAPL',"robinhood")
# df.reset_index(inplace=True)
# df.set_index("begins_at", inplace=True)
# df = df.drop("symbol", axis=1)
## save in the local directory
# df.to_csv('aapl.csv')

df = pd.read_csv('aapl.csv', parse_dates=True, index_col=0)

# feature design
df['HL_PCT'] = (df['high_price'] - df['low_price']) / df['low_price'] * 100.0
df['PCT_change'] = (df['close_price'] - df['open_price']) / df['open_price'] * 100.0
df = df[['close_price', 'HL_PCT', 'PCT_change', 'volume']]

## 预测 forcast_out天后的收盘价
forecast_col = 'close_price'
## 如果有缺失 ，去掉nan
df.dropna(axis=0,inplace = True)
forecast_out = 30
## 收盘价作为预测值，往前shift forecast天，作为那一天的目标值进行预测
df['target'] = df[forecast_col].shift(-forecast_out)
print(df.head())
## 看出往前shift了forecast，后面forecast个数据的'target'变成NAN
print(df.tail(forecast_out+1))

## 训练数据去掉后forecast个，因为没有target，并对尺度预处理，类似matlab的zscore
X = np.array(df.drop(['target'], 1))
X = preprocessing.scale(X)

X = X[:-forecast_out]
df.dropna(axis=0,inplace=True)
y = np.array(df['target'])
## 可以将数据集随机划分成2部分
# 一部分做测试，一部分做训练，这里的0.2，意思是20%的数据用来做测试，80%训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

row,_ = X_train.shape
XX_train =np.hstack([np.ones([row,1]),X_train])

## -------------------梯度下降法----------------------------
# 初始化参数 Beta 和 学习率 alpha
row,col = XX_train.shape
Beta = np.random.random([col,1])
alpha = 0.01

# 梯度的负方向
delta_Beta = 1/row * ( XX_train.T.dot(y_train.reshape(row,1)) \
                       - XX_train.T.dot(XX_train).dot(Beta))

# 更新Beta
new_Beta = Beta + alpha * delta_Beta

loss = []
for idx in range(10000):
    tmp_loss = 1/row * np.linalg.norm(y_train.reshape(row,1)-XX_train.dot(Beta))**2
    loss.append(tmp_loss)
    if tmp_loss < 0.01:
        print(idx)
        break
    else:
        Beta = new_Beta
        delta_Beta = 1/row *( XX_train.T.dot(y_train.reshape(row,1)) \
                              - XX_train.T.dot(XX_train).dot(Beta))
        new_Beta = Beta + alpha * delta_Beta

## -------------------End----------------------------

# 对后30天进行预测
X_lately = X[-forecast_out:]

forecast_set = np.c_[np.ones(len(X_lately)),X_lately].dot(Beta)


## 画图不是重点，可暂时不考虑。为了画的好看，用了unix系统的时间戳来操作
style.use('ggplot')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i[0]]

df['close_price'].plot()
df['Forecast'].plot()
plt.legend(loc=9)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
