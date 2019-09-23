import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
from sklearn import preprocessing
import numpy as np

# style.use('ggplot')
# start = dt.datetime(2018, 1, 1)
# end = dt.datetime.now()
# df = web.DataReader('AAPL',"robinhood")
# df.reset_index(inplace=True)
# df.set_index("begins_at", inplace=True)
# df = df.drop("symbol", axis=1)
# df.to_csv('tsla.csv')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df['HL_PCT'] = (df['high_price'] - df['low_price']) / df['close_price'] * 100.0
df['PCT_change'] = (df['close_price'] - df['open_price']) / df['open_price'] * 100.0
df = df[['close_price', 'HL_PCT', 'PCT_change', 'volume']]

## 预测 forcast_out天后的收盘价
forecast_col = 'close_price'
## 去掉nan
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

# 用笔记里的公式计算
a = np.ones(len(X_train))
XX_train = np.c_[X_train,a]
y_mytest = np.c_[X_test,np.ones(len(X_test))].dot(np.linalg.pinv(XX_train).dot(y_train))

## 取出没有target值的后面forecast个数据进行预测，打印结果
X_lately = X[-forecast_out:]
forecast_set = clf.predict(X_lately)
print(forecast_set, confidence)

#用笔记里的公式计算
forecast_set = np.c_[X_lately,np.ones(len(X_lately))].dot(np.linalg.pinv(XX_train).dot(y_train))

style.use('ggplot')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['close_price'].plot()
df['Forecast'].plot()
plt.legend(loc=8)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()