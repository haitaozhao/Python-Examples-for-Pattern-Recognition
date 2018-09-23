import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# read in data from training files
data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['Score'], inplace=True)
y = data.Score
X = data.drop(['Score','Id'], axis=1).select_dtypes(exclude=['object'])

# split the data into training and validation set
train_X, valid_X, train_y, valid_y = train_test_split(X.values, y.values, test_size=0.25)

# construct gradient tree boosting
my_model = XGBRegressor(n_estimators=1000,learning_rate = 0.05)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(valid_X)
rms = np.sqrt(mean_squared_error(valid_y, predictions))
print(rms)

# make predictions only use the mean of train_y
pred = np.array([4]*8000)
rms1 = np.sqrt(mean_squared_error(valid_y, pred))

# show the results
print('base line prediction rate is {0:.2f}'.format(rms1))
print('Xgboost prediction rate is {0:.2f}'.format(rms))