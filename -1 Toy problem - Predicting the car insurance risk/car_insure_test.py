import pandas as pd
from xgboost import XGBRegressor
import numpy as np

# read in data from training files
data = pd.read_csv('train.csv')
# data.dropna(axis=0, subset=['Score'], inplace=True)

y = data.Score

X = data.drop(['Score','Id'], axis=1)
# using one hot for types of Object
X = pd.get_dummies(X)

# align the one-hot testing data according to the training data
test_data = pd.read_csv('test.csv')
test_X = test_data.drop(['Id'], axis=1)

test_X = pd.get_dummies(test_X)
X,test_X = X.align(test_X,join='left',axis=1)
test_X = test_X.values

train_X,train_y =X.values, y.values

my_model = XGBRegressor(n_estimators=1000,learning_rate = 0.01, early_stopping_rounds =5,n_jobs=4)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)
pre = np.round(predictions)
subm = pd.DataFrame({'Id':test_data.Id,'Score':pre})
subm.to_csv('mysubmission.csv', index = False)
