import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# read in data from training files
data = pd.read_csv('train.csv')
# data.dropna(axis=0, subset=['Score'], inplace=True)
y = data.Score

X = data.drop(['Score','Id'], axis=1)
# using one hot for types of Object
X = pd.get_dummies(X)

# align the one-hot testing data according to the training data
# test_data = pd.read_csv('test.csv')
# test_X = test_data.drop(['Id'], axis=1)
#
# test_X = pd.get_dummies(test_X)
# test_X = X.align(test_X,join='left',axis=1)

# split the data into training and validation set
train_X, validation_X, train_y, validation_y = train_test_split(X.values, y.values, test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
validation_X = my_imputer.transform(validation_X)



my_model = XGBRegressor(n_estimators=1000,learning_rate = 0.01, early_stopping_rounds =5,n_jobs=4)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(validation_X)
rms = np.sqrt(mean_squared_error(validation_y, predictions))

print(rms)