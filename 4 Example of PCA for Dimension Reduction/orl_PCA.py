import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

'''
This program is designed to show that dimension reduction (or feature extraction) 
by PCA can improve the performance of linear regression.
PCA is used to reduce the original dimension (10304) of the training vector to the reduced dimension 9.
It shows that the RMSEs (Root Mean Square Error) of the linear regression are reduced after dimension reduction.
It means we can obtain better performances with the extracted features (the dimension is 9) than with the 
original data (the dimension is 10304). 
'''

MY_DIR = r"c:\facedb\orl"
# function for read in the first 20 images of ORL Dataset
# orl1.bmp to orl10.bmp belong to one person, while orl11.bmp to orl20.bmp belong to another person.

def get_training_data(myDir):
    Training_data = []
    i = 0
    for img in os.listdir(myDir):
        try:
            img_array = cv2.imread(os.path.join(myDir,img) ,cv2.IMREAD_GRAYSCALE)
            label = 0 if i<10 else 1
            Training_data.append([img_array,label])
            i = i+1
        except Exception as e:
            print('error')
            break
        if i >= 20:
            break
    return Training_data

Training_data = get_training_data(MY_DIR)

## read in the images X and the labels y
X = []
y = []
for features,label in Training_data:
    X.append(features)
    y.append(label)

X = np.array(X)/255
y = np.array(y).reshape([len(y),1])
# reshape the images into vectors
X = X.reshape([20,112*92,])

# split 10 training vectors into training set and other 10 for validation
train_X, validation_X, train_y, validation_y = train_test_split(X, y, test_size=0.5)

# Linear Regression
train_X_one = np.c_[train_X,np.ones(len(train_X))]
validation_X_one = np.c_[validation_X,np.ones(len(validation_X))]
forecast_set = validation_X_one.dot(np.linalg.pinv(train_X_one).dot(train_y))
# show the results
forecast = []
for elm in forecast_set:
    if elm > 0.5:
        forecast.append(1)
    else:
        forecast.append(0)
print(100*'-')
print('The regression values of the validation set: {}'.format(forecast_set.T))
print(100*'-')
print('The Labels: {} and The Classification Results: {}'.format(validation_y.T,forecast))
print(100*'-')
print('The absolute errors of the regression: {}'.format(np.abs(forecast_set-validation_y).T))

# Perform PCA on train_X
# 1. Compute the Covariance matrix
Cov = (train_X - train_X.mean(axis=0)).dot((train_X - train_X.mean(axis=0)).T)
# 2. eigen deocompositon of Cov, s contains the eigenvalues and the columns of u are the corresponding eigenvectors
#    dim -1 as the rank of the Cov matrix in this example
s,u = np.linalg.eig(Cov)
u_r = u[:,s.argsort()[1:]]
s_r = s[s.argsort()[1:]]
s_sqrt = np.sqrt(s_r)
# 3. get the transformation matrix for dimension reduction
v_r = np.diag(1/s_sqrt).dot(u_r.T).dot(train_X - train_X.mean(axis=0))

# Perform dimension reduction on both training set and validation set
train_pca = train_X.dot(v_r.T)
validation_pca = validation_X.dot(v_r.T)
train_pca_one = np.c_[train_pca,np.ones(len(train_pca))]
validation_pca_one = np.c_[validation_pca,np.ones(len(validation_pca))]
forecast_set1 = validation_pca_one.dot(np.linalg.pinv(train_pca_one).dot(train_y))

forecast1 = []
for elm in forecast_set1:
    if elm > 0.5:
        forecast1.append(1)
    else:
        forecast1.append(0)
print(100*'=')
print('The regression values of the validation set: {}'.format(forecast_set1.T))
print(100*'-')
print('The Labels: {} and The Classification Results: {}'.format(validation_y.T,forecast1))
print(100*'-')
print('The absolute errors of the regression: {}'.format(np.abs(forecast_set1-validation_y).T))
print(100*'=')
rmse = 1/len(train_y)*np.sqrt((forecast_set-validation_y).T.dot(forecast_set-validation_y))
rmse_pca = 1/len(train_y)*np.sqrt((forecast_set1-validation_y).T.dot(forecast_set1-validation_y))
print('Root mean square error (RMSE) before PCA is {} and after PCA is {}'.format(rmse,rmse_pca))