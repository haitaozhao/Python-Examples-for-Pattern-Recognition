import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

train = pd.read_csv('train_titan.csv')
test = pd.read_csv('test_titan.csv')

# check the columns with nan elements
train.count()

# baseline recognition rate
train.Sex.value_counts()
a = train.groupby(['Survived','Sex']).Sex.count()
rate = (a.loc[1,'female']+a.loc[0,'male'])/train.Survived.count()
## 78.7%

# Using the average age for passengers with no age information
train.loc[train.Age.isnull(),'Age'] = train.Age.mean()
test.loc[test.Age.isnull(),'Age'] = train.Age.mean()

# Let  0 Fare be NaN
train.loc[train.Fare==0,['Fare']]=np.nan
test.loc[test.Fare==0,['Fare']]=np.nan

# Using the average fare of each Pclass for those NaN fares
for idx in range(train.Pclass.max()):
    train.loc[train.Pclass==idx&train.Fare.isnull(),'Fare'] = train.loc[train.Pclass==idx,'Fare'].mean()
    test.loc[test.Pclass==idx&test.Fare.isnull(),'Fare'] = train.loc[train.Pclass==idx,'Fare'].mean()

# Process Cabin problem
train['nCa'] = np.nan
test['nCa'] = np.nan

def myfun(x):
    if type(x) == str:
        return len(str.split(x))

train.nCa = train.Cabin.apply(myfun)
train.loc[train.Cabin.isnull(),'nCa'] = 0

test.nCa = test.Cabin.apply(myfun)
test.loc[test.Cabin.isnull(),'nCa'] = 0

# Embarked place with the most often place

train.loc[train.Embarked.isnull(),'Embarked']=train.Embarked.value_counts().argsort().index[0]
test.loc[test.Embarked.isnull(),'Embarked']=train.Embarked.value_counts().argsort().index[0]

train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
sub1 = test.PassengerId
test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)


y = train.Survived
# using one hot for types of Object
X= train.drop(['Survived'],axis=1)
X = pd.get_dummies(X)

# align the one-hot testing data according to the training data
test_X = test

test_X = pd.get_dummies(test_X)
X,test_X = X.align(test_X,join='left',axis=1)
test_X = test_X.values


train_X, valid_X, train_y, valid_y = train_test_split(X.values, y.values, test_size=0.3)
def xgb_model(train_data, train_label, test_data, test_label):
    ## this function is downloaded from https://www.programcreek.com/python/example/99824/xgboost.XGBClassifier
    clf = XGBClassifier(max_depth=5,
                        min_child_weight=1,
                        learning_rate=0.1,
                        n_estimators=500,
                        silent=True,
                        objective='binary:logistic',
                        gamma=0,
                        max_delta_step=0,
                        subsample=1,
                        colsample_bytree=1,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=0,
                        scale_pos_weight=1,
                        seed=1,
                        missing=None)
    clf.fit(train_data, train_label, eval_metric='auc', verbose=False,
            eval_set=[(test_data, test_label)], early_stopping_rounds=10)
    y_pre = clf.predict(test_data)
    y_pro = clf.predict_proba(test_data)[:, 1]
    print("AUC Score : %f" % metrics.roc_auc_score(test_label, y_pro))
    print("Accuracy : %.4g" % metrics.accuracy_score(test_label, y_pre))
    return clf


my_model = xgb_model(train_X, train_y, valid_X, valid_y )


# make predictions
predictions = my_model.predict(test_X)
pre = np.round(predictions).astype(int)
subm = pd.DataFrame({'PassengerId':sub1,'Survived':pre})
subm.to_csv('mytitan.csv', index = False)
