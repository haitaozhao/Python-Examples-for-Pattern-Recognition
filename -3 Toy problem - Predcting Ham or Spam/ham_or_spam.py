import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics

def xgb_model(train_data, train_label, test_data, test_label):
    ## this function is downloaded from https://www.programcreek.com/python/example/99824/xgboost.XGBClassifier
    clf = XGBClassifier(max_depth=8,
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
    clf.fit(train_data, train_label, eval_metric='auc', verbose=True,
            eval_set=[(test_data, test_label)], early_stopping_rounds=100)
    y_pre = clf.predict(test_data)
    y_pro = clf.predict_proba(test_data)[:, 1]
    print("AUC Score : %f" % metrics.roc_auc_score(test_label, y_pro))
    print("Accuracy : %.4g" % metrics.accuracy_score(test_label, y_pre))
    return clf


data = pd.read_excel('train.xlsx')

token_dict = {}

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
#        stems.append(SnowballStemmer('english').stem(item))
    return stems

y=[]
for f in range(len(data.Text)):
    text = str(data.Text[f])
    if data.Label[f] == 'ham':
        y.append(1)
    else:
        y.append(0)
    token_dict[f] = text.lower().translate(string.punctuation)

# extract tf-idf features
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
X = tfidf.fit_transform(token_dict.values())

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.25)

# construct gradient tree boosting
my_model = xgb_model(train_X, train_y, valid_X, valid_y)

test = pd.read_excel('c:/test1.xlsx')
response = (tfidf.transform(test.Text))
my_pre = my_model.predict(response)
test['Label'] = ''
test.drop(['Text'],axis=1,inplace=True)

for idx in range(len(test.Label)):
    if my_pre[idx] ==1:
        test.loc[idx,'Label']='ham'
    else:
        test.loc[idx,'Label']='spam'

test.to_csv('mysubmission.csv', index = False)


