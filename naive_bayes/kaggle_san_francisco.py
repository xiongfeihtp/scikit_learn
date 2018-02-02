import sklearn
import pandas as pd
import numpy as np

train=pd.read_csv('./kaggle_San/train.csv',parse_dates=['Dates'])
test=pd.read_csv('./kaggle_San/test.csv',parse_dates=['Dates'])

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

leCrime=preprocessing.LabelEncoder()
crime=leCrime.fit_transform(train.Category)


day=pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour=pd.get_dummies(hour)
trainDate=pd.concat([day,district,hour],axis=1)
trainDate['crime']=crime

day=pd.get_dummies(test.DayOfWeek)
district=pd.get_dummies(test.PdDistrict)
hour=test.Dates.dt.hour
hour=pd.get_dummies(hour)
testDate=pd.concat([day,district,hour],axis=1)



from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import time

features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

training, validation = train_test_split(trainDate, train_size=.60)
start_time=time.time()
model=BernoulliNB()
model.fit(training[features],training['crime'])
nb_cost=time.time()-start_time
predict=model.predict_proba(validation[features])
print("朴素贝叶斯建模耗时 %f 秒" %(nb_cost))
print("朴素贝叶斯log损失为 %f" %(log_loss(validation['crime'], predict)))



start_time=time.time()
model=LogisticRegression(C=.01)

model.fit(training[features],training['crime'])
lr_cost=time.time()-start_time
predict=model.predict_proba(validation[features])
print("逻辑回归建模耗时 %f 秒" %(lr_cost))
print("逻辑回归log损失为 %f" % (log_loss(validation['crime'], predict)))







