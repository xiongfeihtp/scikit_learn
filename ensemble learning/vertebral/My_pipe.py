from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier



train_file = '/Users/xiongfei/PycharmProjects/scikit_learn/ensemble learning/vertebral/data/column_3C.dat'
with open(train_file, 'r') as file:
    data = [i for i in csv.reader(file, delimiter=' ')]
    data = data[1:]

#preprocessing for train test
X=np.array([i[:-1] for i in data],dtype=float)
Y=np.array([i[-1] for i in data])

encoder=LabelEncoder()
label=encoder.fit_transform(Y)
X_train, X_test, y_train, y_test=train_test_split(X,label,test_size=.2,random_state=0)

from itertools import islice

range_for_n_estimators=range(10,100,10)

test_score=[]
train_score=[]

for n_estimators in range_for_n_estimators:
    est=GradientBoostingClassifier(n_estimators=n_estimators,max_depth=1)
    est.fit(X_train,y_train)
    train_score.append(est.score(X_train, y_train))
    test_score.append(est.score(X_test,y_test))


alpha=0.4

ax=plt.figure(figsize=(8,5))
ax=plt.gca()
ax.plot(range_for_n_estimators,test_score,label="test",color='r',linewidth=2,alpha=alpha)
ax.plot(range_for_n_estimators,train_score,color='b',label="train",linewidth=2,alpha=alpha)
ax.set_ylabel('accuracy')
ax.set_xlabel('n_estimators')


# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.model_selection import GridSearchCV
# #give the grid
# n_estimator_range = range(10,200,5)
# #pipe_lr grid  by give the pipe_lr.get_params().keys() for the params name, specially fro clf__n_estimators
# param_grid = dict(clf__n_estimators=n_estimator_range)
# pipe_lr = Pipeline([('sc', StandardScaler()),
#                     ('clf', ExtraTreesClassifier(n_estimators=10,random_state=1))
#                     ])
# grid = GridSearchCV(pipe_lr, param_grid, cv=10, scoring='accuracy')
# grid.fit(X_train, y_train)
