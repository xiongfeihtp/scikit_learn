from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
import pandas as pd
import numpy as np


from xgboost.sklearn import XGBClassifier

X,y=make_hastie_10_2(random_state=0)
X=DataFrame(X)
y=DataFrame(y)
y.columns=["label"]
label={-1:0,1:1}
y.label=y.label.map(label)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
y_train.head()

clf=XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bylevel=0.8,
    objective='binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27
)
model_sklearn=clf.fit(X_train,y_train)
y_sklearn=clf.predict_proba(X_test)[:,1]

print("XGBOOST_sklearn接口 AUC score: %f"%(metrics.roc_auc_score(y_test,y_sklearn)))

print("原始train大小",X_train.shape)
print("原始test大小",X_test.shape)

train_new_feature=clf.apply(X_train)
test_new_feature=clf.apply(X_test)


new_feature=clf.fit(train_new_feature,y_train)
y_new_feature=clf.predict_proba(test_new_feature)

y_new_feature=y_new_feature.argmax(axis=1)

metrics.accuracy_score(y_test,y_new_feature)







