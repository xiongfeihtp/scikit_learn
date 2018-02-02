
#give the iteration process and feature_importance
import numpy as np
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

boston=datasets.load_boston()
X,y=shuffle(boston.data,boston.target,random_state=13)
X=X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

n_estimators=500


clf=ensemble.GradientBoostingRegressor(
    n_estimators=n_estimators,
    max_depth=4,
    min_samples_split=2,
    learning_rate=0.01,
    loss='ls'
)
clf.fit(X_train,y_train)
test_predict=clf.predict(X_test)
mse=mean_squared_error(y_test,test_predict)
r2=r2_score(y_test,test_predict) #拟合优度
print("MSE: %.4f and R2: %.4f"%(mse,r2))


import matplotlib.pyplot as plt
test_score=np.zeros((n_estimators),dtype=np.float64)
#迭代
for i,y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i]=clf.loss_(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Deviance')
plt.plot(np.arange(n_estimators)+1,clf.train_score_,'b-',label='Training Set Deviance')
plt.plot(np.arange(n_estimators)+1,test_score,'r',label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting iterations')
plt.ylabel('Deviance')


#描述特征重要性
feature_importance=clf.feature_importances_
feature_importance=100.0*(feature_importance/feature_importance.max())
sorted_idx=np.argsort(feature_importance)
pos=np.arange(sorted_idx.shape[0])
plt.subplot(1,2,2)
plt.barh(pos,feature_importance[sorted_idx],align='center')
plt.yticks(pos,boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.xlabel('Variable Importance')
plt.show()


#xgboost
from sklearn.model_selection import train_test_split
from sklearn import metrics
from  sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
X, y = make_hastie_10_2(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)##test_size测试集合所占比例
clf = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.3, # 如同学习率
min_child_weight=1,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=6, # 构建树的深度，越大越容易过拟合
gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=1, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样
reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#reg_alpha=0, # L1 正则项参数
#scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
#objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
#num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=100, #树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
)
clf.fit(X_train,y_train,eval_metric='auc')
y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))




from sklearn.model_selection import GridSearchCV
tuned_parameters= [{'n_estimators':[100,200,500,1000],
                  'max_depth':[3,5,7], ##range(3,10,2)
                  'learning_rate':[0.5, 1.0],
                  'subsample':[0.75,0.8,0.85,0.9]
                  }]
clf = GridSearchCV(XGBClassifier(silent=0,nthread=4,learning_rate= 0.5,min_child_weight=1, max_depth=3,gamma=0,subsample=1,colsample_bytree=1,reg_lambda=1,seed=1000), param_grid=tuned_parameters,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
clf.fit(X_train, y_train)
##clf.grid_scores_, clf.best_params_, clf.best_score_
print(clf.best_params_)
y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred) )
y_proba=clf.predict_proba(X_test)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_true, y_proba))



