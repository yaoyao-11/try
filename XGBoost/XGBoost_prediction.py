import numpy as np
import pandas as pd
import math
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

data=pd.read_csv('./bind_all_feature/all_train.csv',index_col=0)
data=data.sample(frac=1,random_state=1)
train,test=train_test_split(data,test_size=0.2,random_state=1)
train_x=train.drop(["label"],axis="columns")
train_y=train["label"]
test_x=test.drop(["label"],axis="columns")
test_y=test["label"]
#1.Tuning:n_estimators
cv_params = {'n_estimators': [100,200,300,400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
means=optimized_GBM.cv_results_["mean_test_score"]
params=optimized_GBM.cv_results_['params']
for mean,param in zip(means,params):
    print('%f with %r'%(mean,param))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
cv_params = {'n_estimators': [225,250,275,300,325,350,375]}
other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
means=optimized_GBM.cv_results_["mean_test_score"]
params=optimized_GBM.cv_results_['params']
for mean,param in zip(means,params):
    print('%f with %r'%(mean,param))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#2.Tuning:min_child_weight and max_depth：
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 375, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
means=optimized_GBM.cv_results_["mean_test_score"]
params=optimized_GBM.cv_results_['params']
for mean,param in zip(means,params):
    print('%f with %r'%(mean,param))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#4.Tuning:gamma：
cv_params = {'gamma': [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 375, 'max_depth': 3, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#4.Tuning:subsample and colsample_bytree：
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
gamma=optimized_GBM.best_params_["gamma"]
other_params = {'learning_rate': 0.1, 'n_estimators': 375, 'max_depth': 3, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': gamma, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#5.Tuning:reg_alpha and reg_lambda：
cv_params = {'reg_alpha': [0.05, 0,0.1, 1, 2, 3], 'reg_lambda': [0.05,0, 0.1, 1, 2, 3]}
subsample=optimized_GBM.best_params_["subsample"]
colsample_bytree=optimized_GBM.best_params_["colsample_bytree"]
other_params = {'learning_rate': 0.1, 'n_estimators': 375, 'max_depth': 3, 'min_child_weight': 1, 'seed': 0,
                    'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#6.Tuning:learning_rate：
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
reg_alpha=optimized_GBM.best_params_["reg_alpha"]
reg_lambda=optimized_GBM.best_params_["reg_lambda"]
other_params = {'learning_rate': 0.1, 'n_estimators': 375, 'max_depth': 3, 'min_child_weight': 1, 'seed': 0,
                    'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma, 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,
                    'silent':True, 'objective':'multi:softmax','num_class':2}

model =  xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
means=optimized_GBM.cv_results_["mean_test_score"]
params=optimized_GBM.cv_results_['params']
for mean,param in zip(means,params):
    print('%f with %r'%(mean,param))

# sub_testing results
learning_rate=optimized_GBM.best_params_["learning_rate"]
print(learning_rate)
model = xgb.XGBClassifier(n_estimators=375, max_depth=3, min_child_weight= 1, seed= 0,learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma, reg_alpha=reg_alpha,reg_lambda=reg_lambda, silent=True, objective='multi:softmax',num_class=2)
model.fit(train_x, train_y)
y_test_pred = model.predict(test_x)
y_test_pred=list(y_test_pred)
from sklearn.metrics import accuracy_score
accuracy_score(test_y,y_test_pred)   
newdata={"name":list(test_x.index),
         "true":list(test_y),
         "pre":list(y_test_pred)}
newdata=pd.DataFrame(newdata)
for i in newdata.index:
    if newdata.loc[i,"pre"]==newdata.loc[i,"true"]:#如果预测结果和序列本身的属性一样就记录为T
        newdata.loc[i,"prediction"]="T"
    else:
        newdata.loc[i,"prediction"]="F"


for i in newdata.index:
    if newdata.loc[i,"pre"]==newdata.loc[i,"true"]:#如果预测结果和序列本身的属性一样就记录为T
        newdata.loc[i,"prediction"]="T"
    else:
        newdata.loc[i,"prediction"]="F"
for i in newdata.index:
    if newdata.loc[i,"pre"]==newdata.loc[i,"true"]:#如果预测结果和序列本身的属性一样就记录为T
        newdata.loc[i,"prediction"]="T"
    else:
        newdata.loc[i,"prediction"]="F"
for i in newdata.index:
    if newdata.loc[i,"true"]==1:
        newdata.loc[i,"RBP"]="P"
    else:
        newdata.loc[i,"RBP"]="N"
newdata["ACC_MCC"]="NULL"
for i in newdata.index :
    if newdata.iloc[i,-2]=="P":
        if newdata.iloc[i,-3]=="T":
            newdata.iloc[i,-1]="TP";
        else:
            newdata.iloc[i,-1]="FN";
    if newdata.iloc[i,-2]=="N":
        if newdata.iloc[i,-3]=="T":
            newdata.iloc[i,-1]="TN";
        else:
            newdata.iloc[i,-1]="FP";
prediction= newdata["prediction"].values.tolist()
T_F=dict(zip(*np.unique(prediction, return_counts=True)))
count=T_F["T"]/len(prediction)
ACC_MCC_list= newdata["ACC_MCC"].values.tolist()
ACC_MCC=dict(zip(*np.unique(ACC_MCC_list, return_counts=True)))
TP=ACC_MCC["TP"]
TN=ACC_MCC["TN"]
FP=ACC_MCC["FP"]
FN=ACC_MCC["FN"]
ACC=(TP+TN)/(TP+TN+FN+FP)
MCC=(TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))
SN=TP/(TP+FN)
SP=TN/(TN+FP)
print(ACC,MCC,SN,SP)

# independent testing set
def Deep_forest_prediction(test_file):
    data_test=pd.read_csv(test_file)
    data_test["label"]="NULL"
    data_test.iloc[:1708,-1]=1
    data_test.iloc[1708:,-1]=0
    test_indepent_x=data_test.drop(["label"],axis="columns")
    test_indepent_y=data_test["label"]
    test_indepent_y=test_indepent_y.astype(int)
    result = model.predict(test_indepent_x)
    result=list(result)
    for i in range(len(result)):
        if result[i]>0.5:
            result[i]=1
        if result[i]<=0.5:
            result[i]=0
    test_indepent_y=list(test_indepent_y)
    accuracy_score(test_indepent_y,result)  
    newdata_test={"name":list(test_indepent_x.index),
            "true":list(test_indepent_y),
            "pre":list(result)}
    newdata_test=pd.DataFrame(newdata_test)
    for i in newdata_test.index:
        if newdata_test.loc[i,"pre"]==newdata_test.loc[i,"true"]:#如果预测结果和序列本身的属性一样就记录为T
            newdata_test.loc[i,"prediction"]="T"
        else:
            newdata_test.loc[i,"prediction"]="F"
    for i in newdata_test.index:
        if newdata_test.loc[i,"true"]==1:
            newdata_test.loc[i,"RBP"]="P"
        else:
            newdata_test.loc[i,"RBP"]="N"
    newdata_test["ACC_MCC"]="NULL"
    for i in newdata_test.index :
        if newdata_test.iloc[i,-2]=="P":
            if newdata_test.iloc[i,-3]=="T":
                newdata_test.iloc[i,-1]="TP"
            else:
                newdata_test.iloc[i,-1]="FN"
        if newdata_test.iloc[i,-2]=="N":
            if newdata_test.iloc[i,-3]=="T":
                newdata_test.iloc[i,-1]="TN"
            else:
                newdata_test.iloc[i,-1]="FP"
    prediction= newdata_test["prediction"].values.tolist()
    T_F=dict(zip(*np.unique(prediction, return_counts=True)))
    count=T_F["T"]/len(prediction)
    ACC_MCC_list= newdata_test["ACC_MCC"].values.tolist()
    ACC_MCC=dict(zip(*np.unique(ACC_MCC_list, return_counts=True)))
    TP=ACC_MCC["TP"]
    TN=ACC_MCC["TN"]
    FP=ACC_MCC["FP"]
    FN=ACC_MCC["FN"]
    ACC=(TP+TN)/(TP+TN+FN+FP)
    MCC=(TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))
    SN=TP/(TP+FN)
    SP=TN/(TN+FP)
    return ACC,MCC,SN,SP
test_file="./bind_all_feature/all_test.csv"
ACC,MCC,SN,SP=SVM_prediction(test_file)
