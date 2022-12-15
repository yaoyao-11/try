import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import math

data=pd.read_csv('./bind_all_feature/all_train.csv',index_col=0)
data=data.sample(frac=1,random_state=1)
train,test=train_test_split(data,test_size=0.2,random_state=1)

train_x=train.drop(["label"],axis="columns")
train_y=train["label"]
test_x=test.drop(["label"],axis="columns")
test_y=test["label"]

predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
predictor.fit(train_x, train_y)
result = predictor.predict(test_x)
accuracy_score(test_y,result)   

newdata={"name":list(test_x.index),
         "true":list(test_y),
         "pre":list(result)}
newdata=pd.DataFrame(newdata)
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
            newdata.iloc[i,-1]="TP"
        else:
            newdata.iloc[i,-1]="FN"
    if newdata.iloc[i,-2]=="N":
        if newdata.iloc[i,-3]=="T":
            newdata.iloc[i,-1]="TN"
        else:
            newdata.iloc[i,-1]="FP"

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

# independent testing set
def SVM_prediction(test_file):
    data_test=pd.read_csv(test_file)
    data_test["label"]="NULL"
    data_test.iloc[:1708,-1]=1
    data_test.iloc[1708:,-1]=0
    test_indepent_x=data_test.drop(["label"],axis="columns")
    test_indepent_y=data_test["label"]
    test_indepent_y=test_indepent_y.astype(int)
    result = predictor.predict(test_indepent_x)
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