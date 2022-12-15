import math
import numpy as np
import pandas as pd
import joblib
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#scikit-learn==1.0.2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from matplotlib import rcParams
from pathlib import Path

def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
 
    plt.figure()

    config = {
        "font.family":'serif',
        "font.size": 20,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)

    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',font=Path('TNR.ttf'),fontsize=25)
    plt.ylabel('True Positive Rate',font=Path('TNR.ttf'),fontsize=25)
    plt.title('Receiver operating characteristic example',font=Path('TNR.ttf'),fontsize=25)
    plt.legend(loc="lower right",fontsize=25)

    
    plt.show()
def featureSet(data):
    data_num=len(data)
    Xlist=[]
    for row in range(0,data_num):
        tmp_list=[]
        for ii in range(len(data.columns)):
            tmp_list.append(data.iloc[row][data.columns[ii]])
        Xlist.append(tmp_list)
    return Xlist
data_test=pd.read_csv("./bind_all_feature/all_test.csv",index_col=0)
data_test["label"]="NULL"
data_test.iloc[:1708,-1]=1
data_test.iloc[1708:,-1]=0
data_test.iloc[1700:1710,:]
test_indepent_x=data_test.drop(["label"],axis="columns")
test_indepent_y=data_test["label"]
test_indepent_y=test_indepent_y.astype(int)
data=pd.read_csv('./bind_all_feature/all_train.csv',index_col=0)
data=data.sample(frac=1,random_state=1)
train,test=train_test_split(data,test_size=0.2,random_state=1)
train_x=train.drop(["label"],axis="columns")
train_y=train["label"]
train_x=featureSet(train_x)
y_train=train.label.values
train_x=np.array(train_x)
test_x=data_test.drop(["label"],axis="columns")

model = CascadeForestClassifier(random_state=1)
model.fit(train_x, y_train)
y_test_pred = model.predict_proba(test_x)[:, 1]
acu_curve(test_indepent_y,y_test_pred)
