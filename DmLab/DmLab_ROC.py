import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

data_test=pd.read_csv("./DmLab/DmLab_select_feature/dmlab_200.csv",index_col=0)
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
y_test_pred=list(data_test["pre"])
test_indepent_y=list(data_test["class"])
acu_curve(test_indepent_y,y_test_pred)
