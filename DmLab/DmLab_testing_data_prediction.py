import pandas as pd
import numpy as np
import math

def DmLab_prediction(test_file,rules_file):
    data=pd.read_csv(test_file)
    p=1708
    n=517
    cls=["Positive"]*p+["Negative"]*n
    data=data.set_index("Unnamed: 0")
    newdata=data
    newdata["pre"]=cls
    file=rules_file
    for name in newdata.index:
        a=0
        with open(file,"r") as f:
            for i in f:
                i=i.replace("\n","")
                a=a+1
                data_name="pre_{}".format(a)
                feature_value=[]
                judge=[]
                judge_value=[]
                for features in i.split("=>")[0].split("and"):
                    feature=features.lstrip().replace("(","").replace(")","").split(" ")[0]
                    symbol=features.lstrip().replace("(","").replace(")","").split(" ")[1]
                    need_value=features.lstrip().replace("(","").replace(")","").split(" ")[2]
                    need_value=float(need_value)
                    feature_value.append(newdata.loc[name,feature])
                    judge.append(symbol)
                    judge_value.append(need_value)
                num=len(feature_value)
                bool_list = []
                and_num=0
                while and_num < num:
                    bool_v = eval(f'({feature_value[and_num]} {judge[and_num]} {judge_value[and_num]} )')
                    bool_list.append(bool_v)
                    and_num += 1
                if np.array(bool_list).all():
                    newdata.loc[name,data_name]='Negative'
                else:
                    newdata.loc[name,data_name]='Positive'
    for i in newdata.index:
        pre_last=[]
        for num in range(1,a+1):
            if newdata.loc[i,"pre_{}".format(num)]=="Negative":#如果符合其中的一条规则,就记录为True
                pre_last.append("True")
        if len(pre_last)>0:#只要有一条符合规则就是规则里面的预测结果
            newdata.loc[i,"last"]="Negative"
        else:
            newdata.loc[i,"last"]="Positive"
    for i in newdata.index:
        if newdata.loc[i,"pre"]==newdata.loc[i,"last"]:#如果预测结果和序列本身的属性一样就记录为T
            newdata.loc[i,"prediction"]="T"
        else:
            newdata.loc[i,"prediction"]="F"

    ACC_MCC=["NULL"]*(n+p)
    newdata["ACC_MCC"]=ACC_MCC

    for i in range(p):
        if newdata.iloc[i,-2]=="T":
            newdata.iloc[i,-1]="TP";
        else:
            newdata.iloc[i,-1]="FN";

    for i in range(p,p+n):
        if newdata.iloc[i,-2]=="T":
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

    return ACC,MCC,SN,SP

test_file="./bind_all_feature/all_test.csv"
rules_file="./DmLab/DmLab_get_rules_RI_result/test.txt"

ACC,MCC,SN,SP=DmLab_prediction(test_file,rules_file)