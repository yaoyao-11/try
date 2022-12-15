# Score balance and score imbalance are the prediction results of DeepRBPPred
balance_score =pd.read_csv("./DeepRBPPred/score_balance.csv",index_col=0)
imbalance_score =pd.read_csv("./DeepRBPPred/score_imbalance.csv",index_col=0)

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

data_test=pd.read_csv("./bind_all_feature/all_test.csv",index_col=0)
data_test["label"]="NULL"
data_test.iloc[:1708,-1]=1
data_test.iloc[1708:,-1]=0
data_test.iloc[1700:1710,:]
test_indepent_x=data_test.drop(["label"],axis="columns")
test_indepent_y=data_test["label"]
test_indepent_y=test_indepent_y.astype(int)
balance_y_score=list(balance_score["score"])
acu_curve(test_indepent_y,balance_y_score)
imbalance_y_score=list(imbalance_score["score"])
acu_curve(test_indepent_y,imbalance_y_score)



