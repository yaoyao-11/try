Sys.setenv(JAVA_HOME="F:/JDK1.8")

library("rmcfs")
library("e1071")
library(readr)

train_data<-read.csv("./bind_all_feature/all_train.csv",header=T,row.names=1)
labels<-rep(c(1,-1),times=c(3379,1034))
train_data$class=labels

test_data<-read.csv("./bind_all_feature/all_test.csv",header=T,row.names=1)
valid_data=test_data
valid_data$class=-1
valid_data$class[0:1708]=1
# res <- mcfs(class ~ ., train_data, threadsNumber = 12, featureFreq = 100, cutoffPermutations = 20)
#import data1.csv located within my_data.zip
res <- read.csv("./DmLab/DmLab_get_rules_RI_result/protr_unirep_seqvec_esm_3379_1034_topRanking.csv",header=T,row.names=1)
# top_size <- 1:200
top_size <- 1:400 #you can change the number

acc_vect <- rep(NA, length(top_size))
wacc_vect <- rep(NA, length(top_size)) 
name=res$attribute
for (i in top_size) {
  need_name=name[0:i]
  train_data_filtered <- train_data[,c(need_name,"class")]
  valid_data_filtered <- valid_data[,c(need_name,"class")]
  train_data_filtered=as.data.frame(lapply(train_data_filtered,as.numeric))
  classifier.svm <- e1071::svm(class ~ ., data = train_data_filtered, cost = 1000, gamma = 0.0001)
  pred_200 <- as.character(predict(classifier.svm, valid_data_filtered), type = c("class", "raw"), threshold = 0.001)
  cmat <- table(valid_data_filtered$class, pred_200)
  acc_vect[i] <- sum(diag(cmat)) / sum(cmat)
  wacc_vect[i] <- mean(diag(cmat) / rowSums(cmat))
  cat(".")
  print(i)
}
valid_data_filtered$pre=as.numeric(pred_200)
print("done")
new_data=valid_data_filtered[401:402]
write.csv(new_data,"./DmLab/DmLab_select_feature/dmlab_400.csv")