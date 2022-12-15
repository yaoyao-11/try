#read files
library(dplyr)

prot=read.csv("./Protr_file/train_feature/ALL_protr_train.CSV",check.names = F)
prot=t(prot)
prot=data.frame(prot,stringsAsFactors = F,check.names = F)
colnames(prot)=prot[1,]
prot=prot[-1,]
prot$name=rownames(prot)

UniRep_RBP=read.csv("./UniRep_file/train_feature/bind_1900_RBPs_train.csv")
UniRep_nonRBP=read.csv("./UniRep_file/train_feature/bind_1900_RBPs_train.csv")
rownames(UniRep_RBP)=UniRep_RBP$X
UniRep_RBP=UniRep_RBP[,-1]
UniRep_RBP$name=rownames(UniRep_RBP)
rownames(UniRep_nonRBP)=UniRep_nonRBP$X
UniRep_nonRBP=UniRep_nonRBP[,-1]
UniRep_nonRBP$name=rownames(UniRep_nonRBP)

seqvec=read.csv("./SeqVec_file/SeqVec_train_features.CSV",check.names = F)
seqvec=t(seqvec)
seqvec=data.frame(seqvec,stringsAsFactors = F,check.names = F)
newcolname=paste0("seqvec_features_",1:1024)
colnames(seqvec)=newcolname
seqvec=seqvec[-1,]
seqvec$name=rownames(seqvec)

ESM_1b=read.csv("./ESM-1b/Esm1b_train_features.CSV",check.names = F)
ESM_1b=t(ESM_1b)
ESM_1b=data.frame(ESM_1b,stringsAsFactors = F,check.names = F)
newcolname=paste0("ESM_1b_features_",1:33)
colnames(ESM_1b)=newcolname
ESM_1b=ESM_1b[-1,]
ESM_1b$name=rownames(ESM_1b)

RBP_prot_UniRep=merge(prot,UniRep_RBP,by.x = "name",by.y  = "name",sort = F)
nonRBP_prot_UniRep=merge(prot,UniRep_nonRBP,by.x = "name",by.y  = "name",sort = F)
RBP_prot_UniRep_seqvec=merge(RBP_prot_UniRep,seqvec,by.x = "name",by.y  = "name",sort = F)
nonRBP_prot_UniRep_seqvec=merge(nonRBP_prot_UniRep,seqvec,by.x = "name",by.y  = "name",sort = F)
RBP_all=merge(RBP_prot_UniRep_seqvec,ESM_1b,by.x = "name",by.y  = "name",sort = F)
nonRBP_all=merge(nonRBP_prot_UniRep_seqvec,ESM_1b,by.x = "name",by.y  = "name",sort = F)
rownames(RBP_all)=RBP_all$name
rownames(nonRBP_all)=nonRBP_all$name
RBP_all=RBP_all[,-1]
nonRBP_all=nonRBP_all[,-1]

all_train=rbind(RBP_all,nonRBP_all)
all_train=data.frame(all_train,stringsAsFactors = F,check.names = F)
write.csv(all_train,"./bind_all_feature/all_train.csv",quote = F)
