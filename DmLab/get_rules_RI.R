# output<- "./DmLab/protr_unirep_seqvec_esm_3379_1034.adx"
expr=read.csv("./bind_all_feature/all_train.csv",header = T,row.names = 1)

cls=c(rep("Positive",3379),rep("Negative",1034))
cat(c("attributes
{
class nominal decision(all)",paste(rownames(all_train)," numeric",sep=''),'}','events','{'),file=output,sep='\n')

write.table(cbind(cls,all_train),file=output,sep=',',row.names=F,quote=F,col.names=F,append=T)

cat('}',file=output,sep='\n',append=T)

# Then using dmLab2.3.0
# java -cp dmLab.jar -Xmx6g -Xms6g dmLab.mcfs.MCFS mcfs.run