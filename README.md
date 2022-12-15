# Seq-RBPPred
We develop a machine learning model called Seq-RBPPred, mainly using XGBoost[1] train 6944 features. The analysis in this paper shows that the training and prediction results of RBPs and Non-RBPs of the same species in EuRBPDB[2] and PDB[3].

![image](https://user-images.githubusercontent.com/84023156/207883546-10c179a1-7a0a-430d-9854-259c35370a70.png)
RBPs are from EuRBPDB[2], and Non-RBPs from PDB[3], confirming the same species in RBPs and Non-RBPs. We mainly use Protr[4], UniRep[5], SeqVec[6], ESM-1b[7] to obtain and integrate features. Using DmLab[8], SVM[9], Random forest[10], Deep forest[11] and XGBoost[1] to analyze and predict RBPs.Finally, use the method in Deep-RBPPred [12] and the above machine learning method to test on the same testing set.

# Requirements

- Python
- R
- scikit-learn

# Usage

Positive samples are from EuRBPDB[2],and negative samples are from PDB[3],confirming the same species in positive and negative samples. According to thesequence of each RBP in the RBPlist, it is extracted from totalFa. If there aremultiple corresponding sequences, all are recorded, and the longest one isused. For example, ENSG00000001497 has 5 sequences in totalFa: ENSP00000363940,ENSP00000363944, ENSP00000363937, ENSP00000473471, ENSP00000473630, take thelongest sequence ENSP00000363944, and get 115151 RBPs. An initial set of 2777 Non-RBPs was retrieved using PISCES[13] to retain Non-RBPs of the same species as EuRBPDB from the PDB, cleave protein sequences and remove those whose function is unknown or associated with RNA binding. Redundancy between the initial 115,151 RBPs and 2,777 Non-RBPS was removed using the CD-HIT[14] tool with a sequence identity cutoff of 25%, which yielded a non-redundant set of 6,618 RBPs and 1,565 Non-RBPs. To maintain
length consistency with proteins in the Non-RBPs set, some proteins were removed from the non-redundant set of RBPs, resulting in a set of 6365 RBPs. Two-thirds are used as the training set and one-third as the testing set, obtaining 4243 RBPs for training, 1043 Non-RBPs for training, and 2122 RBPs and 522 Non-RBPs for testing. After the feature extraction of  Protr[4], UniRep[5], SeqVec[6] and ESM-1b[7] , some proteins have no result output, so we discarded these proteins, and finally obtained 3379 RBPs and 1034 Non-RBPs in the training set,1708 RBPs and 517 Non-RBPs in the testing set.

1. Data has our training test and testing test.
2. bind_all _fearure has the code for obtaining final features.

# Contact

Contact:[Huang Tao](http://www.sinh.cas.cn/rcdw/qnyjy/202203/t20220310_6387862.html ),Yan Yuyao

# References

1. ```
   1. Chen, T. and C.Guestrin. *Xgboost: A scalable tree boosting system*. in *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*.2016.
   2. Liao, J.-Y., et al., *EuRBPDB: a comprehensive resource for annotation, functional and oncological investigation of eukaryotic RNA binding proteins (RBPs). * Nucleic Acids Research, 2020. **48**(D1): p. D307-D313.
   3. Berman, H.M., et al., *The Protein Data Bank. * Nucleic Acids Research, 2000. **28**(1): p. 235-242.
   4. Xiao, N., et al., *protr/ProtrWeb: R package and web server for generating various numerical representation schemes of protein sequences. *Bioinformatics, 2015. **31**(11): p.1857-9.
   5. Alley, E.C., et al., *Unified rational protein engineering with sequence-based deep representation learning.* Nature Methods, 2019. **16**(12): p. 1315-1322.
   6. Heinzinger, M., et al., *Modeling aspects of the language of life through transfer-learning protein sequences.* BMC Bioinformatics, 2019. **20**(1): p. 723.
   7. Rives, A., et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* Proceedings of the National Academy of Sciences, 2021. **118**(15): p. e2016239118.
   8. Dramiński, M., et al., *Monte Carlo feature selection for supervised classification.* Bioinformatics, 2008. **24**(1): p. 110-117.
   9. Hearst, M.A., et al., *Support vector machines.* IEEE Intelligent Systems and their applications, 1998. **13**(4): p. 18-28.
   10. Breiman, L., *Random forests.* Machine learning, 2001. **45**(1): p. 5-32.
   11. Zhou, Z.-H. and J. Feng, *Deep forest.* National Science Review, 2019. **6**(1): p. 74-86.
   12. Zheng, J., et al., *Deep-RBPPred: Predicting RNA bindingproteins in the proteome scale based on deep learning.* Scientific Reports,2018. **8**(1): p. 15264.
   13. Wang, G. and R.L. Dunbrack, Jr., *PISCES: a protein sequence culling server.* Bioinformatics, 2003. **19**(12): p. 1589-91.
   14. Li, W. and A. Godzik, *Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences.* Bioinformatics,2006. **22**(13): p. 1658-9.
   ```

