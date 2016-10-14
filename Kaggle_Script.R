# Author:       Uwe Hoenig
# Course:       ??? - Machine learning/Advanced Computing
# Last update:  15.02.16
# Type:         Problemset 1 

#Kaggle Script:
#Check the "OnlineNewsPopularity.txt" file for a detailed overview of the data 
setwd("/Users/uhoenig/Dropbox/Studium 2015-16 GSE/Term 2/Machine learning/Kaggle competition")

####LOADING DATA (starts)
raw_train <- read.csv(file="news_popularity_training.csv", stringsAsFactors = F)
raw_train=raw_train[,-c(1,2)] # get rid of the garbage columns
raw_test <- read.csv(file="news_popularity_test.csv", header=TRUE,sep=",", stringsAsFactors = F)
raw_test = raw_test[,-c(1,2)]
###LOADING DATA (ends)

###SOME NEW CLEANING OPTION (starts)
library(caret)
nzv <- nearZeroVar(raw_train[,-60], saveMetrics = TRUE)
print(paste('Range:',range(nzv$percentUnique)))

print(head(nzv))

print(paste('Column count before cutoff:',ncol(raw_train[,-60])))

dim(nzv[nzv$percentUnique > 0.1,])

Train_nzv <- raw_train[,-60][c(rownames(nzv[nzv$percentUnique > 0.1,])) ]
print(paste('Column count after cutoff:',ncol(Train_nzv)))

dfEvaluate <- cbind(as.data.frame(sapply(Train_nzv, as.numeric)),
                    cluster=raw_train[,60])


###SOME NEW CLEANING OPTION (ends)

####SPLITTING DATA (starts)
library(caTools)
set.seed(3000)
spl = sample.split(raw_train, SplitRatio = 0.75)
Train = subset(raw_train, spl==TRUE)
Test = subset(raw_train, spl==FALSE)
####SPLITTING DATA (ends)


###FEATURE TRANSFORMATION (starts)
##Explanation:  
 #transform the popularity column (label column) because xgboost needs labels
 #starting from 0

labelTrain=Train$popularity-1
labelTest=Test$popularity-1


####FEATURE SLECTION (starts)
##Explanation:
 #Warwick: David Rossel code
 #I run this on the entire training data (col 60 is where my label is at)

library(mombf)
library(robustbase)
fit1 = modelSelection(raw_train[,60],raw_train[,-60])
postProb(fit1)[1:5,]
 #Results:
 #b)-c(3,7,8,11,12,14,16,17,19,20,21,24,25,26,27,30,38,39,41,44) 30%
 #c)-c(3,7,8,11,12,14,16,17,19,20,21,24,25,26,27,30,31,35,38,39,41,44) 24.4%
###FEATURE SELECTION (ends)
###SPLITTING DATA (ends)

###Here is where the magic happens (starts)
##Explanation
 #Place your to be excluded columns here, don't forget to exclude label column 60 ;)
cleanTrainData=Train[,-c(3,7,8,11,12,14,16,17,19,20,21,24,25,26,27,30,38,39,41,44,60)]
cleanTestData=Test[,-c(3,7,8,11,12,14,16,17,19,20,21,24,25,26,27,30,38,39,41,44,60)]

library(xgboost)

h <- sample(nrow(cleanTrainData),5000)

dval <- xgb.DMatrix(data=data.matrix(cleanTrainData[h,]), label=labelTrain[h], missing = NaN)
dtrain <- xgb.DMatrix(data=data.matrix(cleanTrainData[-h,]), label=labelTrain[-h], missing=NaN)

watchlist <- list(val=dval, train=dtrain)
param <- list(
              objective = "multi:softmax",
              #objective = "",
              num_class=5,
              booster = "gbtree",
              eta = 0.02,
              max_depth = 7,
              subsample = 0.8,
              colsample_bytree = 0.8,
              #eval_metric ="merror",
              eval_metric = "mlogloss"
              #eval_metric = "error"
              #scale_pos_weight
)

clf <- xgb.train( params = param,
                  data = dtrain,
                  nrounds = 300,
                  verbose = 1,
                  early.stop.round = 100,
                  watchlist = watchlist,
                  maximize = FALSE
)
a=predict(clf,data.matrix(cleanTestData), missing=NaN)
pred <- factor(predict(clf, data.matrix(cleanTestData), missing=NaN), levels=c(0:4))


library(ROCR)
library(caret)
#ROCRpred = prediction(pred, Test$popularity) #works only for 2 classes
#as.numeric(performance(ROCRpred, "auc")@y.values)

#RESULTS confusin matrix: predicted labels for the test vs actual labels
results <- confusionMatrix(pred, labelTest)
results

####################
####################
#random shot

clfDOM <- xgboost(params = param,
                  data = as.matrix(raw_train[,-60]),
                  nrounds = 1500,
                  verbose = 0,
                  #early.stop.round = 300,
                  #watchlist = watchlist,
                  maximize = FALSE,
                  label=as.integer(raw_train$popularity)-1
)


res= xgb.cv(params= param, data=dtrain, label=labelTrain, nround = 2, nfold= 5, prediction = T)

str(res)

xgb.importance(dtrain, model = clf)

#SAVE THE RESULTS
submission <- data.frame(id = raw_test$id)
submission$popularity <- predict(clfDOM, as.matrix(raw_test))+1 #add 1!!
write.csv(submission, file = "cherrypop.csv", row.names=FALSE)

######


#autoencoder 
install.packages("neuralnet")
library(neuralnet)
library(caret)
