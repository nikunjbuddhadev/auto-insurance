
library(readxl)
library(caTools)
library(rpart)
library(rpart.plot)
library(ROCR)
library(ROSE)
library(xgboost)


# Loading dataset
Dataset <- read_excel("datasets/Dataset.xlsx")
str(Dataset)


# Preprocessing 
Dataset = Dataset[-c(1,7)]

ind = sapply(Dataset, is.character)
Dataset[ind] = lapply(Dataset[ind], as.factor)

str(Dataset)

colSums(is.na(Dataset))
Dataset = na.omit(Dataset)


# Spliting into train and test sets
split = sample.split(Dataset$Response, SplitRatio = 0.75)
train = subset(Dataset, split==TRUE)
test = subset(Dataset, split==FALSE)


# base model with decision tree 
modelrpart = rpart(Response ~ ., data = train, method = "class")
predrpart = predict(modelrpart, newdata = test)
predicrpart = prediction(predrpart[,2], test$Response)
perfrpart = performance(predicrpart, "tpr", "fpr")
plot(perfrpart, print.cutoffs.at=seq(0,1,0.1))
table(test$Response, predrpart[,2]>0.5)


# Oversampling using ROSE
names(train) = make.names(names(train))
names(test) = make.names(names(test))
newtrain = ROSE(Response ~ ., data = train, N=6000, seed = 111)$data
table(newtrain$Response)
str(newtrain)

# testing the oversampled data
newmodel = rpart(Response ~ ., data=newtrain, method = "class")
newpred = predict(newmodel, newdata = test, type = "prob")
newpredic = prediction(newpred[,2], test$Response)
newperf = performance(newpredic, "tpr","fpr")
plot(newperf, print.cutoffs.at= seq(0,1,0.1))
table(test$Response, newpred[,2]>0.6)


ind1 = sapply(newtrain, is.factor)
newtrain[ind1] = lapply(newtrain[ind1], as.numeric)
test[ind] = lapply(test[ind], as.numeric)

table(newtrain$Response)
newtrain$Response = newtrain$Response -1 
test$Response = test$Response -1


# applying xgboost 
newtrain = as.matrix(newtrain)
test = as.matrix(test)

modelxgb = xgboost(data = newtrain[,-3], label = newtrain[,3], nrounds = 200, objective= "binary:logistic")
predxgb = predict(modelxgb, test[,-3] )
predicxgb = prediction(predxgb, test[,3])
perfxgb = performance(predicxgb, "tpr", "fpr")
plot(perfxgb, print.cutoffs.at=seq(0,1,0.1))
table(test[,3], predxgb>0.5)



newtrain = as.data.frame(newtrain)
test = as.data.frame(test)


# Applying Principal Component Analysis
combi = rbind(newtrain, test)
mydata = combi[-c(3)]
pcatrain = mydata[1:nrow(newtrain),]
pcatest = mydata[-(1:nrow(newtrain)),]
prin.comp = prcomp(pcatrain, scale. = T)
names(prin.comp)
biplot(prin.comp, scale = 0)


stddev = prin.comp$sdev
prvar = stddev^2
propvarex = prvar/sum(prvar)
plot(propvarex, xlab="Principal Component Analysis", ylab="Proportion of Variance Explained",type="b")
plot(cumsum(propvarex), xlab="Principal Component Analysis", ylab="Cumulative Proportion of Variance Explained", type="b")

traindata = data.frame(Response=newtrain$Response, prin.comp$x)
traindata = traindata[,1:20]
testdata = predict(prin.comp, newdata = pcatest)
testdata = as.data.frame(testdata)
testdata = testdata[,1:19]


# testing PCAed data with base model with decision tree
modelrp = rpart(Response ~ ., data = traindata, method = "class")
predrp = predict(modelrp, newdata = testdata)
predicrp = prediction(predrp[,2], test$Response)
perfrp = performance(predicrp, "tpr", "fpr")
plot(perfrp, print.cutoffs.at=seq(0,1,0.1))
table(test$Response, predrp[,2]>0.6)


