
library(caTools)
library(mice)
library(rpart)
library(rpart.plot)
library(ROCR)
library(randomForest)



train <- read.csv("datasets/train.csv")

str(train)
train = train[-c(1,4,9)]

train$Sex = as.factor(train$Sex)
train$Embarked = as.factor(train$Embarked)

train$Cabin = substr(train$Cabin,0,1)
train$Cabin = as.factor(train$Cabin)


str(train)

# Missing Value Imputation
mydata = complete(mice(train))
sum(is.na(mydata))

split = sample.split(mydata, SplitRatio = 0.75)
newtrain = subset(mydata, split==TRUE)
newtest = subset(mydata, split==FALSE)

str(newtrain)

# applying rpart
modelrpart = rpart(Survived ~ ., data = newtrain, method = "class")
predrpart = predict(modelrpart, newdata = newtest)
newtest$predrpart = predrpart[,2]
predicrpart = prediction(newtest$predrpart, newtest$Survived)
perf = performance(pred, "tpr", "fpr")
plot(perf, print.cutoffs.at=seq(0,1,0.1))
table(newtest$Survived, newtest$predrpart>0.3)


# applying random forest
modelrf = randomForest(Survived ~ ., data = newtrain, mtry=3, ntree=100)
newtest$predrf = predict(modelrf, newdata = newtest)
predicrf = prediction(newtest$predrf, newtest$Survived)
perfrf = performance(predicrf, "tpr", "fpr")
plot(perfrf, print.cutoffs.at=seq(0,1,0.1))
table(newtest$Survived, newtest$predrf>0.5)


#applying glm
newtrain$Survived = as.factor(newtrain$Survived)
newtest$Survived = as.factor(newtest$Survived)

modelglm = glm(Survived ~ ., data = newtrain, family = binomial)
newtest$predglm = predict(modelglm, newdata = newtest, type = "response")
predicglm = prediction(newtest$predglm, newtest$Survived)
perfglm = performance(predicglm, "tpr","fpr")
plot(perfglm, print.cutoffs.at=seq(0,1,0.1))
table(newtest$Survived, newtest$predglm>0.5)



# applying stacking for improving accuracy
predrpart = predict(modelrpart, type = "prob")
newtrain$predrpart = predrpart[,2]
newtrain$predrf = predict(modelrf, method="class")
newtrain$predglm = predict(modelglm, type = "response")


tlmodel = glm(Survived ~ predrpart+predrf+predglm, data = newtrain, family = binomial)
tlpred = predict(tlmodel, newdata = newtest, type = "response")
tlpredic = prediction(tlpred, newtest$Survived)
tlperf = performance(tlpredic, "tpr", "fpr")
plot(tlperf, print.cutoffs.at= seq(0,1,0.1))
table(newtest$Survived, tlpred>0.4)




# applying Principal Component Analysis

combi = rbind(newtrain,newtest)
newdata = combi[-c(1)]

ind = sapply(newdata, is.factor)
newdata[ind] = lapply(newdata[ind], as.numeric)

pcatrain = newdata[1:nrow(newtrain),]
pcatest = newdata[-(1:nrow(newtrain)),]  

prin.comp = prcomp(pcatrain, scale. = T)  
names(prin.comp)  

biplot(prin.comp, scale = 0)  

stddev = prin.comp$sdev  
prvar = stddev^2
prvar[1:10]
propvarex = prvar/sum(prvar)

plot(propvarex, xlab= "Principal Component",
     ylab = "Proportion of Variance Explained", type="b")
plot(cumsum(propvarex), xlab = "Principal Component",
     ylab="Cumulative Proportion of Variance Explained", type="b")

traindata = data.frame(Y=newtrain$Survived, prin.comp$x)
traindata = traindata[,1:9]

testdata = predict(prin.comp, newdata = pcatest)
testdata = as.data.frame(testdata)
testdata = testdata[,1:8]


rpart.model = rpart(Y ~ ., data = traindata, method = "class")
rpart.pred = predict(rpart.model, newdata = testdata)
rpart.predic = prediction(rpart.pred[,2], newtest$Survived)
rpart.perf = performance(rpart.predic, "tpr", "fpr")
plot(rpart.perf, print.cutoffs.at= seq(0,1,0.1))
table(newtest$Survived, rpart.pred[,2]>0.5)


