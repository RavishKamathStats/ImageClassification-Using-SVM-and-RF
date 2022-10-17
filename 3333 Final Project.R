#### Libraries Required ####
library(jpeg)
library(e1071)
library(randomForest)
library(kernlab)
library(tidyverse)
library(RColorBrewer)
library(party)



#### Data ####
pm = read.csv('/Users/ravishkamath/Desktop/University/2. 
              York Math/1 MATH/1. Statistics /MATH 3333/3. 
              Assessments/Final Project/Supporting files for 
              final project-20220307/photoMetaData.csv')
n <- nrow(pm)
trainFlag <- (runif(n) > 0.5)
y <- as.numeric(pm$category == "outdoor-day")

X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("/Users/ravishkamath/Desktop/University/2. 
                         York Math/1 MATH/1. Statistics /MATH 3333/3. 
                         Assessments/Final Project/Supporting files 
                         for final project-20220307/columbiaImages//",pm$name[j]))
  X[j,] <- apply(img,3,median)
  print(sprintf("%03d / %03d", j, n))
}
dat = data.frame(x=X, y=as.factor(y))

ggplot(data = dat, aes(x = x.2 , y = x.1, color = y, shape = y)) + 
  geom_point(size = 2) +
  scale_color_manual(values=c("#000000", "#FF0000")) +
  theme(legend.position = "none")

ggplot(data = dat, aes(x = x.3 , y = x.1, color = y, shape = y)) + 
  geom_point(size = 2) +
  scale_color_manual(values=c("#000000", "#FF0000")) +
  theme(legend.position = "none")


#### LOGISTIC ####
out <- glm(y ~ X, family=binomial, subset=trainFlag)
out$iter
summary(out)

#ROC#
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
y[order(pred)]
y[!trainFlag][order(pred[!trainFlag])]

mean((as.numeric(pred > 0.5) == y)[trainFlag])
mean((as.numeric(pred > 0.5) == y)[!trainFlag])

roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

#AUC#
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
glmAuc <- auc(r)
glmAuc





#### SVM ####
y = factor(y)
svm_model1 = svm(y~., type = 'C',  kernel = 'linear', data = dat)
summary(svm_model1)
plot(svm_model1, dat, x.2~x.3)
plot(svm_model1, dat, x.1~x.3)

#ROC#
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(svm_model1)))
y[order(pred)]
y[!trainFlag][order(pred[!trainFlag])]

mean((as.numeric(pred > 0.5) == y)[trainFlag])
mean((as.numeric(pred > 0.5) == y)[!trainFlag])

roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
glmAuc <- auc(r)
glmAuc

# Cross Validation #
n = nrow(X)
nt = 600
rep = 10
error_SVM = dim(rep)
neval = n - nt
for (i in 1: rep) {
  training = sample(1:n, nt)
  trainingset = dat[training,]
  testingset = dat[-training,]
  
  # SVM Analysis
  svm_model1 = svm(y~., type = 'C',  kernel = 'linear', data = trainingset) 
  pred_SVM = predict(svm_model1, testingset)
  tableSVM = table(testingset$y, pred_SVM)
  error_SVM[i] = (neval - sum(diag(tableSVM)))/neval
}
mean(error_SVM)





#### RF####
rf_classifier = randomForest(y ~., data = dat, type = classification, 
                             ntree = 100, mtry = 2, importance = TRUE)
rf_classifier

m <- ctree(y ~ ., data=dat)
m
plot(m, type="simple")
table(predict(m), dat$y)

#ROC#
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(rf_classifier)))
y[order(pred)]
y[!trainFlag][order(pred[!trainFlag])]

mean((as.numeric(pred > 0.5) == y)[trainFlag])
mean((as.numeric(pred > 0.5) == y)[!trainFlag])

roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
glmAuc <- auc(r)
glmAuc

# Cross Valdiation # 
n = nrow(X)
nt = 600
rep = 10
error_SVM = dim(rep)
neval = n - nt
for (i in 1: rep) {
  training = sample(1:n, nt)
  trainingset = dat[training,]
  testingset = dat[-training,]
  
  #Random Forest Analysis
  rf_classifier = randomForest(y ~., data = trainingset, type = classification, 
                               ntree = 100, mtry = 2, importance = TRUE)
  prediction_RF = predict(rf_classifier, testingset)
  table_RF = table(testingset$y, prediction_RF)
  error_RF[i] = (neval - sum(diag(table_RF)))/neval
}
mean(error_RF)





















