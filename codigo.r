###################################################################
# Predição da popularidade de um tópico no twitter                #
################################################################### 

library(corrplot)

set.seed(100)

#Inspeção
dataTrain <- read.csv("twitter_training_set.csv", header = TRUE, stringsAsFactors=TRUE)
dataVal <- read.csv("twitter_validation_set.csv", header = TRUE, stringsAsFactors=TRUE)
dataTest <- read.csv("twitter_test_set.csv", header = TRUE, stringsAsFactors=TRUE)

dim(dataTrain)
summary(dataTrain)

dim(dataVal)
summary(dataVal)

dim(dataTest)
summary(dataTest)

any(is.na(dataTrain))
any(is.na(dataVal))
any(is.na(dataTest))

#Normalização - MinMax
min_features <- apply(dataTrain[,1:8], 2, min)
min_features

max_features <- apply(dataTrain[,1:8], 2, max)
max_features

diff <- max_features - min_features
diff

dataTrain[,1:8] <- sweep(dataTrain[,1:8], 2, min_features, "-")
dataTrain[,1:8] <- sweep(dataTrain[,1:8], 2, diff, "/")
summary(dataTrain)

dataVal[,1:8] <- sweep(dataVal[,1:8], 2, min_features, "-")
dataVal[,1:8] <- sweep(dataVal[,1:8], 2, diff, "/")
summary(dataVal)

dataTest[,1:8] <- sweep(dataTest[,1:8], 2, min_features, "-")
dataTest[,1:8] <- sweep(dataTest[,1:8], 2, diff, "/")
summary(dataTest)

# dont Remove Outliers from DataTrain#
summary(dataTrain)
boxplot(dataTrain$target)

outliers <- boxplot(dataTrain$target, plot=FALSE)
dim(outliers)
#dataTrain <- dataTrain[-which(dataTrain$target %in% outliers),]

#boxplot(dataTrain$target)
#summary(dataTrain)

#Correlação
corrplot(cor(dataTrain), method="number")
#PCA
pca <- prcomp(dataTrain[,1:8], scale=F)
summary(pca)


## Baseline ##
baseline <- lm(formula=target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL,
               data=dataTrain)
summary(baseline)

MAE <- function(preds, labels){
  mae_values <- sum(abs(preds-labels))/length(preds)
  return(mae_values)
}




trainPred <- predict(baseline, dataTrain)
valPred <- predict(baseline, dataVal)
testPred <- predict(baseline, dataTest)

mae_train_baseline <- MAE(trainPred, dataTrain$target)
mae_train_baseline

mae_val_baseline <- MAE(valPred, dataVal$target)
mae_val_baseline

mae_test_baseline <- MAE(testPred, dataTest$target)
mae_test_baseline

#Baseline - pca #
## Baseline ##
baseline <- lm(formula=target ~ NCD + AI + AL,
               data=dataTrain)
testPred <- predict(baseline, dataTest)
mae_test_baseline <- MAE(testPred, dataTest$target)
mae_test_baseline

## Combining features ### 

f01 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL 
               + (NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL)^2)

f02 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL 
               + (NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL)^3)

f03 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL 
               + (NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL)^4)


modelsNoCategorical <- c(f01, f02, f03)
total_mae_train_noCat <- c(length(modelsNoCategorical))
total_mae_val_noCat <- c(length(modelsNoCategorical))

i <- 1
for(f in modelsNoCategorical){
  
  model <- lm(formula=f, data=dataTrain)
  
  trainPred <- predict(model, dataTrain)
  valPred <- predict(model, dataVal)
  
  mae_train <- MAE(trainPred, dataTrain$target)
  total_mae_train_noCat[i] <- mae_train
  
  mae_val <- MAE(valPred, dataVal$target)
  total_mae_val_noCat[i] <- mae_val
  i <- i + 1
}

# Performance on TEST SET 
model <- lm(formula=f01, data=dataTrain)

testPred <- predict(model, dataTest)
mae_test <- MAE(testPred, dataTest$target)
mae_test

### Combinação somente das features principais ###
f01 <- formula(target ~ NCD + AI + AL 
               + (NCD + AI + AL)^2)

f02 <- formula(target ~ NCD + AI + AL 
               + (NCD + AI + AL)^3)

f03 <- formula(target ~ NCD + AI + AL 
               + (NCD + AI + AL)^4)


modelsNoCategorical <- c(f01, f02, f03)
total_mae_train_noCat <- c(length(modelsNoCategorical))
total_mae_val_noCat <- c(length(modelsNoCategorical))

i <- 1
for(f in modelsNoCategorical){
  
  model <- lm(formula=f, data=dataTrain)
  
  trainPred <- predict(model, dataTrain)
  valPred <- predict(model, dataVal)
  
  mae_train <- MAE(trainPred, dataTrain$target)
  total_mae_train_noCat[i] <- mae_train
  
  mae_val <- MAE(valPred, dataVal$target)
  total_mae_val_noCat[i] <- mae_val
  i <- i + 1
}

# Performance on TEST SET 
model <- lm(formula=f02, data=dataTrain)

testPred <- predict(model, dataTest)
mae_test <- MAE(testPred, dataTest$target)
mae_test

## Polynomials ### 

f01 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL, data=dataTrain)

f02 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) +
                 I(NAu^2) + I(ADL^2), data=dataTrain)

f03 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3), data=dataTrain)

f04 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4), 
                 data=dataTrain)

f05 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5), data=dataTrain)

f06 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5) + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
                 I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6), data=dataTrain)

f07 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5) + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
                 I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6) + I(NCD^7) + I(AI^7) +
                 I(AL^7) + I(BL^7) + I(AL_C^7) + I(AT_D^7) + I(NAu^7) + I(ADL^7), data=dataTrain)

f08 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5) + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
                 I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6) + I(NCD^7) + I(AI^7) +
                 I(AL^7) + I(BL^7) + I(AL_C^7) + I(AT_D^7) + I(NAu^7) + I(ADL^7) +
                 I(NCD^8) + I(AI^8) + I(AL^8) + I(BL^8) + I(AL_C^8) + I(AT_D^8) +
                 I(NAu^8) + I(ADL^8), data=dataTrain)

f09 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5) + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
                 I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6) + I(NCD^7) + I(AI^7) +
                 I(AL^7) + I(BL^7) + I(AL_C^7) + I(AT_D^7) + I(NAu^7) + I(ADL^7) +
                 I(NCD^8) + I(AI^8) + I(AL^8) + I(BL^8) + I(AL_C^8) + I(AT_D^8) +
                 I(NAu^8) + I(ADL^8) + I(NCD^9) + I(AI^9) + I(AL^9) + I(BL^9) + 
                 I(AL_C^9) + I(AT_D^9) + I(NAu^9) + I(ADL^9), data=dataTrain)

f10 <- formula(target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5) + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
                 I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6) + I(NCD^7) + I(AI^7) +
                 I(AL^7) + I(BL^7) + I(AL_C^7) + I(AT_D^7) + I(NAu^7) + I(ADL^7) +
                 I(NCD^8) + I(AI^8) + I(AL^8) + I(BL^8) + I(AL_C^8) + I(AT_D^8) +
                 I(NAu^8) + I(ADL^8) + I(NCD^9) + I(AI^9) + I(AL^9) + I(BL^9) + 
                 I(AL_C^9) + I(AT_D^9) + I(NAu^9) + I(ADL^9) + I(NCD^10) + I(AI^10) +
                 I(AL^10) + I(BL^10) + I(AL_C^10) + I(AT_D^10) + I(NAu^10) + I(ADL^10), data=dataTrain)

formulas <- list(f01, f02, f03, f04, f05, f06, f07, f08, f09, f10)
total_mae_train_poly <- c(length(formulas))
total_mae_val_poly <- c(length(formulas))

i <- 1
for(i in 1:10){
  model <- lm(formula=formulas[[i]], data=dataTrain)
  
  valPred <- predict(model, dataVal)
  trainPred <- predict(model, dataTrain)
  
  mae_train <- MAE(trainPred, dataTrain$target)
  total_mae_train_poly[i] <- mae_train
  
  mae_val <- MAE(valPred, dataVal$target)
  total_mae_val_poly[i] <- mae_val
  i <- i + 1
  
}

plot(total_mae_val_poly, xlab="Complexity", ylab="Target", 
     ylim=c(65, 71), pch="+", col="blue")

points(total_mae_train_poly, pch="*", col="red")
points(rep(mae_val_baseline, length(total_mae_val_poly)), pch="o", col="green")

lines(total_mae_train_poly, col="red", lty=2)
lines(total_mae_val_poly, col="blue", lty=2)
lines(rep(mae_val_baseline, length(total_mae_val_poly)), col="green", lty=2)

min(total_mae_val_noCat) #67.81487
min(total_mae_val_poly) #66.76273

total_mae_val_poly #Best Model = sixth model 

# Performance on TEST SET 

## Retrain the best model ##
# best_model <-  lm(formula=target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
#                         I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) +
#                         I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
#                         I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
#                         I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) +
#                         I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
#                         I(NAu^5) + I(ADL^5)+  + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
#                    I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6), data=dataTest)

best_model <- lm(formula=target ~ NCD + AI + AL + BL + AL_C + AT_D + NAu + ADL +
                 I(NCD^2) + I(AI^2) + I(AL^2) + I(BL^2) + I(AL_C^2) + I(AT_D^2) + 
                 I(NAu^2) + I(ADL^2) + I(NCD^3) + I(AI^3) + I(AL^3) + I(BL^3) +
                 I(AL_C^3) + I(AT_D^3) + I(NAu^3) + I(ADL^3) + I(NCD^4) + I(AI^4) +
                 I(AL^4) + I(BL^4) + I(AL_C^4) + I(AT_D^4) + I(NAu^4) + I(ADL^4) + 
                 I(NCD^5) + I(AI^5) + I(AL^5) + I(BL^5) + I(AL_C^5) + I(AT_D^5) +
                 I(NAu^5) + I(ADL^5) + I(NCD^6) + I(AI^6) + I(AL^6) + I(BL^6) +
                 I(AL_C^6) + I(AT_D^6) + I(NAu^6) + I(ADL^6), data=dataTrain)
acc <- acc
testPred <- predict(best_model, dataTest)
mae_test <- MAE(testPred, dataTest$target)
mae_test

## Polynomials com Feature Selection ### 

f01 <- formula(target ~ NCD + AI + AL, data=dataTrain)

f02 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2), data=dataTrain)

f03 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) 
                 I(NCD^3) + I(AI^3) + I(AL^3), data=dataTrain)

f04 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4), data=dataTrain)

f05 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4) +  
                 I(NCD^5) + I(AI^5) + I(AL^5), data=dataTrain)

f06 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4) +  
                 I(NCD^5) + I(AI^5) + I(AL^5) +
                 I(NCD^6) + I(AI^6) + I(AL^6), data=dataTrain)

f07 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4) +  
                 I(NCD^5) + I(AI^5) + I(AL^5) +
                 I(NCD^6) + I(AI^6) + I(AL^6) +
                 I(NCD^7) + I(AI^7) + I(AL^7), data=dataTrain)

f08 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4) +  
                 I(NCD^5) + I(AI^5) + I(AL^5) +
                 I(NCD^6) + I(AI^6) + I(AL^6) +
                 I(NCD^7) + I(AI^7) + I(AL^7) +
                 I(NCD^8) + I(AI^8) + I(AL^8), data=dataTrain)

f09 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4) +  
                 I(NCD^5) + I(AI^5) + I(AL^5) +
                 I(NCD^6) + I(AI^6) + I(AL^6) +
                 I(NCD^7) + I(AI^7) + I(AL^7) +
                 I(NCD^8) + I(AI^8) + I(AL^8) +
                 I(NCD^9) + I(AI^9) + I(AL^9), data=dataTrain)

f010 <- formula(target ~ NCD + AI + AL +
                 I(NCD^2) + I(AI^2) + I(AL^2) +
                 I(NCD^3) + I(AI^3) + I(AL^3) +
                 I(NCD^4) + I(AI^4) + I(AL^4) +  
                 I(NCD^5) + I(AI^5) + I(AL^5) +
                 I(NCD^6) + I(AI^6) + I(AL^6) +
                 I(NCD^7) + I(AI^7) + I(AL^7) +
                 I(NCD^8) + I(AI^8) + I(AL^8) +
                 I(NCD^9) + I(AI^9) + I(AL^9) +
                 I(NCD^10) + I(AI^10) + I(AL^10), data=dataTrain)



formulas <- list(f01, f02, f03, f04, f05, f06, f07, f08, f09, f10)
total_mae_train_poly <- c(length(formulas))
total_mae_val_poly <- c(length(formulas))

i <- 1
for(i in 1:10){
  model <- lm(formula=formulas[[i]], data=dataTrain)
  
  valPred <- predict(model, dataVal)
  trainPred <- predict(model, dataTrain)
  
  mae_train <- MAE(trainPred, dataTrain$target)
  total_mae_train_poly[i] <- mae_train
  
  mae_val <- MAE(valPred, dataVal$target)
  total_mae_val_poly[i] <- mae_val
  i <- i + 1
  
}

plot(total_mae_val_poly, xlab="Complexity", ylab="Target", 
     ylim=c(65, 71), pch="+", col="blue")

points(total_mae_train_poly, pch="*", col="red")
points(rep(mae_val_baseline, length(total_mae_val_poly)), pch="o", col="green")

lines(total_mae_train_poly, col="red", lty=2)
lines(total_mae_val_poly, col="blue", lty=2)
lines(rep(mae_val_baseline, length(total_mae_val_poly)), col="green", lty=2)

min(total_mae_val_noCat) #67.81487
min(total_mae_val_poly) #66.76273

total_mae_val_poly #Best Model = sixth model 

# Performance on TEST SET 

## Retrain the best model ##

best_model <- lm(formula=target ~ NCD + AI + AL +
                           I(NCD^2) + I(AI^2) + I(AL^2) +
                           I(NCD^3) + I(AI^3) + I(AL^3) +
                           I(NCD^4) + I(AI^4) + I(AL^4), data=dataTrain)


testPred <- predict(best_model, dataTest)
mae_test <- MAE(testPred, dataTest$target)
mae_test
