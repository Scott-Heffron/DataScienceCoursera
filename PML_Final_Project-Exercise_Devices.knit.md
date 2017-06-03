---
title: "PML Final Project - Exercise Devices"
output: html_document
---



## Backgound of Project

######Using devices such as Jawbone Up, Nike FuelBand, and Fitbitit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

######More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


### Get Needed Data 

```r
set.seed(12345)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), 
                     header = TRUE, 
                     as.is = TRUE,
                     stringsAsFactors = FALSE, 
                     sep = ',', 
                     na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), 
                    header = TRUE, 
                    as.is = TRUE,
                    stringsAsFactors = FALSE, 
                    sep = ',', 
                    na.strings=c("NA","#DIV/0!",""))
```


### Clean up the data and variables for processing

```r
training$classe <- as.factor(training$classe)

## Clean up variables
## Removes columns with NA as the data
findNA <- apply(training, 2, function(x){sum(is.na(x))})
training <- training[, which(findNA == 0)]

findNA <- apply(testing, 2, function(x){sum(is.na(x))})
testing <- testing[, which(findNA == 0)]

v <- which(lapply(training, class) %in% "numeric")
preObj <- preProcess(training[,v], 
                     method=c('knnImpute', 'center', 'scale'))
trainLess1 <- predict(preObj, 
                      training[,v])
trainLess1$classe <- training$classe

testLess1 <- predict(preObj, 
                     testing[,v])

## Remove the non-zero variables
## This is so to get rid of small amounts that will effect the out
## come of the prediction. Only want good numbers to be used in the
## predictions
nzv <- nearZeroVar(trainLess1, saveMetrics=TRUE)
trainLess1 <- trainLess1[, nzv$nzv==FALSE]
nzv <- nearZeroVar(testLess1,saveMetrics=TRUE)
testLess1 <- testLess1[,nzv$nzv==FALSE]

##str(training)
##str(testing)
```

### Slice Data to create training and cross validation data set

```r
set.seed(12345)

inTrain = createDataPartition(trainLess1$classe, 
                              p = 0.75, 
                              list=FALSE)
training = trainLess1[inTrain,]
crossValidation = trainLess1[-inTrain,]
```

### Train Model - Random Forest

```r
modFit <- train(classe ~., 
                method="rf", 
                data=training, 
                trControl=trainControl(method='cv'), 
                number=5, 
                allowParallel=TRUE )

print(modFit)
```

```
## Random Forest 
## 
## 14718 samples
##    27 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13246, 13248, 13245, 13245, 13246, 13246, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9935453  0.9918351
##   14    0.9925258  0.9905461
##   27    0.9895363  0.9867651
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

### Accuracy on Train and Cross Validation data sets

```r
## Training set
trainingPred <- predict(modFit, 
                        training)
confusionMatrix(trainingPred, 
                training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
## Cross Validation data set
crossVldPred <- predict( modFit, 
                         crossValidation)
confusionMatrix(crossVldPred, 
                crossValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395   11    0    0    0
##          B    0  934    6    0    0
##          C    0    4  844    9    2
##          D    0    0    5  794    3
##          E    0    0    0    1  896
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9916         
##                  95% CI : (0.9887, 0.994)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9894         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9842   0.9871   0.9876   0.9945
## Specificity            0.9969   0.9985   0.9963   0.9980   0.9998
## Pos Pred Value         0.9922   0.9936   0.9825   0.9900   0.9989
## Neg Pred Value         1.0000   0.9962   0.9973   0.9976   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1905   0.1721   0.1619   0.1827
## Detection Prevalence   0.2867   0.1917   0.1752   0.1635   0.1829
## Balanced Accuracy      0.9984   0.9913   0.9917   0.9928   0.9971
```

### Results of prediction

```r
resultPred <- predict(modFit, testLess1)
resultPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


