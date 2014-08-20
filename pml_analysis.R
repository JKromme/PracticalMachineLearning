# -------------------------------------------------------------------- #
# ----            Coursera: Practical Machine Learning            ---- #
# -------------------------------------------------------------------- #
rm(list=ls())
# Author = J.Kromme
# Date = 17-8-2014
set.seed(1234)

# ---- Load packages ---- #
library(caret)
library(splines)
library(ggplot2)
library(rattle)
library(rpart)
library(rpart.plot)
library(klaR)
library(foreach)
library(doParallel)
library(randomForest)

# ---- Load data ---- #
# create data folder
if (!file.exists("data")){ dir.create("data")}

# download data
fileUrl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
download.file(fileUrl, destfile = "./data/training.csv")
fileUrl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(fileUrl, destfile = "./data/test.csv")
list.files("./data")
downloadDate <- date()

# load data
# some '#DIV/0!' appears in data, delete those by reading in data
trainingset <- read.csv('./data/training.csv', na.strings = c('#DIV/0!', 'NA') )
testset <- read.csv('./data/test.csv', na.strings = c('#DIV/0!', 'NA') )

# partition train into train / validation set. to validate model as testset doesn't have a DV.
inTrain <- createDataPartition(trainingset[,1], p=0.7, list= FALSE)
trainingset <- trainingset[inTrain,]
validationset <-trainingset[-inTrain,]

# backup data
trainingsetBackup <- trainingset
testsetBackup     <- testset
validationsetBackup <- validationset


# ---- Data preparation ---- #
#summary(trainingset)

# -- missing values
# so many missing values, imputing won't help. therefore delete these
toBeDeleted = list()
for (col in 1:ncol(trainingset)){
  # for each column, count number of missing values
  noOfMissingValues = length(trainingset[!complete.cases(trainingset[,col]),col])
  # if 40% or more is missing, delete variable from traing and testset
  if (noOfMissingValues > nrow(trainingset) * 0.4){
    
    cat('delete due to too many missing values:', names(trainingset)[col], '\n')
    toBeDeleted = c(toBeDeleted, names(trainingset)[col])
  }
}

trainingset <- trainingset[,!names(trainingset) %in% toBeDeleted]
testset <- testset[,!names(testset) %in% toBeDeleted]
validationset <- validationset[,!names(validationset) %in% toBeDeleted]


#summary(trainingset) # no missing values left

# check variable types
#str(trainingset) # correct

# delete due to theoretical reasons: row number and username etc won't explain useful variance. so exclude from model
trainingset <- trainingset[,-c(1:7)]
testset <- testset[,-c(1:7)]
validationset<- validationset[,-c(1:7)]

# delete variables with near zero variance
nearZeroVar(trainingset[,-53], saveMetrics = TRUE) # no variables with NZV

# plot histograms
for (col in 1:(ncol(trainingset) -1)){
  #hist(as.numeric(trainingset[,col]))
  if (col != ncol(trainingset) -1){
    #readline("Please press the Enter")
  }
} # few highly skewed, few with two normal distributions (1 - 4, 10), few with many zero variables

# investigate double distributions further
#featurePlot(x=trainingset[,c(1:4,10)], y = trainingset$classe, plot = 'pairs')

correlations <- abs(cor(trainingset[,c(1:4,10)]))
diag(correlations) <- 0
which(correlations > 0.7, arr.ind = T) # plots and correlation table shows high correlations -> need for PCA as preprocessing proces



# ---- build models ---- #
# -- trees
modelFitTree <- train(trainingset$classe ~., method='rpart', data = trainingset[,-53])
finalModelTree <- modelFitTree$finalModel
modelFitTree
finalModelTree

fancyRpartPlot(finalModelTree)

confusionMatrix(trainingset$classe, predict(modelFitTree, newdata = trainingset))
confusionMatrix(validationset$classe, predict(modelFitTree, newdata = validationset))



# -- random forest - use randomForest package itself which includes combine function for doParallel
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
modelFitRF <- foreach(ntree=rep(50, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
  randomForest(trainingset[,-53], trainingset$classe, ntree=ntree)
  
}
stopCluster(cl)

confusionMatrix(trainingset$classe, predict(modelFitRF, newdata = trainingset))
confusionMatrix(validationset$classe, predict(modelFitRF, newdata = validationset))

varImpPlot(modelFitRF, n = 10)


predict(modelFitRF, newdata = testset)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predict(modelFitRF, newdata = testset))
