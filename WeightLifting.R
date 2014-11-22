library(caret)
set.seed(200801)

train.raw <- read.table("data/pml-training.csv",sep=",",header=TRUE,colClasses=c("character","character","numeric","numeric","Date","character",rep("character",154)))

# test set don't contain the column named "classe" (but "problem_id"), all other points are the same 
# instead we will split the training set to create a train and test sets
submission.raw <- read.table("data/pml-testing.csv",sep=",",header=TRUE)

nTrainNA <- apply( train.raw,2,function(x) sum(is.na(x) | x == "" ) )
# column names where there are more than 15,000 missing values 
almostEmptyColumns <- names(which(nTrainNA > 15000))

# after removing the almost-empty columns and the first 6 columns, left with 54 columns (X, timestamp etc), including the "classe" (y)
train.clean.factor <- train.raw[,!( names(train.raw) %in% almostEmptyColumns ) ][,-c(1:6)]
train.clean <- data.frame( data.matrix(train.clean.factor[,-54]), train.clean.factor[,54])
names(train.clean)[54] <- "class"

# remove from the submission the same columns as the one removed from the training, including the last column which is problem_id
submission.factor <- submission.raw[,!( names(submission.raw) %in% almostEmptyColumns ) ][,-c(1:6)][,-54]
submission <- data.frame( data.matrix( submission.factor ) )

inTrain <- createDataPartition(y=train.clean$class,p=0.75,list=FALSE)
training <- train.clean[inTrain,]
testing <- train.clean[-inTrain,]

prComp <- prcomp(training[,-54])

plot(cumsum(prComp$sdev^2/sum(prComp$sdev^2))) 
# from the plot it's clear that 10 PCA vectors will cover more tha 90% of the deviation

preProc <- preProcess( training[,-54], method="pca", pcaComp = 10)

trainingPC <- predict(preProc,training[,-54])
trainingPCy <- data.frame(trainingPC, training$class)
names(trainingPCy)[11] <- "class"

# rpart ~30% accuracy

# modelFitTree <- train(class ~ ., method = "rpart", data=trainingPCy)
modelFitTree <- train(class ~ ., method = "rf", data=trainingPCy)

# plot(modelFitTree$finalModel, uniform = TRUE, main = "Classification Tree")
# text(modelFitTree$model, use.n = TRUE, cex=.8)

testPC <- predict(preProc,testing[,-54])
predictions <- predict(modelFitTree, testPC)
confusionMatrix(predictions,testing$class)

# Overall Statistics
# 
# Accuracy : 0.9617          
# 95% CI : (0.9559, 0.9669)
# No Information Rate : 0.2845          
# P-Value [Acc > NIR] : < 2.2e-16       

# now applying the model on the submission set
submissionPC <- predict(preProc,submission)
answers <- predict(modelFitTree,submissionPC)


# answers = rep("A", 20)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
