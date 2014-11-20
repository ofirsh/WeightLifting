library(caret)
set.seed(200801)

# train <- read.table("data/pml-training.csv",sep=",",header=TRUE,colClasses=c("numeric","character","numeric","numeric","Date","logical",rep("numeric",154)))
train.raw <- read.table("data/pml-training.csv",sep=",",header=TRUE,colClasses=c("character","character","numeric","numeric","Date","character",rep("character",154)))

# test set don't contain the column named "classe" (but "problem_id"), all other points are the same 
# instead we will split the training set to create a train and test sets
submission.raw <- read.table("data/pml-testing.csv",sep=",",header=TRUE)

nTrainNA <- apply( train.raw,2,function(x) sum(is.na(x) | x == "" ) )
# column names where there are more than 15,000 missing values 
almostEmptyColumns <- names(which(nTrainNA > 15000))

# after removing the almost-empty columns and the first 6 columns, left with 54 columns (X, timestamp etc), including the "classe" (y)
train.clean <- train.raw[,!( names(train.raw) %in% almostEmptyColumns ) ][,-c(1:6)]

# remove from the submission the same columns as the one removed from the training, including the last column which is problem_id
submission <- submission.raw[,!( names(submission.raw) %in% almostEmptyColumns ) ][,-c(1:6)][,-54]

inTrain <- createDataPartition(y=train.clean$classe,p=0.75,list=FALSE)
training <- train.clean[inTrain,]
testing <- train.clean[-inTrain,]

modelFitGlm <- train(classe ~ ., data=training,method = "glm")

# modelFitGlm$finalModel

predictions <- predict(modelFitGlm, newdata = testing)
confusionMatrix(predictions,testing$classe)


