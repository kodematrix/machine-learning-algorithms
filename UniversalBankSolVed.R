library(caret)
library(ISLR)
library(FNN)
library(gmodels)
UniversalBank<-read.csv("C:/Users/suman/Documents/Machine Learning/class_work_probs/UniversalBank.csv")
Bank<-UniversalBank[,c(-1,-5)]
str(Bank)

#creating dummies
library(dummies)
dummy_model <- dummyVars(~Education,data=Bank)
head(predict(dummy_model,Bank))
UBank<- dummy.data.frame(Bank, names = c("Education"), sep= ".")

#normalize the data first: build a model and apply 
norm_model<-preProcess(UBank, method = c('range'))
UBank_normalized<-predict(norm_model,UBank)
UBank_Predictors<-UBank_normalized[,-10]
UBank_labels<-UBank_normalized[,10]


set.seed(15)
inTrain = createDataPartition(UBank_normalized$Personal.Loan,p=0.6, list=FALSE) 
Train_Data = UBank_normalized[inTrain,]
Val_Data = UBank_normalized[-inTrain,]
dim(Train_Data)
summary(Train_Data)
summary(Val_Data)


Train_Predictors<-Train_Data[,-10]
Val_Predictors<-Val_Data[,-10]

Train_labels <-Train_Data[,10] 
Val_labels  <-Val_Data[,10]

Train_labels=as.factor(Train_labels)
Val_labels=as.factor(Val_labels)
UBank_labels<-as.factor(UBank_labels)

#Knn method where k=1
knn.pred <- knn(Train_Predictors,Val_Predictors,cl=Train_labels,k=1,prob = TRUE)
knn.pred

Q1 <- c(40, 10, 84, 2, 2, 0, 1, 0, 0, 0, 0, 1, 1)
knn.pred1 <- knn(Train_Predictors, Q1, cl=Train_labels, k=1, prob = TRUE)
knn.pred1

#2.	What is a choice of k that balances between overfitting and ignoring the predictor information?
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

for(i in 1:14) {
                  knn <- knn(Train_Predictors, Val_Predictors, cl = Train_labels, k = i)
                  accuracy.df[i, 2] <- confusionMatrix(knn, Val_labels)$overall[1] 
                }
accuracy.df

which.max( (accuracy.df$accuracy) )
#3.	Show the confusion matrix for the validation data that results from using the best k.

knn.pred3 <- knn(Train_Predictors,Val_Predictors,cl=Train_labels,k=3,prob = TRUE)
confusionMatrix(knn.pred3,Val_labels)

#4.	Consider the following customer: Classify the customer using the best k

knn.pred4 <- knn(Train_Predictors, Q1, cl=Train_labels, k=3, prob = TRUE)
knn.pred4

knn.pred4 <- knn(UBank_Predictors, Q1, cl=UBank_labels, k=3, prob = TRUE)
knn.pred4

#5.	Repartition the data, this time into training, validation, and test sets (50% : 30% : 20%)


set.seed(15)
Bank_Partition = createDataPartition(UBank_normalized$Personal,p=0.5, list=FALSE) 
Training_Data = UBank_normalized[Bank_Partition,]     #50% of total data assigned to Test data
Test_Valid_Data = UBank_normalized[-Bank_Partition,]

Test_Index = createDataPartition(Test_Valid_Data$Personal.Loan, p=0.6, list=FALSE) 
Validation_Data = Test_Valid_Data[Test_Index,]    #to achieve 50:30:20 ratio among the... 
#.......train, validation and testing, i partioned 60% test_valid_data to test and train
Test_Data = Test_Valid_Data[-Test_Index,] 


Training_Predictors<-Training_Data[,-10]
Test_Predictors<-Test_Data[,-10]
Validation_Predictors<-Validation_Data[,-10]


Training_labels <-Training_Data[,10]
Test_labels <-Test_Data[,10]
Validation_labels <-Validation_Data[,10]


Training_labels=as.factor(Training_labels)
Test_labels<-as.factor(Test_labels)
Validation_labels=as.factor(Validation_labels)

knn.pred5 <- knn(Training_Predictors, Test_Predictors , cl=Training_labels, k=3, prob = TRUE)
knn.pred5
confusionMatrix(knn.pred5,Test_labels)

knn.pred6 <- knn(Validation_Predictors, Test_Predictors, cl=Validation_labels, k=3, prob = TRUE)
knn.pred6
confusionMatrix(knn.pred6,Test_labels)

#Compare the confusion matrix of the test set with that of the training and validation sets. Comment on the differences and their reason.


#             Accuracy : 0.959 for knn.pred5 i.e., for Training set
#             Accuracy : 0.951 for knn.pred6 i.e., for Validation set

# Reason: The Model performs better gives you better accuracy when feed more data to the model. In the above case, the Training set has more data compared to validation set. hence, Accuracy has improved.










