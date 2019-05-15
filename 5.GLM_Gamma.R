### GLM Gamma


###############################################

######### Author : San Lorenzo Marino 

######### Version beginning of April

###############################################


###############################################

#########  load packages and data

###############################################
library(stats)
library(ggpubr)
library(magrittr)
library(data.table)
library(bigmemory)
library(dplyr)
library(psych)
library(base)
library(dplyr)
library(Hmisc)
library(caret)
library(keras)
library(tensorflow)
library(Rtsne) ##### Wrapper for the C++ implementation of Barnes-Hut t-Distributed Stochastic Neighbor Embedding. t-SNE is a method for constructing a low dimensional embedding of high-dimensional data, distances or similarities.
##### Exact t-SNE can be computed by setting theta=0.0.
#C-index
library(devFunc)
library(R2DT)


#The tfruns package provides a suite of tools for tracking, visualizing, and managing TensorFlow training runs and experiments from R:
#Track the hyperparameters, metrics, output, and source code of every training run.
#Compare hyperparmaeters and metrics across runs to find the best performing model.
#Automatically generate reports to visualize individual training runs or comparisons between runs.
#No changes to source code required (run data is automatically captured for all Keras and TF Estimator models).

library(tfruns)  
library(tfestimators)
library(yaml)

library(readxl)
library(DataExplorer)

training_set <- read.csv("C:/Machine Learning/Pricing Game/pg2019/training_set.csv")




################################################

#########  feature pre-processing for NN

###############################################

Data_GLM <- training_set
Data_GLM = Data_GLM %>% data.table


Data_GLM[,Cover     :=as.integer(as.factor(Cover))-1]
Data_GLM[,Split     :=as.integer(as.factor(Split))-1]
Data_GLM[,AgePh    :=as.integer(as.factor(cut2(AgePh,m=10000)))-1]
Data_GLM[,AgeCar    :=as.integer(as.factor(cut2(AgeCar,m=10000)))-1]
Data_GLM[,Power    :=as.integer(as.factor(cut2(Power,m=10000)))-1]
Data_GLM[,Fuel      :=as.integer(as.factor(Fuel))-1]
Data_GLM[,Gender     :=as.integer(as.factor(Gender))-1]
Data_GLM[,Use     :=as.integer(as.factor(Use))-1]
Data_GLM[,Latitude    :=as.integer(as.factor(cut2(Latitude,m=10000)))-1]
Data_GLM[,Longitude    :=as.integer(as.factor(cut2(Longitude,m=10000)))-1]




'set.seed(594)
splits = sample (c (1: nrow ( Data_GLM)), round (0.9* nrow ( Data_GLM)), replace = FALSE )

#setdiff removes the duplicates

learn = Data_GLM[splits ,]
test = Data_GLM[ setdiff (c (1: nrow (Data_GLM)), splits ),]

### Check

learn[,.N]+test[,.N]-Data_GLM[,.N]'

### Fit NN using Keras
require(keras)

glmGamma <- glm(Cclaim / Nclaim ~   Power  + Latitude,  
                             data = training_set[training_set$Nclaim>0,],
                             family=Gamma(link="log"))
summary(glmGamma)

training_set$fit <- fitted (glmGamma )


###############################################

#########  Predictions

###############################################

testing_set <- read.csv("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/testing_set.csv")


Test_embed_2 <- testing_set[,-which(names(testing_set) %in% c("Id_Policyholder"))]

Test_embed_2 = Test_embed %>% data.table

Test_embed_2[,Cover     :=as.integer(as.factor(Cover))-1]
Test_embed_2[,Split     :=as.integer(as.factor(Split))-1]
Test_embed_2[,AgePh    :=as.integer(as.factor(cut2(AgePh,m=10000)))-1]
Test_embed_2[,AgeCar    :=as.integer(as.factor(cut2(AgeCar,m=10000)))-1]
Test_embed_2[,Power    :=as.integer(as.factor(cut2(Power,m=10000)))-1]
Test_embed_2[,Fuel      :=as.integer(as.factor(Fuel))-1]
Test_embed_2[,Gender     :=as.integer(as.factor(Gender))-1]
Test_embed_2[,Use     :=as.integer(as.factor(Use))-1]
Test_embed_2[,Latitude    :=as.integer(as.factor(cut2(Latitude,m=10000)))-1]
Test_embed_2[,Longitude    :=as.integer(as.factor(cut2(Longitude,m=10000)))-1]

Nombre <- c(1:nrow(testing_set))

x = list(
         Power  = power,
         Latitude = latitude)

pred_testing_set_GLM <- predict(glmGamma, Test_embed_2, type = "response")




pred_testing_set_GLM_table = data.table(Id_Policyholder = Nombre, CClaim = pred_testing_set_GLM  )

pred_testing_set_GLM_table  %>% fwrite("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/Severity.csv")
