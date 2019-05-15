
###############################################

#########  load packages and data

###############################################

library(DMwR)

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


library(SMOTE)
library(dplyr)
library(data.table)

require(plyr)

######################################
###### Define function to duplicate rows
######################################

rep.row <- function(r, n){
  colwise(function(x) rep(x, n))(r)
}

###################
###### Import data
###################

path <- "C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/training_set.csv"
training_set <- read.csv(path)

training_set_1 <- training_set %>% filter(Nclaim >1) 

###################
###### Splitting 4
###################

pol_4 <- training_set_1 %>% filter(Nclaim >3) 


pol_4_split <- rep.row(pol_4,pol_4$Nclaim)

pol_4_split$Nclaim <- 1

###################
###### Splitting 3
###################

pol_3 <- training_set_1 %>% filter(Nclaim > 2 , Nclaim <4) 

pol_3_split <- rep.row(pol_3,pol_3$Nclaim)

pol_3_split$Nclaim <- 1

###################
###### Splitting 2
###################

pol_2 <- training_set_1 %>% filter(Nclaim > 1 , Nclaim <3) 

pol_2_split <- rep.row(pol_2,pol_2$Nclaim)

pol_2_split$Nclaim <- 1


###################
###### Binding
###################

training_set_0 <- training_set %>% filter(Nclaim <2) 

training_set_new <- rbind(training_set_0, pol_2_split, pol_3_split, pol_4_split)

###################
###### CHECKS
###################
check <- function(split, split_1) {
  if( (nrow(split) == sum(split$Nclaim) ) & ( sum(split_1$Nclaim) == nrow(split))){
  print("ok")}
    else{
      print("not ok")
    }

}

check(pol_4_split, pol_4)
check(pol_3_split, pol_3)
check(pol_2_split, pol_2)

sum(training_set$Nclaim) - sum(training_set_new$Nclaim)

set.seed(1029)
## Remove rows that do not have target variable values

final <- training_set_new[!(is.na(training_set_new$Nclaim)),]

final$Nclaim <- factor(final$Nclaim)

library(caTools)

split <- sample.split(final$Nclaim, SplitRatio = 0.75)

dresstrain <- subset(final, split == TRUE)
dresstest <- subset(final, split == FALSE)


## Let's check the count of unique value in the target variable
as.data.frame(table(dresstrain$Nclaim))

## Loading DMwr to balance the unbalanced Nclaim
library(DMwR)



## Smote : Synthetic Minority Oversampling Technique To Handle Nclaim Imbalancy In Binary Nclaimification
balanced.data <- SMOTE(Nclaim ~., dresstrain, perc.over = 4800, k = 5, perc.under = 1000)


as.data.frame(table(balanced.data$Nclaim))

