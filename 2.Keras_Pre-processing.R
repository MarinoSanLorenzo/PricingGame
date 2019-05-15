
###############################################

#########  Deep Learning : Neural Networks using pre-processed data by embeddings

######### Author : San Lorenzo Marino 

######### Inspired by : Ronald Richman

######### Version beginning of April

###############################################

###############################################

#########  Enable Keras to Run on GPU and use tensorboard

######### Prior Python, TensorFlow , CUDA, NumpY installation

######### Need NVIDIA GPU 

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

path <- "C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/"
training_set <- read.csv(paste0(path, "training_set.csv", sep=""))


####################################################################

#########  Define Poisson Deviance in Keras as a loss function

####################################################################

pDev <- function(y_true, y_pred){
  2 * k_mean(y_pred - y_true - y_true * (k_log(k_clip(y_pred, k_epsilon(), NULL)) - k_log(k_clip(y_true, k_epsilon(), NULL))), axis = 2)
}


################################################

#########  feature pre-processing for NN

###############################################



data_embed <- training_set[,-which(names(training_set) %in% c("Cclaim"))]

########## EXPERT JUDGMENT ########## 

data_embed <- training_set[training_set$Nclaim<4,]

Data_embed = data_embed %>% data.table


attributes(Data_embed)$names

### prepare data for Embeddings 






Data_embed[,CoverNN     :=as.integer(as.factor(Cover))-1]
Data_embed[,SplitNN     :=as.integer(as.factor(Split))-1]
Data_embed[,AgePhNN    :=as.integer(as.factor(cut2(AgePh,m=1000)))-1]
Data_embed[,AgeCarNN    :=as.integer(as.factor(cut2(AgeCar,m=1000)))-1]
Data_embed[,PowerNN    :=as.integer(as.factor(cut2(Power,m=1100)))-1]
Data_embed[,FuelNN      :=as.integer(as.factor(Fuel))-1]
Data_embed[,GenderNN     :=as.integer(as.factor(Gender))-1]
Data_embed[,UseNN     :=as.integer(as.factor(Use))-1]
Data_embed[,LatitudeNN    :=as.integer(as.factor(cut2(Latitude,m=1000)))-1]
Data_embed[,LongitudeNN    :=as.integer(as.factor(cut2(Longitude,m=1000)))-1]




### prepare dataframe for embeddings


cover_cats = Data_embed[,c("CoverNN","Cover"),with=F] %>% unique %>% setkey(CoverNN)
split_cats = Data_embed[,c("SplitNN","Split"),with=F] %>% unique %>% setkey(SplitNN)
ageph_cats    = Data_embed[,.(AgePh = mean(AgePh)), keyby = AgePhNN]
agecar_cats    = Data_embed[,.(AgeCar = mean(AgeCar)), keyby = AgeCarNN]
power_cats    = Data_embed[,.(Power = mean(Power)), keyby = PowerNN]
fuel_cats = Data_embed[,c("FuelNN","Fuel"),with=F] %>% unique %>% setkey(FuelNN)
gender_cats = Data_embed[,c("GenderNN","Gender"),with=F] %>% unique %>% setkey(GenderNN)
use_cats = Data_embed[,c("UseNN","Use"),with=F] %>% unique %>% setkey(UseNN)
latitude_cats    = Data_embed[,.(Latitude = mean(Latitude)), keyby = LatitudeNN]
longitude_cats    = Data_embed[,.(Longitude = mean(Longitude)), keyby = LongitudeNN]


ageph_dim = ageph_cats[,.N]
agecar_dim = agecar_cats[,.N]
power_dim = power_cats[,.N]
latitude_dim = latitude_cats[,.N]
longitude_dim = longitude_cats[,.N]


embedding_dat <- list(Cover_embed = cover_cats,
                       Split_embed = split_cats,
                       AgePh_embed = ageph_cats,
                       AgeCar_embed = agecar_cats,
                       Power_embed = power_cats,
                       Fuel_embed = fuel_cats,
                       Gender_embed = gender_cats,
                       Use_embed = use_cats,
                       Latitude_embed = latitude_cats,
                       Longitude_embed = longitude_cats  )




set.seed(594)
splits = sample (c (1: nrow ( Data_embed)), round (0.9* nrow ( Data_embed)), replace = FALSE )

#setdiff removes the duplicates

learn = Data_embed[splits ,]
test = Data_embed[ setdiff (c (1: nrow (Data_embed)), splits ),]

### Check

learn[,.N]+test[,.N]-Data_embed[,.N]

### Fit NN using Keras
require(keras)

### Munge data into correct format for embeddings - Train set

cover = learn$CoverNN
split = learn$SplitNN
ageph =  learn$AgePhNN
agecar = learn$AgeCarNN
power = learn$PowerNN
fuel = learn$FuelNN
gender = learn$GenderNN
use =learn$ UseNN
latitude = learn$LatitudeNN
longitude = learn$LongitudeNN


x_train = list(Cover = cover,
         Split = split,
         AgePh = ageph,
         AgeCar = agecar,
         Power  = power,
         Fuel = fuel,
         Gender = gender,
         Use = use,
         Latitude = latitude,
         Longitude = longitude)

y_train = list(N = learn$Nclaim)

### Munge data into correct format for embeddings - Test set

cover = test$CoverNN
split = test$SplitNN
ageph =  test$AgePhNN
agecar = test$AgeCarNN
power = test$PowerNN
fuel = test$FuelNN
gender = test$GenderNN
use =test$ UseNN
latitude = test$LatitudeNN
longitude = test$LongitudeNN


x_test = list(Cover = cover,
         Split = split,
         AgePh = ageph,
         AgeCar = agecar,
         Power  = power,
         Fuel = fuel,
         Gender = gender,
         Use = use,
         Latitude = latitude,
         Longitude = longitude)

y_test = list(N = test$Nclaim)



