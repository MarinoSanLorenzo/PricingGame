
###############################################

#########  Data Pre-processing : PCA + Dumify (manually no recipe object used) 

######### Deep and Machine Learning

######### Author : San Lorenzo Marino and Siham

######### Inspired by : CRAN PCA project +  Ronald Richman

######### Version mid-Nobember

###############################################

#########  Enable Keras to Run on GPU and use tensorboard

######### Prior Python, TensorFlow , CUDA, NumpY installation

######### Need NVIDIA GPU 

###############################################


'install.packages("keras")
library(keras)

install_keras(method = "conda")
# install with GPU version of TensorFlow
# (NOTE: only do this if you have an NVIDIA GPU + CUDA!)  
install_keras(tensorflow = "gpu") # install a specific version of TensorFlow
dir.create("C:/Machine Learning/NN_VIE_2/Projet/2018/1.Codes/LogFilesTensorBoard") #create a directory where to store the log files it generates.
tensorboard("C:/Machine Learning/NN_VIE_2/Projet/2018/1.Codes/LogFilesTensorBoard")'

###############################################

#########  load packages and data

###############################################



library(magrittr)
library(psych)
library(keras)
library(tidyr)
library(ggplot2)
library(lattice)
library(sp)
library(caret)
library(dplyr)
library(base)
library(stats)
library(data.table)
library(onehot)
library(Matrix)
library(h2o)
library(tidyverse)
library(recipes)     # Library for dat.mla processing
library(glue)        # For conveniently concatenating strings
library(zeallot)     # for %<-% operator
library(bigmemory)
library(tensorflow)


data.pca <- as.data.frame(db_train)

#### see variables #########




###############################################

#########  Feature Engineering

###############################################


#############" remove variables deemed insignificant to downsize the dataset ############""

data.pca <- data.pca[,-c(2:6,8,11,22,26,27,30,32:38,40)]

data.pca2 <- db_train[,-which(names(db_train) %in% 
                                c("date_obs",
                                  "num_pol",
                                  "num_risk",
                                  "num_avenant",
                                  "date_starting",
                                  "date_final"))]

attach(data.pca)



#remove exposure#######7



NumDT <- data.pca[,-c(1:7,13:16,20:22)]

NumDT2 <- data.pca2[,which(names(data.pca2) %in%
                             c("driv_age1", "driv_exp1", "driv_age2", "driv_exp2", 
           "veh_age", "veh_power", "veh_weight", "veh_sportivity",
           "veh_value", "geo_postcode_lat", "geo_postcode_lon"))]


'plot_correlation(NumDT)'

NumDT<-NumDT[,-c(5,9)]
attributes(NumDT)$names

#remove drive age young + claims 



###############################################

#########  PCA

###############################################

pca.0 <- prcomp(NumDT,
                center = TRUE,
                scale. = TRUE)



print(pca.0)
summary(pca.0)

pca.2 <- prcomp(NumDT2,
                center = TRUE,
                scale. = TRUE)



print(pca.2)
summary(pca.2)



#######═ stocke les valeurs de la PCA ===> dimension réduite #############

PCA.DT <- predict(pca.0,NumDT)


PCA.DT2 <- predict(pca.2,NumDT2)

#######═ réintroduit les valeurs categorielles que l'on va dumifier  #############


newdata <- data.frame(PCA.DT[,c(1:5)],  data.pca[,c(1:7,13:16,20:23)])


###############################################

#########  Dumify categorical variable

###############################################


#####  PCA FOR NUMERICAL VARIABLES AND THEN DUMIFY FOR CATEGORICAL ################

###### remove model + febiac #########


NewDT<- newdata[,-c(14,16)]

NewDT2 <- data.frame(PCA.DT2[,c(1:6)])

NewDT3 <- cbind.data.frame(PCA.DT2[,c(1:6)],db_train[,which(names(db_train) %in%
                                                               c("channel", "insured_full", "guarantees_type", 
                                                                   "premiums_periodicity", "domiciliation", 
                                                                  "telematic", "driv_number", "driv_age_young",
                                                                   "veh_brand","veh_type",
                                                                   "veh_sgmt_informex", "veh_sgmt_febiac", 
                                                                   "veh_seats", "veh_fuel", "veh_mileage_limit",
                                                                   "veh_trailer", "veh_garage",
                                                                    "geo_province_fr",
                                                                   "geo_mosaic_code_full", "geo_mosaic_code","exposure","claim_nb"))])
 

#####dumify categorical variables  using Keras package ##############################"

form <- claim_nb + exposure ~ (channel + insured_full + guarantees_type + 
                                 domiciliation + 
                                 telematic + driv_number +
                                 veh_brand + 
                                 veh_sgmt_informex  +
                                 veh_fuel  +
                                 geo_province_fr + geo_mosaic_code)

form2 <- claim_nb + exposure ~ (channel + insured_full + guarantees_type + 
                                  premiums_periodicity + domiciliation + 
                                 telematic + driv_number + driv_age_young +
                                 veh_brand +  veh_type +
                                 veh_sgmt_informex  + veh_sgmt_febiac + 
                                  veh_seats + veh_fuel  + veh_mileage_limit +
                                  veh_trailer + veh_garage + 
                                  geo_province_fr +
                                  geo_mosaic_code_full + geo_mosaic_code)
require(caret)

dummyVars <- dummyVars(form, data = NewDT)


oneHotTrain <- data.table(predict(dummyVars, newdata = NewDT))

dummyVars2 <- dummyVars(form2, data = NewDT3)

#remove veh_model and geopostcode + munidipality_fr so that the onehot encoding function can proceed the size of the vector

oneHotTrain2 <- data.table(predict(dummyVars2, newdata = NewDT3))

###############################################

#########  Dumify categorical variable

###############################################

#PCA + Dumify with feature engineering

PCA <- newdata[,c(1:5)]


logexposure <- log(newdata[,c("exposure")])
exposure <- newdata[,c("exposure")]


claim_nb <- newdata[,c("claim_nb")]


NewData.pca_dumFe = cbind.data.frame(PCA,oneHotTrain,logexposure,claim_nb)


#PCA + continous vector without feature engineering 

db_train = db_train %>% data.table

db_train[,ChannelGLM    :=as.integer(as.factor(channel))-1]
db_train[,GuaranteeGLM     :=as.integer(as.factor(insured_full))-1]
db_train[,GuaranteeTypeGLM     :=as.integer(as.factor(guarantees_type))-1]
db_train[,PremiumPeriodicityGLM     :=as.integer(as.factor(premiums_periodicity))-1]
db_train[,DomiciliatioGLMN     :=as.integer(as.factor(domiciliation))-1]
db_train[,TelematicGLM     :=as.integer(as.factor(telematic))-1]
db_train[,DrivNbGLM      :=as.integer(as.factor(driv_number))-1]
db_train[,DrivYoungGLM      :=as.integer(as.factor(driv_age_young))-1]
db_train[,VehBrandGLM      :=as.integer(as.factor(veh_brand))-1]
db_train[,VehModelGLM     :=as.integer(as.factor(veh_model))-1]
db_train[,VehTypeGLM      :=as.integer(as.factor(veh_type))-1]
db_train[,InformexGLM      :=as.integer(as.factor(veh_sgmt_informex))-1]
db_train[,FebiacGLM      :=as.integer(as.factor(veh_sgmt_febiac))-1]
db_train[,VehSeatsGLM      :=as.integer(as.factor(veh_seats))-1]
db_train[,VehFuelGLM      :=as.integer(as.factor(veh_fuel))-1]
db_train[,VehMileageLimitGLM      :=as.integer(as.factor(veh_mileage_limit))-1]
db_train[,VehTrailerGLM      :=as.integer(as.factor(veh_trailer))-1]
db_train[,VehGarageGLM      :=as.integer(as.factor(veh_garage))-1]
db_train[,GeoPostcodeGLM      :=as.integer(as.factor(geo_postcode))-1]
db_train[,GeoMunicipalityGLM      :=as.integer(as.factor(geo_municipality_fr))-1]
db_train[,GeoProvinceGLM      :=as.integer(as.factor(geo_province_fr))-1]
db_train[,GeoMosaicFullGLM      :=as.integer(as.factor(geo_mosaic_code_full))-1]
db_train[,GeoMosaicGLM      :=as.integer(as.factor(geo_mosaic_code))-1]


db_train <- as.data.frame(db_train)

NewData.pca2_contvec <- cbind.data.frame(NewDT2,db_train[,c(43:65)],logexposure,claim_nb)

NewData.pca2_dumFE <- cbind.data.frame(NewDT2,oneHotTrain,logexposure,claim_nb)

NewData.pca2_dumnoFe <- cbind.data.frame(NewDT2,oneHotTrain2,exposure,claim_nb)

PCA_Dum_Fe <- NewData.pca_dumFe
PCA2_Vec_NoFe <- NewData.pca2_contvec
PCA2_Dum_Fe <- NewData.pca2_dumFE
PCA2_Dum_NoFe <- NewData.pca2_dumnoFe

###############################################

#########  Pre-processing Data for H2o

###############################################

#1
ll_PCA_Dum_Fe <- sample(1:nrow(PCA_Dum_Fe), round(0.9 * nrow(PCA_Dum_Fe)), replace = FALSE)
learn_PCA_Dum_Fe  <- PCA_Dum_Fe[ll_PCA_Dum_Fe, ] 
test_PCA_Dum_Fe  <- PCA_Dum_Fe[-ll_PCA_Dum_Fe, ]
#2
ll_PCA2_Vec_NoFe <- sample(1:nrow(PCA2_Vec_NoFe), round(0.9 * nrow(PCA2_Vec_NoFe)), replace = FALSE)
learn_PCA2_Vec_NoFe <- PCA2_Vec_NoFe[ll_PCA2_Vec_NoFe, ] 
test_PCA2_Vec_NoFe <- PCA2_Vec_NoFe[-ll_PCA2_Vec_NoFe, ]
#3
ll_PCA2_Dum_Fe <- sample(1:nrow(PCA2_Dum_Fe), round(0.9 * nrow(PCA2_Dum_Fe)), replace = FALSE)
learn_PCA2_Dum_Fe <- PCA2_Dum_Fe[ll_PCA2_Dum_Fe, ] 
test_PCA2_Dum_Fe <- PCA2_Dum_Fe[-ll_PCA2_Dum_Fe, ]
#4
ll_PCA2_Dum_NoFe <- sample(1:nrow(PCA2_Dum_NoFe), round(0.9 * nrow(PCA2_Dum_NoFe)), replace = FALSE)
learn_PCA2_Dum_NoFe <- PCA2_Dum_NoFe[ll_PCA2_Dum_NoFe, ] 
test_PCA2_Dum_NoFe <- PCA2_Dum_NoFe[-ll_PCA2_Dum_NoFe, ]


learn <- PCA2_Dum_NoFe
test <- PCA2_Dum_NoFe

###############################################

#########  Starting H2o cluster

###############################################

h2o.init(nthreads = 12, port = 11223, max_mem_size = "16G") # Use 6 CPUs and custom port i7 8750h cpu

h2o.no_progress()                    # Disable progress bars for nicer output


learn.h2o <- as.h2o(learn)   # Upload data to h2o
learn.h2o <- learn.h2o[-1,]

test.h2o <- as.h2o(test)
test.h2o <- test.h2o[-1,]


x <- setdiff(colnames(learn.h2o), c("claim_nb", "logexposure")) 

y <- "claim_nb"      # Target variable

offset <- "logexposure"  # log(exposure)

###############################################

#########  Poissong GLM

###############################################


glm_fit2 <- h2o.glm(
  
  x = x, 
  
  y = y,                                          
  
  offset_column = offset,
  
  training_frame = learn.h2o,
  
  validation_frame = test.h2o,
  
  
  
  family = "poisson",
  
  nfolds = 5, # 5 fold cross-validation
  
  seed = 1    # For reproducibility         
  
)

###############################################

#########  Boosting Machines

###############################################


gbm_fit2 <- h2o.gbm(
  
  x = x, 
  
  y = y,                                          
  
  offset_column = offset,
  
  training_frame = learn.h2o,
 
  
  ntrees = 500,
  
  max_depth = 5,
  
  verbose = TRUE,
  
  distribution = "poisson",
  
  nfolds = 5,
  
  keep_cross_validation_predictions = TRUE,
  
  seed = 1) # For reproducibility
  




#Note that all the hyperparameters have been kept at their default values. Unfortunately, the `mean_residual_deviance` cannot be compared directly, since the GBM model in H2O uses another definition of deviance. We can, however, compute the deviance with a function (this is done on the H2O cluster and not in the R session):


get_deviance <- function(y_pred, y_true) {
  
  2 * (sum(y_pred) - sum(y_true) + sum(log((y_true / y_pred) ^ (y_true)))) / nrow(y_pred)
  
}

# Predict on various sets

pred_learn <- predict(gbm_fit2, learn.h2o)$predict


pred_test <- predict(gbm_fit2, test.h2o)$predict

###############################################

#########  Boosting machine autogrid

###############################################

strategy <- list(strategy = "RandomDiscrete",
                 
                 max_runtime_secs = 10800,
                 
                 seed = 1)


# Define grid

gbm_params <- list(learn_rate = seq(0.001, 0.3, 0.001),
                   
                   max_depth = seq(2, 10),
                   
                   sample_rate = c(0.8, 0.9, 1.0),
                   
                   col_sample_rate = seq(0.1, 1.0, 0.1))



# Launch grid search

gbm_grid <- h2o.grid(
  
  "gbm",
  
  x = x,
  
  y = y,
  
  offset_column = offset,
  
  distribution = "poisson",
  
  
  
  training_frame = learn.h2o,
  
  nfolds = 5,
  
  hyper_params = gbm_params,
  
  search_criteria = strategy,
  
  
  
  seed = 1,
  
  ntrees = 10000,
  
  stopping_rounds = 5,           # Early stopping
  
  stopping_tolerance = 0.001,
  
  stopping_metric = "deviance"
  
)

```



After running this grid search, let us examine the results.

gbm_grid


best_gbm <- h2o.getModel(gbm_grid@model_ids[[1]]) # Extract best model from grid

summary(best_gbm) # Show summary


Finally, let us calculate the performance of this model on the test set:
  
  
  
  ```{r}



pred_test_t <- predict(best_gbm, test.h2o)$predict


out_of_sample_t <- get_deviance(pred_test_t, test.h2o$claim_nb)


cat(glue("Out-of-sample deviance: {signif(out_of_sample_t, 3)}"))

###############################################

#########  NN with Keras

###############################################


ll_PCA_Dum_Fe <- sample(1:nrow(PCA2_Dum_NoFe), round(0.9 * nrow(PCA2_Dum_NoFe)), replace = FALSE)
learn_PCA_Dum_Fe  <- PCA2_Dum_NoFe[ll_PCA_Dum_Fe, ] 
test_PCA_Dum_Fe  <- PCA2_Dum_NoFe[-ll_PCA_Dum_Fe, ]

Inputs_Train = learn_PCA_Dum_Fe[,-which(names(learn_PCA_Dum_Fe) %in% c("exposure","claim_nb"))] %>% as.matrix
Inputs_Test = test_PCA_Dum_Fe[,-which(names(learn_PCA_Dum_Fe) %in% c("exposure","claim_nb"))] %>% as.matrix

x_train = list(NumInputs=Inputs_Train, exposure = learn_PCA_Dum_Fe$exposure)
x_test = list(NumInputs=Inputs_Test, exposure = test_PCA_Dum_Fe$exposure)

y_train = list(N = learn_PCA_Dum_Fe$claim_nb)
y_test=list(N = test_PCA_Dum_Fe$claim_nb)

##################################################################

######### Building the NN architecture for Poisson Regression

##################################################################


numinput <- ncol(Inputs_Train)

NumInputs <- layer_input(shape = c(173), dtype = 'float32', name = 'NumInputs')
exposure <- layer_input(shape = c(1), dtype = 'float32', name = 'exposure')

dense = NumInputs %>%  
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(0.05) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(0.05) %>%
  layer_dense(units = 45, activation = 'relu') %>% 
  layer_dropout(0.05) %>%
  layer_dense(units = 10, activation = 'relu') %>% 
  layer_dropout(0.05) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

#Layer that multiplies (element-wise) a list of inputs.
#It takes as input a list of tensors, all of the same shape, and
#returns a single tensor (also of the same shape).

N <- layer_multiply(list(dense,exposure), name = 'N')

model <- keras_model(
  inputs = c(NumInputs,exposure), 
  outputs = c(N))

###define the Poisson Deviance Loss Function

pDev <- function(y_true, y_pred){
  2 * k_mean(y_pred - y_true - y_true * (k_log(k_clip(y_pred, k_epsilon(), NULL)) - k_log(k_clip(y_true, k_epsilon(), NULL))), axis = 2)
}

###Run



model %>% compile(
  optimizer = "nadam",
  loss = function(y_true, y_pred) pDev(y_true, y_pred))

###############################################

#########  Running the moodel

###############################################

fit = model %>% fit(
  x = x_train,
  y = y_train, 
  validation_data = list(x_test, y_test) ,
  epochs = 10,
  batch_size = 256,verbose = 1, shuffle = T)

plot(fit)


learn_PCA_Dum_Fe$NN_no_FE = model %>% predict(x_train)
test_PCA_Dum_Fe$NN_no_FE = model %>% predict(x_test)

###############################################

######### 
#########  In and Out Sample Losses 

###############################################


##### C index

Cindex <- function(y_pred, y_obs){
 for (i in nrow(data)){ if y_pred[i] > y_pred[i+1]
   
 }
}

#GLM_PCA2_Vec_NoFe

cat(glue("In-sample deviance: {signif(h2o.mean_residual_deviance(glm_fit2, train = TRUE), 3)}"))
glm3H2o.insample <- 0.199*100
  cat(glue("Out-of-sample deviance: {signif(h2o.mean_residual_deviance(glm_fit2, valid = TRUE), 3)}"))
glm3H2o.oob <- 0.204*100
  
  results2 = rbind(results2,
                  data.table(Model = "GLM5_PCA2_Dum_NoFe", OutOfSample = glm3H2o.oob, InSample = glm3H2o.insample))

#GBM_PCA2_Vec_NoFe
  
gbm2H2o.insample<- 100*get_deviance(pred_learn, learn.h2o$claim_nb)
  
gbm2H2o.oob <- 100*get_deviance(pred_test, test.h2o$claim_nb)
  
results = rbind(results,
                  data.table(Model = "GBM3_PCA2_Vec_NoFe", OutOfSample = gbm2H2o.oob, InSample = gbm2H2o.insample ))
  
#GBM Tuned AutoGrid
  
gbm2H2o.oob <- 100*get_deviance(pred_test_t, test.h2o$claim_nb)
  
results = rbind(results,
                  data.table(Model = "GBM2_H2o_AutoTune", OutOfSample = gbm2H2o.oob, InSample = "N/A"))
  
#NNKeras_PCA2_Dum_NoFe

in_sample <- 2*( sum ( learn_PCA_Dum_Fe$NN_no_FE )- sum ( learn_PCA_Dum_Fe$claim_nb )
                     + sum ( log (( learn_PCA_Dum_Fe$claim_nb / learn_PCA_Dum_Fe$NN_no_FE )^( learn_PCA_Dum_Fe$claim_nb ))))

average_in_sample <- in_sample*100 / nrow ( learn_PCA_Dum_Fe )

out_of_sample <- 2*( sum ( test_PCA_Dum_Fe$NN_no_FE )- sum ( test_PCA_Dum_Fe$claim_nb )
                     + sum ( log (( test_PCA_Dum_Fe$claim_nb / test_PCA_Dum_Fe$NN_no_FE )^( test_PCA_Dum_Fe$claim_nb ))))

average_out_of_sample <- out_of_sample*100 / nrow ( test_PCA_Dum_Fe )

results2 = rbind(results2,
                data.table(Model = "NN4_Keras_150_45x2_10_DP_0.05", OutOfSample = average_out_of_sample, InSample = average_in_sample))

###############################################

######### Writing the results

###############################################

results2 %>% fwrite("C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/model_results2_v1.csv")

###############################################

######### save the model

###############################################

#save h2o model
h2o.saveModel(glm_fit,"C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/GLM1_H2o")
#save Keras model
save(model, file ="C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/NN0_Keras_PCA1_Dum_No_Fe.rda")
#☻load keras model
load("C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/NN0_Keras_PCA1_Dum_No_Fe.rda")
