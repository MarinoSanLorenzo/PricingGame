
###############################################

#########  Data Analysis : Package H2o

#########  GLM, RandomForest and Gradient Boosting

######### Author : San Lorenzo Marino 

######### Inspired by : Alexander Noll^[PartnerRe Ltd - PartnerRe Holdings Europe Limited]

#- Robert Salzmann^[SIGNAL IDUNA Reinsurance Ltd]

#- Mario V. Wuthrich^[RiskLab, ETH Zurich]

######### Version 29  December 2018

###############################################
###############################################

#########  IMPL package

###############################################

library(iml)

###############################################

#########  load packages and dat.mla

###############################################

library(h2o)

library(tidyverse)

library(recipes)     # Library for dat.mla processing

library(glue)        # For conveniently concatenating strings

library(zeallot)     # for %<-% operator

###############################################

#########  Poisson deviance statistics

###############################################

Poisson.Deviance <- function(pred, obs){
  
  2*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)
  
}

Gamma.Deviance <- function(pred, obs){
  
  2*((sum(obs)-sum(pred)/sum(pred) -sum(log(obs/pred))))/length(pred)
  
  
}

###############################################

#########  feature pre-processing for 

###############################################

path <- "C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/training_set.csv"

training_set <- read.csv(path)

data_embed = training_set %>% data.table

data_embed <- training_set[training_set$Nclaim<4,]

'
AvgCclaim <- data_embed$Cclaim/data_embed$Nclaim

data_embed <- cbind(AvgCclaim, data_embed)'

data_embed <- data_embed[,-which(names(data_embed) %in% c("Cclaim"))]


Data_embed = data_embed %>% data.table



dat.ml <- Data_embed

ll <- sample(1:nrow(dat.ml), round(0.9 * nrow(dat.ml)), replace = FALSE)

learn <- dat.ml[ll, ] 

test <- dat.ml[-ll, ]




# Prepare the recipe object


rec_obj <- recipe(Nclaim ~ .,  # use all other variables as predictors
                  data = learn) %>% 
  step_center(AgeCar, AgePh, Latitude, Longitude, Power) %>%# Subtract column mean 
  step_scale(AgeCar, AgePh, Latitude, Longitude, Power) %>%# Divide column by SD 
  prep(training = learn)                                # Use `learn` set to prepare recipes object


# Use recipe to bake" the final data 


learn_prepped <- bake(rec_obj, new_data = learn) # Bake the recipe

test_prepped <- bake(rec_obj, new_data = test)

###############################################

#########  Starting H2o cluster

###############################################

h2o.init(nthreads = 12, port = 11223, max_mem_size = "16G") # Use 6 CPUs and custom port i7 8750h cpu

h2o.no_progress()                    # Disable progress bars for nicer output

learn.h2o <- as.h2o(learn_prepped)   # Upload data to h2o

test.h2o <- as.h2o(test_prepped)



###############################################

#########  Boosting Machines

###############################################


gbm_fit <- h2o.gbm(
  
  x = x, 
  
  y = y,                                          
  
  
  training_frame = learn.h2o,
  
  
  
  distribution = "poisson",
  
  nfolds = 5,
  
  keep_cross_validation_predictions = TRUE,
  
  seed = 1 # For reproducibility
  
)

plot(gbm_fit)

h2o.varimp_plot(gbm_fit)
#Note that all the hyperparameters have been kept at their default values. Unfortunately, the `mean_residual_deviance` cannot be compared directly, since the GBM model in H2O uses another definition of deviance. We can, however, compute the deviance with a function (this is done on the H2O cluster and not in the R session):


get_deviance <- function(y_pred, y_true) {
  
  2 * (sum(y_pred) - sum(y_true) + sum(log((y_true / y_pred) ^ (y_true)))) / nrow(y_pred)
  
}

# Predict on various sets

pred_learn <- predict(gbm_fit, learn.h2o)$predict

pred_cv <- h2o.cross_validation_holdout_predictions(gbm_fit)

pred_test <- predict(gbm_fit, test.h2o)$predict



# Calculate the deviance measures

in_sample <- get_deviance(pred_learn, learn.h2o$claim_nb)

cv <- get_deviance(pred_cv, learn.h2o$claim_nb)

out_of_sample <- get_deviance(pred_test, test.h2o$claim_nb)



# Show them

cat(glue("In-sample deviance: {signif(in_sample, 3)}\n\n"))

cat(glue("CV-deviance: {signif(cv, 3)}\n\n"))

cat(glue("Out-of-sample deviance: {signif(out_of_sample, 3)}"))

```


h2o.varimp_plot(gbm_fit)

###############################################

#########  Hyperparameter tuning

###############################################

#


# Search parameter



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

#After running this grid search, let us examine the results.

gbm_grid


best_gbm <- h2o.getModel(gbm_grid@model_ids[[1]]) # Extract best model from grid

summary(best_gbm) # Show summary


#Finally, let us calculate the performance of this model on the test set:


h2o.varimp_plot(best_gbm)


pred_test_t <- predict(best_gbm, test.h2o)$predict


out_of_sample_t <- get_deviance(pred_test_t, test.h2o$claim_nb)


cat(glue("Out-of-sample deviance: {signif(out_of_sample_t, 3)}"))


###############################################

#########  Neural Networks using H2o

###############################################

#The API to H2O's neural network algorithm is the same as for the GBM model. We just additionally specify some neural network specific parameters like the number of hidden nodes and the activation function. Note that deeper neural networks can be specified by taking a longer *vector of hidden units* (e.g. `hidden = c(20, 20)`).


deep_model <- h2o.deeplearning(
  
  x = setdiff(colnames(learn.h2o), c("AvgClaim")),
  
  y = "AvgClaim",
  
  distribution = "poisson",
  
  
  
  training_frame = learn.h2o,
  
  nfolds = 5,
  
  seed = 1,
  
  
  
  # Neural network parameters
  
  hidden = c(20),
  
  input_dropout_ratio = 0,
  
  epochs = 5,
  
  activation = "Tanh"
  
)

deep_model

```



#We use this model to calculate the `in_sample` and `out_of_sample` deviance:



```{r}

pred_learn <- predict(deep_model, learn.h2o)

pred_test <- predict(deep_model, test.h2o)







#Next, we perform hyperparameter tuning with random grid search on the following parameters:


#  + `hidden` indicates the architecture of the neural network

# + `activation` specified the activation function

# + `input_dropout_ratio` gives the dropout ratio from input to the first hidden layer

#+  `l1` gives the L1 regularization parameter

#+ `l2` is the L2 regularization parameter

# + `epochs` is the number of passes over the data



# We use the same split into training and validation set as before for the GBM.





dl_grid <- list(
  
  hidden = list(10, 20, 50, 100, c(10, 10), c(10, 20), c(20, 10), c(20, 20),
                
                c(50, 20), c(100, 50), c(10, 10, 10), c(50, 25, 10), c(100, 50, 25),
                
                c(10, 10, 10, 10), c(20, 20, 20, 20), c(50, 50, 30, 20)),
  
  
  
  activation = c("Rectifier", "Tanh", "Maxout", "RectifierWithDropout",
                 
                 "TanhWithDropout", "MaxoutWithDropout"),
  
  
  
  input_dropout_ratio = c(0, 0.05, 0.1),
  
  l1 = seq(0, 1e-4, 1e-6),
  
  l2 = seq(0, 1e-4, 1e-6),
  
  epochs = c(10, 20, 30)
  
)

strategy  <-  list(strategy = "RandomDiscrete",
                   
                   max_runtime_secs =  10800,
                   
                   seed = 1,
                   
                   stopping_rounds = 5,           # Early stopping
                   
                   stopping_tolerance = 0.001,
                   
                   stopping_metric = "deviance")



dl_random_grid <- h2o.grid(
  
  algorithm = "deeplearning",
  
  grid_id = "dl_grid_random",
  
  training_frame = learn.h2o,
  
  nfolds = 5,
  
  seed = 1,
  
  
  
  x = setdiff(colnames(learn.h2o), c("claim_nb", "Offset")),
  
  y = "claim_nb",
  
  offset_column = "Offset",
  
  distribution = "poisson",
  
  
  
  hyper_params = dl_grid,
  
  search_criteria = strategy
  
)



dl_random_grid


We predict on the test set using the best model:
  
  
  
  ```{r}

best_model <- h2o.getModel(dl_random_grid@summary_table$model_ids[[1]])



pred_learn <- predict(best_model, learn.h2o)

pred_test <- predict(best_model, test.h2o)



in_sample <- get_deviance(pred_learn, learn.h2o$claim_nb)

out_of_sample <- get_deviance(pred_test, test.h2o$claim_nb)



cat(glue("In-sample deviance: {signif(in_sample, 3)}\n\n"))

cat(glue("Out-of-sample deviance: {signif(out_of_sample, 3)}"))

###############################################

#########  Random Forest

###############################################

rf0 <- h2o.randomForest(
  
  x = x, 
  
  y = y,
  
  offset_column = offset,
  
  training_frame = learn.h2o,
  
  ntrees = 100,
  
  stopping_metric = "deviance", 
  
  distribution = "poisson",
  
  stopping_rounds = 100,
  
  stopping_tolerance = 0.01,
  
  nfolds = 5,
  
  keep_cross_validation_predictions = TRUE,
  
  seed = 123
)

imp.rf0 <- h2o.varimp_plot(rf0) +  ggtitle("RF")

rf0_Tree <- h2o.getModelTree(rf0, 100)

plot(rf0)


rf0

# Predict on various sets

pred_learn <- predict(rf0, learn.h2o)$predict

pred_test <- predict(rf0, test.h2o)$predict





###############################################

#########  In and Out Sample Losses

###############################################


#GLM

cat(glue("In-sample deviance: {signif(h2o.mean_residual_deviance(glm_fit, train = TRUE), 3)}"))
glm1H2o.insample <-  0.197*100
cat(glue("Out-of-sample deviance: {signif(h2o.mean_residual_deviance(glm_fit, valid = TRUE), 3)}"))
glm1H2o.oob <- 0.196*100

results2 = data.table(Model = "GLM_H2o", OutOfSample = glm1H2o.oob, InSample = glm1H2o.insample )
#GBM

gbm1H2o.insample<- 100*get_deviance(pred_learn, learn.h2o$claim_nb)
gbm1H2o.cv <- get_deviance(pred_cv, learn.h2o$claim_nb)
gbm1H2o.oob <- 100*get_deviance(pred_test, test.h2o$claim_nb)

results2 = rbind(results,
                 data.table(Model = "GBM1_H2o", OutOfSample = gbm1H2o.oob, InSample = gbm1H2o.insample ))
#GBM Tuned AutoGrid

gbm2H2o.oob <- 100*get_deviance(pred_test_t, test.h2o$claim_nb)

results2 = rbind(results,
                 data.table(Model = "GBM2_H2o_AutoTune", OutOfSample = gbm2H2o.oob, InSample = "N/A"))
#NN

NN0_H2o.insample <- 100*get_deviance(pred_learn, learn.h2o$claim_nb)
NN0_H2o.oob <- 100*get_deviance(pred_test, test.h2o$claim_nb)

results = rbind(results,
                data.table(Model = "NN0_H2o", OutOfSample = NN0_H2o.oob, InSample = NN0_H2o.insample))
#NN AutoTune

NN1_H2o.insample <- 100*get_deviance(pred_learn, learn.h2o$claim_nb)
NN1_H2o.oob <- 100*get_deviance(pred_test, test.h2o$claim_nb)

results2 = rbind(results2,
                 data.table(Model = "NN2_H2o_AutoTune", OutOfSample = NN1_H2o.oob, InSample = NN1_H2o.insample))

#Random Forest

RF0_H2o.insample <- 100*get_deviance(pred_learn, learn.h2o$claim_nb)
RF0_H2o.oob <- 100*get_deviance(pred_test, test.h2o$claim_nb)

results = rbind(results,
                data.table(Model = "RF0", OutOfSample = RF0_H2o.oob, InSample = RF0_H2o.insample))

###############################################

######### Writing the results

###############################################

results2 %>% fwrite("C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/model_results2_v1.csv")

###############################################

######### save the model

###############################################

h2o.saveModel(best_model,"C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/NN2_H2o_AutoTune")


savedModel <- h2o.loadModel("C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/Models/GLM1_H2o/GLM_model_R_1546045621795_11")


