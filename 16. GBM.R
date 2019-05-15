
###############################################

#########  Data Analysis : Package H2o

#########  GBM, RandomForest and Gradient Boosting

######### Author : San Lorenzo Marino 

######### Inspired by : Alexander Noll^[PartnerRe Ltd - PartnerRe Holdings Europe Limited]

#- Robert Salzmann^[SIGNAL IDUNA Reinsurance Ltd]

#- Mario V. Wuthrich^[RiskLab, ETH Zurich]

######### Version 29  December 2018

###############################################
###############################################

#########  IMPL package

###############################################

'install.packages("tidyverse", lib = "C:/Users/G38249/Downloads/Extra/")
install.packages("recipes", lib = "C:/Users/G38249/Downloads/Extra/")
install.packages("dplyr", lib = "C:/Users/G38249/Downloads/Extra/")
install.packages("glue", lib = "C:/Users/G38249/Downloads/Extra/")
install.packages("rvest", lib = "C:/Users/G38249/Downloads/Extra/")
'
'library(tidyverse, lib.loc = "C:/Users/G38249/Downloads/Extra/")
library(recipes, lib.loc = "C:/Users/G38249/Downloads/Extra/")
library(lava, lib.loc = "C:/Users/G38249/Downloads/Extra/")
library(dplyr, lib.loc = "C:/Users/G38249/Downloads/Extra/")
library(glue, lib.loc = "C:/Users/G38249/Downloads/Extra/")
library(rvest, lib.loc = "C:/Users/G38249/Downloads/Extra/")
library(iml)'

###############################################

#########  load packages and dat.mla

###############################################

library(h2o)

library(data.table)

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

index_poisson <- "GBM_Poisson_16042019_0909"


'path <- "C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/.csv"'

path <- "C:/Users/G38249/Downloads/Extra/"

training_set <- read.csv(paste0(path,"training_set.csv", sep=""))

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

#########  Poissong GBM

###############################################

#After performing these preprocessing and setup steps, we are now ready to train a first model. For the sake of explaining the H2O API, we first fit a **Poisson GBM with offset** (see Section 3). We start with a few words on the general API used in H2O. All machine learning algorithms in H2O have a similar interface: first of all, the model is called with `h2o.xxx` where `xxx` is the model we want to fit (e.g. `h2o.GBM` or `h2o.gbm`, see [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html) for a list of all algorithms). Then the **predictor columns** are specified as the `x` argument, the target variable as the `y` argument and the dataset as `training_frame`. Also, the offset (logarithm of `Exposure`) can be set with the `offset_column` argument.



#Optionally, a **validation frame** can be specified with the `validation_frame` argument on which performance metrics are calculated. Alternatively, there is the `nfolds` argument for performing cross-validation (see Section 4.3). Most of the other arguments then specify the **hyperparameters** of the ML algorithm. Note that `h2o.GBM` does not reproduce the results of the `GBM` function in R (for example, some regularization is applied by default; if you want to know more, it is worthwhile reading through the default arguments). The next code chunk shows an example:



# Use all columns except target and offset

x <- setdiff(colnames(learn.h2o), c("Nclaim")) 

y <- "Nclaim"      # Target variable


###############################################

#########  Hyperparameter tuning

###############################################

#


# Search parameter



hyper_params <- list(ntrees = seq(10, 100, 1),
                     learn_rate = seq(0.0001, 0.3, 0.001),
                     max_depth = seq(1, 20, 1),
                     sample_rate = seq(0.5, 1.0, 0.0001),
                     col_sample_rate = seq(0.2, 1.0, 0.001))
search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 15, 
                        seed = 1)


gbm_fit_poisson <- h2o.grid(   #h2o.grid
  
  "gbm",                      #"gbm",
  
  x = x,
  
  y = y,
  
  
  distribution = "poisson",
  
  
  training_frame = learn.h2o,
  
  nfolds = 5,
  
  hyper_params = hyper_params,                             # hyper_params = gbm_params,
  
  search_criteria = search_criteria,                            #  search_criteria = strategy,
  
  max_runtime_secs = 180000,
  
  score_each_iteration = TRUE,
  
  seed = 1,
  
  
  stopping_rounds = 10,           # Early stopping
  
  stopping_tolerance = 0.00001,
  
  stopping_metric = "deviance"
)


grid_poisson <- h2o.getGrid(grid_id = gbm_fit_poisson@grid_id, sort_by = "residual_deviance", decreasing = FALSE)
grid_top_model_poisson <- grid_poisson@summary_table[1, "model_ids"]

best_gbm_fit_poisson <- h2o.getModel(grid_top_model_poisson)

summary(gbm_fit_poisson)

###############################################

#########  GBM Predictions

###############################################


testing_set <- read.csv(paste0(path,"testing_set.csv", sep=""))


Test_embed <- testing_set[,-which(names(testing_set) %in% c("Id_Policyholder"))]

Test_embed = Test_embed %>% data.table
str(Test_embed)


# Prepare the recipe object


rec_obj_test <- recipe(~ .,  # use all other variables as predictors
                       data = Test_embed) %>% 
  step_center(AgeCar, AgePh, Latitude, Longitude, Power) %>%# Subtract column mean 
  step_scale(AgeCar, AgePh, Latitude, Longitude, Power) %>%# Divide column by SD 
  prep(training = Test_embed)                                # Use `learn` set to prepare recipes object


# Use recipe to bake" the final data 


test_prepped <- bake(rec_obj_test, new_data = Test_embed ) # Bake the recipe

test_prepped <- as.h2o(test_prepped)

x = list(Cover = cover,
         Split = split,
         AgePh = ageph,
         AgeCar = agecar,
         Power  = power,
         Fuel = fuel,
         Gender = gender,
         Use = use,
         Latitude = latitude,
         Longitude = longitude)


Nombre <- c(1:nrow(testing_set))

pred_GBM_Poisson <- as.vector(predict(best_gbm_fit_poisson, newdata = test_prepped))

pred_GBM_Poisson_table = data.table(Id_Policyholder = Nombre, NClaim = pred_GBM_Poisson  )


pred_GBM_Poisson_table  %>% fwrite(paste0(path,index_poisson,".csv", sep=""))

'exposure <- rep(1,nrow(training_set))
table_conc <- data.table(observed = training_set$Nclaim, predicted = pred_GBM_Poisson, exposure = exposure)

concprob <- concProb(table_conc,0,3)



c_index <- data.table(concprob$concProbGlobal)

c_index  %>% fwrite("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/export/best/c_index/c_index_Poisson_14042019_0948.csv")

'

#Note that the results, both in coefficients and performance, are different from Section 3 for two reasons:



# + In the paper, the features were further processed (e.g. numeric columns were binned)

# + By default, H2O fits models with non-zero regularization parameter


###############################################

#########  feature pre-processing for GBM GAMMA

###############################################

index_nn_gamma <- "GBM_Gamma_18042019_1111"
  

training_set <- read.csv(paste0(path,"training_set.csv", sep=""))

data_embed = training_set %>% data.table

'order <- data_embed[order(Cclaim, decreasing = TRUE),]'

data_embed_gamma <-  filter(data_embed, Cclaim < 20000,  Nclaim > 0)


AvgCclaim <- data_embed_gamma$Cclaim/data_embed_gamma$Nclaim

data_embed_gamma <- cbind(AvgCclaim, data_embed_gamma)

data_embed_gamma <- data_embed_gamma[,-which(names(data_embed_gamma) %in% c("Cclaim", "Nclaim"))]





dat.ml <- data_embed_gamma

ll <- sample(1:nrow(dat.ml), round(0.9 * nrow(dat.ml)), replace = FALSE)

learn <- dat.ml[ll, ] 

test <- dat.ml[-ll, ]




# Prepare the recipe object


rec_obj <- recipe(AvgCclaim ~ .,  # use all other variables as predictors
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

#########  Poissong GBM

###############################################

#After performing these preprocessing and setup steps, we are now ready to train a first model. For the sake of explaining the H2O API, we first fit a **Poisson GBM with offset** (see Section 3). We start with a few words on the general API used in H2O. All machine learning algorithms in H2O have a similar interface: first of all, the model is called with `h2o.xxx` where `xxx` is the model we want to fit (e.g. `h2o.GBM` or `h2o.gbm`, see [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html) for a list of all algorithms). Then the **predictor columns** are specified as the `x` argument, the target variable as the `y` argument and the dataset as `training_frame`. Also, the offset (logarithm of `Exposure`) can be set with the `offset_column` argument.



#Optionally, a **validation frame** can be specified with the `validation_frame` argument on which performance metrics are calculated. Alternatively, there is the `nfolds` argument for performing cross-validation (see Section 4.3). Most of the other arguments then specify the **hyperparameters** of the ML algorithm. Note that `h2o.GBM` does not reproduce the results of the `GBM` function in R (for example, some regularization is applied by default; if you want to know more, it is worthwhile reading through the default arguments). The next code chunk shows an example:



# Use all columns except target and offset

x <- setdiff(colnames(learn.h2o), c("AvgCclaim")) 

y <- "AvgCclaim"      # Target variable




hyper_params <- list(ntrees = seq(10, 100, 1),
                     learn_rate = seq(0.0001, 0.2, 0.001),
                     max_depth = seq(1, 20, 1),
                     sample_rate = seq(0.5, 1.0, 0.0001),
                     col_sample_rate = seq(0.2, 1.0, 0.001))
search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 15, 
                        seed = 1)


gbm_fit_gamma <- h2o.grid(   #h2o.grid
  
  "gbm",                      #"gbm",
  
  x = x,
  
  y = y,
  
  
  distribution = "gamma",
  
  
  training_frame = learn.h2o,
  
  nfolds = 5,
  
  hyper_params = hyper_params,                             # hyper_params = gbm_params,
  
  search_criteria = search_criteria,                            #  search_criteria = strategy,
  
  max_runtime_secs = 180000,
  
  score_each_iteration = TRUE,
  
  seed = 1,
  
  
  stopping_rounds = 10,           # Early stopping
  
  stopping_tolerance = 0.00001,
  
  stopping_metric = "deviance"
)


grid_gamma <- h2o.getGrid(grid_id = gbm_fit_gamma@grid_id, sort_by = "residual_deviance", decreasing = FALSE)
grid_top_model_gamma <- grid_gamma@summary_table[1, "model_ids"]

best_gbm_fit_gamma <- h2o.getModel(grid_top_model_gamma)

summary(gbm_fit_gamma)



####☻IML representation ######


###############################################

#########  GBM Predictions

###############################################


testing_set <- read.csv(paste0(path,"testing_set.csv", sep=""))


Test_embed <- testing_set[,-which(names(testing_set) %in% c("Id_Policyholder"))]

Test_embed = Test_embed %>% data.table
str(Test_embed)


# Prepare the recipe object


rec_obj_test <- recipe(~ .,  # use all other variables as predictors
                       data = Test_embed) %>% 
  step_center(AgeCar, AgePh, Latitude, Longitude, Power) %>%# Subtract column mean 
  step_scale(AgeCar, AgePh, Latitude, Longitude, Power) %>%# Divide column by SD 
  prep(training = Test_embed)                                # Use `learn` set to prepare recipes object


# Use recipe to bake" the final data 


test_prepped <- bake(rec_obj_test, new_data = Test_embed ) # Bake the recipe

test_prepped <- as.h2o(test_prepped)

x = list(Cover = cover,
         Split = split,
         AgePh = ageph,
         AgeCar = agecar,
         Power  = power,
         Fuel = fuel,
         Gender = gender,
         Use = use,
         Latitude = latitude,
         Longitude = longitude)


Nombre <- c(1:nrow(testing_set))

pred_GBM_Gamma <- as.vector(predict(best_gbm_fit_gamma, newdata = test_prepped))

pred_GBM_Gamma_table = data.table(Id_Policyholder = Nombre, Cclaim = pred_GBM_Gamma  )

pred_GBM_Gamma_table  %>% fwrite(paste0(path,index_nn_gamma,".csv", sep=""))

# Load previously saved new frequency

pred_GBM_Poisson_table  <- read.csv(paste0(path,index_poisson,".csv", sep=""))



proxy <- cbind(pred_GBM_Poisson_table, pred_GBM_Gamma_table[,2])

purepremium_alone <- proxy$NClaim*proxy$Cclaim


'purepremium_stacked <- 0.5*proxy$NClaim*proxy$CClaim + 0.5*best_pred_1[,2]'



sampleSubmission <- data.table(Id_Policyholder = proxy$Id_Policyholder, pure_premium = purepremium_alone  )

'sampleSubmission <- data.table(Id_Policyholder = proxy$Id_Policyholder, pure_premium= purepremium_stacked )'

############### TEST #####################



check_freq <- function(table_freq, table_sev, epsilon) {
  freq_mean <-0.0564
  sev_mean <- 2478.13
  mean_freq <- apply(table_freq, 2, mean)
  
  mean_sev <- apply(table_sev, 2, mean)
  
  perc_freq <- 100*round(abs( (mean_freq[2]-freq_mean)/freq_mean ),3)
  perc_sev <- 100*round(abs( (mean_sev[2]-sev_mean)/sev_mean ),3)
  
  
  
  if( perc_freq > epsilon ) {
    print(paste(mean_freq[2],"Nclaim average is not ok", perc_freq, " "))
  }
  else{
    print(paste(mean_freq[2],"Nclaim average is  ok", perc_freq, " "))
  }
  if( perc_sev > epsilon ) {
    print(paste(mean_sev[2], "Cclaim average is not ok", perc_sev, " "))
  }
  else{
    print(paste(mean_sev[2], "Cclaim average is  ok", perc_sev,  " "))
  }
  
  
}

epsilon <- 0.01
check_freq(pred_GBM_Poisson_table,pred_GBM_Gamma_table , epsilon )


summary(proxy) # Nclaim 0.0564 ; Avg Claim 2478.13
summary(sampleSubmission) # pure premium 139.9


sampleSubmission  %>% fwrite(paste0(path,"sampleSubmission.csv", sep=""))


