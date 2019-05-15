pDev <- function(y_true, y_pred){
  2 * k_mean(y_pred - y_true - y_true * (k_log(k_clip(y_pred, k_epsilon(), NULL)) - k_log(k_clip(y_true, k_epsilon(), NULL))), axis = 2)
}


FLAGS <- flags(
  
  ### EMBEDDINGS LAYERS
  
  flag_integer("cover_emb", 2),
  flag_integer("split_emb", 2),
  flag_integer("ageph_emb", 25),
  flag_integer("agecar_emb", 10),
  flag_integer("power_emb", 40),
  flag_integer("fuel_emb", 1),
  flag_integer("gender_emb", 1),
  flag_integer("use_emb", 1),
  flag_integer("latitude_emb", 75),
  flag_integer("longitude_emb", 75), #2++2+25+10+25+1+1+1+50+50 = 167
  
  
  ### LAYER 1
  
  flag_integer("dense_units1", 400),
  flag_numeric("dropout1", 0),
  flag_numeric("regularizer_l1_1", 0.00),
  flag_numeric("regularizer_l2_1", 0.0001),
  flag_string("activation1", "tanh"),                  # relu
  
  
  ### LAYER 2
  
  flag_integer("dense_units2", 300),
  flag_numeric("dropout2", 0.2),
  flag_numeric("regularizer_l1_2", 0.00),
  flag_numeric("regularizer_l2_2", 0.0001),
  flag_string("activation2", "tanh"),
  
  ### LAYER 3
  
  flag_integer("dense_units3", 175),
  flag_numeric("dropout3", 0.25),
  flag_numeric("regularizer_l1_3", 0.0),
  flag_numeric("regularizer_l2_3", 0.0001),
  flag_string("activation3", "tanh"),
  
  ### LAYER 4
  
  flag_integer("dense_units4", 90),
  flag_numeric("dropout4", 0.3),
  flag_numeric("regularizer_l1_4", 0),
  flag_numeric("regularizer_l2_4", 0.0001),
  flag_string("activation4", "tanh"),
  
  ### LAYER 4
  
  flag_integer("dense_units5", 30),
  flag_numeric("dropout5", 0.35),
  flag_numeric("regularizer_l1_5", 0),
  flag_numeric("regularizer_l2_5", 0.0001),
  flag_string("activation5", "tanh"),
  
  
  

  
  ### MIDDLE LAYER
  
  flag_integer("mainoutput", 1),
  flag_string("activation_output", "sigmoid"), #sigmoid
  
  
  ### HYPERPARAMETERS
  
  flag_integer("epochs", 3),
  flag_integer("batch_size", 16),
  flag_string("optimizer", "nadam")
  
)




###############################################

#########  Building embeddings layers

###############################################


Cover <- layer_input(shape = c(1), dtype = 'int32', name = 'Cover')
Cover_embed = Cover %>% 
layer_embedding(input_dim = 3, output_dim = FLAGS$cover_emb, input_length = 1, name = 'Cover_embed') %>%
keras::layer_flatten()

  Split <- layer_input(shape = c(1), dtype = 'int32', name = 'Split')
Split_embed = Split %>% 
  layer_embedding(input_dim = 4, output_dim = FLAGS$split_emb, input_length = 1, name = 'Split_embed') %>%
  keras::layer_flatten()

   AgePh <- layer_input(shape = c(1), dtype = 'int32', name = 'AgePh')
AgePh_embed = AgePh %>% 
  layer_embedding(input_dim = 59, output_dim = FLAGS$ageph_emb, input_length = 1, name = 'AgePh_embed') %>%
  keras::layer_flatten()

  AgeCar <- layer_input(shape = c(1), dtype = 'int32', name = 'AgeCar')
AgeCar_embed = AgeCar %>% 
  layer_embedding(input_dim = 19, output_dim = FLAGS$agecar_emb, input_length = 1, name = 'AgeCar_embed') %>%
  keras::layer_flatten()

  Power <- layer_input(shape = c(1), dtype = 'int32', name = 'Power')
Power_embed = Power %>% 
  layer_embedding(input_dim = 56, output_dim = FLAGS$power_emb, input_length = 1, name = 'Power_embed') %>%
  keras::layer_flatten()

    Fuel <- layer_input(shape = c(1), dtype = 'int32', name = 'Fuel')
Fuel_embed = Fuel %>% 
  layer_embedding(input_dim = 2, output_dim = FLAGS$fuel_emb, input_length = 1, name = 'Fuel_embed') %>%
  keras::layer_flatten()

  Gender <- layer_input(shape = c(1), dtype = 'int32', name = 'Gender')
Gender_embed = Gender %>% 
  layer_embedding(input_dim = 2, output_dim = FLAGS$gender_emb, input_length = 1, name = 'Gender_embed') %>%
  keras::layer_flatten()

    Use <- layer_input(shape = c(1), dtype = 'int32', name = 'Use')
Use_embed = Use %>% 
  layer_embedding(input_dim = 2, output_dim = FLAGS$use_emb, input_length = 1, name = 'Use_embed') %>%
  keras::layer_flatten()

  Latitude <- layer_input(shape = c(1), dtype = 'int32', name = 'Latitude')
Latitude_embed = Latitude %>% 
  layer_embedding(input_dim = 98, output_dim = FLAGS$latitude_emb, input_length = 1, name = 'Latitude_embed') %>%
  keras::layer_flatten()

  Longitude <- layer_input(shape = c(1), dtype = 'int32', name = 'Longitude')
Longitude_embed = Longitude %>% 
  layer_embedding(input_dim = 116, output_dim = FLAGS$longitude_emb, input_length = 1, name = 'Longitude_embed') %>%
  keras::layer_flatten()


###############################################

#########  Building the NN architecture

###############################################


middle_layer <- layer_concatenate(list(Cover_embed,
                                       Split_embed,
                                       AgePh_embed,
                                       AgeCar_embed,
                                       Power_embed,
                                       Fuel_embed,
                                       Gender_embed,
                                       Use_embed,
                                       Latitude_embed,
                                       Longitude_embed)) %>% 
  layer_dense(units = FLAGS$dense_units1, activation = FLAGS$activation1, 
              kernel_regularizer = regularizer_l1_l2(l1= FLAGS$regularizer_l1_1, l2 = FLAGS$regularizer_l2_1)) %>% 
  layer_dropout(FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$dense_units2, activation = FLAGS$activation2, 
              kernel_regularizer = regularizer_l1_l2(l1= FLAGS$regularizer_l1_2, l2 = FLAGS$regularizer_l2_2)) %>% 
  layer_dropout(FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$dense_units3, activation = FLAGS$activation3,
              kernel_regularizer = regularizer_l1_l2(l1= FLAGS$regularizer_l1_3, l2 = FLAGS$regularizer_l2_3)) %>% 
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(units = FLAGS$dense_units4, activation = FLAGS$activation4,
              kernel_regularizer = regularizer_l1_l2(l1= FLAGS$regularizer_l1_4, l2 = FLAGS$regularizer_l2_4)) %>% 
  layer_dropout(rate = FLAGS$dropout4)
  layer_dense(units = FLAGS$dense_units5, activation = FLAGS$activation5,
              kernel_regularizer = regularizer_l1_l2(l1= FLAGS$regularizer_l1_5, l2 = FLAGS$regularizer_l2_5)) %>% 
    layer_dropout(rate = FLAGS$dropout5, name='features')

N = middle_layer  %>% 
  layer_dense(units = FLAGS$mainoutput, activation = FLAGS$activation_output, name = 'N')




model_fr <- keras_model(
  inputs = c(Cover, Split, AgePh, AgeCar, Power, Fuel, Gender, Use, Latitude, Longitude),
  outputs = c(N))



###############################################

#########  Compiling the moodel

###############################################


set.seed(1140)

model_fr %>% compile(
  optimizer = FLAGS$optimizer,
  loss =  "poisson",#function(y_true, y_pred) pDev(y_true, y_pred), #function(y_true, y_pred) pDev(y_true, y_pred)#function(y_true, y_pred) pDev(y_true, y_pred)
  metric = c("mae")) 

### CALLBACKS

dir.create("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/Tensorboard/Tensorbord_directory_frequency")

callbacks_list <- list(
  
  callback_early_stopping(
    monitor = "val_loss",
    
    min_delta = 0.00001, 
    
    patience = 2, 
    
    verbose = TRUE, mode = c("auto"), restore_best_weights = TRUE),
  
  callback_model_checkpoint(
    
    filepath = "model_fr.h5",
    monitor = "val_loss",
    save_best_only = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 10
  )
)


###############################################

#########  Create directory for model checkpoints

###############################################

'checkpoint_dir <- "C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/models/model_fr"

dir.create(checkpoint_dir, showWarnings = FALSE)

filepath <- file.path(checkpoint_dir, "model_fr.hdf5")


'
###############################################

#########  Compiling the moodel

###############################################

history <-  model_fr %>% fit(
  x = x_train,
  y = y_train, validation_data = list(x_test,y_test),
  epochs = FLAGS$epochs ,
  batch_size = FLAGS$batch_size,
  view_metrics = TRUE,
  callbacks = callbacks_list,
  verbose = 1,
  shuffle = T)



'
save_model_hdf5(model_fr, filepath)
model_fr <- load_model_hdf5("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/models/model_fr.hdf5")

training_run("3.Keras_runs.R")
tensorboard("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/Tensorboard/Tensorbord_directory_frequency")
View(ls_runs())'

'runs <- ls_runs()
str(runs)
plot(runs$metric_val_loss[runs$metric_val_loss < 0.33])

model_fr <- load_model_hdf5(filepath = "C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/model_fr.h5")'





'