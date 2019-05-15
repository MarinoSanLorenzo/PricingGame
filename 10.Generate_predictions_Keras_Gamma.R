
###############################################

######### Generating Predictions Keras

###############################################


###############################################

decision_poisson <- c("New")
decision_gamma <- c("New")

index_nn_gamma <- "Gamma_04182019"


testing_set <- read.csv(paste0(path, "testing_set.csv", sep=""))


Test_embed <- testing_set[,-which(names(testing_set) %in% c("Id_Policyholder"))]

Test_embed = Test_embed %>% data.table



attributes(Test_embed)$names

### prepare data for Embeddings

Test_embed[,CoverNN     :=as.integer(as.factor(Cover))-1]
Test_embed[,SplitNN     :=as.integer(as.factor(Split))-1]
Test_embed[,AgePhNN    :=as.integer(as.factor(cut2(AgePh,m=17000)))-1]
Test_embed[,AgeCarNN    :=as.integer(as.factor(cut2(AgeCar,m=17000)))-1]
Test_embed[,PowerNN    :=as.integer(as.factor(cut2(Power,m=17000)))-1]
Test_embed[,FuelNN      :=as.integer(as.factor(Fuel))-1]
Test_embed[,GenderNN     :=as.integer(as.factor(Gender))-1]
Test_embed[,UseNN     :=as.integer(as.factor(Use))-1]
Test_embed[,LatitudeNN    :=as.integer(as.factor(cut2(Latitude,m=17000)))-1]
Test_embed[,LongitudeNN    :=as.integer(as.factor(cut2(Longitude,m=17000)))-1]


### prepare dataframe for embeddings


cover_cats = Test_embed[,c("CoverNN","Cover"),with=F] %>% unique %>% setkey(CoverNN)
split_cats = Test_embed[,c("SplitNN","Split"),with=F] %>% unique %>% setkey(SplitNN)
ageph_cats    = Test_embed[,.(AgePh = mean(AgePh)), keyby = AgePhNN]
agecar_cats    = Test_embed[,.(AgeCar = mean(AgeCar)), keyby = AgeCarNN]
power_cats    = Test_embed[,.(Power = mean(Power)), keyby = PowerNN]
fuel_cats = Test_embed[,c("FuelNN","Fuel"),with=F] %>% unique %>% setkey(FuelNN)
gender_cats = Test_embed[,c("GenderNN","Gender"),with=F] %>% unique %>% setkey(GenderNN)
use_cats = Test_embed[,c("UseNN","Use"),with=F] %>% unique %>% setkey(UseNN)
latitude_cats    = Test_embed[,.(Latitude = mean(Latitude)), keyby = LatitudeNN]
longitude_cats    = Test_embed[,.(Longitude = mean(Longitude)), keyby = LongitudeNN]

ageph_dim = ageph_cats[,.N]
agecar_dim = agecar_cats[,.N]
power_dim = power_cats[,.N]
latitude_dim = latitude_cats[,.N]
longitude_dim = longitude_cats[,.N]





### Fit NN using Keras
require(keras)

### Munge data into correct format for embeddings - Train set

cover = Test_embed$CoverNN
split = Test_embed$SplitNN
ageph =  Test_embed$AgePhNN
agecar = Test_embed$AgeCarNN
power = Test_embed$PowerNN
fuel = Test_embed$FuelNN
gender = Test_embed$GenderNN
use = Test_embed$ UseNN
latitude = Test_embed$LatitudeNN
longitude = Test_embed$LongitudeNN


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

### make severity predictions with new model

'model_sev <- load_model_hdf5("model_sev.h5")'

pred_testing_set_Keras_Gamma <- as.vector(predict(model_sev, x)) 

# data table

# Load previously saved new frequency OR gAMMA

pred_testing_set_Keras_table <- read.csv(paste0(path, "export/",index_nn_poisson,".csv", sep =""))

pred_Poisson <- function(decision){
  
  if(decision == "New"){
    
  NClaim = pred_testing_set_Keras_table = data.table(Id_Policyholder = Nombre, NClaim = pred_testing_set_Keras  )
    
    # Save severity
    
   pred_testing_set_Keras_table  %>% fwrite(paste0(path, "/export/",index_nn_poisson,".csv", sep=""))
    
    print('New model selected')
    
    return( pred_testing_set_Keras_table )
    
  } else {
    pred_testing_set_Keras_Gamma_table <- read.csv("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/export/best/NN/Frequency_04092019_2221_75_34900.csv")
    
    print("Old model selected")
    
    return( pred_testing_set_Keras_table )
  }
  
  
}


pred_Gamma <- function(decision){
  
  if(decision == "New"){
    
    pred_testing_set_Keras_Gamma_table = data.table(Id_Policyholder = Nombre, Cclaim = pred_testing_set_Keras_Gamma  )
    
    # Save severity
    
    pred_testing_set_Keras_Gamma_table  %>% fwrite(paste0(path, "/export/",index_nn_gamma,".csv", sep=""))
    
    print('New model selected')
    return( pred_testing_set_Keras_Gamma_table )
    
  } else {
    pred_testing_set_Keras_Gamma_table <- read.csv("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/export/best/GBM/GBM_Gamma_16042019_0909__7719.csv")
    
    print("Old model selected")
    
    return( pred_testing_set_Keras_Gamma_table )
  }
  
  
}

pred_p <- pred_Poisson(decision_poisson)

pred_g <- pred_Gamma(decision_gamma)

proxy <- cbind(pred_p, pred_g[,2])

purepremium_alone <- proxy$NClaim*proxy$Cclaim

'purepremium_stacked <- 0.5*proxy$NClaim*proxy$CClaim + 0.5*best_pred_1[,2]'



sampleSubmission <- data.table(Id_Policyholder = proxy$Id_Policyholder, pure_premium = purepremium_alone )

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
check_freq(pred_testing_set_Keras_table, pred_testing_set_Keras_Gamma_table , epsilon )


summary(proxy) # Nclaim 0.0564 ; Avg Claim 2478.13
summary(sampleSubmission) # pure premium 139.9

sampleSubmission  %>% fwrite("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/export/sampleSubmission.csv")
