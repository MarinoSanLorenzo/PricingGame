
###############################################

######### Generating Predictions Keras

###############################################


###############################################

index_nn_poisson <- "Poisson_04202019_2110"

testing_set <- read.csv(paste0(path,"testing_set.csv", sep=""))


Test_embed <- testing_set[,-which(names(testing_set) %in% c("Id_Policyholder"))]

Test_embed = Test_embed %>% data.table



attributes(Test_embed)$names

### prepare data for Embeddings

Test_embed[,CoverNN     :=as.integer(as.factor(Cover))-1]
Test_embed[,SplitNN     :=as.integer(as.factor(Split))-1]
Test_embed[,AgePhNN    :=as.integer(as.factor(cut2(AgePh,m=1000)))-1]
Test_embed[,AgeCarNN    :=as.integer(as.factor(cut2(AgeCar,m=1000)))-1]
Test_embed[,PowerNN    :=as.integer(as.factor(cut2(Power,m=1050)))-1]
Test_embed[,FuelNN      :=as.integer(as.factor(Fuel))-1]
Test_embed[,GenderNN     :=as.integer(as.factor(Gender))-1]
Test_embed[,UseNN     :=as.integer(as.factor(Use))-1]
Test_embed[,LatitudeNN    :=as.integer(as.factor(cut2(Latitude,m=1000)))-1]
Test_embed[,LongitudeNN    :=as.integer(as.factor(cut2(Longitude,m=1000)))-1]


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

model_fr <- load_model_hdf5("model_fr.h5")

pred_testing_set_Keras <- as.vector(predict(model_fr, x))




pred_testing_set_Keras_table = data.table(Id_Policyholder = Nombre, NClaim = pred_testing_set_Keras  )

pred_testing_set_Keras_table  %>% fwrite(paste0(path, "/export/",index_nn_poisson,".csv", sep=""))

