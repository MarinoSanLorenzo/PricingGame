
###############################################

######### Generating Predictions Keras

###############################################


###############################################



testing_set <- read.csv("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/testing_set.csv")


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



pred_testing_set_Keras <- as.vector(predict(model_fr, x))




pred_testing_set_Keras_table = data.table(Id_Policyholder = Nombre, NClaim = pred_testing_set_Keras  )

pred_testing_set_Keras_table  %>% fwrite("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/export/Frequency_04122019_2250_log.csv")

################# GLM Poisson ################# 
'
glmPoisson<- glm(Nclaim ~ Split + Latitude + Longitude +  Cover +  AgePh + AgeCar + Power + Fuel + Gender + Use, 
data = training_set, family = poisson((link = "log")))


training_set$GLM <- fitted (glmPoisson )

df.frequency <- as.data.frame(cbind(training_set, NN = pred_testing_set_Keras))


################# PLOT ################# 


cover_data <- df.frequency %>%             # group policies who faced one claims & sum it to derive the total #claims by policy

group_by(Cover) %>% summarise(
GLM = mean(GLM),
NN = mean(NN)
)

color = data.table(GLM = "blue", NN = "red")

ggplot(data = cover_data) + 
geom_point(mapping = aes(x = Cover, y = GLM, color = "GLM")) + 
geom_point(mapping = aes(x = Cover, y = NN, shape = "NN")) 
ggtitle("Model GLM vs NN" ) 


'
exposure <- rep(1,nrow(training_set))
table_conc <- data.table(observed = training_set$Nclaim, predicted = pred_testing_set_Keras, exposure = exposure)

concprob <- concProb(table_conc,0,1)

c_index <- concprob$concProbGlobal
c_index