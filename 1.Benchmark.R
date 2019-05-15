
###############################################

#########  Data Analysis : Benchmark

#########  GLM, Regression Trees and Boosting

######### Author : San Lorenzo Marino and Siham

######### Inspired by : Mario Wuthrich

######### Version 28 December 2018

###############################################




###############################################

#########  load packages and data

###############################################

library(MASS)

library(stats)

library(data.table)

library(plyr)

library(rpart)

library(rpart.plot)

library(Hmisc)

library(magrittr)

library(bigmemory)

library(dplyr)

library(psych)

library(base)

library(caret)

library(keras)

library(visreg)

library(mgcv)

library(caret)

library(plyr)

library(ggplot2)

library(gridExtra)

library(parallel)




###############################################

#########  Poisson deviance statistics

###############################################

Poisson.Deviance <- function(pred, obs){
  
  2*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)
  
}

###############################################

#########  feature pre-processing for GLM

###############################################

dat <- db_train

dat<- dat[,-which(names(db_train) %in% 
                                 c("date_obs",
                                   "num_pol",
                                   "num_risk",
                                   "num_avenant",
                                   "date_starting",
                                   "date_final"))]

dat = dat %>% data.table

dat[,ChannelGLM    :=as.integer(as.factor(channel))-1]
dat[,GuaranteeGLM     :=as.integer(as.factor(insured_full))-1]
dat[,GuaranteeTypeGLM     :=as.integer(as.factor(guarantees_type))-1]
dat[,PremiumPeriodicityGLM     :=as.integer(as.factor(premiums_periodicity))-1]
dat[,DomiciliatioGLMN     :=as.integer(as.factor(domiciliation))-1]
dat[,TelematicGLM     :=as.integer(as.factor(telematic))-1]
dat[,DrivNbGLM      :=as.integer(as.factor(driv_number))-1]
dat[,DrivAge1GLM    :=as.integer(as.factor(cut2(driv_age1,m=10000)))-1]
dat[,DrivExp1GLM    :=as.integer(as.factor(cut2(driv_exp1,m=10000)))-1]
dat[,DrivAge2GLM    :=as.integer(as.factor(cut2(driv_age2,m=10000)))-1]
dat[,DrivExp2GLM    :=as.integer(as.factor(cut2(driv_exp2,m=10000)))-1]
dat[,DrivYoungGLM      :=as.integer(as.factor(driv_age_young))-1]
dat[,VehBrandGLM      :=as.integer(as.factor(veh_brand))-1]
dat[,VehModelGLM     :=as.integer(as.factor(veh_model))-1]
dat[,VehTypeGLM      :=as.integer(as.factor(veh_type))-1]
dat[,InformexGLM      :=as.integer(as.factor(veh_sgmt_informex))-1]
dat[,FebiacGLM      :=as.integer(as.factor(veh_sgmt_febiac))-1]
dat[,VehAgeGLM    :=as.integer(as.factor(cut2(veh_age,m=10000)))-1]
dat[,VehPowerGLM    :=as.integer(as.factor(cut2(veh_power,m=10000)))-1]
dat[,VehWeightGLM    :=as.integer(as.factor(cut2(veh_weight,m=10000)))-1]
dat[,VehSportivityGLM    :=as.integer(as.factor(cut2(veh_sportivity,m=10000)))-1]
dat[,VehValueGLM    :=as.integer(as.factor(cut2(veh_value,m=10000)))-1]
dat[,VehSeatsGLM      :=as.integer(as.factor(veh_seats))-1]
dat[,VehFuelGLM      :=as.integer(as.factor(veh_fuel))-1]
dat[,VehMileageLimitGLM      :=as.integer(as.factor(veh_mileage_limit))-1]
dat[,VehTrailerGLM      :=as.integer(as.factor(veh_trailer))-1]
dat[,VehGarageGLM      :=as.integer(as.factor(veh_garage))-1]
dat[,GeoPostcodeGLM      :=as.integer(as.factor(geo_postcode))-1]
dat[,GeoPostcodeLatGLM    :=as.integer(as.factor(cut2(geo_postcode_lat,m=10000)))-1]
dat[,GeoPostcodeLonGLM    :=as.integer(as.factor(cut2(geo_postcode_lon,m=10000)))-1]
dat[,GeoMunicipalityGLM      :=as.integer(as.factor(geo_municipality_fr))-1]
dat[,GeoProvinceGLM      :=as.integer(as.factor(geo_province_fr))-1]
dat[,GeoMosaicFullGLM      :=as.integer(as.factor(geo_mosaic_code_full))-1]
dat[,GeoMosaicGLM      :=as.integer(as.factor(geo_mosaic_code))-1]

dat = as.data.frame(dat)



###############################################

#########  choosing learning and test sample

###############################################

set.seed(231)

ll <- sample(c(1:nrow(dat)), round(0.9*nrow(dat)), replace = FALSE)

learn <- dat[ll,]

test <- dat[-ll,]

(n_l <- nrow(learn))

(n_t <- nrow(test))

###############################################

#########  GLM analysis

###############################################


### Model GLM0

glm0 <- glm(claim_nb ~ offset(log(exposure)),
            data = learn,family=poisson())



learn$fitGLM0 <- fitted(glm0)

test$fitGLM0 <- predict(glm0, newdata=test, type="response")

### Model GLM1

glm1 <- glm(claim_nb ~ ChannelGLM + GuaranteeGLM  + GuaranteeTypeGLM + PremiumPeriodicityGLM +
             DomiciliatioGLMN  + TelematicGLM  + DrivNbGLM + DrivAge1GLM + DrivExp1GLM +       
            DrivAge2GLM + DrivExp2GLM + DrivYoungGLM + VehBrandGLM + VehModelGLM +         
             VehTypeGLM + InformexGLM + FebiacGLM + VehAgeGLM + VehPowerGLM +      
             VehWeightGLM + VehSportivityGLM + VehValueGLM + VehSeatsGLM + VehFuelGLM +         
             VehMileageLimitGLM + VehTrailerGLM + VehGarageGLM + GeoPostcodeGLM + GeoPostcodeLatGLM + 
             GeoPostcodeLonGLM + GeoMunicipalityGLM + GeoProvinceGLM + GeoMosaicFullGLM + GeoMosaicGLM, 
              
              data=learn, offset=log(exposure), family=poisson())

'plot(glm1)   ### plot residuals
summary(glm0)   
anova(glm0)
summary(glm1)
anova(glm0, glm1, test="Chisq")'


learn$fitGLM1 <- fitted(glm1) ### fitted values on training set

test$fitGLM1 <- predict(glm1, newdata=test, type="response") ### fitted values on test set (predict)

### Model GLM2 : backward approach : removing parameters less significant

glm2 <- glm(claim_nb ~ GuaranteeGLM  + GuaranteeTypeGLM + PremiumPeriodicityGLM +
              DomiciliatioGLMN  + TelematicGLM  + DrivNbGLM + DrivAge1GLM + DrivExp1GLM +       
              DrivExp2GLM + DrivYoungGLM +          
              VehTypeGLM + FebiacGLM + VehAgeGLM + VehPowerGLM +      
              VehSportivityGLM + VehFuelGLM +         
              VehMileageLimitGLM + GeoPostcodeGLM + 
              GeoMunicipalityGLM + GeoMosaicFullGLM, 
            
            data=learn, offset=log(exposure), family=poisson())

summary(glm2)


learn$fitGLM2 <- fitted(glm2) ### fitted values on training set

test$fitGLM2 <- predict(glm2, newdata=test, type="response") ### fitted values on test set (predict)


(glm2$deviance)
(glm2$null.deviance)
###############################################

#########  Regressionn tree analysis

###############################################


tree1 <- rpart(cbind(exposure,claim_nb) ~ ChannelGLM + GuaranteeGLM  + GuaranteeTypeGLM + PremiumPeriodicityGLM +
                 DomiciliatioGLMN  + TelematicGLM  + DrivNbGLM + DrivAge1GLM + DrivExp1GLM +       
                 DrivAge2GLM + DrivExp2GLM + DrivYoungGLM + VehBrandGLM + VehModelGLM +         
                 VehTypeGLM + InformexGLM + FebiacGLM + VehAgeGLM + VehPowerGLM +      
                 VehWeightGLM + VehSportivityGLM + VehValueGLM + VehSeatsGLM + VehFuelGLM +         
                 VehMileageLimitGLM + VehTrailerGLM + VehGarageGLM + GeoPostcodeGLM + GeoPostcodeLatGLM + 
                 GeoPostcodeLonGLM + GeoMunicipalityGLM + GeoProvinceGLM + GeoMosaicFullGLM + GeoMosaicGLM, 
               
               learn, method="poisson",
               
               control=rpart.control(xval=1, minbucket=10000, cp=0.00001)) 



'rpart.plot(tree1)        # plot tree

tree1                    # show tree with all binary splits

printcp(tree1)           # cost-complexit statistics
'

learn$fitRT1 <- predict(tree1)*learn$exposure

test$fitRT1 <- predict(tree1, newdata=test)*test$exposure


'average_loss <- cbind(tree1$cptable[,2], tree1$cptable[,3], tree1$cptable[,3]* tree1$frame$dev[1] / n_l)

plot(x=average_loss[,1], y=average_loss[,3]*100, type="l", col="blue", xlab="number of splits", ylim = c(19.85,20.30), ylab="average in-sample loss (in 10^(-2))", main="decrease of in-sample loss")

points(x=average_loss[,1], y=average_loss[,3]*100, pch=19, col="blue")

abline(h=c(glm2.insample), col="green", lty=2)

legend(x="topright", col=c("blue", "green"), lty=c(1,2), lwd=c(1,1), pch=c(19,-1), legend=c("Model RT1", "Model GLM2"))
'

# cross-validation and cost-complexity pruning

K <- 10                  # K-fold cross-validation value

set.seed(357)

xgroup <- rep(1:K, length = nrow(learn))

xfit <- xpred.rpart(tree1, xgroup)

(n_subtrees <- dim(tree1$cptable)[1])

std1 <- numeric(n_subtrees)

err1 <- numeric(n_subtrees)

err_group <- numeric(K)

for (i in 1:n_subtrees){
  
  for (k in 1:K){
    
    ind_group <- which(xgroup ==k)  
    
    err_group[k] <- Poisson.Deviance(learn[ind_group,"exposure"]*xfit[ind_group,i],learn[ind_group,"claim_nb"])
    
  }
  
  err1[i] <- mean(err_group)             
  
  std1[i] <- sd(err_group)
  
}



'x1 <- log10(tree1$cptable[,1])

xmain <- "cross-validation error plot"

xlabel <- "cost-complexity parameter (log-scale)"

ylabel <- "CV error (in 10^(-2))"

errbar(x=x1, y=err1*100, yplus=(err1+std1)*100, yminus=(err1-std1)*100, xlim=rev(range(x1)), col="blue", main=xmain, ylab=ylabel, xlab=xlabel)

lines(x=x1, y=err1*100, col="blue")

abline(h=c(min(err1+std1)*100), lty=1, col="orange")

abline(h=c(min(err1)*100), lty=1, col="magenta")

abline(h=c(glm2.insample), col="green", lty=2)

legend(x="topright", col=c("blue", "orange", "magenta", "green"), lty=c(1,1,1,2), lwd=c(1,1,1,1), pch=c(19,-1,-1,-1), legend=c("tree1", "1-SD rule", "min.CV rule", "Model GLM1"))

'

# prune to appropriate cp constant

printcp(tree1)

tree2 <- prune(tree1, cp=0.00003)

printcp(tree2)



learn$fitRT2 <- predict(tree2)*learn$exposure

test$fitRT2 <- predict(tree2, newdata=test)*test$exposure


###############################################

#########  Poisson regression tree boosting

###############################################



### Model PBM3

J0 <- 3       # depth of tree

M0 <- 50      # iterations

nu <- 1       # shrinkage constant 



learn$fitPBM1 <- learn$exposure

test$fitPBM1  <- test$exposure



for (m in 1:M0){
  
  PBM.1 <- rpart(cbind(fitPBM1,claim_nb) ~ ChannelGLM + GuaranteeGLM  + GuaranteeTypeGLM + PremiumPeriodicityGLM +
                   DomiciliatioGLMN  + TelematicGLM  + DrivNbGLM + DrivAge1GLM + DrivExp1GLM +       
                   DrivAge2GLM + DrivExp2GLM + DrivYoungGLM + VehBrandGLM + VehModelGLM +         
                   VehTypeGLM + InformexGLM + FebiacGLM + VehAgeGLM + VehPowerGLM +      
                   VehWeightGLM + VehSportivityGLM + VehValueGLM + VehSeatsGLM + VehFuelGLM +         
                   VehMileageLimitGLM + VehTrailerGLM + VehGarageGLM + GeoPostcodeGLM + GeoPostcodeLatGLM + 
                   GeoPostcodeLonGLM + GeoMunicipalityGLM + GeoProvinceGLM + GeoMosaicFullGLM + GeoMosaicGLM, 
                 
                 data=learn, method="poisson",
                 
                 control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, minbucket=10000, cp=0.00001))     
  
  if(m>1){
    
    learn$fitPBM1 <- learn$fitPBM1 * predict(PBM.1)^nu
    
    test$fitPBM1 <- test$fitPBM1 * predict(PBM.1, newdata=test)^nu
    
  } else {
    
    learn$fitPBM1 <- learn$fitPBM1 * predict(PBM.1)
    
    test$fitPBM1 <- test$fitPBM1 * predict(PBM.1, newdata=test)
    
  }
  
}

##############################################

#########  feature pre-processing for GAM

###############################################

dat <- db_train

dat<- dat[,-which(names(db_train) %in% 
                    c("date_obs",
                      "num_pol",
                      "num_risk",
                      "num_avenant",
                      "date_starting",
                      "date_final"))]

dat = dat %>% data.table

dat[,ChannelGLM    :=as.integer(as.factor(channel))-1]
dat[,GuaranteeGLM     :=as.integer(as.factor(insured_full))-1]
dat[,GuaranteeTypeGLM     :=as.integer(as.factor(guarantees_type))-1]
dat[,PremiumPeriodicityGLM     :=as.integer(as.factor(premiums_periodicity))-1]
dat[,DomiciliatioGLMN     :=as.integer(as.factor(domiciliation))-1]
dat[,TelematicGLM     :=as.integer(as.factor(telematic))-1]
dat[,DrivNbGLM      :=as.integer(as.factor(driv_number))-1]
dat[,DrivAge1GLM    :=as.integer(as.factor(cut2(driv_age1,m=10000)))-1]
dat[,DrivExp1GLM    :=as.integer(as.factor(cut2(driv_exp1,m=10000)))-1]
dat[,DrivAge2GLM    :=as.integer(as.factor(cut2(driv_age2,m=10000)))-1]
dat[,DrivExp2GLM    :=as.integer(as.factor(cut2(driv_exp2,m=10000)))-1]
dat[,DrivYoungGLM      :=as.integer(as.factor(driv_age_young))-1]
dat[,VehBrandGLM      :=as.integer(as.factor(veh_brand))-1]
dat[,VehModelGLM     :=as.integer(as.factor(veh_model))-1]
dat[,VehTypeGLM      :=as.integer(as.factor(veh_type))-1]
dat[,InformexGLM      :=as.integer(as.factor(veh_sgmt_informex))-1]
dat[,FebiacGLM      :=as.integer(as.factor(veh_sgmt_febiac))-1]
dat[,VehAgeGLM    :=as.integer(as.factor(cut2(veh_age,m=10000)))-1]
dat[,VehPowerGLM    :=as.integer(as.factor(cut2(veh_power,m=10000)))-1]
dat[,VehWeightGLM    :=as.integer(as.factor(cut2(veh_weight,m=10000)))-1]
dat[,VehSportivityGLM    :=as.integer(as.factor(cut2(veh_sportivity,m=10000)))-1]
dat[,VehValueGLM    :=as.integer(as.factor(cut2(veh_value,m=10000)))-1]
dat[,VehSeatsGLM      :=as.integer(as.factor(veh_seats))-1]
dat[,VehFuelGLM      :=as.integer(as.factor(veh_fuel))-1]
dat[,VehMileageLimitGLM      :=as.integer(as.factor(veh_mileage_limit))-1]
dat[,VehTrailerGLM      :=as.integer(as.factor(veh_trailer))-1]
dat[,VehGarageGLM      :=as.integer(as.factor(veh_garage))-1]
dat[,GeoPostcodeGLM      :=as.integer(as.factor(geo_postcode))-1]
dat[,GeoPostcodeLatGLM    :=as.integer(as.factor(cut2(geo_postcode_lat,m=10000)))-1]
dat[,GeoPostcodeLonGLM    :=as.integer(as.factor(cut2(geo_postcode_lon,m=10000)))-1]
dat[,GeoMunicipalityGLM      :=as.integer(as.factor(geo_municipality_fr))-1]
dat[,GeoProvinceGLM      :=as.integer(as.factor(geo_province_fr))-1]
dat[,GeoMosaicFullGLM      :=as.integer(as.factor(geo_mosaic_code_full))-1]
dat[,GeoMosaicGLM      :=as.integer(as.factor(geo_mosaic_code))-1]

dat = as.data.frame(dat)

#reincluding numerical continous variables for splines

dat <-cbind(dat, db_train$driv_age1, db_train$driv_exp1, db_train$driv_age2, db_train$driv_exp2, 
            db_train$veh_age, db_train$veh_power, db_train$veh_weight, db_train$veh_sportivity,
            db_train$veh_value, db_train$geo_postcode_lat, db_train$geo_postcode_lon)

# Illustration of the backfitting algorithm

## First iteration

# First we start with a Poisson regression with only an intercept.

###############################################

#########  choosing learning and test sample

###############################################

set.seed(231)

ll <- sample(c(1:nrow(dat)), round(0.9*nrow(dat)), replace = FALSE)

learn <- dat[ll,]

test <- dat[-ll,]

(n_l <- nrow(learn))

(n_t <- nrow(test))



###############################################

#########  Building the model

###############################################

#GAM0

gam0_fit <-gam(claim_nb~1, data= learn, family=poisson(), offset=log(exposure))

learn$fitGAM0 <- fitted(gam0_fit)

test$fitGAM0 <- predict(gam0_fit, newdata= test, type="response")

#We fit a model with the discrete variables. (e.g. model from the GLM session)


#GAM1

#### splines on continous covariates

gam1_fit <- gam(claim_nb ~  ChannelGLM + GuaranteeGLM  + GuaranteeTypeGLM + PremiumPeriodicityGLM +
                  DomiciliatioGLMN  + TelematicGLM  + DrivNbGLM + s(driv_age1) + s(driv_age2) +       
                  s(driv_exp1) + s(driv_exp2) + DrivYoungGLM + VehBrandGLM + VehModelGLM +         
                  VehTypeGLM + InformexGLM + FebiacGLM + s(veh_age) + s(veh_power) +      
                  s(veh_weight) + s(veh_sportivity) + s(veh_value) + VehSeatsGLM + VehFuelGLM +         
                  VehMileageLimitGLM + VehTrailerGLM + VehGarageGLM + GeoPostcodeGLM + s(geo_postcode_lat) + 
                  s(geo_postcode_lon) + GeoMunicipalityGLM + GeoProvinceGLM + GeoMosaicFullGLM + GeoMosaicGLM,
                data = learn,
                offset = log(exposure),
                family=poisson())

learn$fitGAM1 <- fitted(gam1_fit)

test$fitGAM1 <- predict(gam1_fit, newdata= test, type="response")

#GAM2

gam2_fit <- gam(claim_nb ~ s(driv_age1) +      
                  s(driv_exp1) + s(driv_exp2) +         
                  VehTypeGLM+ FebiacGLM + s(veh_age) + s(veh_power) +      
                  s(veh_weight) + s(veh_sportivity) + VehFuelGLM +         
                  VehMileageLimitGLM + GeoPostcodeGLM,
                data = learn,
                offset = log(exposure),
                family=poisson())

learn$fitGAM2 <- fitted(gam2_fit)

test$fitGAM2 <- predict(gam2_fit, newdata= test, type="response")


#GAM3

gam3_fit <- gam(claim_nb ~  GuaranteeTypeGLM + PremiumPeriodicityGLM +
                  DomiciliatioGLMN  + TelematicGLM  + DrivNbGLM + s(driv_age1) +      
                  s(driv_exp1) + s(driv_exp2) + DrivYoungGLM +          
                  VehTypeGLM+ FebiacGLM + s(veh_age) + s(veh_power) +      
                  s(veh_weight) + s(veh_sportivity, driv_age1) + VehFuelGLM +         
                  VehMileageLimitGLM + GeoPostcodeGLM + GeoMunicipalityGLM,
                data = learn,
                offset = log(exposure),
                family=poisson())

learn$fitGAM3 <- fitted(gam3_fit)

test$fitGAM3 <- predict(gam3_fit, newdata= test, type="response")


###############################################

#########  In and Out Sample Losses

###############################################

##GLM
(glm0.insample <- 100*Poisson.Deviance(learn$fitGLM0, learn$claim_nb))
(glm0.oob <- 100*Poisson.Deviance(test$fitGLM0, test$claim_nb))

(glm1.insample <- 100*Poisson.Deviance(learn$fitGLM1, learn$claim_nb))
(glm1.oob <- 100*Poisson.Deviance(test$fitGLM1, test$claim_nb))

(glm2.insample <- 100*Poisson.Deviance(learn$fitGLM2, learn$claim_nb))
(glm2.oob <- 100*Poisson.Deviance(test$fitGLM2, test$claim_nb))

##Regression trees
(rt1.insample <- 100*Poisson.Deviance(learn$fitRT1, learn$claim_nb))
(rt1.oob <- 100*Poisson.Deviance(test$fitRT1, test$claim_nb))

(rt2.insample <- 100*Poisson.Deviance(learn$fitRT2, learn$claim_nb))
(rt2.oob <- 100*Poisson.Deviance(test$fitRT2, test$claim_nb))
##Poisson regression Boosting
(pbm1.insample <-100*Poisson.Deviance(learn$fitPBM1, learn$claim_nb))
(pbm1.oob <- 100*Poisson.Deviance(test$fitPBM1, test$claim_nb))
#GAM0_NULL
(gam0.insample <- 100*Poisson.Deviance(learn$fitGAM0, learn$claim_nb))
(gam0.oob <- 100*Poisson.Deviance(test$fitGAM0, test$claim_nb))
#GAM1

(gam1.insample <- 100*Poisson.Deviance(learn$fitGAM1, learn$claim_nb))
(gam1.oob <- 100*Poisson.Deviance(test$fitGAM1, test$claim_nb))



###############################################

######### Writing the results

###############################################


results2 = data.table(Model = "GLM0_NULL", OutOfSample = glm0.oob, InSample = glm0.insample )


results = rbind(results,
                data.table(Model = "GLM1", OutOfSample = glm1.oob, InSample = glm1.insample ))
results = rbind(results,
                data.table(Model = "GLM2", OutOfSample = glm2.oob, InSample = glm2.insample ))


results = rbind(results,
                data.table(Model = "RT1", OutOfSample = rt1.oob, InSample = rt1.insample ))
results = rbind(results,
                data.table(Model = "RT2", OutOfSample = rt2.oob, InSample = rt2.insample ))


results = rbind(results,
                data.table(Model = "PBM1", OutOfSample = pbm1.oob, InSample = pbm1.insample ))

results = rbind(results,
                data.table(Model = "GAM0_NULL", OutOfSample = gam0.oob, InSample = gam0.insample))

results = rbind(results,
                data.table(Model = "GAM1", OutOfSample = gam1.oob, InSample = gam1.insample))


results %>% fwrite("C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/model_results.csv")
###############################################

######### save the best model

###############################################


save(PBM.1, file ="C:/Machine Learning/NN_VIE_2/Projet/2018/3.Models/PBM1.6rda")





