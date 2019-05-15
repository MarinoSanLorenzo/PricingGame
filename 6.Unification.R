####unification

proxy <- cbind(pred_testing_set_Keras_table, pred_testing_set_GLM_table[,2])

sampleSubmission <- proxy$NClaim*proxy$CClaim

sampleSubmission <- data.table(Id_Policyholder = proxy$Id_Policyholder, pure_premium= sampleSubmission)

head(sampleSubmission)

sampleSubmission  %>% fwrite("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/export/sampleSubmission.csv")

training_set_0 <- training_set %>%
  filter(Cclaim > 0) 

cclaim_0 <- hist(training_set_0["Cclaim"])

training_set_15k <- training_set %>%
  filter(Cclaim > 15000) 

cclaim_15k <- hist(training_set_15k["Cclaim"])

freq_mean <- mean(training_set$Nclaim)
sev_mean <- mean(training_set$Cclaim)

pp_theoric <- freq_mean*sev_mean

