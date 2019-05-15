##############################################

#########  IMPORT DATA

###############################################


library(readxl)
library(DataExplorer)

training_set <- read.csv("C:/Machine Learning/Pricing Game/Pricing Game (UCL-ULB)/pg2019/training_set.csv")

create_report(training_set, y= "Nclaim")
create_report(training_set, y= "Cclaim")
