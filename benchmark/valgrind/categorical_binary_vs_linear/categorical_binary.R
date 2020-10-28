cat("Load package\n")

devtools::load_all()
source("sim_data.R")

cat("Train compboost with binary base-learner")

cboost = Compboost$new(mydata, "y", loss = LossQuadratic$new())
cboost$addBaselearner("x", "category", BaselearnerCategoricalBinary)
cboost$train(mstop, trace = mstop/4)
