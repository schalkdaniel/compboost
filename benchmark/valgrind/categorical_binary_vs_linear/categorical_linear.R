cat("Load package\n")

devtools::load_all()
source("sim_data.R")

cat("Train compboost with polynomial base-learner\n")

cboost = Compboost$new(mydata, "y", loss = LossQuadratic$new())
cboost$addBaselearner("x", "category", BaselearnerPolynomial, intercept = FALSE)
cboost$train(mstop, trace = mstop/4)
