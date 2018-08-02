n.rows = 10000
p.feat = 200
iterations = 200
knots = 100

y = rnorm(n.rows)
X = matrix(rnorm(n.rows * p.feat), nrow = n.rows) 

mydf = data.frame(y = y, X)

devtools::load_all()
library(mboost)

options(mboost_useMatrix = FALSE)

mboost.formula = paste0("y ~ ", paste(paste0("bbs(", names(mydf)[-1], ", knots = ", knots, ")"), collapse = " + "))

microbenchmark::microbenchmark(
	mod.mboost = gamboost(as.formula(mboost.formula), data = mydf, control = boost_control(mstop = iterations, nu = 0.05, trace = TRUE)),
	mod.cboost = boostSplines(data = mydf, target = "y", n.knots = knots, loss = LossQuadratic$new(), iterations = iterations),
	times = 2L
)