# ============================================================================ #
#                                                                              #
#                        Benchmark of compboost vs mboost                      #
#                                                                              #
# ============================================================================ #

# This file is intended to create microbenchmarks by hand to quickly get an
# idea about the performance.

library(mboost)
library(compboost)

# Function to get formula for mboost:
# -------------------------------------------------
getMboostFormula = function (data, target, learner = "spline") {
  data.names = setdiff(names(data), target)
  if (learner == "spline") {
    myformula = paste0(
      target, " ~ ",
      paste(
        paste0("bbs(", data.names, ", knots = 20, degree = 3, differences = 2, lambda = 2)"),
        collapse = " + "
        )
      )  
  } 
  if (learner == "linear") {
    myformula = paste0(
      target, " ~ ",
      paste(
        paste0("bols(", data.names, ")"),
        collapse = " + "
        )
      )  
  }
  return (as.formula(myformula))
}

# Function to simulate Data:
# -------------------------------------------------

simulateData = function (nrows, ncols, seed) {
  set.seed(seed)

  return (
    data.frame(
      target     = rnorm(n = nrows, mean = 100, sd = 10),
      # y.classification = sample(x = c(-1, 1), size = nrows, replace = TRUE),
      data             = as.data.frame(
        matrix(
          runif(
            n   = nrows * ncols,
            min = seq(-10, 10, length.out = ncols),
            max = seq(10, 20, length.out = ncols)
            ), nrow = nrows, ncol = ncols, byrow = TRUE
          )
        )
      )
    )
}

set.seed(314159)

# Fix dataset:
mydata = simulateData(
  nrows = 500000,
  ncol = 100,
  seed = round(pi * 1000)
  )

# Fix parameters:
iters = 500
learning.rate = 0.05
penalty = 4

check = FALSE

if (check) {
  # Compare spline learner:
  # -------------------------------------------------

  cboost.mod.spline = boostSplines(data = mydata, target = "target", 
    loss = LossQuadratic$new(), iterations = iters, penalty = 2)
  mboost.mod.spline = gamboost(getMboostFormula(mydata, "target"), data = mydata, 
    control = boost_control(mstop = iters, nu = learning.rate, trace = TRUE))

  # Are the results the same:
  all.equal(cboost.mod.spline$predict(), predict(mboost.mod.spline))

  # Compare linear learner:
  # -------------------------------------------------

  cboost.mod.linear = boostLinear(data = mydata, target = "target", 
    loss = LossQuadratic$new(), iterations = iters)
  mboost.mod.linear = glmboost(target ~ ., data = mydata, 
    control = boost_control(mstop = iters, nu = learning.rate, trace = TRUE))

  # Are the results the same:
  all.equal(cboost.mod.linear$predict(), predict(mboost.mod.linear), check.attributes = FALSE)
}
# Small benchmark:
# -------------------------------------------------

microbenchmark::microbenchmark(
  "compboost.spline" = boostSplines(data = mydata, target = "target", loss = LossQuadratic$new(), iterations = iters, penalty = 2, trace = FALSE),
  "mboost.spline" = gamboost(getMboostFormula(mydata, "target"), data = mydata, control = boost_control(mstop = iters, nu = learning.rate)),
  # "compboost.linear" = boostLinear(data = mydata, target = "target", loss = LossQuadratic$new(), iterations = iters, trace = FALSE),
  # "mboost.linear" = glmboost(target ~ ., data = mydata, control = boost_control(mstop = iters, nu = learning.rate)),
  # "mboost.linear.plain" = mboost(getMboostFormula(mydata, "target", "linear"), data = mydata, control = boost_control(mstop = iters, nu = learning.rate)),
  times = 1L
  )
