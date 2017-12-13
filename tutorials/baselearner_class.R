# Install the package:
# ====================

library(Rcpp)

compileAttributes()
devtools::load_all()

# Define a linear baselearner object:
# ===================================

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

# Create new object:
bl = BaselearnerWrapper$new("linear", "l1", X)

# Get identifier:
bl$GetIdentifier()

# Get the data:
bl$GetData()

# Train with given y:
bl$train(y)
mod.r = lm(y ~ 0 + X)

# Get Paramter:
bl$GetParameter()
coef(mod.r)

# Predict:
bl$predict()
predict(mod.r)


X = matrix(1:1000000, ncol = 1)
y = 3 * as.numeric(X) + rnorm(1000000, 0, 200)

# Create new object:
bl = BaselearnerWrapper$new("linear", "baselearner 2", X)
bl$GetIdentifier()

# Benchmark parameter calculation:
microbenchmark::microbenchmark(
  "C++"    = bl$train(y),
  "R"      = lm(y ~ 0 + X),
  "fastLm" = RcppArmadillo::fastLm(X, y),
  times = 20
)

profvis::profvis({
  cpp_run    = bl$train(y)
  r_run      = lm(y ~ 0 + X)
  fastLm_run = RcppArmadillo::fastLm(X, y)
})

