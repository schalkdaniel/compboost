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
bl = LinearWrapper$new(X, "linear baselearner 1")

# Get identifier:
bl$GetId()

# Get the data:
bl$GetData()

# Train:
bl$train(y)

# Predict:
bl$predict(y)

X = matrix(1:1000000, ncol = 1)
y = 3 * as.numeric(X) + rnorm(1000000, 0, 200)

# Create new object:
bl = LinearWrapper$new(X, "linear baselearner 2")
bl$GetId()

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

