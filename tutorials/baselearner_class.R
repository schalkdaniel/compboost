# Define a linear baselearner object:
# ===================================

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

# Create new object (Note that we call a polynomial with degree 1). Note that
# the data identifier (third argument) has to be exactly the same string as
# the corresponding colname in a dataframe later on:
bl = BaselearnerWrapper$new("l1", X, "my_name_is_important", 1)

# Get identifier:
bl$GetIdentifier()

# Get the type:
bl$GetBaselearnerType()

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
bl = BaselearnerWrapper$new("l2", X, "my_name_is_important", 1)
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

# The same for a quadratic baselearner:
# =====================================

X = matrix(1:10, ncol = 1)
y = 2 * (1:10)^2 + rnorm(10, 0, 20)

# Create new object:
bl = BaselearnerWrapper$new("q1", X, "my_name_is_also_important", 2)

# Get identifier:
bl$GetIdentifier()

# Get the data:
bl$GetData()

# Train with given y:
bl$train(y)

X0 = X^2
mod.r = lm(y ~ 0 + X0)

# Get Paramter:
bl$GetParameter()
coef(mod.r)

# Predict:
bl$predict()
predict(mod.r)

# Define a custom baselearner object:
# ===================================

instantiateDataFun = function (X) {
  return(X)
}

trainFun = function (y, X) {
  return(lm(y ~ 0 + X))
}

predictFun = function (model, newdata) {
  return(as.matrix(predict(model, as.data.frame(newdata))))
}

extractParameter = function (model) {
  return(as.matrix(coef(model)))
}

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

# Create custom baselearner:
bl = BaselearnerWrapper$new("linear custom 1", X, "and_my_name_too", instantiateDataFun, trainFun, 
  predictFun, extractParameter)

# Get identifier:
bl$GetIdentifier()

# Get the data:
bl$GetData()

# Train with given y:
bl$train(y)
mod = lm(y ~ 0 + X)

# Since we specified an 'extractParameter' function we have some parameters 
# here:
bl$GetParameter()
coef(mod)

# But we can predict:
bl$predict()
predict(mod)

# Comparison between custom and inline baselearner:
# =================================================

X = matrix(1:1000000, ncol = 1)
y = 3 * as.numeric(X) + rnorm(1000000, 0, 200)

# Create inline baselearner:
bl.inline = BaselearnerWrapper$new("inline l1", X, "x_variable", 1)
bl.inline$GetIdentifier()
bl.inline$train(y)

# Create custom baselearner
bl.custom = BaselearnerWrapper$new("linear custom", X, "x_variable", instantiateDataFun, 
  trainFun, predictFun, extractParameter)
bl.custom$GetIdentifier()
bl.custom$train(y)

# Do both baselearner compute the same?
all.equal(bl.inline$predict(), bl.custom$predict())

# Benchmark the two versions:
microbenchmark::microbenchmark(
  "inline" = bl.inline$train(y),
  "custom" = bl.custom$train(y),
  "R"      = lm(y ~ 0 + X),
  times = 20
)
