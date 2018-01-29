# ============================================================================ #
#                                   EXAMPLES                                   #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Run Compboost
# ---------------------------------------------------------------------------- #

# Prepare Data:
# -------------

# Some data:
df = mtcars

# # Create new variable to check the polynomial baselearner with degree 2:
# df$hp2 = df[["hp"]]^2

# Data for the baselearner are matrices:
X.hp = cbind(1, df[["hp"]])
X.wt = cbind(1, df[["wt"]])

# Target variable:
y = df[["mpg"]]

# Next lists are the same as the used data. Then we can have a look if the oob
# and inbag logger and the train prediction and prediction on newdata are doing
# the same.

# List for oob logging:
eval.oob.test = list(
  "hp" = X.hp,
  "wt" = X.wt
)

# List to test prediction on newdata:
eval.data = eval.oob.test


# Prepare compboost:
# ------------------

## Baselearner

# Create new linear baselearner of hp and wt:
linear.factory.hp = PolynomialBlearnerFactory$new(X.hp, "hp", 1)
linear.factory.wt = PolynomialBlearnerFactory$new(X.wt, "wt", 1)

# Create new quadratic baselearner of hp:
quadratic.factory.hp = PolynomialBlearnerFactory$new(X.hp, "hp", 2)

# Create new factory list:
factory.list = BlearnerFactoryList$new()

# Register factorys:
factory.list$registerFactory(linear.factory.hp)
factory.list$registerFactory(linear.factory.wt)
factory.list$registerFactory(quadratic.factory.hp)

# Print the registered factorys:
factory.list$printRegisteredFactorys()

# Print model.frame:
factory.list$getModelFrame()


## Loss

# Use quadratic loss:
loss.quadratic = QuadraticLoss$new()


## Optimizer

# Use the greedy optimizer:
optimizer = GreedyOptimizer$new()

## Logger

# Define logger. We want just the iterations as stopper but also track the
# time, inbag risk and oob risk:
log.iterations = LogIterations$new(TRUE, 500)
log.time       = LogTime$new(FALSE, 500, "microseconds")
log.inbag      = LogInbagRisk$new(FALSE, loss.quadratic, 0.05)
log.oob        = LogOobRisk$new(FALSE, loss.quadratic, 0.05, eval.oob.test, y)

# Define new logger list:
logger.list = LoggerList$new()

# Register the logger:
logger.list$registerLogger(log.iterations)
logger.list$registerLogger(log.time)
logger.list$registerLogger(log.inbag)
logger.list$registerLogger(log.oob)

logger.list$printRegisteredLogger()

# Run compboost:
# --------------

# Initialize object:
cboost = Compboost$new(
  response      = y,
  learning_rate = 0.05,
  stop_if_all_stopper_fulfilled = FALSE,
  factory_list = factory.list,
  loss         = loss.quadratic,
  logger_list  = logger.list,
  optimizer    = optimizer
)

# Train the model (we want to print the trace):
cboost$train(trace = TRUE)

# Get some results:
cboost$getEstimatedParameter()
cboost$getSelectedBaselearner()

cboost$getEstimatedParameterOfIteration(40)
parameter.matrix = cboost$getParameterMatrix()

all.equal(cboost$getPrediction(), cboost$predict(eval.data))

# ---------------------------------------------------------------------------- #
# Extend Compboost
# ---------------------------------------------------------------------------- #

# Linear model as benchmark:
mod = lm(mpg ~ hp, data = mtcars)

# R Functions:
# ------------

instantiateData = function (X)
{
  return(X);
}
trainFun = function (y, X) {
  return(solve(t(X) %*% X) %*% t(X) %*% y)
}
predictFun = function (model, newdata) {
  return(newdata %*% model)
}
extractParameter = function (model) {
  return(model)
}

custom.learner = CustomBlearner$new(X.hp, "hp", instantiateData, trainFun, predictFun,
  extractParameter)

# Train learner:
custom.learner$train(y)
custom.learner$getParameter()

coef(mod)

# Create fatorys is very similar:
custom.factory = CustomBlearnerFactory$new(X.hp, "hp", instantiateData, trainFun,
  predictFun, extractParameter)

# C++ Functions:
# --------------

Rcpp::sourceCpp(file = "tutorials/custom_cpp_learner.cpp")

custom.cpp.learner = CustomCppBlearner$new(X.hp, "hp", dataFunSetter(), trainFunSetter(),
  predictFunSetter())

# Train learner:
custom.cpp.learner$train(y)
custom.cpp.learner$getParameter()

coef(mod)

# Create fatorys is very similar:
custom.cpp.factory = CustomCppBlearnerFactory$new(X.hp, "hp", dataFunSetter(),
  trainFunSetter(), predictFunSetter())

# Small (unfair) Benchmark:
# -------------------------

linear.learner = PolynomialBlearner$new(X.hp, "hp", 1)

microbenchmark::microbenchmark(
  "Linear Model in R"       = lm(mpg ~ hp, data = mtcars),
  "Custom Learner (R)"      = custom.learner$train(y),
  "Custom Learner (C++)"    = custom.cpp.learner$train(y),
  "Implemented Baselearner" = linear.learner$train(y)
)

# ---------------------------------------------------------------------------- #
# Small Comparison with mboost
# ---------------------------------------------------------------------------- #

df[["hp2"]] = df[["hp"]]^2

library(mboost)

mod = mboost(
  formula = mpg ~ bols(hp) + bols(wt) + bols(hp2),
  data    = df,
  control = boost_control(mstop = 500, nu = 0.05)
)

# Does compboost the same as mboost?
# ----------------------------------

# Estimated Paramter:
coef(mod)
cboost$getEstimatedParameter()

# Prediction:
all.equal(predict(mod), cboost$getPrediction())

# Selected Baselearner:
cboost.xselect = match(
  x     = cboost$getSelectedBaselearner(),
  table = c(
    "hp: polynomial with degree 1",
    "wt: polynomial with degree 1",
    "hp: polynomial with degree 2"
  )
)

all.equal(mod$xselect(), cboost.xselect)

# Benchmark:
microbenchmark::microbenchmark(
  "compboost" = cboost$train(FALSE),
  "mboost"    = mboost(
    formula = mpg ~ bols(hp) + bols(wt) + bols(hp2),
    data    = df,
    control = boost_control(mstop = 500, nu = 0.05)
  )
)
