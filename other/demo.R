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
X.hp = as.matrix(df[["hp"]])
X.wt = as.matrix(df[["wt"]])

# Target variable:
y = df[["mpg"]]

data.source.hp = InMemoryData$new(X.hp, "hp")
data.source.wt = InMemoryData$new(X.wt, "wt")

data.target.hp1 = InMemoryData$new()
data.target.hp2 = InMemoryData$new()
data.target.wt1 = InMemoryData$new()
data.target.wt2 = InMemoryData$new()

# Next lists are the same as the used data. Then we can have a look if the oob
# and inbag logger and the train prediction and prediction on newdata are doing
# the same.

# List for oob logging:
oob.data = list(data.source.hp, data.source.wt)

# List to test prediction on newdata:
test.data = oob.data


# Prepare compboost:
# ------------------

## Baselearner

# Create new linear baselearner of hp and wt:
linear.factory.hp = BaselearnerPolynomialFactory$new(data.source.hp, data.target.hp1, 1)
linear.factory.wt = BaselearnerPolynomialFactory$new(data.source.wt, data.target.wt1, 1)

# Create new quadratic baselearner of hp:
quadratic.factory.hp = BaselearnerPolynomialFactory$new(data.source.hp, data.target.hp2, 2)

# Create spline factory for wt with:
#   - degree: 2
#   - 10 knots
#   - penalty parameter: 2
#   - differences: 2
spline.factory.wt = BaselearnerPSplineFactory$new(data.source.wt, data.target.wt2, 3, 10, 2, 2)


# Create new factory list:
factory.list = BlearnerFactoryList$new()

# Register factorys:
factory.list$registerFactory(linear.factory.hp)
factory.list$registerFactory(linear.factory.wt)
factory.list$registerFactory(quadratic.factory.hp)
factory.list$registerFactory(spline.factory.wt)

# Print the registered factorys:
factory.list$printRegisteredFactories()

# Print model.frame:
factory.list$getModelFrame()


## Loss

# Use quadratic loss:
loss.quadratic = QuadraticLoss$new()


## Optimizer

# Use the greedy optimizer:
optimizer = OptimizerCoordinateDescent$new()

## Logger

# Define logger. We want just the iterations as stopper but also track the
# time, inbag risk and oob risk:
log.iterations = IterationLogger$new(TRUE, 500)
log.time       = TimeLogger$new(FALSE, 500, "microseconds")
log.inbag      = InbagRiskLogger$new(FALSE, loss.quadratic, 0.05)
log.oob        = OobRiskLogger$new(FALSE, loss.quadratic, 0.05, oob.data, y)

# Define new logger list:
logger.list = LoggerList$new()

logger.list

# Register the logger:
logger.list$registerLogger(" iterations", log.iterations)
logger.list$registerLogger("time", log.time)
logger.list$registerLogger("inbag.risk", log.inbag)
logger.list$registerLogger("oob.risk", log.oob)

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

cboost

# Continue Training:
# Define logger. We want just the iterations as stopper but also track the
# time, inbag risk and oob risk:
log.iterations.continue = IterationLogger$new(TRUE, 200)
log.time.continue       = TimeLogger$new(FALSE, 500, "microseconds")

# Define new logger list:
logger.list.continue = LoggerList$new()

# Register the logger:
logger.list.continue$registerLogger("iterations", log.iterations.continue)
logger.list.continue$registerLogger("time", log.time.continue)

cboost$continueTraining(trace = TRUE, logger = logger.list.continue)

# We also could just use cboost$setToIteration(700) which automatically 
# continue training for 200 iterations. The iterations logger is created 
# automatically.

# Get some results:
cboost$getEstimatedParameter()
cboost$getSelectedBaselearner()

cboost$getParameterAtIteration(50)
parameter.matrix = cboost$getParameterMatrix()

all.equal(cboost$getPrediction(), cboost$predict(test.data))

# Set model to a specific iteration. This is faster than calling the 
# predictAtIteration function since predictAtIteration calculates the 
# parameter at iteration k every time, while in setToIteration this is done
# once and is then reused for the other functions:
cboost$setToIteration(10)

all.equal(
  cboost$getEstimatedParameter(),
  cboost$getParameterAtIteration(10)
)
all.equal(
  cboost$getPrediction(),
  cboost$predictAtIteration(oob.data, 10)
)

microbenchmark::microbenchmark(
  "After 'setToIteration'"  = cboost$getPrediction(),
  "Before 'setToIteration'" = cboost$predictAtIteration(oob.data, 10)
)

# If one sets the iteration to a higher value than already trained learner,
# then the model is automatically retrained.



## The factories should work, the baselearner needs to be adopt to the new
## data class!


# # ---------------------------------------------------------------------------- #
# # Extend Compboost
# # ---------------------------------------------------------------------------- #
# 
# # Linear model as benchmark:
# mod = lm(mpg ~ 0 + hp, data = mtcars)
# 
# # R Functions:
# # ------------
# 
# instantiateData = function (X)
# {
#   return(X);
# }
# trainFun = function (y, X) {
#   return(solve(t(X) %*% X) %*% t(X) %*% y)
# }
# predictFun = function (model, newdata) {
#   return(newdata %*% model)
# }
# extractParameter = function (model) {
#   return(model)
# }
# 
# data.hp3 = IdentityData$new(X.hp, "hp")
# custom.learner = BaselearnerCustom$new(data.hp3, instantiateData, trainFun, predictFun,
#   extractParameter)
# 
# # Train learner:
# custom.learner$train(y)
# custom.learner$getParameter()
# 
# coef(mod)
# 
# # Create fatorys is very similar:
# custom.factory = BaselearnerCustomFactory$new(data.hp3, instantiateData, trainFun,
#   predictFun, extractParameter)
# 
# # C++ Functions:
# # --------------
# 
# file.edit("tutorials/custom_cpp_learner.cpp")
# Rcpp::sourceCpp(file = "tutorials/custom_cpp_learner.cpp")
# 
# custom.cpp.learner = BaselearnerCustomCpp$new(X.hp, "hp", dataFunSetter(), trainFunSetter(),
#   predictFunSetter())
# 
# # Train learner:
# custom.cpp.learner$train(y)
# custom.cpp.learner$getParameter()
# 
# coef(mod)
# 
# # Create fatorys is very similar:
# custom.cpp.factory = BaselearnerCustomCppFactory$new(data.hp3, dataFunSetter(),
#   trainFunSetter(), predictFunSetter())
# 
# # Small (unfair) Benchmark:
# # -------------------------
# 
# linear.learner = BaselearnerPolynomial$new(X.hp, "hp", 1)
# 
# microbenchmark::microbenchmark(
#   "Linear Model in R"       = lm(mpg ~ hp, data = mtcars),
#   "Custom Learner (R)"      = custom.learner$train(y),
#   "Custom Learner (C++)"    = custom.cpp.learner$train(y),
#   "Implemented Baselearner" = linear.learner$train(y)
# )

# ---------------------------------------------------------------------------- #
# Small Comparison with mboost
# ---------------------------------------------------------------------------- #

df[["hp2"]] = df[["hp"]]^2

library(mboost)

mod = mboost(
  formula = mpg ~ bols(hp, intercept = FALSE) + 
    bols(wt, intercept = FALSE) + 
    bols(hp2, intercept = FALSE) +
    bbs(wt, knots = 10, degree = 3, differences = 2, lambda = 2),
  data    = df,
  control = boost_control(mstop = 700, nu = 0.05)
)

cboost$setToIteration(700)

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
    "hp: polynomial with degree 2",
    "wt: spline with degree 3"
  )
)

all.equal(mod$xselect(), cboost.xselect)

# Benchmark:
microbenchmark::microbenchmark(
  "compboost" = cboost$train(FALSE),
  "mboost"    = mboost(
    formula = mpg ~ bols(hp, intercept = FALSE) + bols(wt, intercept = FALSE) + bols(hp2, intercept = FALSE),
    data    = df,
    control = boost_control(mstop = 500, nu = 0.05)
  ),
  "glmboost" = glmboost(mpg ~ hp + wt + hp2, 
    data = df, 
    control = boost_control(mstop = 500, nu = 0.05)
  )
)
