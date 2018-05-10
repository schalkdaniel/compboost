# ============================================================================ #
#                                   EXAMPLES                                   #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Run Compboost
# ---------------------------------------------------------------------------- #

# Prepare Data:
# -------------

# Binary classification if mpg is greater than 20 or not. We want to log using
# the auc measure of mlr. Therefore we define a custom loss (Just the loss
# fct is required here):

# Note that a tweak is necessary since the logging 
aucLoss = function (truth, response) {
  # Convert response on f basis to probs using sigmoid:
  probs = 1 / (1 + exp(-response))
  
  #  Calculate AUC:
  mlr:::measureAUC(probabilities = probs, truth = truth, negative = -1, positive = 1) 
}

gradDummy = function (trutz, response) { return (NA) }
constInitDummy = function (truth, response) { return (NA) }

# Define loss:
auc.loss = CustomLoss$new(aucLoss, gradDummy, constInitDummy)

# Test loss:
response = rnorm(10)
truth = rbinom(10, 1, 0.3) * 2 - 1

auc.loss$testLoss(truth, response)



# Some data:
df = mtcars
df$mpg.cat = ifelse(df$mpg > 20, 1, -1)

# # Create new variable to check the polynomial baselearner with degree 2:
# df$hp2 = df[["hp"]]^2

# Data for the baselearner are matrices:
X.hp = as.matrix(df[["hp"]])
X.wt = as.matrix(df[["wt"]])

# Target variable:
y = df[["mpg.cat"]]

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
linear.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target.hp1, 1)
linear.factory.wt = PolynomialBlearnerFactory$new(data.source.wt, data.target.wt1, 1)

# Create new quadratic baselearner of hp:
quadratic.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target.hp2, 2)

# Create spline factory for wt with:
#   - degree: 2
#   - 10 knots
#   - penalty parameter: 2
#   - differences: 2
spline.factory.wt = PSplineBlearnerFactory$new(data.source.wt, data.target.wt2, 3, 10, 2, 2)


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
loss.bin = BinomialLoss$new()


## Optimizer

# Use the greedy optimizer:
optimizer = GreedyOptimizer$new()

## Logger

# Define logger. We want just the iterations as stopper but also track the
# time, inbag risk and oob risk:
log.iterations  = IterationLogger$new(TRUE, 500)
log.time        = TimeLogger$new(FALSE, 500, "microseconds")
log.inbag       = InbagRiskLogger$new(FALSE, loss.bin, 0.05)
log.oob         = OobRiskLogger$new(FALSE, loss.bin, 0.05, oob.data, y)
log.auc.inbag   = InbagRiskLogger$new(FALSE, auc.loss, 0.05)
log.auc.oob     = OobRiskLogger$new(FALSE, auc.loss, 0.05, oob.data, y)

# Define new logger list:
logger.list = LoggerList$new()

logger.list

# Register the logger:
logger.list$registerLogger(" iteration.logger", log.iterations)
logger.list$registerLogger("time.logger", log.time)
logger.list$registerLogger("inbag.binomial", log.inbag)
logger.list$registerLogger("oob.binomial", log.oob)
logger.list$registerLogger("inbag.auc", log.auc.inbag)
logger.list$registerLogger("oob.auc", log.auc.oob)

logger.list$printRegisteredLogger()

# Run compboost:
# --------------

# Initialize object:
cboost = Compboost$new(
  response      = y,
  learning_rate = 0.05,
  stop_if_all_stopper_fulfilled = FALSE,
  factory_list = factory.list,
  loss         = loss.bin,
  logger_list  = logger.list,
  optimizer    = optimizer
)

# Train the model (we want to print the trace):
cboost$train(trace = TRUE)

cboost

# Confusion matrix:
conf.mat = table(real = y, pred = ifelse(cboost$getPrediction() > 0, 1, -1))
conf.mat

# Store logger data and check if final auc is correct:
log.data = cboost$getLoggerData()

auc.loss$testLoss(y, cboost$getPrediction())
log.data$logger.data[, log.data$logger.names == "inbag.auc"]
