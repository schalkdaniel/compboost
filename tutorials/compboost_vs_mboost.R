# ============================================================================ #
#                            compboost vs mboost                               #
# ============================================================================ #

# Just a small comparison, maybe more a check if compboost does the same
# as mboost using mtcars.

# Prepare Data:
# -------------

df = mtcars

# Create new variable to check the polynomial baselearner with degree 2:
df$hp2 = df[["hp"]]^2

# Data for compboost:
X.hp = as.matrix(df[["hp"]], ncol = 1)
X.wt = as.matrix(df[["wt"]], ncol = 1)

y = df[["mpg"]]

# Hyperparameter for the algorithm:
learning.rate = 0.05
iter.max = 500


# Prepare compboost:
# ------------------

# Create new linear baselearner of hp and wt:
linear.factory.hp = PolynomialFactory$new(X.hp, "hp", 1)
linear.factory.wt = PolynomialFactory$new(X.wt, "wt", 1)

# Create new quadratic baselearner of hp:
quadratic.factory.hp = PolynomialFactory$new(X.hp, "hp", 2)

# Create new factory list:
factory.list = FactoryList$new()

# Register factorys:
factory.list$registerFactory(linear.factory.hp)
factory.list$registerFactory(linear.factory.wt)
factory.list$registerFactory(quadratic.factory.hp)

factory.list$printRegisteredFactorys()

# We use quadratic loss:
loss = QuadraticLoss$new()

# Run compboost:
# --------------

# Initialize object (Response, learning rate, maximal iterations, stop if all
# stopper are fulfilled?, maximal microseconds, factory list):
cboost = Compboost$new(y, learning.rate, iter.max, TRUE, 0, factory.list, loss)

# Train the model (here we don't need the trace):
cboost$train(FALSE)

# Get vector selected baselearner:
cboost$getSelectedBaselearner()
# cboost$GetModelFrame()
cboost$getLoggerData()


# Do the same with mboost:
# ------------------------

library(mboost)

mod = mboost(
  formula = mpg ~ bols(hp, intercept = FALSE) + 
    bols(wt, intercept = FALSE) +
    bols(hp2, intercept = FALSE), 
  data    = df, 
  control = boost_control(mstop = iter.max, nu = learning.rate)
)

# Check if the selected baselearner are the same:
# -----------------------------------------------

cboost.xselect = match(
  x     = cboost$getSelectedBaselearner(), 
  table = c(
    "hp: polynomial with degree 1", 
    "wt: polynomial with degree 1", 
    "hp: polynomial with degree 2"
  )
)

all.equal(predict(mod), cboost$getPrediction())

# Check if the prediction is the same:
# ------------------------------------

all.equal(mod$xselect(), cboost.xselect)
# cboost$GetParameter()

# Benchmark:
# ----------

# Time comparison:
microbenchmark::microbenchmark(
  "compboost" = cboost$train(FALSE),
  "mboost"    = mboost(
    formula = mpg ~ bols(hp, intercept = FALSE) + 
      bols(wt, intercept = FALSE) +
      bols(hp2, intercept = FALSE), 
    data    = df, 
    control = boost_control(mstop = iter.max, nu = learning.rate)
  ),
  times = 10L
)

# Profiling to compare used memory:
p = profvis::profvis({
  cboost$train(FALSE)
  mboost(
    formula = mpg ~ bols(hp, intercept = FALSE) + 
      bols(wt, intercept = FALSE) +
      bols(hp2, intercept = FALSE), 
    data    = df, 
    control = boost_control(mstop = iter.max, nu = learning.rate)
  )
})

print(p)

