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
bl.linear.hp = BaselearnerWrapper$new("l1", X.hp, "hp", 1)
bl.linear.wt = BaselearnerWrapper$new("l2", X.wt, "wt", 1)

# Create new quadratic baselearner of hp:
bl.quadratic.hp = BaselearnerWrapper$new("q1", X.hp, "hp", 2)

# Register factorys:
bl.linear.hp$RegisterFactory("linear factory 1")
bl.linear.wt$RegisterFactory("linear factory 2")
bl.quadratic.hp$RegisterFactory("quadratic factory 1")

printRegisteredFactorys()

# Run compboost:
# --------------

# Initialize object (Response, maximal iterations, learning maximal seconds):
cboost = CompboostWrapper$new(y, iter.max, learning.rate, 0)

# Train the model:
cboost$Train()

# Get vector selected baselearner:
cboost$GetSelectedBaselearner()
# cboost$GetModelFrame()
cboost$GetLoggerData()

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
  x     = cboost$GetSelectedBaselearner(), 
  table = c(
    "hp: polynomial with degree 1", 
    "wt: polynomial with degree 1", 
    "hp: polynomial with degree 2"
    )
  )

all.equal(predict(mod), cboost$GetPrediction())

# Check if the prediction is the same:
# ------------------------------------

all.equal(mod$xselect(), cboost.xselect)
# cboost$GetParameter()

# Benchmark:
# ----------

# Time comparison:
microbenchmark::microbenchmark(
  "compboost" = cboost$Train(),
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
  cboost$Train()
  mboost(
    formula = mpg ~ bols(hp, intercept = FALSE) + 
      bols(wt, intercept = FALSE) +
      bols(hp2, intercept = FALSE), 
    data    = df, 
    control = boost_control(mstop = iter.max, nu = learning.rate)
  )
})

print(p)
