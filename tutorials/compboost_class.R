# mboost comparison:
set.seed(pi)

n.test = 10

X = matrix(runif(n.test, 0, 2), ncol = 1)
y = 3 * as.numeric(X) + rnorm(n.test, 0, 2)

learning.rate = 0.05
iter.max = 20

# for mboost:
df = data.frame(y = y, x1 = X, x2 = X^2)

# Create new object (Note that we call a polynomial with degree 1):
bl.linear = BaselearnerWrapper$new("l1", X, "var1", 1)

# Create new object (Note that we call a polynomial with degree 2):
bl.quadratic = BaselearnerWrapper$new("q1", X, "var1", 2)

bl.linear$RegisterFactory("x")
bl.quadratic$RegisterFactory("x1")

printRegisteredFactorys()

# compboost:
cboost = CompboostWrapper$new(y, iter.max, learning.rate)
cboost$Train()

cboost$GetSelectedBaselearner()
# cboost$GetModelFrame()

# mboost:
library(mboost)

mod = mboost(
  formula = y ~ bols(x1, intercept = FALSE) + bols(x2, intercept = FALSE), 
  data    = df, 
  control = boost_control(mstop = iter.max, nu = learning.rate)
)

all.equal(predict(mod), cboost$GetPrediction())
# cboost$GetParameter()

# Benchmark:
microbenchmark::microbenchmark(
  "compboost" = cboost$Train(),
  "mboost"    = mboost(
    formula = y ~ bols(x1, intercept = FALSE) + bols(x2, intercept = FALSE), 
    data    = df, 
    control = boost_control(mstop = iter.max, nu = learning.rate)
  )
)

p = profvis::profvis({
  cboost$Train()
  mboost(
    formula = y ~ bols(x1, intercept = FALSE) + bols(x2, intercept = FALSE),
    data    = df, 
    control = boost_control(mstop = iter.max, nu = learning.rate)
  )
})

p
