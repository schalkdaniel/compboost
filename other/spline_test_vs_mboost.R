splines.cpp = "C:/Users/schal/OneDrive/github_repos/compboost/other/splines.cpp"

if (file.exists(splines.cpp)) {
  Rcpp::sourceCpp(file = splines.cpp, rebuild = TRUE)
}

library(Matrix)

n.sim = 2000

x = sort(c(1, seq(0, 11, length.out = n.sim), 10.9))
y = 3 + 1.4 * x + 4 * sin(x) + rnorm(length(x), 0, 1)

plot(x,y)

n.knots = 30
degree = 3

differences = 2

# myknots = seq(min(x), max(x), length.out = n.knots + 2)
# 
# knot.range = diff(myknots)[1]
# myknots = c(min(x) - degree:1 * knot.range, myknots, max(x) + 1:degree * knot.range)

penalty = 2

myspline = function (penalty, difference, ...) {
  
  # Do all by hand:
  knots = createKnots(values = x, n_knots = n.knots, degree = degree)
  basis = createBasis(values = x, degree = degree, knots = knots)
  
  param.estimate = estimateSplines(response = y, x = x, degree = degree, 
    n_knots = n.knots, differences = differences, penalty = penalty)
  
  y.hat = basis %*% param.estimate
  
  lines(x, y.hat, ...)
}

plot(x, y)
myspline(10, 1, col = "blue")
myspline(10, 2, col = "green")
myspline(10, 3, col = "red")


# small benchmark with r's splines library:
microbenchmark::microbenchmark(
  "My C++" = createBasis(values = x, degree = degree, knots = knots),
  "R splines" = splines::splineDesign(knots, x = x, outer.ok = TRUE, ord = degree + 1), 
  times = 3L
)







# Compare with mboosts bbs:

n.sim = 2000

x = sort(c(1, seq(0, 11, length.out = n.sim), 10.9))
y = 3 + 1.4 * x + 4 * sin(x) + rnorm(length(x), 0, 1)

df = 4
# For differences unequal of 2 the results do slightly differ ???
differences = 2
n.knots = 30

degree = 3

weights = rep(1, length(y))

D = diff(diag(n.knots + (degree + 1)), differences = differences)
pen.mat = t(D) %*% D


# Create spline learner, extract basis and convert df to lambda:

library(mboost)

spline1 = bbs(x, knots = n.knots, df = df, boundary.knots = range(x))
bb.mboost = extract(spline1, "design")

penalty = mboost:::df2lambda(bb.mboost, df = df, dmat = pen.mat,
  weights = weights)

penalty.lambda = penalty["lambda"]



### compute base-model
mod.mboost = spline1$dpp(weights)

# Fit the base learner:
fit = mod.mboost$fit(y)

# Parameter estimator:
param.mboost = as.matrix(fit$model)

dim.mboost = dim(param.mboost)
attributes(param.mboost) = NULL
dim(param.mboost) = dim.mboost

# Predictions:
predictions.mboost = fit$fitted()


# My model:
knots = createKnots(values = x, n_knots = n.knots, degree = degree)
basis = createBasis(values = x, degree = degree, knots = knots)

param.estimate = estimateSplines(response = y, x = x, degree = degree, 
  n_knots = n.knots, differences = differences, penalty = penalty.lambda)
predictions.own = basis %*% param.estimate

# Check if it is the same:
all.equal(param.mboost, param.estimate)
all.equal(predictions.mboost, as.numeric(predictions.own))


# param.mboost.own = solve(t(mboost.mat) %*% mboost.mat + penalty.lambda * pen.mat) %*% t(mboost.mat) %*% y
# param.own.own = solve(t(basis) %*% basis + penalty.lambda * pen.mat) %*% t(basis) %*% y
# 
# 
# mboost.mat = as.matrix(bb.mboost)
# mb.dim = dim(mboost.mat)
# attributes(mboost.mat) = NULL
# dim(mboost.mat) = mb.dim
# 
# all.equal(mboost.mat, basis)
