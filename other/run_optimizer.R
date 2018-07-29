set.seed(pi)
X = as.matrix(runif(100, -4, 4))

y.linear = as.numeric(32 * X)
y.cubic  = as.numeric(16 * X^3)
y.pow5   = as.numeric(8 * X^5)

# Create new linear baselearner of hp and wt:
linear.factory = PolynomialBlearnerFactory$new(X, "X", 1)
cubic.factory  = PolynomialBlearnerFactory$new(X, "X", 3)
pow5.factory   = PolynomialBlearnerFactory$new(X, "X", 5)

# Create new factory list:
factory.list = FactoryList$new()

# Register factorys:
factory.list$registerFactory(linear.factory)
factory.list$registerFactory(cubic.factory)
factory.list$registerFactory(pow5.factory)

# Optimizer:
greedy.optimizer = OptimizerCoordinateDescent$new()

res.linear = greedy.optimizer$testOptimizer(y.linear, factory.list)
res.cubic  = greedy.optimizer$testOptimizer(y.cubic, factory.list)
res.pow5   = greedy.optimizer$testOptimizer(y.pow5, factory.list)

res.cubic
res.linear
