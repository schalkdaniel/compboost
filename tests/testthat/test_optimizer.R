context("The optimizer works")

test_that("greedy optimizer works", {
  
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
  factory.list = BlearnerFactoryList$new()
  
  # Register factorys:
  factory.list$registerFactory(linear.factory)
  factory.list$registerFactory(cubic.factory)
  factory.list$registerFactory(pow5.factory)
  
  # Optimizer:
  greedy.optimizer = GreedyOptimizer$new()
  
  res.linear = greedy.optimizer$testOptimizer(y.linear, factory.list)
  res.cubic  = greedy.optimizer$testOptimizer(y.cubic, factory.list)
  res.pow5   = greedy.optimizer$testOptimizer(y.pow5, factory.list)
  
  # Tests:
  # ------
  expect_equal(res.linear$selected.learner, "(test run) polynomial with degree 1")
  expect_equal(res.cubic$selected.learner, "(test run) polynomial with degree 3")
  expect_equal(res.pow5$selected.learner, "(test run) polynomial with degree 5")
  
  expect_equal(as.numeric(res.linear$parameter), 32)
  expect_equal(as.numeric(res.cubic$parameter), 16)
  expect_equal(as.numeric(res.pow5$parameter), 8)
})
