context("The optimizer workds")

test_that("greedy optimizer works", {
  
  X = as.matrix(runif(100))
  y.linear = as.numeric(4 * X)
  y.quadratic = as.numeric(8 * X^2)
  y.cubic = as.numeric(16 * X^3)
  
  # Create new linear baselearner of hp and wt:
  linear.factory    = PolynomialFactory$new(X, "X", 1)
  quadratic.factory = PolynomialFactory$new(X, "X", 2)
  cubic.factory     = PolynomialFactory$new(X, "X", 3)
  
  # Create new factory list:
  factory.list = FactoryList$new()
  
  # Register factorys:
  factory.list$registerFactory(linear.factory)
  factory.list$registerFactory(quadratic.factory)
  factory.list$registerFactory(cubic.factory)
  
  # Optimizer:
  greedy.optimizer = GreedyOptimizer$new()
  
  res.linear    = greedy.optimizer$testOptimizer(y.linear, factory.list)
  res.quadratic = greedy.optimizer$testOptimizer(y.quadratic, factory.list)
  res.cubic     = greedy.optimizer$testOptimizer(y.cubic, factory.list)
  
  # Tests:
  # ------
  expect_equal(res.linear$selected.learner, "(test run) polynomial with degree 1")
  expect_equal(res.quadratic$selected.learner, "(test run) polynomial with degree 2")
  expect_equal(res.cubic$selected.learner, "(test run) polynomial with degree 3")
  
  expect_equal(as.numeric(res.linear$parameter), 4)
  expect_equal(as.numeric(res.quadratic$parameter), 8)
  expect_equal(as.numeric(res.cubic$parameter), 16)
})