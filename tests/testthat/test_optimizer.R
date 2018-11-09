context("Optimizer of 'compboost'")

test_that("Coordinate Descent with line search works", {
  n.train = 1000L

  used.optimizer = OptimizerCoordinateDescent$new()
  
  cboost = Compboost$new(data = mtcars, target = "mpg", optimizer = used.optimizer, loss = LossQuadratic$new(), learning.rate = 0.05)
  
  cboost$addBaselearner("wt", "linear", BaselearnerPolynomial)
  cboost$addBaselearner("disp", "linear", BaselearnerPolynomial)
  cboost$addBaselearner("hp", "linear", BaselearnerPolynomial)
  
  cboost$train(n.train)
  
  used.optimizer.ls = OptimizerCoordinateDescentLineSearch$new()
  
  cboost1 = Compboost$new(data = mtcars, target = "mpg", optimizer = used.optimizer.ls, loss = LossQuadratic$new(), learning.rate = 0.05)
  
  cboost1$addBaselearner("wt", "linear", BaselearnerPolynomial)
  cboost1$addBaselearner("disp", "linear", BaselearnerPolynomial)
  cboost1$addBaselearner("hp", "linear", BaselearnerPolynomial)
  
  cboost1$train(n.train)
  
  expect_equal(cboost$predict(), cboost1$predict())
  expect_equal(cboost$getInbagRisk(), cboost1$getInbagRisk())
  expect_true(all(abs(used.optimizer.ls$getStepSize() - 1) < 1e10))
  expect_true(var(used.optimizer.ls$getStepSize()) < 1e-10)
})