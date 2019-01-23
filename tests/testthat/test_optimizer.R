context("Optimizer of 'compboost'")

test_that("Coordinate Descent with line search works", {
  n_train = 1000L

  used_optimizer = OptimizerCoordinateDescent$new()
  
  cboost = Compboost$new(data = mtcars, target = "mpg", optimizer = used_optimizer, loss = LossQuadratic$new(), learning_rate = 0.05)
  
  cboost$addBaselearner("wt", "linear", BaselearnerPolynomial)
  cboost$addBaselearner("disp", "linear", BaselearnerPolynomial)
  cboost$addBaselearner("hp", "linear", BaselearnerPolynomial)
  
  nuisance = capture.output(suppressWarnings({
    cboost$train(n_train)
  }))
  
  used_optimizer_ls = OptimizerCoordinateDescentLineSearch$new()
  
  nuisance = capture.output(suppressWarnings({
    cboost1 = Compboost$new(data = mtcars, target = "mpg", optimizer = used_optimizer_ls, loss = LossQuadratic$new(), learning_rate = 0.05)
  }))

  cboost1$addBaselearner("wt", "linear", BaselearnerPolynomial)
  cboost1$addBaselearner("disp", "linear", BaselearnerPolynomial)
  cboost1$addBaselearner("hp", "linear", BaselearnerPolynomial)
  
  nuisance = capture.output(suppressWarnings({
    cboost1$train(n_train)
  }))

  expect_equal(cboost$predict(), cboost1$predict())
  expect_equal(cboost$getInbagRisk(), cboost1$getInbagRisk())
  expect_true(all(abs(used_optimizer_ls$getStepSize() - 1) < 1e10))
  expect_true(var(used_optimizer_ls$getStepSize()) < 1e-10)
})