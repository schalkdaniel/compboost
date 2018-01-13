context("Factorys of 'compboost'")

test_that("polynomial factory works", {
  
  # Data X and response y:
  X.linear = 1:10
  X.cubic  = X.linear^3
  
  X.test = as.matrix(runif(200))
  
  y = 3 * X.linear + rnorm(10, 0, 2)
  
  # Create and train test baselearner:
  linear.factory = PolynomialFactory$new(as.matrix(X.linear), "my_variable_name", 1)
  linear.factory$testTrain(y)
  
  cubic.factory = PolynomialFactory$new(as.matrix(X.linear), "my_variable_name", 3)
  cubic.factory$testTrain(y)
  
  # lm as benchmark:
  mod.linear = lm(y ~ 0 + X.linear)
  mod.cubic  = lm(y ~ 0 + X.cubic)
  
  # Tests:
  # ------
  expect_equal(
    linear.factory$getData(), 
    as.matrix(mod.linear$model[["X.linear"]])
  )
  expect_equal(
    linear.factory$testGetParameter(), 
    as.matrix(unname(mod.linear$coef))
  )
  expect_equal(
    as.numeric(linear.factory$testPredict()), 
    unname(mod.linear$fitted.values)
  )
  expect_equal(
    as.numeric(linear.factory$testPredictNewdata(X.test)), 
    unname(predict(mod.linear, data.frame(X.linear = X.test[,1])))
  )
  
  expect_equal(
    cubic.factory$getData(), 
    as.matrix(mod.cubic$model[["X.cubic"]])
  )
  expect_equal(
    cubic.factory$testGetParameter(), 
    as.matrix(unname(mod.cubic$coef))
  )
  expect_equal(
    as.numeric(cubic.factory$testPredict()), 
    unname(mod.cubic$fitted.values)
  )
  expect_equal(
    as.numeric(cubic.factory$testPredictNewdata(X.test)), 
    unname(predict(mod.cubic, data.frame(X.cubic = X.test[,1])))
  )
})

test_that("custom factory works", {
  
  # Define the custom functions:
  instantiateDataFun = function (X) {
    return(X)
  }
  trainFun = function (y, X) {
    X = data.frame(y = y, x = as.numeric(X))
    return(rpart::rpart(y ~ x, data = X))
  }
  predictFun = function (model, newdata) {
    newdata = data.frame(x = as.numeric(newdata))
    return(as.matrix(predict(model, newdata)))
  }
  extractParameter = function (model) {
    return(as.matrix(NA))
  }
  
  # Data X and response y:
  X = matrix(1:10, ncol = 1)
  y = sin(as.numeric(X)) + rnorm(10, 0, 0.6)
  
  X.test = as.matrix(runif(200))
  
  mod.test = trainFun(y, X)
  
  # Create and train test baselearner:
  custom.factory = CustomFactory$new(X, "variable_1", instantiateDataFun, trainFun, 
    predictFun, extractParameter)
  custom.factory$testTrain(y)
  
  # Test:
  # -----
  expect_equal(
    custom.factory$getData(), 
    instantiateDataFun(X)
  )
  expect_equal(
    custom.factory$testGetParameter(), 
    as.matrix(NA_real_)
  )
  expect_equal(
    as.numeric(custom.factory$testPredict()), 
    unname(predict(mod.test))
  )
  expect_equal(
    custom.factory$testPredictNewdata(X.test), 
    predictFun(mod.test, X.test)
  )
})
