context("Baselearners of 'compboost'")

test_that("linear baselearner works", {
  
  set.seed(pi)
  
  # Data X and response y:
  X = matrix(1:10, ncol = 1)
  y = 3 * as.numeric(X) + rnorm(10, 0, 2)
  
  # Create Baselearner:
  bl = BaselearnerWrapper$new("linear", "test", X)
  
  # Check the baselearner type:
  expect_equal(bl$GetBaselearnerType(), "linear")
  
  # Check stored data:
  expect_equal(bl$GetData(), X)
  
  # Check train:
  bl$train(y)
  expect_equal(bl$GetParameter(), solve(t(X) %*% X) %*% t(X) %*% y)
  
  # Check prediction:
  expect_equal(bl$predict(), X %*% solve(t(X) %*% X) %*% t(X) %*% y)
})

test_that("custom baselearner works", {
  
  # Define the custom functions:
  transformDataFun = function (X) {
    return(X)
  }
  trainFun = function (y, X) {
    return(rpart::rpart(y ~ X))
  }
  predictFun = function (model) {
    return(as.matrix(predict(model)))
  }
  predictNewdataFun = function (model, newdata) {
    return(as.matrix(predict(model, newdata)))
  }
  extractParameter = function (model) {
    return(as.matrix(NA))
  }
  
  set.seed(pi)
  
  # Data X and response y:
  X = matrix(1:10, ncol = 1)
  y = sin(as.numeric(X)) + rnorm(10, 0, 0.6)
  
  # Create baselearner:
  bl = BaselearnerWrapper$new("linear custom test", X, transformDataFun, trainFun, 
    predictFun, predictNewdataFun, extractParameter)
  
  # Test type:
  expect_equal(bl$GetBaselearnerType(), "custom")
  
  # Test data structure:
  expect_equal(bl$GetData(), transformDataFun(X))
  
  # Test the parameter setter:
  bl$train(y)
  expect_equal(bl$GetParameter(), as.matrix(NA_real_))
  
  # Test the prediction (and therefore implicit if the model works):
  expect_equivalent(bl$predict(), predictFun(rpart::rpart(y ~ X)))
})