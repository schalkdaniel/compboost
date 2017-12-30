context("Baselearners of 'compboost'")

test_that("linear baselearner works", {
  
  # Data X and response y:
  X = matrix(1:10, ncol = 1)
  y = 3 * as.numeric(X) + rnorm(10, 0, 2)
  
  # Create Baselearner:
  bl = BaselearnerWrapper$new("l1", X, "variable_1", 1)
  
  # Check the baselearner type:
  expect_equal(bl$GetBaselearnerType(), "polynomial with degree 1")
  
  # Check the identifier:
  expect_equal(bl$GetIdentifier(), "l1")
  
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
  instantiateDataFun = function (X) {
    return(X)
  }
  trainFun = function (y, X) {
    return(rpart::rpart(y ~ X))
  }
  predictFun = function (model, newdata) {
    return(as.matrix(predict(model, as.data.frame(newdata))))
  }
  extractParameter = function (model) {
    return(as.matrix(NA))
  }
  
  # Data X and response y:
  X = matrix(1:10, ncol = 1)
  y = sin(as.numeric(X)) + rnorm(10, 0, 0.6)
  
  # Create baselearner:
  bl = BaselearnerWrapper$new("linear custom test", X, "variable_1", instantiateDataFun, trainFun, 
    predictFun, extractParameter)
  
  # Test type:
  expect_equal(bl$GetBaselearnerType(), "custom")
  
  # Check identifier:
  expect_equal(bl$GetIdentifier(), "linear custom test")
  
  # Test data structure:
  expect_equal(bl$GetData(), instantiateDataFun(X))
  
  # Test the parameter setter:
  bl$train(y)
  expect_equal(bl$GetParameter(), as.matrix(NA_real_))
  
  # Test the prediction (and therefore implicit if the model works):
  expect_equivalent(bl$predict(), as.matrix(predict(rpart::rpart(y ~ X))))
})