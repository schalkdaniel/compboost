context("Baselearner works")

test_that("polynomial baselearner works correctly", {
  
  x = 1:10
  X = matrix(x, ncol = 1)
  y = 3 * 1:10 + rnorm(10)
  newdata = runif(10, 1, 10)
  
  linear = Polynomial$new(X, "myvariable", 1)
  
  linear$train(y)
  mod = lm(y ~ 0 + x)
  
  expect_equal(as.numeric(linear$getParameter()), unname(coef(mod)))
  expect_equal(linear$predict(), as.matrix(unname(predict(mod))))
  expect_equal(
    linear$predictNewdata(as.matrix(newdata, ncol = 1)),
    as.matrix(unname(predict(mod, newdata = data.frame(x = newdata))))
  )
  
})

test_that("custom baselearner works correctly", {
  
  trainFun = function (y, X) {
    return(solve(t(X) %*% X) %*% t(X) %*% y)
  }
  predictFun = function (model, newdata) {
    return(newdata %*% model)
  }
  extractParameter = function (model) {
    return(model)
  }
  
  x = 1:10
  X = matrix(x, ncol = 1)
  y = 3 * 1:10 + rnorm(10)
  newdata = runif(10, 1, 10)
  
  custom = Custom$new(X, "myvariable", trainFun, predictFun, extractParameter)
  
  custom$train(y)
  mod = lm(y ~ 0 + x)
  
  expect_equal(as.numeric(custom$getParameter()), unname(coef(mod)))
  expect_equal(custom$predict(), as.matrix(unname(predict(mod))))
  expect_equal(
    custom$predictNewdata(as.matrix(newdata, ncol = 1)),
    as.matrix(unname(predict(mod, newdata = data.frame(x = newdata))))
  )
})