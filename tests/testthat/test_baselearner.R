context("Baselearner works")

test_that("polynomial baselearner works correctly", {
  
  x       = 1:10
  x.cubic = x^3
  X = matrix(x, ncol = 1)
  y = 3 * 1:10 + rnorm(10)
  y.cubic = 0.5 * x.cubic + rnorm(10)
  newdata = runif(10, 1, 10)
  
  linear = Polynomial$new(X, "myvariable", 1)
  cubic  = Polynomial$new(X, "myvariable", 3)
  
  linear$train(y)
  cubic$train(y.cubic)
  
  mod = lm(y ~ 0 + x)
  mod.cubic = lm(y.cubic ~ 0 + x.cubic)
  
  expect_equal(linear$getData(), X)
  expect_equal(as.numeric(linear$getParameter()), unname(coef(mod)))
  expect_equal(linear$predict(), as.matrix(unname(predict(mod))))
  expect_equal(
    linear$predictNewdata(as.matrix(newdata, ncol = 1)),
    as.matrix(unname(predict(mod, newdata = data.frame(x = newdata))))
  )
  
  expect_equal(cubic$getData(), X^3)
  expect_equal(as.numeric(cubic$getParameter()), unname(coef(mod.cubic)))
  expect_equal(cubic$predict(), as.matrix(unname(predict(mod.cubic))))
  expect_equal(
    cubic$predictNewdata(as.matrix(newdata, ncol = 1)),
    as.matrix(unname(predict(mod.cubic, newdata = data.frame(x.cubic = newdata^3))))
  )
  
})

test_that("custom baselearner works correctly", {
  
  instantiateData = function (X)
  {
    return(X);
  }
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
  
  custom = Custom$new(X, "myvariable", instantiateData, trainFun, predictFun, extractParameter)
  
  custom$train(y)
  mod = lm(y ~ 0 + x)
  
  expect_equal(custom$getData(), X)
  expect_equal(as.numeric(custom$getParameter()), unname(coef(mod)))
  expect_equal(custom$predict(), as.matrix(unname(predict(mod))))
  expect_equal(
    custom$predictNewdata(as.matrix(newdata, ncol = 1)),
    as.matrix(unname(predict(mod, newdata = data.frame(x = newdata))))
  )
})