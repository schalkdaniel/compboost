context("Factories of 'compboost'")

test_that("polynomial factory works", {

  # Data X and response y:
  X.linear = 1:10
  X.cubic  = X.linear^3

  y = 3 * X.linear + rnorm(10, 0, 2)
  
  expect_silent({ data.source = InMemoryData$new(as.matrix(X.linear), "my_variable") })
  expect_silent({ data.target.lin = InMemoryData$new() })
  expect_silent({ data.target.cub = InMemoryData$new() })
  expect_silent({ linear.factory = BaselearnerPolynomial$new(data.source, data.target.lin, 1, FALSE) })
  expect_silent({ cubic.factory = BaselearnerPolynomial$new(data.source, data.target.cub, 3, FALSE) })

  mod.linear = lm(y ~ 0 + X.linear)
  mod.cubic  = lm(y ~ 0 + X.cubic)

  expect_equal(
    linear.factory$getData(),
    as.matrix(mod.linear$model[["X.linear"]])
  )
  expect_equal(
    cubic.factory$getData(),
    as.matrix(mod.cubic$model[["X.cubic"]])
  )
  expect_equal(
    linear.factory$getData(),
    linear.factory$transformData(data.source$getData())
  )
  expect_equal(
    cubic.factory$getData(),
    cubic.factory$transformData(data.source$getData())
  )
})

test_that("custom factory works", {

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

  X = matrix(1:10, ncol = 1)
  y = sin(as.numeric(X)) + rnorm(10, 0, 0.6)

  expect_silent({ data.source = InMemoryData$new(X, "variable_1") })
  expect_silent({ data.target = InMemoryData$new() })
  expect_silent({
    custom.factory = BaselearnerCustom$new(data.source, data.target, 
      instantiateDataFun, trainFun, predictFun, extractParameter)
  })
  expect_equal(
    custom.factory$getData(),
    instantiateDataFun(X)
  )
  expect_equal(
    custom.factory$getData(),
    custom.factory$transformData(data.source$getData())
  )
})


test_that("custom cpp factory works", {

  expect_output(Rcpp::sourceCpp(code = getCustomCppExample()))

  set.seed(pi)
  X = matrix(1:10, ncol = 1)
  y = 3 * as.numeric(X) + rnorm(10, 0, 2)

  expect_silent({ data.source = InMemoryData$new(X, "my_variable_name") })
  expect_silent({ data.target = InMemoryData$new() })
  expect_silent({
    custom.cpp.factory = BaselearnerCustomCpp$new(data.source, data.target, 
      dataFunSetter(), trainFunSetter(), predictFunSetter())
  })
  expect_equal(custom.cpp.factory$getData(), X)  
  expect_equal(
    custom.cpp.factory$getData(),
    custom.cpp.factory$transformData(data.source$getData())
  )
})
