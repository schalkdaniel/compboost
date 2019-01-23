context("Factories of 'compboost'")

test_that("polynomial factory works", {

  # Data X and response y:
  X_linear = 1:10
  X_cubic  = X_linear^3

  y = 3 * X_linear + rnorm(10, 0, 2)
  
  expect_silent({ data_source = InMemoryData$new(as.matrix(X_linear), "my_variable") })
  expect_silent({ data_target_lin = InMemoryData$new() })
  expect_silent({ data_target_cub = InMemoryData$new() })
  expect_error({ linear_factory = BaselearnerPolynomial$new(data_source, data_target_lin, 
    list(1, FALSE)) })
  expect_error({ cubic_factory = BaselearnerPolynomial$new(data_source, data_target_cub, 
    list(degree = "test", intercept = FALSE)) })  
  expect_warning({ cubic_factory = BaselearnerPolynomial$new(data_source, data_target_cub, 
    list(degree = 5, intercept = FALSE, nuisance = 10)) })  
  expect_silent({ linear_factory = BaselearnerPolynomial$new(data_source, data_target_lin, 
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ cubic_factory = BaselearnerPolynomial$new(data_source, data_target_cub, 
    list(degree = 3, intercept = FALSE)) })

  mod_linear = lm(y ~ 0 + X_linear)
  mod_cubic  = lm(y ~ 0 + X_cubic)

  expect_equal(
    linear_factory$getData(),
    as.matrix(mod_linear$model[["X_linear"]])
  )
  expect_equal(
    cubic_factory$getData(),
    as.matrix(mod_cubic$model[["X_cubic"]])
  )
  expect_equal(
    linear_factory$getData(),
    linear_factory$transformData(data_source$getData())
  )
  expect_equal(
    cubic_factory$getData(),
    cubic_factory$transformData(data_source$getData())
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

  expect_silent({ data_source = InMemoryData$new(X, "variable_1") })
  expect_silent({ data_target = InMemoryData$new() })
  expect_silent({
    custom_factory = BaselearnerCustom$new(data_source, data_target, 
      list(instantiate_fun = instantiateDataFun, train_fun =  trainFun, predict_fun = predictFun, 
        param_fun = extractParameter))
  })
  expect_equal(
    custom_factory$getData(),
    instantiateDataFun(X)
  )
  expect_equal(
    custom_factory$getData(),
    custom_factory$transformData(data_source$getData())
  )
})


test_that("custom cpp factory works", {

  expect_output(Rcpp::sourceCpp(code = getCustomCppExample()))

  set.seed(pi)
  X = matrix(1:10, ncol = 1)
  y = 3 * as.numeric(X) + rnorm(10, 0, 2)

  expect_silent({ data_source = InMemoryData$new(X, "my_variable_name") })
  expect_silent({ data_target = InMemoryData$new() })
  expect_silent({
    custom_cpp_factory = BaselearnerCustomCpp$new(data_source, data_target, 
      list(instantiate_ptr = dataFunSetter(), train_ptr = trainFunSetter(), 
        predict_ptr = predictFunSetter()))
  })
  expect_equal(custom_cpp_factory$getData(), X)  
  expect_equal(
    custom_cpp_factory$getData(),
    custom_cpp_factory$transformData(data_source$getData())
  )
})
