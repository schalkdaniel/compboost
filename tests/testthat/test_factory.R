context("Factorys of 'compboost'")

test_that("polynomial factory works", {

  # Data X and response y:
  X.linear = 1:10
  X.cubic  = X.linear^3

  y = 3 * X.linear + rnorm(10, 0, 2)
  
  data.source = InMemoryData$new(as.matrix(X.linear), "my_variable")
  
  data.target.lin = InMemoryData$new()
  data.target.cub = InMemoryData$new()

  # Create and train test baselearner:
  linear.factory = PolynomialBlearnerFactory$new(data.source, data.target.lin, 1)
  # linear.factory$testTrain(y)

  cubic.factory = PolynomialBlearnerFactory$new(data.source, data.target.cub, 3)

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
  set.seed(pi)
  X = matrix(1:10, ncol = 1)
  y = sin(as.numeric(X)) + rnorm(10, 0, 0.6)

  data.source = InMemoryData$new(X, "variable_1")
  data.target = InMemoryData$new()

  # Create and train test baselearner:
  custom.factory = CustomBlearnerFactory$new(data.source, data.target, 
    instantiateDataFun, trainFun, predictFun, extractParameter)
  # custom.factory$testTrain(y)

  # Test:
  # -----
  expect_equal(
    custom.factory$getData(),
    instantiateDataFun(X)
  )
  
  expect_equal(
    custom.factory$getData(),
    custom.factory$transformData(data.source$getData())
  )
  # expect_equal(
  #   custom.factory$testGetParameter(),
  #   as.matrix(NA_real_)
  # )
  # expect_equal(
  #   as.numeric(custom.factory$testPredict()),
  #   unname(predict(mod.test))
  # )
  # expect_equal(
  #   custom.factory$testPredictNewdata(X.test),
  #   predictFun(mod.test, X.test)
  # )
})


test_that("custom cpp factory works", {

  Rcpp::sourceCpp(code = '
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>

    typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
    typedef arma::mat (*trainFunPtr) (const arma::vec& y, const arma::mat& X);
    typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);


    // instantiateDataFun:
    // -------------------

    arma::mat instantiateDataFun (const arma::mat& X)
    {
    return X;
    }

    // trainFun:
    // -------------------

    arma::mat trainFun (const arma::vec& y, const arma::mat& X)
    {
    return arma::solve(X, y);
    }

    // predictFun:
    // -------------------

    arma::mat predictFun (const arma::mat& newdata, const arma::mat& parameter)
    {
    return newdata * parameter;
    }


    // Setter function:
    // ------------------

    // [[Rcpp::export]]
    Rcpp::XPtr<instantiateDataFunPtr> dataFunSetter ()
    {
    return Rcpp::XPtr<instantiateDataFunPtr> (new instantiateDataFunPtr (&instantiateDataFun));
    }

    // [[Rcpp::export]]
    Rcpp::XPtr<trainFunPtr> trainFunSetter ()
    {
    return Rcpp::XPtr<trainFunPtr> (new trainFunPtr (&trainFun));
    }

    // [[Rcpp::export]]
    Rcpp::XPtr<predictFunPtr> predictFunSetter ()
    {
    return Rcpp::XPtr<predictFunPtr> (new predictFunPtr (&predictFun));
    }'
  )

  set.seed(pi)
  X = matrix(1:10, ncol = 1)
  y = 3 * as.numeric(X) + rnorm(10, 0, 2)

  data.source = InMemoryData$new(X, "my_variable_name")
  data.target = InMemoryData$new()

  custom.cpp.factory = CustomCppBlearnerFactory$new(data.source, data.target, 
    dataFunSetter(), trainFunSetter(), predictFunSetter())

  expect_equal(custom.cpp.factory$getData(), X)
  
  expect_equal(
    custom.cpp.factory$getData(),
    custom.cpp.factory$transformData(data.source$getData())
  )
})
