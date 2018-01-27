context("Factorys of 'compboost'")

test_that("polynomial factory works", {
  
  # Data X and response y:
  X.linear = 1:10
  X.cubic  = X.linear^3
  
  set.seed(pi)
  X.test = as.matrix(runif(200))
  
  y = 3 * X.linear + rnorm(10, 0, 2)
  
  # Create and train test baselearner:
  linear.factory = PolynomialFactory$new(as.matrix(X.linear), "my_variable_name", 1)
  # linear.factory$testTrain(y)
  
  cubic.factory = PolynomialFactory$new(as.matrix(X.linear), "my_variable_name", 3)
  # cubic.factory$testTrain(y)
  
  # lm as benchmark:
  mod.linear = lm(y ~ 0 + X.linear)
  mod.cubic  = lm(y ~ 0 + X.cubic)
  
  # Tests:
  # ------
  expect_equal(
    linear.factory$getData(), 
    as.matrix(mod.linear$model[["X.linear"]])
  )
  # expect_equal(
  #   linear.factory$testGetParameter(), 
  #   as.matrix(unname(mod.linear$coef))
  # )
  # expect_equal(
  #   as.numeric(linear.factory$testPredict()), 
  #   unname(mod.linear$fitted.values)
  # )
  # expect_equal(
  #   as.numeric(linear.factory$testPredictNewdata(X.test)), 
  #   unname(predict(mod.linear, data.frame(X.linear = X.test[,1])))
  # )
  
  expect_equal(
    cubic.factory$getData(), 
    as.matrix(mod.cubic$model[["X.cubic"]])
  )
  # expect_equal(
  #   cubic.factory$testGetParameter(), 
  #   as.matrix(unname(mod.cubic$coef))
  # )
  # expect_equal(
  #   as.numeric(cubic.factory$testPredict()), 
  #   unname(mod.cubic$fitted.values)
  # )
  # expect_equal(
  #   as.numeric(cubic.factory$testPredictNewdata(X.test)), 
  #   unname(predict(mod.cubic, data.frame(X.cubic = X.test[,1]^3)))
  # )
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
  
  X.test = as.matrix(runif(200))
  
  mod.test = trainFun(y, X)
  
  # Create and train test baselearner:
  custom.factory = CustomFactory$new(X, "variable_1", instantiateDataFun, trainFun, 
    predictFun, extractParameter)
  # custom.factory$testTrain(y)
  
  # Test:
  # -----
  expect_equal(
    custom.factory$getData(), 
    instantiateDataFun(X)
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
    
    typedef arma::mat (*instantiateDataFunPtr) (arma::mat& X);
    typedef arma::mat (*trainFunPtr) (arma::vec& y, arma::mat& X);
    typedef arma::mat (*predictFunPtr) (arma::mat& newdata, arma::mat& parameter);
    
    
    // instantiateDataFun:
    // -------------------
    
    arma::mat instantiateDataFun (arma::mat& X)
    {
    return X;
    }
    
    // trainFun:
    // -------------------
    
    arma::mat trainFun (arma::vec& y, arma::mat& X)
    {
    return arma::solve(X, y);
    }
    
    // predictFun:
    // -------------------
    
    arma::mat predictFun (arma::mat& newdata, arma::mat& parameter)
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

  X.test = as.matrix(runif(200))

  custom.cpp.factory = CustomCppFactory$new(X, "my_variable_name", dataFunSetter(),
    trainFunSetter(), predictFunSetter())

  expect_equal(custom.cpp.factory$getData(), X)
})