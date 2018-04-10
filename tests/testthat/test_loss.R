context("The implemented loss object")

test_that("Quadratic loss works", {
  true.value = runif(100)
  prediction = runif(100)
  
  quadratic.loss = QuadraticLoss$new()
  quadratic.loss.custom = QuadraticLoss$new(2)
  
  # Tests:
  # -----------
  expect_equal(
    quadratic.loss$testLoss(true.value, prediction), 
    0.5 * as.matrix(true.value - prediction)^2
  )
  expect_equal(
    quadratic.loss$testGradient(true.value, prediction),
    as.matrix(prediction - true.value)
  )
  expect_equal(
    quadratic.loss$testConstantInitializer(true.value),
    mean.default(true.value)
  )
  expect_equal(quadratic.loss.custom$testConstantInitializer(true.value), 2)
})

test_that("Absolute loss works", {
  true.value = runif(100)
  prediction = runif(100)
  
  absolute.loss = AbsoluteLoss$new()
  absolute.loss.custom = AbsoluteLoss$new(pi)
  
  # Tests:
  # -----------
  expect_equal(
    absolute.loss$testLoss(true.value, prediction), 
    as.matrix(abs(true.value - prediction))
  )
  expect_equal(
    absolute.loss$testGradient(true.value, prediction),
    as.matrix(sign(prediction - true.value))
  )
  expect_equal(
    absolute.loss$testConstantInitializer(true.value),
    median.default(true.value)
  )
  expect_equal(absolute.loss.custom$testConstantInitializer(true.value), pi)
})

test_that("Binomial loss works", {
  true.value = rbinom(100, 1, 0.4) * 2 - 1
  prediction = runif(100, -1, 1)
  
  binomial.loss = BinomialLoss$new()
  binomial.loss.custom = BinomialLoss$new(0.7)
  
  # # This forces travis to fail:
  # suppressWarnings({
  #   binomial.loss.warning = BinomialLoss$new(2)
  # })
  
  # Tests:
  # -----------
  expect_equal(
    binomial.loss$testLoss(true.value, prediction), 
    as.matrix(log(1 + exp(-true.value * prediction)))
  )
  expect_equal(
    binomial.loss$testGradient(true.value, prediction),
    as.matrix(-true.value / (1 + exp(true.value * prediction)))
  )
  expect_equal(
    binomial.loss$testConstantInitializer(true.value),
    log(mean(true.value > 0) / (1 - mean(true.value > 0))) / 2
  )
  expect_equal(binomial.loss.custom$testConstantInitializer(true.value), 0.7)
  # expect_equal(
  #   binomial.loss.warning$testConstantInitializer(true.value),
  #   log(mean(true.value > 0) / (1 - mean(true.value > 0))) / 2
  # )
})

test_that("Custom loss works", {
  true.value = runif(100)
  prediction = runif(100)
  
  myLossFun = function (true.value, prediction) { return(0.25 * (true.value - prediction)^4) }
  myGradientFun = function (true.value, prediction) { return((prediction - true.value)^3) }
  myConstantInitializerFun = function (true.value) {
    suppressWarnings(
      unname(
        optim(par = list(c = 0), fn = function (c){
          return (sum(myGradientFun(true.value, c)))
        })$par
      )
    )
  }
  
  custom.loss = CustomLoss$new(myLossFun, myGradientFun, myConstantInitializerFun)
  
  # Tests:
  # -----------
  expect_equal(
    custom.loss$testLoss(true.value, prediction), 
    as.matrix(myLossFun(true.value, prediction))
  )
  expect_equal(
    custom.loss$testGradient(true.value, prediction),
    as.matrix(myGradientFun(true.value, prediction))
  )
  expect_equal(
    custom.loss$testConstantInitializer(true.value),
    myConstantInitializerFun(true.value)
  )
})


test_that("Custom cpp loss works", {
  
  Rcpp::sourceCpp(code = '
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>
    
    typedef arma::vec (*lossFunPtr) (const arma::vec& true_value, const arma::vec& prediction);
    typedef arma::vec (*gradFunPtr) (const arma::vec& true_value, const arma::vec& prediction);
    typedef double (*constInitFunPtr) (const arma::vec& true_value);
    
    arma::vec lossFun (const arma::vec& true_value, const arma::vec& prediction)
    {
    return arma::pow(true_value - prediction, 2) / 2;
    }
    
    arma::vec gradFun (const arma::vec& true_value, const arma::vec& prediction)
    {
    return prediction - true_value;
    }
    
    double constInitFun (const arma::vec& true_value)
    {
    return arma::mean(true_value);
    }
    
    // [[Rcpp::export]]
    Rcpp::XPtr<lossFunPtr> lossFunSetter ()
    {
      return Rcpp::XPtr<lossFunPtr> (new lossFunPtr (&lossFun));
    }
    
    // [[Rcpp::export]]
    Rcpp::XPtr<gradFunPtr> gradFunSetter ()
    {
      return Rcpp::XPtr<gradFunPtr> (new gradFunPtr (&gradFun));
    }
    
    // [[Rcpp::export]]
    Rcpp::XPtr<constInitFunPtr> constInitFunSetter ()
    {
      return Rcpp::XPtr<constInitFunPtr> (new constInitFunPtr (&constInitFun));
    }
  ')

  true.value = rnorm(100)
  prediction = rnorm(100)
  
  custom.cpp.loss = CustomCppLoss$new(lossFunSetter(), gradFunSetter(), constInitFunSetter())
  
  # Test Functionality:
  # ----------------------------
  expect_equal(
    as.matrix((true.value - prediction)^2 / 2), 
    custom.cpp.loss$testLoss(true.value, prediction)
  )
  
  expect_equal(
    as.matrix(prediction - true.value), 
    custom.cpp.loss$testGradient(true.value, prediction)
  )
  
  expect_equal(custom.cpp.loss$testConstantInitializer(true.value), mean(true.value))
  
  
  # Test Printer:
  # ----------------------------
  tc = textConnection(NULL, "w")
  sink(tc)
  
  test.custom.cpp.printer = show(custom.cpp.loss)
  
  sink()
  close(tc)
  
  expect_equal(test.custom.cpp.printer, "CustomCppLossPrinter")
})