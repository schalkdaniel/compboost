context("The implemented loss object")

test_that("Quadratic loss works", {
  
  expect_silent({ quadratic.loss = QuadraticLoss$new() })
  expect_silent({ quadratic.loss.custom = QuadraticLoss$new(2) })

})

test_that("Absolute loss works", {

  expect_silent({ absolute.loss = AbsoluteLoss$new() })
  expect_silent({ absolute.loss.custom = AbsoluteLoss$new(pi) })

})

test_that("Binomial loss works", {

  expect_silent({ binomial.loss = BinomialLoss$new() })
  expect_silent({ binomial.loss.custom = BinomialLoss$new(0.7) })

})

test_that("Custom loss works", {
 
  myLossFun = function (true.value, prediction) { return(0.25 * (true.value - prediction)^4) }
  myGradientFun = function (true.value, prediction) { return((prediction - true.value)^3) }
  myConstantInitializerFun = function (true.value) {
    suppressWarnings(
      unname(
        optim(par = list(const = 0), fn = function (const){
          return (sum(myGradientFun(true.value, const)))
        })$par
      )
    )
  }
  expect_silent({ custom.loss = CustomLoss$new(myLossFun, myGradientFun, myConstantInitializerFun) })
})


test_that("Custom cpp loss works", {
  
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })  
  expect_silent({ custom.cpp.loss = CustomCppLoss$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
})