context("The implemented loss object")

test_that("Quadratic loss works", {
  
  expect_silent({ quadratic.loss = LossQuadratic$new() })
  expect_silent({ quadratic.loss.custom = LossQuadratic$new(2) })

})

test_that("Absolute loss works", {

  expect_silent({ absolute.loss = LossAbsolute$new() })
  expect_silent({ absolute.loss.custom = LossAbsolute$new(pi) })

})

test_that("Binomial loss works", {

  expect_silent({ binomial.loss = LossBinomial$new() })
  expect_silent({ binomial.loss.custom = LossBinomial$new(0.7) })

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
  expect_silent({ custom.loss = LossCustom$new(myLossFun, myGradientFun, myConstantInitializerFun) })
})


test_that("Custom cpp loss works", {
  
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })  
  expect_silent({ custom.cpp.loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
})