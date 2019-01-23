context("The implemented loss object")

test_that("Quadratic loss works", {
  expect_silent({ quadratic_loss = LossQuadratic$new() })
  expect_silent({ quadratic_loss_custom = LossQuadratic$new(2) })
})

test_that("Absolute loss works", {
  expect_silent({ absolute_loss = LossAbsolute$new() })
  expect_silent({ absolute_loss_custom = LossAbsolute$new(pi) })
})

test_that("Binomial loss works", {
  expect_silent({ binomial_loss = LossBinomial$new() })
  expect_silent({ binomial_loss_custom = LossBinomial$new(0.7) })
})

test_that("Custom loss works", {
 
  myLossFun = function (true_value, prediction) { return(0.25 * (true_value - prediction)^4) }
  myGradientFun = function (true_value, prediction) { return((prediction - true_value)^3) }
  myConstantInitializerFun = function (true_value) {
    suppressWarnings(
      unname(
        optim(par = list(const = 0), fn = function (const){
          return (sum(myGradientFun(true_value, const)))
        })$par
      )
    )
  }
  expect_silent({ custom_loss = LossCustom$new(myLossFun, myGradientFun, myConstantInitializerFun) })
})


test_that("Custom cpp loss works", {  
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })  
  expect_silent({ custom_cpp_loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
})