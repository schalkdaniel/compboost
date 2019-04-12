context("The implemented loss object")

test_that("Quadratic loss works", {
  expect_silent({ quadratic_loss = LossQuadratic$new() })
  expect_silent({ quadratic_loss_custom = LossQuadratic$new(2) })

  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = quadratic_loss) })
})

test_that("Absolute loss works", {
  expect_silent({ absolute_loss = LossAbsolute$new() })
  expect_silent({ absolute_loss_custom = LossAbsolute$new(pi) })

  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = absolute_loss) })
})

test_that("Quantile loss works", {
  expect_silent({ quantile_loss = LossQuantile$new() })
  expect_silent({ quantile_loss = LossQuantile$new(0.3) })
  expect_silent({ quantile_loss_custom = LossQuantile$new(2, 0.3) })

  expect_error({ LossQuantile$new(10) })
  expect_error({ LossQuantile$new(10, 10) })

  expect_equal(quantile_loss$getQuantile(), 0.3)

  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = quantile_loss) })
})

test_that("Huber loss works", {
  expect_silent({ huber_loss = LossHuber$new() })
  expect_silent({ huber_loss = LossHuber$new(0.3) })
  expect_silent({ huber_loss_custom = LossHuber$new(2, 0.3) })

  expect_error({ LossHuber$new(-10) })
  expect_error({ LossHuber$new(10, -10) })

  expect_equal(huber_loss$getDelta(), 0.3)

  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = huber_loss) })
})

test_that("Binomial loss works", {
  expect_silent({ binomial_loss = LossBinomial$new() })
  expect_silent({ binomial_loss_custom = LossBinomial$new(0.7) })

  cars$speed = ifelse (cars$speed > median(cars$speed), "fast", "slow")
  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = binomial_loss) })
})

test_that("Custom loss works", {

  myLossFun = function (true_value, prediction) { return(0.5 * (true_value - prediction)^2) }
  myGradientFun = function (true_value, prediction) { return(prediction - true_value) }
  myConstantInitializerFun = function (true_value) { matrix(mean(true_value)) }
  expect_silent({ custom_loss = LossCustom$new(myLossFun, myGradientFun, myConstantInitializerFun) })
  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = custom_loss) })
})


test_that("Custom cpp loss works", {
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })
  expect_silent({ custom_cpp_loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter())) })
})