context("The implemented loss object")

test_that("Quadratic loss works", {
  quadratic_loss = expect_silent(LossQuadratic$new())
  quadratic_loss_custom = expect_silent(LossQuadratic$new(2))

  expect_output(boostLinear(data = cars, target = "speed", loss = quadratic_loss))

  # Save and load:
  cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossQuadratic$new()))
  file = "cboost.json"
  expect_silent(cboost$model$saveJson(file))

  ln = expect_silent(LossQuadratic$new())
  expect_silent(ln$loadFromJson(file))

  expect_equal(class(ln), class(cboost$loss))

  file.remove(file)
})

test_that("Absolute loss works", {
  absolute_loss = expect_silent(LossAbsolute$new())
  absolute_loss_custom = expect_silent(LossAbsolute$new(pi))

  expect_output(boostLinear(data = cars, target = "speed", loss = absolute_loss))

  # Save and load:
  cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossAbsolute$new()))
  file = "cboost.json"
  expect_silent(cboost$model$saveJson(file))

  ln = expect_silent(LossAbsolute$new())
  expect_silent(ln$loadFromJson(file))

  expect_equal(class(ln), class(cboost$loss))

  file.remove(file)
})

test_that("Quantile loss works", {
  quantile_loss = expect_silent(LossQuantile$new())
  quantile_loss = expect_silent(LossQuantile$new(0.3))
  quantile_loss_custom = expect_silent(LossQuantile$new(2, 0.3))

  expect_error(LossQuantile$new(10))
  expect_error(LossQuantile$new(10, 10))

  expect_equal(quantile_loss$getQuantile(), 0.3)

  expect_output(boostLinear(data = cars, target = "speed", loss = quantile_loss))

  # Save and load:
  cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossQuantile$new(0.3)))
  file = "cboost.json"
  expect_silent(cboost$model$saveJson(file))

  ln = expect_silent(LossQuantile$new())
  expect_silent(ln$loadFromJson(file))

  expect_equal(class(ln), class(cboost$loss))
  expect_equal(ln$getQuantile(), cboost$loss$getQuantile())

  file.remove(file)
})

test_that("Huber loss works", {
  huber_loss = expect_silent(LossHuber$new())
  huber_loss = expect_silent(LossHuber$new(0.3))
  huber_loss_custom = expect_silent(LossHuber$new(2, 0.3))

  expect_error(LossHuber$new(-10))
  expect_error(LossHuber$new(10, -10))

  expect_equal(huber_loss$getDelta(), 0.3)

  expect_output(boostLinear(data = cars, target = "speed", loss = huber_loss))

  # Save and load:
  cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossHuber$new(0.3)))
  file = "cboost.json"
  expect_silent(cboost$model$saveJson(file))

  ln = expect_silent(LossHuber$new())
  expect_silent(ln$loadFromJson(file))

  expect_equal(class(ln), class(cboost$loss))
  expect_equal(ln$getDelta(), cboost$loss$getDelta())

  file.remove(file)
})

test_that("Binomial loss works", {
  binomial_loss = expect_silent(LossBinomial$new())
  binomial_loss_custom = expect_silent(LossBinomial$new(0.7))

  cars$speed = ifelse (cars$speed > median(cars$speed), "fast", "slow")
  expect_output(boostLinear(data = cars, target = "speed", loss = binomial_loss))

  # Save and load:
  cboost = expect_output(boostSplines(iris[1:100, ], "Species", loss = LossBinomial$new()))
  file = "cboost.json"
  expect_silent(cboost$model$saveJson(file))

  ln = expect_silent(LossBinomial$new())
  expect_silent(ln$loadFromJson(file))

  expect_equal(class(ln), class(cboost$loss))

  file.remove(file)
})

test_that("Custom loss works", {

  myLossFun = function (true_value, prediction) { return(0.5 * (true_value - prediction)^2) }
  myGradientFun = function (true_value, prediction) { return(prediction - true_value) }
  myConstantInitializerFun = function (true_value) { matrix(mean(true_value)) }
  expect_silent({ custom_loss = LossCustom$new(myLossFun, myGradientFun, myConstantInitializerFun) })
  expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = custom_loss) })
})


#test_that("Custom cpp loss works", {
  #expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })
  #expect_silent({ custom_cpp_loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
  #expect_output({ cboost = boostLinear(data = cars, target = "speed", loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter())) })
#})
