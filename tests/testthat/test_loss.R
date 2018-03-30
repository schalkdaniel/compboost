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

test_that("Bernoulli loss works", {
  true.value = rbinom(100, 1, 0.4) * 2 - 1
  prediction = runif(100, -1, 1)
  
  bernoulli.loss = BernoulliLoss$new()
  bernoulli.loss.custom = BernoulliLoss$new(0.7)
  suppressWarnings({
    bernoulli.loss.warning = BernoulliLoss$new(2)
  })
  
  # Tests:
  # -----------
  expect_equal(
    bernoulli.loss$testLoss(true.value, prediction), 
    as.matrix(log(1 + exp(-true.value * prediction)))
  )
  expect_equal(
    bernoulli.loss$testGradient(true.value, prediction),
    as.matrix(-true.value / (1 + exp(true.value * prediction)))
  )
  expect_equal(
    bernoulli.loss$testConstantInitializer(true.value),
    log(mean(true.value > 0) / (1 - mean(true.value > 0))) / 2
  )
  expect_equal(bernoulli.loss.custom$testConstantInitializer(true.value), 0.7)
  expect_equal(
    bernoulli.loss.warning$testConstantInitializer(true.value),
    log(mean(true.value > 0) / (1 - mean(true.value > 0))) / 2
  )
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
