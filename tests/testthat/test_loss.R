context("The implemented loss object")

test_that("Quadratic loss works", {
  true.value = runif(100)
  prediction = runif(100)
  
  quadratic.loss = QuadraticLoss$new()
  
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
})

test_that("Absolute loss works", {
  true.value = runif(100)
  prediction = runif(100)
  
  absolute.loss = AbsoluteLoss$new()
  
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