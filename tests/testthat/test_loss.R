context("The implemented loss object")

test_that("Quadratic loss works", {
  
  # true.value = runif(100)
  # prediction = runif(100)
  
  expect_silent({ quadratic.loss = QuadraticLoss$new() })
  expect_silent({ quadratic.loss.custom = QuadraticLoss$new(2) })
  # expect_equal(
  #   quadratic.loss$testLoss(true.value, prediction), 
  #   0.5 * as.matrix(true.value - prediction)^2
  # )
  # expect_equal(
  #   quadratic.loss$testGradient(true.value, prediction),
  #   as.matrix(prediction - true.value)
  # )
  # expect_equal(
  #   quadratic.loss$testConstantInitializer(true.value),
  #   mean.default(true.value)
  # )
  # expect_equal(quadratic.loss.custom$testConstantInitializer(true.value), 2)
})

test_that("Absolute loss works", {

  # true.value = runif(100)
  # prediction = runif(100)
  
  expect_silent({ absolute.loss = AbsoluteLoss$new() })
  expect_silent({ absolute.loss.custom = AbsoluteLoss$new(pi) })
  # expect_equal(
  #   absolute.loss$testLoss(true.value, prediction), 
  #   as.matrix(abs(true.value - prediction))
  # )
  # expect_equal(
  #   absolute.loss$testGradient(true.value, prediction),
  #   as.matrix(sign(prediction - true.value))
  # )
  # expect_equal(
  #   absolute.loss$testConstantInitializer(true.value),
  #   median.default(true.value)
  # )
  # expect_equal(absolute.loss.custom$testConstantInitializer(true.value), pi)
})

test_that("Binomial loss works", {

  # true.value = rbinom(100, 1, 0.4) * 2 - 1
  # prediction = runif(100, -1, 1)
  
  expect_silent({ binomial.loss = BinomialLoss$new() })
  expect_silent({ binomial.loss.custom = BinomialLoss$new(0.7) })
  # expect_equal(
  #   binomial.loss$testLoss(true.value, prediction), 
  #   as.matrix(log(1 + exp(-true.value * prediction)))
  # )
  # expect_equal(
  #   binomial.loss$testGradient(true.value, prediction),
  #   as.matrix(-true.value / (1 + exp(true.value * prediction)))
  # )
  # expect_equal(
  #   binomial.loss$testConstantInitializer(true.value),
  #   log(mean(true.value > 0) / (1 - mean(true.value > 0))) / 2
  # )
  # expect_equal(binomial.loss.custom$testConstantInitializer(true.value), 0.7)
})

test_that("Custom loss works", {

  # true.value = runif(100)
  # prediction = runif(100)
  
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
  # expect_equal(
  #   custom.loss$testLoss(true.value, prediction), 
  #   as.matrix(myLossFun(true.value, prediction))
  # )
  # expect_equal(
  #   custom.loss$testGradient(true.value, prediction),
  #   as.matrix(myGradientFun(true.value, prediction))
  # )
  # expect_equal(
  #   custom.loss$testConstantInitializer(true.value),
  #   myConstantInitializerFun(true.value)
  # )
})


test_that("Custom cpp loss works", {
  
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })

  # true.value = rnorm(100)
  # prediction = rnorm(100)
  
  expect_silent({ custom.cpp.loss = CustomCppLoss$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
  # expect_equal(
  #   as.matrix((true.value - prediction)^2 / 2), 
  #   custom.cpp.loss$testLoss(true.value, prediction)
  # )
  # expect_equal(
  #   as.matrix(prediction - true.value), 
  #   custom.cpp.loss$testGradient(true.value, prediction)
  # )
  # expect_equal(custom.cpp.loss$testConstantInitializer(true.value), mean(true.value))
  # expect_output({ test.custom.cpp.printer = show(custom.cpp.loss) })
  # expect_equal(test.custom.cpp.printer, "CustomCppLossPrinter")
})