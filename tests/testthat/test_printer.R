context("Printer works")

test_that("Loss printer works", {
  
  quadratic.loss = QuadraticLoss$new()
  absolute.loss  = AbsoluteLoss$new()
  
  # Function for Custom Loss:
  myLossFun = function (true.value, prediction) NULL
  myGradientFun = function (true.value, prediction) NULL
  myConstantInitializerFun = function (true.value) NULL
  
  custom.loss = CustomLoss$new(myLossFun, myGradientFun, myConstantInitializerFun)
  
  # A hack to suppress console output:
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  test.quadratic.printer = show(quadratic.loss) 
  test.absolute.printer  = show(absolute.loss)
  test.custom.printer    = show(custom.loss)
  
  sink() 
  close(tc) 
  
  # Test:
  # --------
  
  expect_equal(test.quadratic.printer, "QuadraticPrinter")
  expect_equal(test.absolute.printer, "AbsolutePrinter")
  expect_equal(test.custom.printer, "CustomPrinter")
  
})