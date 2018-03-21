context("Data works")

test_that("polynomial baselearner works correctly", {
  
  X = as.matrix(1:10)
  
  data.source = InMemoryData$new(X, "x")
  data.target = InMemoryData$new()
  
  expect_equal(data.source$getData(), X)
  expect_equal(data.source$getIdentifier(), "x")
  
  expect_equal(data.target$getData(), as.matrix(0))
  expect_equal(data.target$getIdentifier(), "")
  
  lin.factory = PolynomialBlearnerFactory$new(data.source, data.target, 3)
  
  expect_equal(data.target$getData(), X^3)
  expect_equal(data.target$getIdentifier(), "x")
})