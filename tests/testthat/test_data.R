context("Data works")

test_that("data objects works correctly", {
  
  X = as.matrix(1:10)
  
  expect_silent({ data.source = InMemoryData$new(X, "x") })
  expect_silent({ data.target = InMemoryData$new() })
  
  expect_equal(data.source$getData(), X)
  expect_equal(data.source$getIdentifier(), "x")
  
  expect_equal(data.target$getData(), as.matrix(0))
  expect_equal(data.target$getIdentifier(), "")
  
  expect_silent({ lin.factory = BaselearnerPolynomial$new(data.source, data.target, 3, FALSE) })
  
  expect_equal(data.target$getData(), X^3)
  expect_equal(data.target$getIdentifier(), "x")
  
})