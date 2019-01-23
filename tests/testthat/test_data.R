context("Data works")

test_that("data objects works correctly", {
  
  X = as.matrix(1:10)
  
  expect_silent({ data_source = InMemoryData$new(X, "x") })
  expect_silent({ data_target = InMemoryData$new() })
  
  expect_equal(data_source$getData(), X)
  expect_equal(data_source$getIdentifier(), "x")
  
  expect_equal(data_target$getData(), as.matrix(0))
  expect_equal(data_target$getIdentifier(), "")
  
  expect_silent({ lin.factory = BaselearnerPolynomial$new(data_source, data_target, 
    list(degree = 3, intercept = FALSE)) })
  
  expect_equal(data_target$getData(), X^3)
  expect_equal(data_target$getIdentifier(), "x")
  
})