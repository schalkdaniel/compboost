context("Cholesky and inverse solver doing the same (for simple data)")

test_that("cholesky does the same as inverse", {
  expect_output({
    cboost_chol = boostSplines(data = iris, target = "Sepal.Width", loss = LossQuadratic$new(), cache_type = "cholesky")
  })
  expect_output({
    cboost_inv = boostSplines(data = iris, target = "Sepal.Width", loss = LossQuadratic$new(), cache_type = "inverse")
  })
  expect_equal(cboost_chol$getEstimatedCoef(), cboost_inv$getEstimatedCoef())
})
