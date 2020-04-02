context("Binning works")

test_that("Binning works", {
	expect_output({
	  mod = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(), use_binning = TRUE)
  })
})
