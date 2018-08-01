context("Wrapper boostLinear works")

test_that("boostLinear function works", {
	expect_output({
	  mod = boostLinear(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()) 
  })

	expect_length(mod$getBaselearnerNames(), 6)
  expect_length(mod$getSelectedBaselearner(), 100)
  expect_length(mod$getInbagRisk(), 101)
  expect_length(mod$predict(), nrow(iris))

  expect_output(mod$train(150))

  expect_length(mod$getSelectedBaselearner(), 150)
  expect_length(mod$getInbagRisk(), 151)
  expect_length(mod$predict(), nrow(iris))
  expect_equal(mod$predict(), mod$predict(iris))
})
