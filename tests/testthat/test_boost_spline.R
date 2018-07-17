context("Wrapper boostSpline works")

test_that("boostSpline function works", {
	expect_output({
	  mod = boostSplines(data = iris, target = "Sepal.Length", loss = QuadraticLoss$new()) 
  })

	expect_length(mod$getBaselearnerNames(), 6)
  expect_length(mod$selected(), 100)
  expect_length(mod$risk(), 101)
  expect_length(mod$predict(), nrow(iris))

  expect_output(mod$train(150))

  expect_length(mod$selected(), 150)
  expect_length(mod$risk(), 151)
  expect_length(mod$predict(), nrow(iris))
  expect_equal(mod$predict(), mod$predict(iris))
})