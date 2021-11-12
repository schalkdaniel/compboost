context("Prediction of individual feature contribution works")

test_that("Prediction of individual feature contribution works", {
	expect_output({
	  mod = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(),
      iterations = 2000L)
  })
  expect_silent({ ind_pred = mod$model$predictIndividualTrainData() })
  expect_silent({ pred = mod$getEstimatedCoef()$offset + Reduce("+", ind_pred) })
  expect_equal(pred, as.vector(mod$predict()))
  expect_equal(pred, as.vector(mod$predict(iris)))
  expect_equal(mod$model$predictIndividualTrainData(), mod$model$predictIndividual(mod$prepareData(iris)))
  expect_silent({
    ind_predX = lapply(names(ind_pred), function(fn) as.vector(mod$model$predictFactoryTrainData(fn)))
  })
  names(ind_predX) = names(ind_pred)
  expect_equal(ind_pred, ind_predX)
})
