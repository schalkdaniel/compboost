testCboostJson = function(cboost, cboost2, new_iter = NULL) {

  old_iter = cboost$getCurrentIteration()
  if (! is.null(new_iter)) {
    if (old_iter >= new_iter) {
      expect_silent(cboost$model$setToIteration(new_iter, 0))
      expect_silent(cboost2$setToIteration(new_iter, 0))
    } else {
      expect_output(cboost$model$setToIteration(new_iter, 0))
      expect_output(cboost2$setToIteration(new_iter, 0))
    }
  }

  expect_equal(cboost$model$isTrained(), cboost2$isTrained())
  expect_equal(cboost$model$getOffset(), cboost2$getOffset())
  expect_equal(cboost$model$getRiskVector(), cboost2$getRiskVector())
  if ((! is.null(new_iter)) && (old_iter >= new_iter)) {
    expect_equal(cboost$model$getLoggerData(), cboost2$getLoggerData())
  }
  expect_equal(cboost$model$getSelectedBaselearner(), cboost2$getSelectedBaselearner())
  expect_equal(cboost$model$getPrediction(TRUE), cboost2$getPrediction(TRUE))
  expect_equal(cboost$model$getEstimatedParameter(), cboost2$getEstimatedParameter())
  for (i in seq(10, 90, 10)) {
    expect_equal(cboost$model$getParameterAtIteration(i), cboost2$getParameterAtIteration(i))
  }

  if (! is.null(new_iter)) {
    expect_equal(cboost2$getEstimatedParameter(), cboost2$getParameterAtIteration(new_iter))
  }

  expect_equal(cboost$model$predictFactoryTrainData("Petal.Length_spline"),
    cboost2$predictFactoryTrainData("Petal.Length_spline"))
  expect_equal(cboost$model$predictIndividualTrainData(), cboost2$predictIndividualTrainData())

  if (is.null(new_iter)) {
    expect_equal(cboost$model$getParameterMatrix(), cboost2$getParameterMatrix())
  }

  dl = cboost$prepareData(iris)

  expect_equal(cboost$model$predictFactoryNewData("Petal.Length_spline", dl),
    cboost2$predictFactoryNewData("Petal.Length_spline", dl))
  expect_equal(cboost$model$predictIndividual(dl), cboost2$predictIndividual(dl))
}
