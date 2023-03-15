testCboostJson = function(cboost, cboost2, new_iter = NULL, blp = "Petal.Length_spline") {

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
    if (i <= old_iter)
      expect_equal(cboost$model$getParameterAtIteration(i), cboost2$getParameterAtIteration(i))
  }

  if (! is.null(new_iter)) {
    expect_equal(cboost2$getEstimatedParameter(), cboost2$getParameterAtIteration(new_iter))
  }

  expect_equal(cboost$model$predictFactoryTrainData(blp), cboost2$predictFactoryTrainData(blp))
  expect_equal(cboost$model$predictIndividualTrainData(), cboost2$predictIndividualTrainData())

  if (is.null(new_iter)) {
    expect_equal(cboost$model$getParameterMatrix(), cboost2$getParameterMatrix())
  }

  dl = cboost$prepareData(iris)

  expect_equal(cboost$model$predictFactoryNewData(blp, dl), cboost2$predictFactoryNewData(blp, dl))
  expect_equal(cboost$model$predictIndividual(dl), cboost2$predictIndividual(dl))
}

testCboostJsonAPI = function(cb, file = "cboost.json") {

  expect_silent(cb$saveToJson(file))

  cboost = expect_silent(Compboost$new(file = file))

  expect_equal(cboost$getLoggerData(), cb$getLoggerData())
  expect_equal(cboost$getInbagRisk(), cb$getInbagRisk())
  expect_equal(cboost$getCurrentIteration(), cb$getCurrentIteration())
  expect_equal(cboost$getCoef(), cb$getCoef())
  expect_equal(cboost$positive, cb$positive)
  expect_equal(cboost$stop_all, cb$stop_all)
  expect_equal(cboost$target, cb$target)
  expect_equal(cboost$getSelectedBaselearner(), cb$getSelectedBaselearner())

  a = expect_silent(cboost$transformData(iris[, -1]))
  b = expect_silent(cb$transformData(iris[, -1]))
  for (bn in names(a)) {
    expect_equal(a[[bn]], b[[bn]])
    expect_equal(cboost$transformData(iris[, -1], bn), cb$transformData(iris[, -1], bn))
  }
  expect_equal(cboost$calculateFeatureImportance(), cb$calculateFeatureImportance())
  expect_equal(cboost$learning_rate, cb$learning_rate)
  expect_equal(sort(cboost$getBaselearnerNames()), sort(cb$getBaselearnerNames()))

  expect_equal(cboost$predict(), cb$predict())
  expect_equal(cboost$predict(iris), cb$predict(iris))
  expect_equal(cboost$predictIndividual(iris[2,]), cb$predictIndividual(iris[2,]))

  dnames = names(cb$data)
  fAsS = function(d) {
    for (fn in names(d)) if (is.factor(d[[fn]])) d[[fn]] = as.character(d[[fn]])
    return(d)
  }
  expect_equal(cboost$prepareData(iris)[dnames], cb$prepareData(iris)[dnames])
  d1 = fAsS(cboost$data[, dnames])
  d2 = fAsS(cb$data[, dnames])
  rownames(d1) = NULL
  rownames(d2) = NULL
  expect_equal(d1, d2)

  expect_equal(class(plotBaselearnerTraces(cboost)), c("gg", "ggplot"))
  expect_equal(class(plotFeatureImportance(cboost, aggregate = TRUE)), c("gg", "ggplot"))
  expect_error(plotFeatureImportance(cboost, num_feats = 10))
  expect_equal(class(plotFeatureImportance(cboost, aggregate = FALSE)), c("gg", "ggplot"))
  expect_equal(class(plotRisk(cboost)), c("gg", "ggplot"))
  expect_equal(class(plotIndividualContribution(cboost, iris[1, ])), c("gg", "ggplot"))
  expect_equal(class(plotIndividualContribution(cboost, iris[1, ], offset = FALSE)), c("gg", "ggplot"))
  expect_equal(class(plotIndividualContribution(cboost, iris[1, ], offset = FALSE, colbreaks = c(-Inf, Inf), collabels = "Test")), c("gg", "ggplot"))


  #expect_equal(class(plotBaselearner(cboost, "Petal.Width_spline")), c("gg", "ggplot"))
  #plotPEUni(cboost, "Petal.Length")
  #plotPEUni(cb, "Species")

  #file.remove(file)
  return(invisible(cboost))
}
