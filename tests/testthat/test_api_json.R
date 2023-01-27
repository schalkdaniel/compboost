context("Test if JSON loading via the R6 API is working")

test_that("Basic save and load works", {
  file = "cboost.json"
  cb = expect_output(boostSplines(iris, "Sepal.Length"))
  expect_silent(cb$model$saveJson(file))

  cboost = expect_silent(Compboost$new(file = file))

  expect_equal(cboost$getLoggerData(), cb$getLoggerData())
  expect_equal(cboost$getInbagRisk(), cb$getInbagRisk())
  expect_equal(cboost$getCurrentIteration(), cb$getCurrentIteration())
  expect_equal(cboost$getCoef(), cb$getCoef())
  expect_equal(cboost$positive, cb$positive)
  expect_equal(cboost$stop_all, cb$stop_all)
  expect_equal(cboost$target, cb$target)

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
  expect_equal(cboost$predict(), cboost$predict(iris))
  expect_equal(cboost$predictIndividual(iris[2,]), cb$predictIndividual(iris[2,]))

  dnames = names(cb$data)
  fAsS = function(d) {
    for (fn in names(d)) if (is.factor(d[[fn]])) d[[fn]] = as.character(d[[fn]])
    return(d)
  }
  expect_equal(cboost$prepareData(iris)[dnames], cb$prepareData(iris)[dnames])
  expect_equal(fAsS(cboost$data[, dnames]), fAsS(cb$data[, dnames]))

  expect_equal(class(plotBaselearnerTraces(cboost)), c("gg", "ggplot"))
  expect_equal(class(plotFeatureImportance(cboost)), c("gg", "ggplot"))
  expect_equal(class(plotRisk(cboost)), c("gg", "ggplot"))
  expect_equal(class(plotBaselearner(cboost, "Petal.Width_spline")), c("gg", "ggplot"))
  expect_equal(class(plotIndividualContribution(cboost, iris[1, ])), c("gg", "ggplot"))
  expect_equal(class(plotIndividualContribution(cboost, iris[1, ], offset = FALSE)), c("gg", "ggplot"))
  expect_equal(class(plotIndividualContribution(cboost, iris[1, ], offset = FALSE, colbreaks = c(-Inf, Inf), collabels = "Test")), c("gg", "ggplot"))

  #plotPEUni(cboost, "Species")
  #plotPEUni(cb, "Species")

  file.remove(file)
})
