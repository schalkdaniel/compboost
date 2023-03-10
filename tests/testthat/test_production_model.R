context("Save and load model in production mode")

test_that("rm_data removes the data", {
  iterations = 100L
  cboost = expect_output(boostSplines(iris, "Sepal.Width", iterations = iterations))

  file = "cboost.json"
  cboost$saveToJson(file, rm_data = TRUE)
  cboostp = expect_silent(Compboost$new(file = file))
  file.remove(file)

  expect_error(cboostp$predict())
  expect_error(cboostp$train(iterations + 1))

  for (i in seq_len(iterations)) {
    cboost$train(i)
    cboostp$train(i)
    expect_equal(cboost$predict(iris), cboostp$predict(iris))
    expect_equal(cboost$getCoef(), cboostp$getCoef())
  }
  expect_equal(cboost$calculateFeatureImportance(), cboostp$calculateFeatureImportance())
  expect_equal(cboost$getSelectedBaselearner(), cboostp$getSelectedBaselearner())

  a = suppressWarnings(cboost$transformData(iris))
  b = suppressWarnings(cboostp$transformData(iris))
  for (nm in names(a)) {
    if (nm %in% unique(cboost$getSelectedBaselearner())) {
      f1 = cboost$baselearner_list[[nm]]$factory
      f2 = cboostp$baselearner_list[[nm]]$factory
      expect_equal(a[[nm]], b[[nm]])
      expect_equal(f1$getMinMax(), f2$getMinMax())
      expect_equal(f1$getMeta(), f2$getMeta())

      ggf1 = plotBaselearner(cboost, nm)
      ggf2 = plotBaselearner(cboostp, nm)
      expect_equal(ggf1$data, ggf2$data)

      ggf1 = plotPEUni(cboost, strsplit(nm, "_")[[1]][1])
      ggf2 = plotPEUni(cboostp, strsplit(nm, "_")[[1]][1])
      expect_equal(ggf1$layers[[1]]$data, ggf2$layers[[1]]$data)
    }
  }
})
