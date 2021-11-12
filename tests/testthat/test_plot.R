context("Plot function produce a ggplot")

test_that("plotting univariate partial effects works", {
  expect_silent({
    cboost = Compboost$new(data = iris, target = "Petal.Length",
      loss = LossQuadratic$new())
  })
  expect_silent({cboost$addComponents("Sepal.Width")})
  expect_silent({cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge)})
  expect_output({cboost$train(500L)})

  expect_silent({ gg_num = plotPEUni(cboost, "Sepal.Width") })
  expect_silent({ gg_cat = plotPEUni(cboost, "Species") })

  expect_true(inherits(gg_num, "gg"))
  expect_true(inherits(gg_cat, "gg"))

  expect_error(plotPEUni(1:10, "bla"))
  expect_error(plotPEUni(cboost, "bla"))
  expect_error(plotPEUni(cboost, "Petal.Width", npoints = 1L))

  cboost$model = NULL
  expect_error(plotPEUni(cboost, "Petal.Width"))
})

test_that("base learner works", {
  expect_output({
    cboost = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(),
      iterations = 2000L, df = 4)
  })

  expect_silent({ gg_num = plotBaselearner(cboost, "Petal.Width_spline") })
  expect_silent({ gg_cat = plotBaselearner(cboost, "Species_ridge") })

  expect_true(inherits(gg_num, "gg"))
  expect_true(inherits(gg_cat, "gg"))

  expect_error(plotBaselearner(1:10, "bla"))
  expect_error(plotBaselearner(cboost, "bla"))
  expect_error(plotBaselearner(cboost, "Petal.Width_spline", npoints = 1L))

  cboost$model = NULL
  expect_error(plotBaselearner(cboost, "Petal.Width_spline"))
})

test_that("base learner traces can be plotted", {

  expect_silent({
    cboost = Compboost$new(data = iris, target = "Petal.Length",
      loss = LossQuadratic$new())
  })
  expect_silent({ cboost$addComponents("Sepal.Width", df = 3) })
  expect_silent({ cboost$addComponents("Sepal.Length", df = 3) })
  expect_silent({ cboost$addComponents("Petal.Width", df = 3) })
  expect_silent({ cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge) })
  expect_output({ cboost$train(500L)})
  expect_silent({ gg = plotBaselearnerTraces(cboost) })
  expect_true({ inherits(gg, "ggplot")})

  expect_error(plotBaselearnerTraces(cboost, value = 1:2))
  expect_error(plotBaselearnerTraces(cboost, n_legend = "bla"))

  cboost$model = NULL
  expect_error(plotBaselearnerTraces(cboost))
})

test_that("individual predictions can be plotted", {

  expect_silent({
    cboost = Compboost$new(data = iris, target = "Petal.Length",
      loss = LossQuadratic$new())
  })
  expect_silent({ cboost$addComponents("Sepal.Width", df = 3) })
  expect_silent({ cboost$addComponents("Sepal.Length", df = 3) })
  expect_silent({ cboost$addComponents("Petal.Width", df = 3) })
  expect_silent({ cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge) })
  expect_output({ cboost$train(500L)})
  expect_silent({ gg = plotIndividualContribution(cboost, iris[10,]) })
  expect_silent({ gg = plotIndividualContribution(cboost, iris[10,], offset = FALSE) })
  expect_silent({ gg = plotIndividualContribution(cboost, iris[10,], colbreaks = NULL) })
  expect_silent({ gg = plotIndividualContribution(cboost, iris[10,], collabels = NULL) })
  expect_silent({ gg = plotIndividualContribution(cboost, iris[10,], colbreaks = NULL, collabels = NULL) })
  expect_true({ inherits(gg, "ggplot")})

  expect_error(plotIndividualContribution(cboost, iris))
  expect_error(plotIndividualContribution(cboost, collabels = c("a", "b", "c")))

  cboost$model = NULL
  expect_error(plotIndividualContribution(cboost))
})

test_that("risk plotting works", {

  expect_output({ cboost_no_valdat = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()) })
  expect_output({ cboost_valdat = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(), oob_fraction = 0.3) })

  expect_silent({ gg1 = plotRisk(cboost_no_valdat) })
  expect_silent({ gg2 = plotRisk(cboost_valdat) })
  expect_true({ inherits(gg1, "ggplot")})
  expect_true({ inherits(gg2, "ggplot")})

  cboost_no_valdat$model = NULL
  expect_error(plotRisk(cboost_no_valdat))
})

test_that("feature importance plotting works", {

  expect_output({ cboost = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()) })

  expect_silent({ gg = plotFeatureImportance(cboost) })
  expect_silent({ gg = plotFeatureImportance(cboost, num_feats = 2) })
  expect_silent({ gg = plotFeatureImportance(cboost, aggregate = FALSE) })

  expect_error({ plotFeatureImportance(cboost, num_feats = 100) })
  expect_error({ plotFeatureImportance(cboost, aggregate = 4) })

  cboost$model = NULL
  expect_error(plotFeatureImportance(cboost))
})


