context("Plot function produce a ggplot")

test_that("plotting univariate partial effects works", {
  cboost = expect_silent(Compboost$new(data = iris, target = "Petal.Length",
    loss = LossQuadratic$new()))
  expect_silent(cboost$addComponents("Sepal.Width"))
  expect_silent(cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge))
  expect_output(cboost$train(500L))

  gg_num = expect_silent(plotPEUni(cboost, "Sepal.Width"))
  gg_cat = expect_silent(plotPEUni(cboost, "Species"))

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

  cboost = expect_silent(Compboost$new(data = iris, target = "Petal.Length",
    loss = LossQuadratic$new()))
  expect_silent(cboost$addComponents("Sepal.Width", df = 3))
  expect_silent(cboost$addComponents("Sepal.Length", df = 3))
  expect_silent(cboost$addComponents("Petal.Width", df = 3))
  expect_silent(cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge))
  expect_output(cboost$train(500L))

  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ]))
  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ], aggregate = FALSE))
  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ], offset = FALSE))
  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ], offset = FALSE, aggregate = FALSE))
  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ], colbreaks = NULL))
  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ], collabels = NULL))
  gg = expect_silent(plotIndividualContribution(cboost, iris[10, ], colbreaks = NULL, collabels = NULL))
  expect_true(inherits(gg, "ggplot"))

  expect_error(plotIndividualContribution(cboost, iris))
  expect_error(plotIndividualContribution(cboost, collabels = c("a", "b", "c")))

  cboost$model = NULL
  expect_error(plotIndividualContribution(cboost))
})

test_that("risk plotting works", {

  cboost_no_valdat = expect_output(boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()))
  cboost_valdat = expect_output(boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(), oob_fraction = 0.3))

  gg1 = expect_silent(plotRisk(cboost_no_valdat))
  gg2 = expect_silent(plotRisk(cboost_valdat))

  expect_true(inherits(gg1, "ggplot"))
  expect_true(inherits(gg2, "ggplot"))

  cboost_no_valdat$model = NULL
  expect_error(plotRisk(cboost_no_valdat))
})

test_that("feature importance plotting works", {

  expect_output({ cboost = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()) })

  gg = expect_silent(plotFeatureImportance(cboost))
  gg = expect_silent(plotFeatureImportance(cboost, num_feats = 2))
  gg = expect_silent(plotFeatureImportance(cboost, aggregate = FALSE))

  expect_error(plotFeatureImportance(cboost, num_feats = 100))
  expect_error(plotFeatureImportance(cboost, aggregate = 4))

  cboost$model = NULL
  expect_error(plotFeatureImportance(cboost))
})

test_that("bivariate tensors are working", {
  set.seed(31415)
  iris$g2 = sample(LETTERS[1:3], 150, TRUE)

  expect_silent({
    cboost = Compboost$new(data = iris, target = "Petal.Length",
      loss = LossQuadratic$new())

    cboost$addBaselearner("Sepal.Width", "spline", BaselearnerPSpline, df = 4)
    cboost$addBaselearner("Sepal.Length", "spline", BaselearnerPSpline, df = 4)
    cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge)
    cboost$addBaselearner("g2", "ridge", BaselearnerCategoricalRidge)
  })

  expect_silent({cboost$addTensor("Sepal.Width", "Sepal.Length", df1 = 4, df2 = 4) })
  expect_output(cboost$train(100L))

  expect_silent({ gg = plotTensor(cboost, "Sepal.Width_Sepal.Length_tensor") })
  expect_silent({ gg = plotTensor(cboost, "Sepal.Width_Sepal.Length_tensor", npoints = 10L) })
  expect_silent({ gg = plotTensor(cboost, "Sepal.Width_Sepal.Length_tensor", nbins = NULL) })
  expect_true({inherits(gg, "ggplot")})

  expect_silent({
    cboost = Compboost$new(data = iris, target = "Petal.Length",
      loss = LossQuadratic$new())

    cboost$addBaselearner("Sepal.Width", "spline", BaselearnerPSpline, df = 4)
    cboost$addBaselearner("Sepal.Length", "spline", BaselearnerPSpline, df = 4)
    cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge)
    cboost$addBaselearner("g2", "ridge", BaselearnerCategoricalRidge)
  })
  expect_silent({ cboost$addTensor("Sepal.Width", "Species", df1 = 4, df2 = 2) })
  expect_output(cboost$train(1000L))

  expect_silent({ gg = plotTensor(cboost, "Sepal.Width_Species_tensor") })
  expect_true({inherits(gg, "ggplot")})


  expect_silent({
    cboost = Compboost$new(data = iris, target = "Petal.Length",
      loss = LossQuadratic$new())

    cboost$addBaselearner("Sepal.Width", "spline", BaselearnerPSpline, df = 4)
    cboost$addBaselearner("Sepal.Length", "spline", BaselearnerPSpline, df = 4)
    cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge)
    cboost$addBaselearner("g2", "ridge", BaselearnerCategoricalRidge)
  })
  expect_silent({ cboost$addTensor("g2", "Species", df1 = 2, df2 = 2) })
  expect_output(cboost$train(100L))
  expect_error(plotTensor(cboost, "g2_Species_tensor", nbins = NULL))

  expect_output(cboost$train(1000L))
  expect_silent({ gg = plotTensor(cboost, "g2_Species_tensor", nbins = NULL) })
  expect_true({inherits(gg, "ggplot")})
})
