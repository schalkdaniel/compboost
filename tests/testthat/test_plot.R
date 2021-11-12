context("Plotting Univariate effects works")

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
