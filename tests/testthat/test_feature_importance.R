context("Feature importance of 'compboost'")

test_that("feature importance works", {
  nuisance = capture.output(suppressWarnings({
    cboost = boostSplines(data = mtcars, target = "mpg", loss = LossQuadratic$new())
  }))
  expect_error(cboost$calculateFeatureImportance(20L))
  expect_silent(cboost$calculateFeatureImportance())

  #expect_silent({ gg_vip = cboost$plotFeatureImportance() })
  #expect_true(inherits(gg_vip, "ggplot"))
})
