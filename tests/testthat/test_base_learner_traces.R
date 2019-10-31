context("Base-learner traces works")

test_that("Visualization works", {

  expect_silent({
    cboost = Compboost$new(mtcars, "mpg", loss = LossQuadratic$new())
    cboost$addBaselearner("hp", "spline", BaselearnerPSpline)
    cboost$addBaselearner(c("hp", "wt"), "quadratic", BaselearnerPolynomial)
    cboost$addBaselearner("wt", "linear", BaselearnerPolynomial)
  })

  expect_error(cboost$plot("hp_spline"))

  expect_output(cboost$train(2000, trace = 0))

  expect_error(cboost$plotBlearnerTraces(n_legend = "bls"))
  expect_error(cboost$plotBlearnerTraces(value = "bls"))
  expect_error(cboost$plotBlearnerTraces(value = c(1,2)))

  expect_silent({gg = cboost$plotBlearnerTraces()})
  expect_s3_class(cboost$plotBlearnerTraces(), "ggplot")
})