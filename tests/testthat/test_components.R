context("Centered base learner")

test_that("Centering works consistently with S4 and R6", {
  x = runif(100, 0, 10)
  y = 2 * sin(x) + 2 * x + rnorm(100, 0, 0.5)
  dat = data.frame(x, y)

  # S4 wrapper
  data_mat = cbind(x)
  data_source = InMemoryData$new(data_mat, "x")

  bl_lin = BaselearnerPolynomial$new(data_source, list(degree = 1, intercept = TRUE))
  bl_sp = BaselearnerPSpline$new(data_source, list(n_knots = 15, df = 5))

  bl_ctr = expect_silent(BaselearnerCentered$new(bl_sp, bl_lin, "ctr"))

  # Recognize, that the data matrix of this base learner has
  # `nrow(bl_sp$getData()) - ncol(bl_lin$getData())` columns:
  expect_equal(ncol(bl_ctr$getData()), nrow(bl_sp$getData()) - ncol(bl_lin$getData()))
  expect_equal(t(bl_sp$getData()) %*% bl_ctr$getRotation(), bl_ctr$getData())

  # Transform "new data". Internally, the basis of the spline is build and
  # then rotated by the rotation matrix to subtract the linear part:
  newdata = list(data_source)
  expect_equal(bl_ctr$transformData(newdata)$design, bl_ctr$getData())

  # R6 wrapper
  cboost = Compboost$new(dat, "y")
  expect_silent(cboost$addComponents("x", n_knots = 15, df = 5, df = 5))
  expect_equal(bl_ctr$getData(), cboost$baselearner_list$x_x_spline_centered$factory$getData())
  expect_output(cboost$train(200, 0))

  cboost = Compboost$new(dat, "y")
  expect_silent(cboost$addComponents("x", n_knots = 15, df = 5, df = 5, bin_root = 2))
  expect_output(cboost$train(200, 0))
})
