context("mlr learners")

test_that("Basic parameter are working", {
  task = mlr3::tsk("german_credit")

  l = expect_silent(mlr3::lrn("classif.compboost"))
  expect_silent(l$train(task))
  expect_true(checkmate::testR6(l$model, "Compboost"))

  l = expect_silent(mlr3::lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
    n_knots = 10, learning_rate = 0.01, show_output = TRUE))
  expect_output(l$train(task))
  sp = expect_silent(l$model$baselearner_list$duration_spline$factory)
  ri = expect_silent(l$model$baselearner_list$job_ridge$factory)

  expect_equal(as.numeric(sp$getMeta()$df), 3)
  expect_equal(as.numeric(ri$getMeta()$df), 2)
  expect_equal(ncol(sp$getData()), floor(sqrt(task$nrow)))
  expect_equal(nrow(sp$getData()), 14)
  expect_equal(length(l$model$getBaselearnerNames()), length(task$feature_names))
  expect_equal(l$model$model$getLearningRate(), 0.01)
  expect_equal(l$importance(), l$model$calculateFeatureImportance())
  expect_equal(l$selected_features(), sort(unique(l$model$getSelectedBaselearner())))

  pred = expect_silent(l$predict(task))
  expect_true(checkmate::checkMatrix(pred$prob, any.missing = FALSE, nrows = task$nrow))

  l = expect_error(mlr3::lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
    n_knots = 10, learning_rate = 0.01, baselearner = "linear"))
  l = expect_silent(mlr3::lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
    learning_rate = 0.01, baselearner = "linear"))
  expect_silent(l$train(task))
  li = expect_silent(l$model$baselearner_list$duration_linear$factory)
  ri = expect_silent(l$model$baselearner_list$job_ridge$factory)

  expect_equal(as.numeric(ri$getMeta()$df), 2)
  expect_equal(nrow(li$getData()), task$nrow)
  expect_equal(ncol(li$getData()), 2)
  expect_equal(length(l$model$getBaselearnerNames()), length(task$feature_names))

  ## Components:
  l = expect_silent(mlr3::lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
    n_knots = 10, learning_rate = 0.01, baselearner = "components"))
  expect_silent(l$train(task))
  expect_equal(length(l$model$getBaselearnerNames()),
    2 * sum(task$feature_types$type == "integer") + sum(task$feature_types$type != "integer"))
  sp = expect_silent(l$model$baselearner_list$duration_duration_spline_centered$factory)

  expect_equal(as.numeric(sp$getMeta()$df), 3)
  expect_equal(nrow(sp$getData()), floor(sqrt(task$nrow)))
  expect_equal(ncol(sp$getData()), 14 - 2)
})

test_that("Interactions can be included correctly", {
  task = mlr3::tsk("german_credit")
  ints = data.frame(feat1 = rep("duration", 2), feat2 = c("age", "job"))

  l = expect_silent(mlr3::lrn("classif.compboost", interactions = ints))
  expect_silent(l$train(task))
  expect_equal(length(l$model$getBaselearnerNames()), length(task$feature_names) + 2)
  t1 = expect_silent(l$model$baselearner_list$duration_age_tensor$factory)
  t2 = expect_silent(l$model$baselearner_list$duration_job_tensor$factory)

  expect_equal(nrow(t1$getData()), 24^2)
  expect_equal(nrow(t2$getData()), 24 * nlevels(task$data(cols = "job")[[1]]))

  l = expect_silent(mlr3::lrn("classif.compboost", interactions = ints, just_interactions = TRUE))
  expect_silent(l$train(task))
  expect_equal(length(l$model$getBaselearnerNames()), 2)
})

test_that("Early stopping works", {
  task = mlr3::tsk("mtcars")

  l = expect_silent(mlr3::lrn("regr.compboost", oob_fraction = 0, iterations = 1000, early_stop = TRUE))
  expect_error(l$train(task))

  set.seed(314)
  l = mlr3::lrn("regr.compboost", oob_fraction = 0.3, iterations = 1000, early_stop = TRUE, patience = 1)
  expect_silent(l$train(task))
  lgs = expect_silent(l$model$logs)
  expect_true(nrow(lgs) < 1000)

  set.seed(314)
  l = mlr3::lrn("regr.compboost", oob_fraction = 0.3, iterations = 1000, early_stop = TRUE, patience = 10)
  expect_silent(l$train(task))
  lgs2 = expect_silent(l$model$logs)
  expect_true(nrow(lgs2) > nrow(lgs))

  l = expect_error(mlr3::lrn("regr.compboost", oob_fraction = 0.3, iterations = 1000, patience = 10))
})

test_that("Custom loss for oob works", {
  task = mlr3::tsk("mtcars")

  # patience = 100 ensures that all learners are trained at least 100 iterations.

  loss_oob = expect_silent(LossQuadratic$new(0))
  set.seed(31415)
  l1 = expect_silent({mlr3::lrn("regr.compboost", oob_fraction = 0.3, iterations = 100, early_stop = TRUE,
    patience = 100, loss_oob = loss_oob)})
  expect_silent(l1$train(task))
  offset = l1$model$getCoef()$offset
  pmat = cbind(l1$predict_newdata(l1$model$data_oob)$response)
  expect_equal(l1$model$response_oob$getPrediction() + offset, pmat)

  loss_oob = expect_silent(LossQuadratic$new(offset[1]))
  set.seed(31415)
  l2 = expect_silent({mlr3::lrn("regr.compboost", oob_fraction = 0.3, iterations = 100, early_stop = TRUE,
    patience = 100, loss_oob = loss_oob)})
  expect_silent(l2$train(task))
  expect_equal(l2$model$response_oob$getPrediction(), pmat)
  expect_equal(l1$model$data_oob, l2$model$data_oob)
  expect_equal(l1$predict(task), l2$predict(task))

  loss_oob = expect_silent(LossQuadratic$new(pmat, TRUE))
  set.seed(31415)
  l3 = expect_silent({mlr3::lrn("regr.compboost", oob_fraction = 0.3, iterations = 100, early_stop = TRUE,
    patience = 100, loss_oob = loss_oob)})
  expect_silent(l3$train(task))
  expect_equal(l3$model$response_oob$getPrediction() - pmat + offset, pmat)
  expect_equal(l1$model$data_oob, l3$model$data_oob)
  expect_equal(l1$predict(task), l3$predict(task))
})
