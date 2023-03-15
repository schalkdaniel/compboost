context("mlr learners")

if (FALSE) {
  devtools::load_all()
}

test_that("Basic parameter are working", {
  task = tsk("german_credit")

  l = expect_silent(lrn("classif.compboost"))
  expect_silent(l$train(task))
  expect_true(checkmate::testR6(l$model, "Compboost"))

  l = expect_silent(lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
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

  l = expect_error(lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
    n_knots = 10, learning_rate = 0.01, baselearner = "linear"))
  l = expect_silent(lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
    learning_rate = 0.01, baselearner = "linear"))
  expect_silent(l$train(task))
  li = expect_silent(l$model$baselearner_list$duration_linear$factory)
  ri = expect_silent(l$model$baselearner_list$job_ridge$factory)

  expect_equal(as.numeric(ri$getMeta()$df), 2)
  expect_equal(nrow(li$getData()), task$nrow)
  expect_equal(ncol(li$getData()), 2)
  expect_equal(length(l$model$getBaselearnerNames()), length(task$feature_names))

  ## Components:
  l = expect_silent(lrn("classif.compboost", bin_root = 2, df = 3, df_cat = 2,
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
  task = tsk("german_credit")
  ints = data.frame(feat1 = rep("duration", 2), feat2 = c("age", "job"))

  l = expect_silent(lrn("classif.compboost", interactions = ints))
  expect_silent(l$train(task))
  expect_equal(length(l$model$getBaselearnerNames()), length(task$feature_names) + 2)
  t1 = expect_silent(l$model$baselearner_list$duration_age_tensor$factory)
  t2 = expect_silent(l$model$baselearner_list$duration_job_tensor$factory)

  expect_equal(nrow(t1$getData()), 24^2)
  expect_equal(nrow(t2$getData()), 24 * nlevels(task$data(cols = "job")[[1]]))

  l = expect_silent(lrn("classif.compboost", interactions = ints, just_interactions = TRUE))
  expect_silent(l$train(task))
  expect_equal(length(l$model$getBaselearnerNames()), 2)
})

test_that("Early stopping works", {
  task = tsk("mtcars")

  l = expect_silent(lrn("regr.compboost", oob_fraction = 0, iterations = 1000, early_stop = TRUE))
  expect_error(l$train(task))

  set.seed(314)
  l = lrn("regr.compboost", oob_fraction = 0.3, iterations = 1000, early_stop = TRUE, patience = 1)
  expect_silent(l$train(task))
  lgs = expect_silent(l$model$logs)
  expect_true(nrow(lgs) < 1000)

  set.seed(314)
  l = lrn("regr.compboost", oob_fraction = 0.3, iterations = 1000, early_stop = TRUE, patience = 10)
  expect_silent(l$train(task))
  lgs2 = expect_silent(l$model$logs)
  expect_true(nrow(lgs2) > nrow(lgs))

  l = expect_error(lrn("regr.compboost", oob_fraction = 0.3, iterations = 1000, patience = 10))
})
