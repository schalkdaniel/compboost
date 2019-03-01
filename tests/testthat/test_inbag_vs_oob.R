context("Intrinsic inbag vs. oob works")

test_that("Internal oob is the same as the logger", {

  df = mtcars
  target_var = "mpg"
  char_vars = c("cyl", "vs", "am", "gear", "carb")

  for (feature in char_vars) {
    df[[feature]] = as.factor(df[[feature]])
  }

  n_data = nrow(df)

  set.seed(31415)
  idx_test = sample(x = seq_len(n_data), size = floor(n_data * 0.25))
  idx_train = setdiff(x = seq_len(n_data), idx_test)

  cboost = Compboost$new(data = df[idx_train, ], target = target_var,
    loss = LossQuadratic$new(), learning_rate = 0.005)

  for (feature_name in setdiff(names(df), target_var)) {
    if (feature_name %in% char_vars) {
      cboost$addBaselearner(feature = feature_name, id = "category",
        bl_factory = BaselearnerPolynomial, intercept = FALSE)
    } else {
      cboost$addBaselearner(feature = feature_name, id = "spline",
        bl_factory = BaselearnerPSpline, degree = 3, n_knots = 10)
    }
  }

  oob_data = cboost$prepareData(df[idx_test,])
  oob_response = ResponseRegr$new("oob_response", as.matrix(df[[target_var]][idx_test]))

  cboost$addLogger(logger = LoggerOobRisk, logger_id = "oob_risk",
    used_loss = LossQuadratic$new(), eps_for_break = 0, patience = 5,
    oob_data = oob_data, oob_response = oob_response)

  nuisance = capture.output(suppressWarnings({
    cboost$train(6000)
  }))
  set.seed(31415)
  nuisance = capture.output(suppressWarnings({
    cboost1 = boostSplines(data = df, target = target_var, loss = LossQuadratic$new(), learning_rate = 0.005,
      iterations = 6000L, degree = 3, n_knots = 10, oob_fraction = 0.25)
  }))
  expect_equal(rownames(df)[idx_train], rownames(cboost1$data))
  expect_equal(rownames(df)[idx_test], rownames(cboost1$data_oob))
  expect_equal(cboost$getLoggerData(), cboost1$getLoggerData())

  gg = cboost$plotInbagVsOobRisk()
  gg1 = cboost1$plotInbagVsOobRisk()

  expect_true(inherits(gg, "ggplot"))
  expect_true(inherits(gg1, "ggplot"))
  expect_equal(gg, gg1)
})
