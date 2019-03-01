context("Compboost internal")

test_that("Compboost loggs correctly", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X_hp = as.matrix(df[["hp"]], ncol = 1)
  X_wt = as.matrix(df[["wt"]], ncol = 1)

  y = df[["mpg"]]
  response = ResponseRegr$new("mpg", as.matrix(y))
  response_oob = ResponseRegr$new("mpg_oog", as.matrix(y))

  expect_silent({ data_source_hp = InMemoryData$new(X_hp, "hp") })
  expect_silent({ data_source_wt = InMemoryData$new(X_wt, "wt") })

  expect_silent({ data_target_hp1 = InMemoryData$new() })
  expect_silent({ data_target_hp2 = InMemoryData$new() })
  expect_silent({ data_target_wt  = InMemoryData$new() })

  eval_oob_test = list(data_source_hp, data_source_wt)
  learning_rate = 0.05
  iter_max      = 500

  expect_silent({ linear_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target_hp1,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ linear_factory_wt = BaselearnerPolynomial$new(data_source_wt, data_target_wt,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ quadratic_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target_hp2,
    list(degree = 2, intercept = FALSE)) })
  expect_silent({ factory_list = BlearnerFactoryList$new() })
  expect_silent({ factory_list$registerFactory(linear_factory_hp) })
  expect_silent({ factory_list$registerFactory(linear_factory_wt) })
  expect_silent({ factory_list$registerFactory(quadratic_factory_hp) })
  expect_silent({ loss_quadratic = LossQuadratic$new() })
  expect_silent({ optimizer = OptimizerCoordinateDescent$new() })
  expect_silent({ log_iterations = LoggerIteration$new(" iterations", TRUE, iter_max) })
  expect_silent({ log_time_ms    = LoggerTime$new("time_microseconds", TRUE, 200000, "microseconds") })
  expect_silent({ log_time_sec   = LoggerTime$new("time_seconds", TRUE, 10, "seconds") })
  expect_silent({ log_time_min   = LoggerTime$new("time_minutes", TRUE, 10, "minutes") })
  expect_silent({ log_inbag      = LoggerInbagRisk$new("inbag_risk", FALSE, loss_quadratic, 0.01) })
  expect_silent({ log_oob        = LoggerOobRisk$new("oob_risk", FALSE, loss_quadratic, 0.01, 5, eval_oob_test, response_oob) })
  expect_silent({ logger_list = LoggerList$new() })
  expect_silent({ logger_list$registerLogger(log_iterations) })
  expect_silent({ logger_list$registerLogger(log_time_ms) })
  expect_silent({ logger_list$registerLogger(log_time_sec) })
  expect_silent({ logger_list$registerLogger(log_time_min) })
  expect_silent({ logger_list$registerLogger(log_inbag) })
  expect_silent({ logger_list$registerLogger(log_oob) })

  expect_output(show(log_inbag))
  expect_output(show(log_oob))

  expect_output(logger_list$printRegisteredLogger())
  expect_silent({
    cboost = Compboost_internal$new(
      response      = response,
      learning_rate = learning_rate,
      stop_if_all_stopper_fulfilled = FALSE,
      factory_list = factory_list,
      loss         = loss_quadratic,
      logger_list  = logger_list,
      optimizer    = optimizer
    )
  })
  expect_output({ cboost$train(trace = 1) })
  expect_silent({ logger_data = cboost$getLoggerData() })
  expect_equal(logger_list$getNumberOfRegisteredLogger(), 6)
  expect_equal(dim(logger_data$logger_data), c(iter_max, logger_list$getNumberOfRegisteredLogger()))
  expect_equal(cboost$getLoggerData()$logger_data[, 1], 1:500)
  expect_equal(cboost$getLoggerData()$logger_data[, 2], cboost$getLoggerData()$logger_data[, 3])

})

test_that("compboost does the same as mboost", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X_hp = as.matrix(df[["hp"]], ncol = 1)
  X_wt = as.matrix(df[["wt"]], ncol = 1)

  y = df[["mpg"]]
  response = ResponseRegr$new("mpg", as.matrix(y))

  expect_silent({ data_source_hp = InMemoryData$new(X_hp, "hp") })
  expect_silent({ data_source_wt = InMemoryData$new(X_wt, "wt") })
  expect_silent({ data_target_hp1 = InMemoryData$new() })
  expect_silent({ data_target_hp2 = InMemoryData$new() })
  expect_silent({ data_target_wt  = InMemoryData$new() })

  eval_oob_test = list(data_source_hp, data_source_wt)

  learning_rate = 0.05
  iter_max = 500

  expect_silent({ linear_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target_hp1,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ linear_factory_wt = BaselearnerPolynomial$new(data_source_wt, data_target_wt,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ quadratic_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target_hp2,
    list(degree = 2, intercept = FALSE)) })
  expect_silent({ factory_list = BlearnerFactoryList$new() })

  # Register factorys:
  expect_silent(factory_list$registerFactory(linear_factory_hp))
  expect_silent(factory_list$registerFactory(linear_factory_wt))
  expect_silent(factory_list$registerFactory(quadratic_factory_hp))
  expect_silent({ loss_quadratic = LossQuadratic$new() })
  expect_silent({ optimizer = OptimizerCoordinateDescent$new() })
  expect_silent({ log_iterations = LoggerIteration$new(" iterations", TRUE, iter_max) })
  expect_silent({ log_time       = LoggerTime$new("time_ms", FALSE, 500, "microseconds") })
  expect_silent({ logger_list = LoggerList$new() })
  expect_silent({ logger_list$registerLogger(log_iterations) })
  expect_silent({ logger_list$registerLogger(log_time) })
  expect_silent({
    cboost = Compboost_internal$new(
      response      = response,
      learning_rate = learning_rate,
      stop_if_all_stopper_fulfilled = TRUE,
      factory_list = factory_list,
      loss         = loss_quadratic,
      logger_list  = logger_list,
      optimizer    = optimizer
    )
  })
  expect_output(cboost$train(trace = 100))
  suppressWarnings({
    library(mboost)

    mod = mboost(
      formula = mpg ~ bols(hp, intercept = FALSE) +
        bols(wt, intercept = FALSE) +
        bols(hp2, intercept = FALSE),
      data    = df,
      control = boost_control(mstop = iter_max, nu = learning_rate)
    )
  })
  # Create vector of selected baselearner:
  # --------------------------------------

  cboost_xselect = match(
    x     = cboost$getSelectedBaselearner(),
    table = c(
      "hp_polynomial_degree_1",
      "wt_polynomial_degree_1",
      "hp_polynomial_degree_2"
    )
  )
  expect_equal(predict(mod), cboost$getPrediction(FALSE))
  expect_equal(mod$xselect(), cboost_xselect)
  expect_equal(
    unname(
      unlist(
        mod$coef()[
          order(
            unlist(
              lapply(names(unlist(mod$coef()[1:3])), function (x) {
                strsplit(x, "[.]")[[1]][2]
              })
            )
          )
        ]
      )
    ),
    unname(unlist(cboost$getEstimatedParameter()))
  )
  expect_equal(dim(cboost$getLoggerData()$logger_data), c(500, 2))
  expect_equal(cboost$getLoggerData()$logger_data[, 1], 1:500)
  expect_equal(length(cboost$getLoggerData()$logger_data[, 2]), 500)

  # Check if paraemter getter of smaller iteration works:
  suppressWarnings({
    mod_reduced = mboost(
      formula = mpg ~ bols(hp, intercept = FALSE) +
        bols(wt, intercept = FALSE) +
        bols(hp2, intercept = FALSE),
      data    = df,
      control = boost_control(mstop = 200, nu = learning_rate)
    )
  })

  expect_equal(
    unname(
      unlist(
        mod_reduced$coef()[
          order(
            unlist(
              lapply(names(unlist(mod_reduced$coef()[1:3])), function (x) {
                strsplit(x, "[.]")[[1]][2]
              })
            )
          )
          ]
      )
    ),
    unname(unlist(cboost$getParameterAtIteration(200)))
  )

  idx = 2:4 * 120
  matrix_compare = matrix(NA_real_, nrow = 3, ncol = 3)

  for (i in seq_along(idx)) {
    expect_silent({ matrix_compare[i, ] = unname(unlist(cboost$getParameterAtIteration(idx[i]))) })
  }
  expect_equal(cboost$getParameterMatrix()$parameter_matrix[idx, ], matrix_compare)
  expect_equal(cboost$predict(eval_oob_test, FALSE), predict(mod, df))
  expect_silent(cboost$setToIteration(200, -1))
  expect_equal(cboost$predict(eval_oob_test, FALSE), predict(mod_reduced, df))

  suppressWarnings({
    mod_new = mboost(
      formula = mpg ~ bols(hp, intercept = FALSE) +
        bols(wt, intercept = FALSE) +
        bols(hp2, intercept = FALSE),
      data    = df,
      control = boost_control(mstop = 700, nu = learning_rate)
    )
  })
  expect_output(cboost$setToIteration(700, -1))
  expect_equal(cboost$getPrediction(FALSE), predict(mod_new))
})

