context("Compboost works")

test_that("Compboost loggs correctly", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X.hp = as.matrix(df[["hp"]], ncol = 1)
  X.wt = as.matrix(df[["wt"]], ncol = 1)

  y = df[["mpg"]]

  expect_silent({ data.source.hp = InMemoryData$new(X.hp, "hp") })
  expect_silent({ data.source.wt = InMemoryData$new(X.wt, "wt") })

  expect_silent({ data.target.hp1 = InMemoryData$new() })
  expect_silent({ data.target.hp2 = InMemoryData$new() })
  expect_silent({ data.target.wt  = InMemoryData$new() })

  eval.oob.test = list(data.source.hp, data.source.wt)
  learning.rate = 0.05
  iter.max      = 500

  expect_silent({ linear.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target.hp1, 1, FALSE) })
  expect_silent({ linear.factory.wt = BaselearnerPolynomial$new(data.source.wt, data.target.wt, 1, FALSE) })
  expect_silent({ quadratic.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target.hp2, 2, FALSE) })
  expect_silent({ factory.list = BlearnerFactoryList$new() })
  expect_silent({ factory.list$registerFactory(linear.factory.hp) })
  expect_silent({ factory.list$registerFactory(linear.factory.wt) })
  expect_silent({ factory.list$registerFactory(quadratic.factory.hp) })
  expect_silent({ loss.quadratic = LossQuadratic$new() })
  expect_silent({ optimizer = OptimizerCoordinateDescent$new() })
  expect_silent({ log.iterations = LoggerIteration$new(TRUE, iter.max) })
  expect_silent({ log.time.ms    = LoggerTime$new(TRUE, 200000, "microseconds") })
  expect_silent({ log.time.sec   = LoggerTime$new(TRUE, 10, "seconds") })
  expect_silent({ log.time.min   = LoggerTime$new(TRUE, 10, "minutes") })
  expect_silent({ log.inbag      = LoggerInbagRisk$new(FALSE, loss.quadratic, 0.01) })
  expect_silent({ log.oob        = LoggerOobRisk$new(FALSE, loss.quadratic, 0.01, eval.oob.test, y) })
  expect_silent({ logger.list = LoggerList$new() })
  expect_silent({ logger.list$registerLogger(" iterations", log.iterations) })
  expect_silent({ logger.list$registerLogger("time.microseconds", log.time.ms) })
  expect_silent({ logger.list$registerLogger("time.seconds", log.time.sec) })
  expect_silent({ logger.list$registerLogger("time.minutes", log.time.min) })
  expect_silent({ logger.list$registerLogger("inbag.risk", log.inbag) })
  expect_silent({ logger.list$registerLogger("oob.risk", log.oob) })
  
  expect_output(show(log.inbag))
  expect_output(show(log.oob))

  expect_output(logger.list$printRegisteredLogger())
  expect_silent({
    cboost = Compboost_internal$new(
      response      = y,
      learning_rate = learning.rate,
      stop_if_all_stopper_fulfilled = FALSE,
      factory_list = factory.list,
      loss         = loss.quadratic,
      logger_list  = logger.list,
      optimizer    = optimizer
    )
  })
  expect_output({ cboost$train(trace = 1) })
  expect_silent({ logger.data = cboost$getLoggerData() })
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 6)
  expect_equal(dim(logger.data$logger.data), c(iter.max, logger.list$getNumberOfRegisteredLogger()))
  expect_equal(cboost$getLoggerData()$logger.data[, 1], 1:500)
  expect_equal(cboost$getLoggerData()$logger.data[, 2], cboost$getLoggerData()$logger.data[, 3])
  
})

test_that("compboost does the same as mboost", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X.hp = as.matrix(df[["hp"]], ncol = 1)
  X.wt = as.matrix(df[["wt"]], ncol = 1)

  y = df[["mpg"]]

  expect_silent({ data.source.hp = InMemoryData$new(X.hp, "hp") })
  expect_silent({ data.source.wt = InMemoryData$new(X.wt, "wt") })
  expect_silent({ data.target.hp1 = InMemoryData$new() })
  expect_silent({ data.target.hp2 = InMemoryData$new() })
  expect_silent({ data.target.wt  = InMemoryData$new() })

  eval.oob.test = list(data.source.hp, data.source.wt)

  learning.rate = 0.05
  iter.max = 500

  expect_silent({ linear.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target.hp1, 1, FALSE) })
  expect_silent({ linear.factory.wt = BaselearnerPolynomial$new(data.source.wt, data.target.wt, 1, FALSE) })
  expect_silent({ quadratic.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target.hp2, 2, FALSE) })
  expect_silent({ factory.list = BlearnerFactoryList$new() })

  # Register factorys:
  expect_silent(factory.list$registerFactory(linear.factory.hp))
  expect_silent(factory.list$registerFactory(linear.factory.wt))
  expect_silent(factory.list$registerFactory(quadratic.factory.hp))
  expect_silent({ loss.quadratic = LossQuadratic$new() })
  expect_silent({ optimizer = OptimizerCoordinateDescent$new() })
  expect_silent({ log.iterations = LoggerIteration$new(TRUE, iter.max) })
  expect_silent({ log.time       = LoggerTime$new(FALSE, 500, "microseconds") })
  expect_silent({ logger.list = LoggerList$new() })
  expect_silent({ logger.list$registerLogger(" iterations", log.iterations) })
  expect_silent({ logger.list$registerLogger("time.ms", log.time) })
  expect_silent({
    cboost = Compboost_internal$new(
      response      = y,
      learning_rate = learning.rate,
      stop_if_all_stopper_fulfilled = TRUE,
      factory_list = factory.list,
      loss         = loss.quadratic,
      logger_list  = logger.list,
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
      control = boost_control(mstop = iter.max, nu = learning.rate)
    )
  })
  # Create vector of selected baselearner:
  # --------------------------------------

  cboost.xselect = match(
    x     = cboost$getSelectedBaselearner(),
    table = c(
      "hp_polynomial_degree_1",
      "wt_polynomial_degree_1",
      "hp_polynomial_degree_2"
    )
  )
  expect_equal(predict(mod), cboost$getPrediction(FALSE))
  expect_equal(mod$xselect(), cboost.xselect)
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
  expect_equal(dim(cboost$getLoggerData()$logger.data), c(500, 2))
  expect_equal(cboost$getLoggerData()$logger.data[, 1], 1:500)
  expect_equal(length(cboost$getLoggerData()$logger.data[, 2]), 500)

  # Check if paraemter getter of smaller iteration works:
  suppressWarnings({
    mod.reduced = mboost(
      formula = mpg ~ bols(hp, intercept = FALSE) +
        bols(wt, intercept = FALSE) +
        bols(hp2, intercept = FALSE),
      data    = df,
      control = boost_control(mstop = 200, nu = learning.rate)
    )
  })

  expect_equal(
    unname(
      unlist(
        mod.reduced$coef()[
          order(
            unlist(
              lapply(names(unlist(mod.reduced$coef()[1:3])), function (x) {
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
  matrix.compare = matrix(NA_real_, nrow = 3, ncol = 3)

  for (i in seq_along(idx)) {
    expect_silent({ matrix.compare[i, ] = unname(unlist(cboost$getParameterAtIteration(idx[i]))) })
  }
  expect_equal(cboost$getParameterMatrix()$parameter.matrix[idx, ], matrix.compare)
  expect_equal(cboost$predict(eval.oob.test, FALSE), predict(mod, df))
  expect_equal(cboost$predictAtIteration(eval.oob.test, 200, FALSE), predict(mod.reduced, df))

  suppressWarnings({
    mod.new = mboost(
      formula = mpg ~ bols(hp, intercept = FALSE) +
        bols(wt, intercept = FALSE) +
        bols(hp2, intercept = FALSE),
      data    = df,
      control = boost_control(mstop = 700, nu = learning.rate)
    )
  })
  expect_output(cboost$setToIteration(700))
  expect_equal(cboost$getPrediction(FALSE), predict(mod.new))
})

