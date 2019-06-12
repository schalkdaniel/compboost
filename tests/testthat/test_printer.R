context("Printer works")

test_that("data printer works", {
  X = as.matrix(1:10)

  expect_silent({ data_source = InMemoryData$new(X, "x") })
  expect_silent({ data_target = InMemoryData$new() })

  expect_output({ test_source = show(data_source) })
  expect_output({ test_target = show(data_target) })

  expect_equal(test_source, "InMemoryDataPrinter")
  expect_equal(test_target, "InMemoryDataPrinter")
})

test_that("factory list printer works", {

  expect_silent({ factory_list = BlearnerFactoryList$new() })
  expect_output({ test_factory_list_printer = show(factory_list) })
  expect_equal(test_factory_list_printer, "BlearnerFactoryListPrinter")

})

test_that("Loss printer works", {

  expect_silent({ quadratic_loss = LossQuadratic$new() })
  expect_silent({ absolute_loss  = LossAbsolute$new() })
  expect_silent({ binomial_loss = LossBinomial$new() })
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })

  myLossFun = function (true_value, prediction) NULL
  myGradientFun = function (true_value, prediction) NULL
  myConstantInitializerFun = function (true_value) NULL

  expect_silent({ custom_cpp_loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
  expect_silent({ custom_loss = LossCustom$new(myLossFun, myGradientFun, myConstantInitializerFun) })

  expect_output({ test_quadratic_printer  = show(quadratic_loss) })
  expect_output({ test_absolute_printer   = show(absolute_loss) })
  expect_output({ test_custom_printer     = show(custom_loss) })
  expect_output({ test_custom_cpp_printer = show(custom_cpp_loss) })
  expect_output({ test_binomialprinter    = show(binomial_loss) })

  expect_equal(test_quadratic_printer, "LossQuadraticPrinter")
  expect_equal(test_absolute_printer, "LossAbsolutePrinter")
  expect_equal(test_binomialprinter, "LossBinomialPrinter")
  expect_equal(test_custom_cpp_printer, "LossCustomCppPrinter")
  expect_equal(test_custom_printer, "LossCustomPrinter")

})

test_that("Baselearner factory printer works", {

  df = mtcars

  X_hp = cbind(1, df[["hp"]])
  X_hp_sp = as.matrix(df[["hp"]])

  expect_silent({ data_source    = InMemoryData$new(X_hp, "hp") })
  expect_silent({ data_source_sp = InMemoryData$new(X_hp_sp, "hp") })
  expect_silent({ data_target    = InMemoryData$new() })

  expect_silent({ linear_factory_hp = BaselearnerPolynomial$new(data_source, data_target,
    list(degree = 1, intercept = FALSE)) })
  expect_output({ linear_factory_hp_printer = show(linear_factory_hp) })
  expect_equal(linear_factory_hp_printer, "BaselearnerPolynomialPrinter")

  expect_silent({ quad_factory_hp = BaselearnerPolynomial$new(data_source, data_target,
    list(degree = 2, intercept = FALSE)) })
  expect_output({ quad_factory_hp_printer = show(quad_factory_hp) })
  expect_equal(quad_factory_hp_printer, "BaselearnerPolynomialPrinter")

  expect_silent({ cubic_factory_hp = BaselearnerPolynomial$new(data_source, data_target,
    list(degree = 3, intercept = FALSE)) })
  expect_output({ cubic_factory_hp_printer = show(cubic_factory_hp) })
  expect_equal(cubic_factory_hp_printer, "BaselearnerPolynomialPrinter")

  expect_silent({ poly_factory_hp = BaselearnerPolynomial$new(data_source, data_target,
    list(degree = 4, intercept = FALSE)) })
  expect_output({ poly_factory_hp_printer = show(poly_factory_hp) })
  expect_equal(poly_factory_hp_printer, "BaselearnerPolynomialPrinter")

  expect_silent({ spline_factory = BaselearnerPSpline$new(data_source_sp, data_target,
    list(degree = 3, n_knots = 5, penalty = 2.5, differences = 2)) })
  expect_output({ spline_printer = show(spline_factory) })
  expect_equal(spline_printer, "BaselearnerPSplinePrinter")

  instantiateData = function (X)
  {
    return(X);
  }
  trainFun = function (y, X) {
    return(solve(t(X) %*% X) %*% t(X) %*% y)
  }
  predictFun = function (model, newdata) {
    return(newdata %*% model)
  }
  extractParameter = function (model) {
    return(model)
  }

  expect_silent({
    custom_factory = BaselearnerCustom$new(data_source, data_target,
      list(instantiate_fun = instantiateData, train_fun = trainFun,
        predict_fun = predictFun, param_fun = extractParameter))
  })
  expect_output({ custom_factory_printer = show(custom_factory) })

  expect_equal(custom_factory_printer, "BaselearnerCustomPrinter")
  expect_output(Rcpp::sourceCpp(code = getCustomCppExample()))
  expect_silent({
    custom_cpp_factory = BaselearnerCustomCpp$new(data_source, data_target,
      list(instantiate_ptr = dataFunSetter(), train_ptr = trainFunSetter(),
        predict_ptr = predictFunSetter()))
  })
  expect_output({ custom_cpp_factory_printer = show(custom_cpp_factory) })
  expect_equal(custom_cpp_factory_printer, "BaselearnerCustomCppPrinter")
})

test_that("Optimizer printer works", {

  expect_silent({ greedy_optimizer = OptimizerCoordinateDescent$new() })
  expect_output({ greedy_optimizer_printer = show(greedy_optimizer) })
  expect_equal(greedy_optimizer_printer, "OptimizerCoordinateDescentPrinter")

  expect_silent({ greedy_optimizer_ls = OptimizerCoordinateDescentLineSearch$new() })
  expect_output({ greedy_optimizer_printer_ls = show(greedy_optimizer_ls) })
  expect_equal(greedy_optimizer_printer_ls, "OptimizerCoordinateDescentLineSearchPrinter")

})

test_that("Logger(List) printer works", {

  expect_silent({ loss_quadratic = LossQuadratic$new() })

  expect_silent({
    eval_oob_test = list(
      InMemoryData$new(as.matrix(NA_real_), "hp"),
      InMemoryData$new(as.matrix(NA_real_), "wt")
    )
  })

  y = NA_real_
  response_oob = ResponseRegr$new("mpg_oog", as.matrix(y))

  expect_silent({ log_iterations = LoggerIteration$new("iterations", TRUE, 500) })
  expect_silent({ log_time       = LoggerTime$new("time", FALSE, 500, "microseconds") })
  expect_silent({ log_inbag      = LoggerInbagRisk$new("inbag_risk", FALSE, loss_quadratic, 0.05) })
  expect_silent({ log_oob        = LoggerOobRisk$new("oob_risk", FALSE, loss_quadratic, 0.05, 5, eval_oob_test, response_oob) })
  expect_silent({ logger_list = LoggerList$new() })
  expect_output({ logger_list_printer = show(logger_list) })

  expect_equal(logger_list_printer, "LoggerListPrinter")

  expect_silent(logger_list$registerLogger(log_iterations))
  expect_silent(logger_list$registerLogger(log_time))
  expect_silent(logger_list$registerLogger(log_inbag))
  expect_silent(logger_list$registerLogger(log_oob))

  expect_output({ log_iterations_printer = show(log_iterations) })
  expect_output({ log_time_printer       = show(log_time) })
  expect_output({ log_inbag              = show(log_inbag) })
  expect_output({ log_oob                = show(log_oob) })

  expect_output({ logger_list_printer    = show(logger_list) })

  expect_equal(log_iterations_printer, "LoggerIterationPrinter")
  expect_equal(log_time_printer, "LoggerTimePrinter")
  expect_equal(log_inbag, "LoggerInbagRiskPrinter")
  expect_equal(log_oob, "LoggerOobRiskPrinter")

  expect_equal(logger_list_printer, "LoggerListPrinter")
})

test_that("Compboost printer works", {

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
  iter_max = 500

  expect_silent({ linear_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target_hp1,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ linear_factory_wt = BaselearnerPolynomial$new(data_source_wt, data_target_wt,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ quadratic_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target_hp2,
    list(degree = 2, intercept = FALSE)) })
  expect_silent({ factory_list = BlearnerFactoryList$new() })

  expect_silent(factory_list$registerFactory(linear_factory_hp))
  expect_silent(factory_list$registerFactory(linear_factory_wt))
  expect_silent(factory_list$registerFactory(quadratic_factory_hp))

  expect_silent({ loss_quadratic = LossQuadratic$new() })
  expect_silent({ optimizer = OptimizerCoordinateDescent$new() })

  expect_silent({ log_iterations = LoggerIteration$new("iterations", TRUE, iter_max) })
  expect_silent({ log_time_ms    = LoggerTime$new("time_ms", TRUE, 50000, "microseconds") })
  expect_silent({ log_time_sec   = LoggerTime$new("time_sec", TRUE, 2, "seconds") })
  expect_silent({ log_time_min   = LoggerTime$new("time_min", TRUE, 1, "minutes") })
  expect_silent({ log_inbag      = LoggerInbagRisk$new("inbag_risk", FALSE, loss_quadratic, 0.01) })
  expect_silent({ log_oob        = LoggerOobRisk$new("oob_risk", FALSE, loss_quadratic, 0.01, 5, eval_oob_test, response_oob) })

  expect_silent({ logger_list = LoggerList$new() })
  expect_silent({ logger_list$registerLogger(log_iterations) })
  expect_silent({ logger_list$registerLogger(log_time_ms) })
  expect_silent({ logger_list$registerLogger(log_time_sec) })
  expect_silent({ logger_list$registerLogger(log_time_min) })
  expect_silent({ logger_list$registerLogger(log_inbag) })
  expect_silent({ logger_list$registerLogger(log_oob) })

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
  expect_output(cboost$train(trace = 0))
  expect_output({ cboost_printer = show(cboost) })
  expect_equal(cboost_printer, "CompboostInternalPrinter")

})
