context("Printer works")

test_that("data printer works", {
  X = as.matrix(1:10)

  expect_silent({ data.source = InMemoryData$new(X, "x") })
  expect_silent({ data.target = InMemoryData$new() })

  expect_output({ test.source = show(data.source) })
  expect_output({ test.target = show(data.target) })

  expect_equal(test.source, "InMemoryDataPrinter")
  expect_equal(test.target, "InMemoryDataPrinter")
})

test_that("factory list printer works", {

  expect_silent({ factory.list = BlearnerFactoryList$new() })
  expect_output({ test.factory.list.printer = show(factory.list) })
  expect_equal(test.factory.list.printer, "BlearnerFactoryListPrinter")

})

test_that("Loss printer works", {

  expect_silent({ quadratic.loss = LossQuadratic$new() })
  expect_silent({ absolute.loss  = LossAbsolute$new() })
  expect_silent({ binomial.loss = LossBinomial$new() })
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)) })  

  myLossFun = function (true.value, prediction) NULL
  myGradientFun = function (true.value, prediction) NULL
  myConstantInitializerFun = function (true.value) NULL

  expect_silent({ custom.cpp.loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
  expect_silent({ custom.loss = LossCustom$new(myLossFun, myGradientFun, myConstantInitializerFun) })

  expect_output({ test.quadratic.printer  = show(quadratic.loss) })
  expect_output({ test.absolute.printer   = show(absolute.loss) })
  expect_output({ test.custom.printer     = show(custom.loss) })
  expect_output({ test.custom.cpp.printer = show(custom.cpp.loss) })
  expect_output({ test.binomialprinter    = show(binomial.loss) })

  expect_equal(test.quadratic.printer, "LossQuadraticPrinter")
  expect_equal(test.absolute.printer, "LossAbsolutePrinter")
  expect_equal(test.binomialprinter, "LossBinomialPrinter")
  expect_equal(test.custom.cpp.printer, "LossCustomCppPrinter")
  expect_equal(test.custom.printer, "LossCustomPrinter")

})

test_that("Baselearner factory printer works", {

  df = mtcars

  X.hp = cbind(1, df[["hp"]])
  X.hp.sp = as.matrix(df[["hp"]])

  expect_silent({ data.source    = InMemoryData$new(X.hp, "hp") })
  expect_silent({ data.source.sp = InMemoryData$new(X.hp.sp, "hp") })
  expect_silent({ data.target    = InMemoryData$new() })
  
  expect_silent({ linear.factory.hp = BaselearnerPolynomial$new(data.source, data.target, 1, FALSE) })
  expect_output({ linear.factory.hp.printer = show(linear.factory.hp) })
  expect_equal(linear.factory.hp.printer, "BaselearnerPolynomialPrinter")
  
  expect_silent({ quad.factory.hp = BaselearnerPolynomial$new(data.source, data.target, 2, FALSE) })
  expect_output({ quad.factory.hp.printer = show(quad.factory.hp) })
  expect_equal(quad.factory.hp.printer, "BaselearnerPolynomialPrinter")
  
  expect_silent({ cubic.factory.hp = BaselearnerPolynomial$new(data.source, data.target, 3, FALSE) })
  expect_output({ cubic.factory.hp.printer = show(cubic.factory.hp) })
  expect_equal(cubic.factory.hp.printer, "BaselearnerPolynomialPrinter")
  
  expect_silent({ poly.factory.hp = BaselearnerPolynomial$new(data.source, data.target, 4, FALSE) })
  expect_output({ poly.factory.hp.printer = show(poly.factory.hp) })
  expect_equal(poly.factory.hp.printer, "BaselearnerPolynomialPrinter")
  
  expect_silent({ spline.factory = BaselearnerPSpline$new(data.source.sp, data.target, 3, 5, 2.5, 2) })
  expect_output({ spline.printer = show(spline.factory) })
  expect_equal(spline.printer, "BaselearnerPSplinePrinter")

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
    custom.factory = BaselearnerCustom$new(data.source, data.target,
      instantiateData, trainFun, predictFun, extractParameter)
  })
  expect_output({ custom.factory.printer = show(custom.factory) })

  expect_equal(custom.factory.printer, "BaselearnerCustomPrinter")
  expect_output(Rcpp::sourceCpp(code = getCustomCppExample()))
  expect_silent({
    custom.cpp.factory = BaselearnerCustomCpp$new(data.source, data.target,
      dataFunSetter(), trainFunSetter(), predictFunSetter())
  })
  expect_output({ custom.cpp.factory.printer = show(custom.cpp.factory) })
  expect_equal(custom.cpp.factory.printer, "BaselearnerCustomCppPrinter")
})

test_that("Optimizer printer works", {

  expect_silent({ greedy.optimizer = OptimizerCoordinateDescent$new() })
  expect_output({ greedy.optimizer.printer = show(greedy.optimizer) })
  expect_equal(greedy.optimizer.printer, "OptimizerCoordinateDescentPrinter")

})

test_that("Logger(List) printer works", {

  expect_silent({ loss.quadratic = LossQuadratic$new() })

  expect_silent({
    eval.oob.test = list(
      InMemoryData$new(as.matrix(NA_real_), "hp"),
      InMemoryData$new(as.matrix(NA_real_), "wt")
    )
  })

  y = NA_real_

  expect_silent({ log.iterations = LoggerIteration$new(TRUE, 500) })
  expect_silent({ log.time       = LoggerTime$new(FALSE, 500, "microseconds") })
  expect_silent({ log.inbag      = LoggerInbagRisk$new(FALSE, loss.quadratic, 0.05) })
  expect_silent({ log.oob        = LoggerOobRisk$new(FALSE, loss.quadratic, 0.05, eval.oob.test, y) })
  expect_silent({ logger.list = LoggerList$new() })
  expect_output({ logger.list.printer = show(logger.list) })

  expect_equal(logger.list.printer, "LoggerListPrinter")

  expect_silent(logger.list$registerLogger("iterations", log.iterations))
  expect_silent(logger.list$registerLogger("time", log.time))
  expect_silent(logger.list$registerLogger("inbag.risk", log.inbag))
  expect_silent(logger.list$registerLogger("oob.risk", log.oob))

  expect_output({ log.iterations.printer = show(log.iterations) })
  expect_output({ log.time.printer       = show(log.time) })
  expect_output({ log.inbag              = show(log.inbag) })
  expect_output({ log.oob                = show(log.oob) })

  expect_output({ logger.list.printer    = show(logger.list) })

  expect_equal(log.iterations.printer, "LoggerIterationPrinter")
  expect_equal(log.time.printer, "LoggerTimePrinter")
  expect_equal(log.inbag, "LoggerInbagRiskPrinter")
  expect_equal(log.oob, "LoggerOobRiskPrinter")

  expect_equal(logger.list.printer, "LoggerListPrinter")
})

test_that("Compboost printer works", {

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

  expect_silent(factory.list$registerFactory(linear.factory.hp))
  expect_silent(factory.list$registerFactory(linear.factory.wt))
  expect_silent(factory.list$registerFactory(quadratic.factory.hp))

  expect_silent({ loss.quadratic = LossQuadratic$new() })
  expect_silent({ optimizer = OptimizerCoordinateDescent$new() })

  expect_silent({ log.iterations = LoggerIteration$new(TRUE, iter.max) })
  expect_silent({ log.time.ms    = LoggerTime$new(TRUE, 50000, "microseconds") })
  expect_silent({ log.time.sec   = LoggerTime$new(TRUE, 2, "seconds") })
  expect_silent({ log.time.min   = LoggerTime$new(TRUE, 1, "minutes") })
  expect_silent({ log.inbag      = LoggerInbagRisk$new(FALSE, loss.quadratic, 0.01) })
  expect_silent({ log.oob        = LoggerOobRisk$new(FALSE, loss.quadratic, 0.01, eval.oob.test, y) })

  expect_silent({ logger.list = LoggerList$new() })
  expect_silent({ logger.list$registerLogger("iterations", log.iterations) })
  expect_silent({ logger.list$registerLogger("time.ms", log.time.ms) })
  expect_silent({ logger.list$registerLogger("time.sec", log.time.sec) })
  expect_silent({ logger.list$registerLogger("time.min", log.time.min) })
  expect_silent({ logger.list$registerLogger("inbag.risk", log.inbag) })
  expect_silent({ logger.list$registerLogger("oob.risk", log.oob) })

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
  expect_output(cboost$train(trace = 0))
  expect_output({ cboost.printer = show(cboost) })
  expect_equal(cboost.printer, "CompboostInternalPrinter")

})
