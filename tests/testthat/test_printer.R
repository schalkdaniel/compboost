context("Printer works")

test_that("data printer works", {
  X = as.matrix(1:10)

  data.source = InMemoryData$new(X, "x")
  data.target = InMemoryData$new()

  tc = textConnection(NULL, "w")
  sink(tc)

  test.source = show(data.source)
  test.target = show(data.target)

  sink()
  close(tc)

  expect_equal(test.source, "InMemoryDataPrinter")
  expect_equal(test.target, "InMemoryDataPrinter")
})

test_that("factory list printer works", {

  factory.list = BlearnerFactoryList$new()

  # A hack to suppress console output:
  tc = textConnection(NULL, "w")
  sink(tc)

  test.factory.list.printer = show(factory.list)

  sink()
  close(tc)

  # Test:
  # ---------

  expect_equal(test.factory.list.printer, "BlearnerFactoryListPrinter")

})

test_that("Loss printer works", {

  quadratic.loss = QuadraticLoss$new()
  absolute.loss  = AbsoluteLoss$new()
  binomial.loss = BinomialLoss$new()

  # Function for Custom Loss:
  myLossFun = function (true.value, prediction) NULL
  myGradientFun = function (true.value, prediction) NULL
  myConstantInitializerFun = function (true.value) NULL

  custom.loss = CustomLoss$new(myLossFun, myGradientFun, myConstantInitializerFun)

  # A hack to suppress console output:
  tc = textConnection(NULL, "w")
  sink(tc)

  test.quadratic.printer = show(quadratic.loss)
  test.absolute.printer  = show(absolute.loss)
  test.custom.printer    = show(custom.loss)
  test.binomialprinter  = show(binomial.loss)

  sink()
  close(tc)

  # Test:
  # --------

  expect_equal(test.quadratic.printer, "QuadraticLossPrinter")
  expect_equal(test.absolute.printer, "AbsoluteLossPrinter")
  expect_equal(test.binomialprinter, "BinomialLossPrinter")
  expect_equal(test.custom.printer, "CustomLossPrinter")

})

test_that("Baselearner printer works", {

  x = 1:10
  X = matrix(x, ncol = 1)

  data.source = InMemoryData$new(X, "myvariable")
  data.target = InMemoryData$new()


  # Polynomial Baselearner:
  # ---------------------------------
  linear = PolynomialBlearner$new(data.source, data.target, 1)

  tc = textConnection(NULL, "w")
  sink(tc)

  linear.printer = show(linear)

  sink()
  close(tc)

  expect_equal(linear.printer, "PolynomialBlearnerPrinter")


  # Spline Baselearner:
  # ---------------------------------

  spline.learner = PSplineBlearner$new(data.source, data.target, 3, 5, 2.5, 2)

  tc = textConnection(NULL, "w")
  sink(tc)

  spline.printer = show(spline.learner)

  sink()
  close(tc)

  expect_equal(spline.printer, "PSplineBlearnerPrinter")

  # Custom Baselearner:
  # ---------------------------------
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

  custom.blearner = CustomBlearner$new(data.source, data.target,
    instantiateData, trainFun, predictFun, extractParameter)

  tc = textConnection(NULL, "w")
  sink(tc)

  custom.printer = show(custom.blearner)

  sink()
  close(tc)

  expect_equal(custom.printer, "CustomBlearnerPrinter")

  # Custom Cpp Baselearner:
  # ---------------------------------
  Rcpp::sourceCpp(code = '
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>

    typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
    typedef arma::mat (*trainFunPtr) (const arma::vec& y, const arma::mat& X);
    typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);


    // instantiateDataFun:
    // -------------------

    arma::mat instantiateDataFun (const arma::mat& X)
    {
    return X;
    }

    // trainFun:
    // -------------------

    arma::mat trainFun (const arma::vec& y, const arma::mat& X)
    {
    return arma::solve(X, y);
    }

    // predictFun:
    // -------------------

    arma::mat predictFun (const arma::mat& newdata, const arma::mat& parameter)
    {
    return newdata * parameter;
    }


    // Setter function:
    // ------------------

    // [[Rcpp::export]]
    Rcpp::XPtr<instantiateDataFunPtr> dataFunSetter ()
    {
    return Rcpp::XPtr<instantiateDataFunPtr> (new instantiateDataFunPtr (&instantiateDataFun));
    }

    // [[Rcpp::export]]
    Rcpp::XPtr<trainFunPtr> trainFunSetter ()
    {
    return Rcpp::XPtr<trainFunPtr> (new trainFunPtr (&trainFun));
    }

    // [[Rcpp::export]]
    Rcpp::XPtr<predictFunPtr> predictFunSetter ()
    {
    return Rcpp::XPtr<predictFunPtr> (new predictFunPtr (&predictFun));
    }'
  )

  custom.cpp.blearner = CustomCppBlearner$new(data.source, data.target,
    dataFunSetter(), trainFunSetter(), predictFunSetter())


  tc = textConnection(NULL, "w")
  sink(tc)

  custom.cpp.printer = show(custom.cpp.blearner)

  sink()
  close(tc)

  expect_equal(custom.cpp.printer, "CustomCppBlearnerPrinter")

})

test_that("Baselearner factory printer works", {

  df = mtcars

  X.hp = cbind(1, df[["hp"]])
  X.hp.sp = as.matrix(df[["hp"]])

  data.source    = InMemoryData$new(X.hp, "hp")
  data.source.sp = InMemoryData$new(X.hp.sp, "hp")
  data.target    = InMemoryData$new()

  # Polynomial Baselearner Factory:
  # ------------------------------------
  linear.factory.hp = PolynomialBlearnerFactory$new(data.source, data.target, 1)

  tc = textConnection(NULL, "w")
  sink(tc)

  linear.factory.hp.printer = show(linear.factory.hp)

  sink()
  close(tc)

  expect_equal(linear.factory.hp.printer, "PolynomialBlearnerFactoryPrinter")

  # Spline Baselearner Factory:
  # ------------------------------------
  spline.factory = PSplineBlearnerFactory$new(data.source.sp, data.target, 3, 5, 2.5, 2)

  tc = textConnection(NULL, "w")
  sink(tc)

  spline.printer = show(spline.factory)

  sink()
  close(tc)

  expect_equal(spline.printer, "PSplineBlearnerFactoryPrinter")

  # Custom Baselearner Factory:
  # ------------------------------------
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

  # Create fatorys is very similar:
  custom.factory = CustomBlearnerFactory$new(data.source, data.target,
    instantiateData, trainFun, predictFun, extractParameter)

  tc = textConnection(NULL, "w")
  sink(tc)

  custom.factory.printer = show(custom.factory)

  sink()
  close(tc)

  expect_equal(custom.factory.printer, "CustomBlearnerFactoryPrinter")


  # Custom Cpp Baselearner Factory:
  # ------------------------------------
  Rcpp::sourceCpp(code = '
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>

    typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
    typedef arma::mat (*trainFunPtr) (const arma::vec& y, const arma::mat& X);
    typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);


    // instantiateDataFun:
    // -------------------

    arma::mat instantiateDataFun (const arma::mat& X)
    {
    return X;
    }

    // trainFun:
    // -------------------

    arma::mat trainFun (const arma::vec& y, const arma::mat& X)
    {
    return arma::solve(X, y);
    }

    // predictFun:
    // -------------------

    arma::mat predictFun (const arma::mat& newdata, const arma::mat& parameter)
    {
    return newdata * parameter;
    }


    // Setter function:
    // ------------------

    // [[Rcpp::export]]
    Rcpp::XPtr<instantiateDataFunPtr> dataFunSetter ()
    {
    return Rcpp::XPtr<instantiateDataFunPtr> (new instantiateDataFunPtr (&instantiateDataFun));
    }

    // [[Rcpp::export]]
    Rcpp::XPtr<trainFunPtr> trainFunSetter ()
    {
    return Rcpp::XPtr<trainFunPtr> (new trainFunPtr (&trainFun));
    }

    // [[Rcpp::export]]
    Rcpp::XPtr<predictFunPtr> predictFunSetter ()
    {
    return Rcpp::XPtr<predictFunPtr> (new predictFunPtr (&predictFun));
    }'
  )

  custom.cpp.factory = CustomCppBlearnerFactory$new(data.source, data.target,
    dataFunSetter(), trainFunSetter(), predictFunSetter())

  tc = textConnection(NULL, "w")
  sink(tc)

  custom.cpp.factory.printer = show(custom.cpp.factory)

  sink()
  close(tc)

  expect_equal(custom.cpp.factory.printer, "CustomCppBlearnerFactoryPrinter")
})

test_that("Optimizer printer works", {

  greedy.optimizer = GreedyOptimizer$new()

  tc = textConnection(NULL, "w")
  sink(tc)

  greedy.optimizer.printer = show(greedy.optimizer)

  sink()
  close(tc)

  expect_equal(greedy.optimizer.printer, "GreedyOptimizerPrinter")

})

test_that("Logger(List) printer works", {

  loss.quadratic = QuadraticLoss$new()

  eval.oob.test = list(
    InMemoryData$new(as.matrix(NA_real_), "hp"),
    InMemoryData$new(as.matrix(NA_real_), "wt")
  )

  y = NA_real_

  log.iterations = IterationLogger$new(TRUE, 500)
  log.time       = TimeLogger$new(FALSE, 500, "microseconds")
  log.inbag      = InbagRiskLogger$new(FALSE, loss.quadratic, 0.05)
  log.oob        = OobRiskLogger$new(FALSE, loss.quadratic, 0.05, eval.oob.test, y)

  # Define new logger list:
  logger.list = LoggerList$new()

  # Test empty printer:
  tc = textConnection(NULL, "w")
  sink(tc)

  logger.list.printer = show(logger.list)

  sink()
  close(tc)

  expect_equal(logger.list.printer, "LoggerListPrinter")

  # Register the logger:
  logger.list$registerLogger("iterations", log.iterations)
  logger.list$registerLogger("time", log.time)
  logger.list$registerLogger("inbag.risk", log.inbag)
  logger.list$registerLogger("oob.risk", log.oob)

  tc = textConnection(NULL, "w")
  sink(tc)

  log.iterations.printer = show(log.iterations)
  log.time.printer       = show(log.time)
  log.inbag              = show(log.inbag)
  log.oob                = show(log.oob)

  logger.list.printer    = show(logger.list)

  sink()
  close(tc)

  expect_equal(log.iterations.printer, "IterationLoggerPrinter")
  expect_equal(log.time.printer, "TimeLoggerPrinter")
  expect_equal(log.inbag, "InbagRiskLoggerPrinter")
  expect_equal(log.oob, "OobRiskLoggerPrinter")

  expect_equal(logger.list.printer, "LoggerListPrinter")
})

test_that("Compboost printer works", {

  df = mtcars

  # Create new variable to check the polynomial baselearner with degree 2:
  df$hp2 = df[["hp"]]^2

  # Data for compboost:
  X.hp = as.matrix(df[["hp"]], ncol = 1)
  X.wt = as.matrix(df[["wt"]], ncol = 1)

  y = df[["mpg"]]

  data.source.hp = InMemoryData$new(X.hp, "hp")
  data.source.wt = InMemoryData$new(X.wt, "wt")

  data.target.hp1 = InMemoryData$new()
  data.target.hp2 = InMemoryData$new()
  data.target.wt  = InMemoryData$new()

  eval.oob.test = list(data.source.hp, data.source.wt)

  # Hyperparameter for the algorithm:
  learning.rate = 0.05
  iter.max = 500

  # Prepare compboost:
  # ------------------

  # Create new linear baselearner of hp and wt:
  linear.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target.hp1, 1)
  linear.factory.wt = PolynomialBlearnerFactory$new(data.source.wt, data.target.wt, 1)

  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target.hp2, 2)

  # Create new factory list:
  factory.list = BlearnerFactoryList$new()

  # Register factorys:
  factory.list$registerFactory(linear.factory.hp)
  factory.list$registerFactory(linear.factory.wt)
  factory.list$registerFactory(quadratic.factory.hp)

  # Use quadratic loss:
  loss.quadratic = QuadraticLoss$new()

  # Use the greedy optimizer:
  optimizer = GreedyOptimizer$new()

  # Define logger. We want just the iterations as stopper but also track the
  # time:
  log.iterations = IterationLogger$new(TRUE, iter.max)
  log.time.ms    = TimeLogger$new(TRUE, 50000, "microseconds")
  log.time.sec   = TimeLogger$new(TRUE, 2, "seconds")
  log.time.min   = TimeLogger$new(TRUE, 1, "minutes")
  log.inbag      = InbagRiskLogger$new(FALSE, loss.quadratic, 0.01)
  log.oob        = OobRiskLogger$new(FALSE, loss.quadratic, 0.01, eval.oob.test, y)

  logger.list = LoggerList$new()
  logger.list$registerLogger("iterations", log.iterations)
  logger.list$registerLogger("time.ms", log.time.ms)
  logger.list$registerLogger("time.sec", log.time.sec)
  logger.list$registerLogger("time.min", log.time.min)
  logger.list$registerLogger("inbag.risk", log.inbag)
  logger.list$registerLogger("oob.risk", log.oob)

  # logger.list$printRegisteredLogger()

  # Run compboost:
  # --------------

  # Initialize object (Response, learning rate, stop if all stopper are fulfilled?,
  # factory list, used loss, logger list):
  cboost = Compboost_internal$new(
    response      = y,
    learning_rate = learning.rate,
    stop_if_all_stopper_fulfilled = FALSE,
    factory_list = factory.list,
    loss         = loss.quadratic,
    logger_list  = logger.list,
    optimizer    = optimizer
  )

  # Train the model (we want to print the trace):
  tc = textConnection(NULL, "w")
  sink(tc)

  cboost$train(trace = TRUE)
  cboost.printer = show(cboost)

  sink()
  close(tc)

  expect_equal(cboost.printer, "CompboostInternalPrinter")

})
