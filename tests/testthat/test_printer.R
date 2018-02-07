context("Printer works")

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
  
  sink() 
  close(tc) 
  
  # Test:
  # --------
  
  expect_equal(test.quadratic.printer, "QuadraticPrinter")
  expect_equal(test.absolute.printer, "AbsolutePrinter")
  expect_equal(test.custom.printer, "CustomPrinter")
  
})

test_that("Baselearner printer works", {
  
  x = 1:10
  X = matrix(x, ncol = 1)
  
  linear = PolynomialBlearner$new(X, "myvariable", 1)
  
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  linear.printer = show(linear)
  
  sink() 
  close(tc) 
  
  expect_equal(linear.printer, "PolynomialBlearnerPrinter")
  
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
  
  custom.blearner = CustomBlearner$new(X, "myvariable", instantiateData, 
    trainFun, predictFun, extractParameter)
  
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  custom.printer = show(custom.blearner)
  
  sink() 
  close(tc) 
  
  expect_equal(custom.printer, "CustomBlearnerPrinter")
  
  
  Rcpp::sourceCpp(code = '
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>
    
    typedef arma::mat (*instantiateDataFunPtr) (arma::mat& X);
    typedef arma::mat (*trainFunPtr) (arma::vec& y, arma::mat& X);
    typedef arma::mat (*predictFunPtr) (arma::mat& newdata, arma::mat& parameter);
    
    
    // instantiateDataFun:
    // -------------------
    
    arma::mat instantiateDataFun (arma::mat& X)
    {
    return X;
    }
    
    // trainFun:
    // -------------------
    
    arma::mat trainFun (arma::vec& y, arma::mat& X)
    {
    return arma::solve(X, y);
    }
    
    // predictFun:
    // -------------------
    
    arma::mat predictFun (arma::mat& newdata, arma::mat& parameter)
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
  
  custom.cpp.blearner = CustomCppBlearner$new(X, "my_variable_name", dataFunSetter(),
    trainFunSetter(), predictFunSetter())
  
  
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  custom.cpp.printer = show(custom.cpp.blearner)
  
  sink() 
  close(tc) 
  
  expect_equal(custom.cpp.printer, "CustomCppBlearnerPrinter")
  
})

test_that("Baselearner factory printer works", {
  
  df = mtcars
  
  # # Create new variable to check the polynomial baselearner with degree 2:
  # df$hp2 = df[["hp"]]^2
  
  # Data for the baselearner are matrices:
  X.hp = cbind(1, df[["hp"]])
  X.wt = cbind(1, df[["wt"]])
  
  # Create new linear baselearner of hp and wt:
  linear.factory.hp = PolynomialBlearnerFactory$new(X.hp, "hp", 1)
  linear.factory.wt = PolynomialBlearnerFactory$new(X.wt, "wt", 1)
  
  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialBlearnerFactory$new(X.hp, "hp", 2)
  
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  linear.factory.hp.printer = show(linear.factory.hp)
  linear.factory.wt.printer = show(linear.factory.wt)
  quadratic.factory.hp.printer = show(quadratic.factory.hp)
  
  sink() 
  close(tc) 
  
  expect_equal(linear.factory.hp.printer, "PolynomialBlearnerFactoryPrinter")
  expect_equal(linear.factory.wt.printer, "PolynomialBlearnerFactoryPrinter")
  expect_equal(quadratic.factory.hp.printer, "PolynomialBlearnerFactoryPrinter")
  
  
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
  custom.factory = CustomBlearnerFactory$new(X.hp, "hp", instantiateData, trainFun,
    predictFun, extractParameter)
  
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  custom.factory.printer = show(custom.factory)
  
  sink() 
  close(tc) 
  
  expect_equal(custom.factory.printer, "CustomBlearnerFactoryPrinter")
  
  
  
  Rcpp::sourceCpp(code = '
    // [[Rcpp::depends(RcppArmadillo)]]
    #include <RcppArmadillo.h>
    
    typedef arma::mat (*instantiateDataFunPtr) (arma::mat& X);
    typedef arma::mat (*trainFunPtr) (arma::vec& y, arma::mat& X);
    typedef arma::mat (*predictFunPtr) (arma::mat& newdata, arma::mat& parameter);
    
    
    // instantiateDataFun:
    // -------------------
    
    arma::mat instantiateDataFun (arma::mat& X)
    {
    return X;
    }
    
    // trainFun:
    // -------------------
    
    arma::mat trainFun (arma::vec& y, arma::mat& X)
    {
    return arma::solve(X, y);
    }
    
    // predictFun:
    // -------------------
    
    arma::mat predictFun (arma::mat& newdata, arma::mat& parameter)
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
  
  custom.cpp.factory = CustomCppBlearnerFactory$new(X.hp, "hp", dataFunSetter(),
    trainFunSetter(), predictFunSetter())
  
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
    "hp" = as.matrix(NA_real_),
    "wt" = as.matrix(NA_real_)
  )
  
  y = NA_real_
  
  log.iterations = LogIterations$new(TRUE, 500)
  log.time       = LogTime$new(FALSE, 500, "microseconds")
  log.inbag      = LogInbagRisk$new(FALSE, loss.quadratic, 0.05)
  log.oob        = LogOobRisk$new(FALSE, loss.quadratic, 0.05, eval.oob.test, y)
  
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
  logger.list$registerLogger(log.iterations)
  logger.list$registerLogger(log.time)
  logger.list$registerLogger(log.inbag)
  logger.list$registerLogger(log.oob)
  
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  log.iterations.printer = show(log.iterations)
  log.time.printer       = show(log.time)
  log.inbag              = show(log.inbag)
  log.oob                = show(log.oob)
  
  logger.list.printer    = show(logger.list)
  
  sink() 
  close(tc) 
  
  expect_equal(log.iterations.printer, "LogIterationsPrinter")
  expect_equal(log.time.printer, "LogTimePrinter")
  expect_equal(log.inbag, "LogInbagRiskPrinter")
  expect_equal(log.oob, "LogOobRiskPrinter")
  
  expect_equal(logger.list.printer, "LoggerListPrinter")
})

test_that("Compboost printer works", {
  
  df = mtcars
  
  # # Create new variable to check the polynomial baselearner with degree 2:
  # df$hp2 = df[["hp"]]^2
  
  # Data for the baselearner are matrices:
  X.hp = cbind(1, df[["hp"]])
  X.wt = cbind(1, df[["wt"]])
  
  # Target variable:
  y = df[["mpg"]]
  
  # Next lists are the same as the used data. Then we can have a look if the oob
  # and inbag logger and the train prediction and prediction on newdata are doing
  # the same.
  
  # List for oob logging:
  eval.oob.test = list(
    "hp" = X.hp,
    "wt" = X.wt
  )
  
  # List to test prediction on newdata:
  eval.data = eval.oob.test
  
  
  # Prepare compboost:
  # ------------------
  
  ## Baselearner
  
  # Create new linear baselearner of hp and wt:
  linear.factory.hp = PolynomialBlearnerFactory$new(X.hp, "hp", 1)
  linear.factory.wt = PolynomialBlearnerFactory$new(X.wt, "wt", 1)
  
  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialBlearnerFactory$new(X.hp, "hp", 2)
  
  # Create new factory list:
  factory.list = BlearnerFactoryList$new()
  
  # Register factorys:
  factory.list$registerFactory(linear.factory.hp)
  factory.list$registerFactory(linear.factory.wt)
  factory.list$registerFactory(quadratic.factory.hp)
  
  ## Loss
  
  # Use quadratic loss:
  loss.quadratic = QuadraticLoss$new()
  
  
  ## Optimizer
  
  # Use the greedy optimizer:
  optimizer = GreedyOptimizer$new()
  
  ## Logger
  
  # Define logger. We want just the iterations as stopper but also track the
  # time, inbag risk and oob risk:
  log.iterations = LogIterations$new(TRUE, 500)
  log.time       = LogTime$new(FALSE, 500, "microseconds")
  log.inbag      = LogInbagRisk$new(FALSE, loss.quadratic, 0.05)
  log.oob        = LogOobRisk$new(FALSE, loss.quadratic, 0.05, eval.oob.test, y)
  
  # Define new logger list:
  logger.list = LoggerList$new()
  
  # Register the logger:
  logger.list$registerLogger(log.iterations)
  logger.list$registerLogger(log.time)
  logger.list$registerLogger(log.inbag)
  logger.list$registerLogger(log.oob)
  
  logger.list$printRegisteredLogger()
  
  # Run compboost:
  # --------------
  
  # Initialize object:
  cboost = Compboost$new(
    response      = y,
    learning_rate = 0.05,
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
  
  expect_equal(cboost.printer, "CompboostPrinter")
  
})