context("Plotter works")

test_that("Parameter plotter works", {
  
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
  log.iterations = IterationLogger$new(TRUE, 500)
  log.time       = TimeLogger$new(FALSE, 500, "microseconds")
  log.inbag      = InbagRiskLogger$new(FALSE, loss.quadratic, 0.05)
  log.oob        = OobRiskLogger$new(FALSE, loss.quadratic, 0.05, eval.oob.test, y)
  
  # Define new logger list:
  logger.list = LoggerList$new()
  
  # Register the logger:
  logger.list$registerLogger(log.iterations)
  logger.list$registerLogger(log.time)
  logger.list$registerLogger(log.inbag)
  logger.list$registerLogger(log.oob)
  
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
  
  suppressWarnings({
    failed.plotter = plotCompboostParameter(cboost)
  })
  
  # Train the model (we want to print the trace):
  cboost$train(trace = FALSE)
  
  plotter = plotCompboostParameter(cboost)
  
  # Test:
  # ---------
  
  expect_equal(failed.plotter, 1)
  expect_equal(plotter, 0)
  
})
