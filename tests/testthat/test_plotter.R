context("Plotter works")

test_that("Parameter plotter works", {

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
  linear.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target.hp1, 1, FALSE)
  linear.factory.wt = PolynomialBlearnerFactory$new(data.source.wt, data.target.wt, 1, FALSE)

  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target.hp2, 2, FALSE)

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
