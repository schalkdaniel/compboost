
mydata = na.omit(hflights::hflights)
mydata = rbind(mydata, mydata, mydata)

time = proc.time()
size.compboost = pryr::mem_change({

  # Data for the baselearner are matrices:
  X.arrdelay = cbind(1, mydata[["ArrDelay"]])
  X.taxiin   = cbind(1, mydata[["TaxiIn"]])
  X.dist     = cbind(1, mydata[["Distance"]])
  X.aet      = cbind(1, mydata[["ActualElapsedTime"]])
  
  X.dist.sp     = as.matrix(mydata[["Distance"]])
  X.arrdelay.sp = as.matrix(mydata[["ArrDelay"]])
  X.aet.sp      = as.matrix(mydata[["ActualElapsedTime"]])
  
  # Target variable:
  y = mydata[["DepDelay"]]
  
  data.source.arrdelay = InMemoryData$new(X.arrdelay, "ArrDelay") 
  data.source.taxiin   = InMemoryData$new(X.taxiin, "TaxiIn")
  data.source.dist     = InMemoryData$new(X.dist, "Distance") 
  data.source.aet      = InMemoryData$new(X.aet, "ActualElapsedTime")
  
  data.source.dist.sp     = InMemoryData$new(X.dist.sp, "Distance") 
  data.source.arrdelay.sp = InMemoryData$new(X.arrdelay.sp, "ArrDelay")
  data.source.aet.sp      = InMemoryData$new(X.aet.sp, "ActualElapsedTime")
  
  data.target.arrdelay = InMemoryData$new()
  data.target.taxiin   = InMemoryData$new()
  data.target.dist     = InMemoryData$new()
  data.target.aet      = InMemoryData$new()
  
  data.target.dist.sp     = InMemoryData$new()
  data.target.arrdelay.sp = InMemoryData$new()
  data.target.aet.sp      = InMemoryData$new()
  
  
  # Prepare compboost:
  # ------------------
  
  ## Baselearner
  
  # Just linear:
  
  linear.factory.arrdelay = PolynomialBlearnerFactory$new(data.source.arrdelay, data.target.arrdelay, 1)
  linear.factory.taxiin   = PolynomialBlearnerFactory$new(data.source.taxiin, data.target.taxiin, 1)
  linear.factory.dist     = PolynomialBlearnerFactory$new(data.source.dist, data.target.dist, 1)
  linear.factory.aet      = PolynomialBlearnerFactory$new(data.source.aet, data.target.aet, 1)
  
  # Splines:
  
  spline.factory.dist     = PSplineBlearnerFactory$new(data.source.dist.sp, data.target.dist.sp, 3, 10, 2, 2)
  spline.factory.arrdelay = PSplineBlearnerFactory$new(data.source.arrdelay.sp, data.target.arrdelay.sp, 3, 10, 2, 2)
  spline.factory.aet      = PSplineBlearnerFactory$new(data.source.aet.sp, data.target.aet.sp, 3, 10, 2, 2)
  
  # Create new factory list:
  factory.list = BlearnerFactoryList$new()
  
  # Register factorys:
  factory.list$registerFactory(linear.factory.arrdelay)
  factory.list$registerFactory(linear.factory.taxiin)
  factory.list$registerFactory(linear.factory.dist)
  factory.list$registerFactory(linear.factory.aet)
  factory.list$registerFactory(spline.factory.dist)
  factory.list$registerFactory(spline.factory.arrdelay)
  factory.list$registerFactory(spline.factory.aet)
  
  # Print the registered factorys:
  factory.list$printRegisteredFactories()
  
  
  ## Loss
  
  # Use quadratic loss:
  loss.quadratic = QuadraticLoss$new()
  
  
  ## Optimizer
  
  # Use the greedy optimizer:
  optimizer = CoordinateDescent$new()
  
  ## Logger
  
  # Define logger. We want just the iterations as stopper but also track the
  # time, inbag risk and oob risk:
  log.iterations = IterationLogger$new(TRUE, 500)
  log.time       = TimeLogger$new(FALSE, 500, "seconds")
  # log.inbag      = InbagRiskLogger$new(FALSE, loss.quadratic, 0.05)
  # log.oob        = OobRiskLogger$new(FALSE, loss.quadratic, 0.05, oob.data, y)
  
  # Define new logger list:
  logger.list = LoggerList$new()
  
  logger.list
  
  # Register the logger:
  logger.list$registerLogger(log.iterations)
  # logger.list$registerLogger(log.time)
  # logger.list$registerLogger(log.inbag)
  # logger.list$registerLogger(log.oob)
  
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
  cboost$train(trace = TRUE)
})
time.compboost = proc.time() - time

# used memory while running 3.1 GB, bevore 2.8 GB, after 3.1 GB

library(mboost)

time = proc.time()
size.mboost = pryr::mem_change({
  mod = mboost(
    formula = DepDelay ~ bols(ArrDelay) + bols(TaxiIn) + bols(Distance) + 
      bols(ActualElapsedTime) + 
      bbs(Distance, knots = 10, degree = 3, differences = 2, lambda = 2) +
      bbs(ArrDelay, knots = 10, degree = 3, differences = 2, lambda = 2) +
      bbs(ActualElapsedTime, knots = 10, degree = 3, differences = 2, lambda = 2),
    data    = mydata,
    control = boost_control(mstop = 500, nu = 0.05, trace = TRUE)
  )
})
time.mboost = proc.time() - time

# used memory while running 3.5 GB, bevore 3.2 GB, after 3.4 GB


microbenchmark::microbenchmark(
  "compboost" = {
    cboost$train(trace = FALSE)
  },
  "mboost"    = {
    mboost(
      formula = DepDelay ~ bols(ArrDelay) + bols(TaxiIn) + bols(Distance) + 
        bols(ActualElapsedTime) + 
        bbs(Distance, knots = 10, degree = 3, differences = 2, lambda = 2) +
        bbs(ArrDelay, knots = 10, degree = 3, differences = 2, lambda = 2) +
        bbs(ActualElapsedTime, knots = 10, degree = 3, differences = 2, lambda = 2),
      data    = mydata,
      control = boost_control(mstop = 1000, nu = 0.05, trace = FALSE)
    )
  }, times = 3L
)

