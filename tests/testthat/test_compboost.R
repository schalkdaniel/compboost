context("Compboost works")

test_that("compboost does the same as mboost", {
  
  # Prepare Data:
  # -------------
  
  df = mtcars
  
  # Create new variable to check the polynomial baselearner with degree 2:
  df$hp2 = df[["hp"]]^2
  
  # Data for compboost:
  X.hp = as.matrix(df[["hp"]], ncol = 1)
  X.wt = as.matrix(df[["wt"]], ncol = 1)
  
  y = df[["mpg"]]
  
  # Hyperparameter for the algorithm:
  learning.rate = 0.05
  iter.max = 500
  
  # Prepare compboost:
  # ------------------
  
  # Create new linear baselearner of hp and wt:
  linear.factory.hp = PolynomialFactory$new(X.hp, "hp", 1)
  linear.factory.wt = PolynomialFactory$new(X.wt, "wt", 1)
  
  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialFactory$new(X.hp, "hp", 2)
  
  # Create new factory list:
  factory.list = FactoryList$new()
  
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
  log.iterations = LogIterations$new(TRUE, iter.max)
  log.time       = LogTime$new(FALSE, 500, "microseconds")
  
  logger.list = LoggerList$new()
  logger.list$registerLogger(log.iterations)
  logger.list$registerLogger(log.time)
  
  # Run compboost:
  # --------------
  
  # Initialize object (Response, learning rate, stop if all stopper are fulfilled?,
  # factory list, used loss, logger list):
  cboost = Compboost$new(
    response      = y, 
    learning_rate = learning.rate, 
    stop_if_all_stopper_fulfilled = TRUE, 
    factory_list = factory.list, 
    loss         = loss.quadratic, 
    logger_list  = logger.list,
    optimizer    = optimizer
  )
  
  # Train the model (we want to print the trace):
  tc = textConnection(NULL, "w") 
  sink(tc) 
  
  cboost$train(trace = TRUE)
  
  sink() 
  close(tc) 
  
  
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
      "hp: polynomial with degree 1", 
      "wt: polynomial with degree 1", 
      "hp: polynomial with degree 2"
    )
  )
  
  # Tests:
  # ------
  expect_equal(predict(mod), cboost$getPrediction())
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
    unname(unlist(cboost$getEstimatedParameterOfIteration(200)))
  )
  
  idx = 2:4 * 120
  matrix.compare = matrix(NA_real_, nrow = 3, ncol = 3)
  
  for (i in seq_along(idx)) {
    matrix.compare[i, ] = unname(unlist(cboost$getEstimatedParameterOfIteration(idx[i])))
  }

  expect_equal(cboost$getParameterMatrix()$parameter.matrix[idx, ], matrix.compare)
})

