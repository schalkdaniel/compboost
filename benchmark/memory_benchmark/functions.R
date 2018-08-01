# ============================================================================ #
#                                                                              #
#                     Functions to run the Memory Benchmark                    #
#                                                                              #
# ============================================================================ #

# Simulate data:
# ----------------------------

simData = function (n, p, seed) {

  set.seed(seed)

  vars = p

  #create beta distributed correlations
  corrs = rbeta(n = (vars * (vars - 1))/2, shape1 = 1, shape2 = 8)
  corrs = sample(c(-1, 1), size = length(corrs), replace = TRUE) * corrs

  sigma = matrix(1, nrow = vars, ncol = vars)
  sigma[upper.tri(sigma)] = corrs
  sigma[lower.tri(sigma)] = t(sigma)[lower.tri(sigma)]

  data = as.data.frame(mvtnorm::rmvnorm(n = n, sigma = sigma, method = "svd"))

  betas = runif(p + 1, min = -2, max = 2)
  data$y = rnorm(n = n, mean = as.matrix(cbind(1, data[,1:p])) %*% betas)

  return (data)
}

# Compboost:
# ----------------------------

memBenchmarkCompboost = function (mydata, iters, learner) {

  if (! learner %in% c("spline", "linear")) {
    stop("No valid learner!")
  }

  y.idx = which(names(mydata) == "y")
  data.names = names(mydata[, -y.idx])

  cboost.objects = list(source = list(), target = list(), factory = list())
  factory.list   = BlearnerFactoryList$new()
  for (i in data.names) {
    cboost.objects[["source"]][[i]] = list()
    if (learner == "spline") {
      cboost.objects[["source"]][[i]] = InMemoryData$new(as.matrix(mydata[, i]), i)
    }
    if (learner == "linear") {
      cboost.objects[["source"]][[i]] = InMemoryData$new(cbind(1, mydata[, i]), i)
    }

    cboost.objects[["target"]][[i]] = list()
    cboost.objects[["target"]][[i]] = InMemoryData$new()

    cboost.objects[["factory"]][[i]] = list()
    if (learner == "spline") {
      cboost.objects[["factory"]][[i]] = PSplineBlearnerFactory$new(
        cboost.objects[["source"]][[i]], cboost.objects[["target"]][[i]], 3, 20, 2, 2
      )
    }
    if (learner == "linear") {
      cboost.objects[["factory"]][[i]] = PolynomialBlearnerFactory$new(
        cboost.objects[["source"]][[i]], cboost.objects[["target"]][[i]], 1
      )
    }

    factory.list$registerFactory(cboost.objects[["factory"]][[i]])
    cboost.objects[["source"]][[i]] = NULL
  }

  iteration.logger = IterationLogger$new(TRUE, iters)
  logger.list = LoggerList$new()
  logger.list$registerLogger("iteration", iteration.logger)

  myloss      = QuadraticLoss$new()
  myoptimizer = GreedyOptimizer$new()

  # Initialize object:
  cboost = Compboost$new(
    response      = mydata[, y.idx],
    learning_rate = 0.05,
    stop_if_all_stopper_fulfilled = FALSE,
    factory_list = factory.list,
    loss         = myloss,
    logger_list  = logger.list,
    optimizer    = myoptimizer
  )
  cboost$train(trace = TRUE)
}

# Mboost:
# ----------------------------

memBenchmarkMboost = function (mydata, iters, learner) {

  y.idx = which(names(mydata) == "y")
  data.names = names(mydata[, -y.idx])
  if (learner == "spline") {
    myformula = paste0(
      "y ~ ",
      paste(
        paste0("bbs(", data.names, ", knots = 20, degree = 3, differences = 2, lambda = 2)"),
        collapse = " + "
      )
    )
  }
  if (learner == "linear") {
    myformula = paste0(
      "y ~ ",
      paste(
        paste0("bols(", data.names, ")"),
        collapse = " + "
      )
    )
  }
  if (! learner %in% c("spline", "linear")) {
    stop("No valid learner!")
  }

  mod = mboost(formula = as.formula(myformula), data = mydata,
    control = boost_control(mstop = iters, nu = 0.05, trace = TRUE))
}
