context("Check if tensors works properly")

test_that("Tensors with S4 API exported by the modules (registration, training, predicting, logging, early stopping)", {

  src = "
  arma::mat rowWiseKronecker (const arma::mat& A, const arma::mat& B)
  {
    // Variables
    arma::mat out;
    arma::rowvec vecA = arma::rowvec(A.n_cols, arma::fill::ones);
    arma::rowvec vecB = arma::rowvec(B.n_cols, arma::fill::ones);

    // Multiply both kronecker products element-wise
    out = arma::kron(A,vecB) % arma::kron(vecA, B);

    return out;
  }"
  Rcpp::cppFunction(src, "RcppArmadillo")

  # Simulate test data:
  x1 = runif(1000L)
  x2 = runif(1000L)
  x3 = runif(1000L)
  x4 = runif(1000L)

  f1  = function(x1, x2) (1 - x1) * x2^2 + x1 * sin(pi * x2)
  f2 = function(x1, x2) x1^2 + x2^2
  y  = f1(x1, x2) + f2(x3, x4) + rnorm(1000L, 0, 0.1)

  df = data.frame(x1, x2, x3, x4, y, cat = sample(LETTERS[1:3], 1000, TRUE))

  # Define data:
  ds1 = InMemoryData$new(cbind(x1), "x1")
  ds2 = InMemoryData$new(cbind(x2), "x2")
  ds3 = InMemoryData$new(cbind(x3), "x3")
  ds4 = InMemoryData$new(cbind(x4), "x4")
  ds5 = CategoricalDataRaw$new(df$cat, "ct")

  fac1 = BaselearnerPSpline$new(ds1, "spline", list(df = 4, n_knots = 10))
  fac2 = BaselearnerPSpline$new(ds2, "spline", list(df = 4))
  fac3 = BaselearnerPSpline$new(ds3, "spline", list(df = 4))
  fac4 = BaselearnerPSpline$new(ds4, "spline", list(df = 4))
  fac5 = BaselearnerCategoricalRidge$new(ds5, "category")

  # Get spline designs from factories:
  X1test = t(fac1$getData())
  X2test = t(fac2$getData())
  X3test = t(fac3$getData())
  X4test = t(fac4$getData())

  tensors = list(
    x1_x2_tensor = rowWiseKronecker(X1test, X2test),
    x3_x4_tensor = rowWiseKronecker(X3test, X4test)
  )

  fl = BlearnerFactoryList$new()

  ## First tensor:
  expect_silent({
    tensor1 = BaselearnerTensor$new(fac1, fac2, "tensor")
    fl$registerFactory(tensor1)
  })
  expect_equal(fl$getRegisteredFactoryNames(), "x1_x2_tensor")
  expect_equal(tensor1$getData(), t(tensors$x1_x2_tensor))

  ## Second tensor:
  expect_silent({
    tensor2 = BaselearnerTensor$new(fac3, fac4, "tensor")
    fl$registerFactory(tensor2)
  })
  expect_true("x3_x4_tensor" %in% fl$getRegisteredFactoryNames())
  expect_equal(tensor2$getData(), t(tensors$x3_x4_tensor))


  loss = LossQuadratic$new()
  optimizer = OptimizerCoordinateDescent$new()

  newdata_full  = list(ds1, ds2, ds3, ds4)
  response_full = ResponseRegr$new("y", cbind(y))

  idx_reduced      = sample(seq_len(nrow(df)), 10)
  response_reduced = ResponseRegr$new("y", cbind(y[idx_reduced]))
  newdata_reduced  = list(
    InMemoryData$new(cbind(x1[idx_reduced]), "x1"),
    InMemoryData$new(cbind(x2[idx_reduced]), "x2"),
    InMemoryData$new(cbind(x3[idx_reduced]), "x3"),
    InMemoryData$new(cbind(x4[idx_reduced]), "x4")
  )


  log_iterations = LoggerIteration$new("iter", TRUE, 500)
  log_time       = LoggerTime$new("time", FALSE, 500, "microseconds")
  log_inbag      = LoggerInbagRisk$new("inbag", FALSE, loss, 0.05, 10)
  log_oob        = LoggerOobRisk$new("oob", FALSE, loss, 0.001, 10, newdata_full, response_full)
  log_oob_red    = LoggerOobRisk$new("oob_reduced", FALSE, LossQuadratic$new(mean(y)), 0.001, 10, newdata_reduced, response_reduced)

  # Define new logger list:
  logger_list = LoggerList$new()

  # Register the logger:
  logger_list$registerLogger(log_iterations)
  logger_list$registerLogger(log_time)
  logger_list$registerLogger(log_inbag)
  logger_list$registerLogger(log_oob)
  logger_list$registerLogger(log_oob_red)

  expect_silent({
    cboost = Compboost_internal$new(
      response      = ResponseRegr$new("y", cbind(y)),
      learning_rate = 0.05,
      stop_if_all_stopper_fulfilled = FALSE,
      factory_list = fl,
      loss         = loss,
      logger_list  = logger_list,
      optimizer    = optimizer
    )
  })

  # Train the model (we want to print the trace):
  expect_output(cboost$train(trace = TRUE))

  expect_silent({
    pred_oob         = as.vector(cboost$predict(newdata_full, TRUE))
    pred_oob_reduced = as.vector(cboost$predict(newdata_reduced, TRUE))
    pred             = as.vector(cboost$getPrediction(TRUE))
  })
  cf = cboost$getEstimatedParameter()
  pred_raw = as.vector(cboost$getOffset()) + rowSums(do.call(cbind, lapply(names(cf), function(pn) tensors[[pn]] %*% cf[[pn]])))

  expect_equal(pred, pred_oob)
  expect_equal(pred, pred_raw)

  expect_equal(pred_oob_reduced, pred[idx_reduced])
  expect_equal(pred_oob_reduced, pred_oob[idx_reduced])
  expect_equal(pred_oob_reduced, pred_raw[idx_reduced])

  expect_equal(pred, as.vector(response_full$getPrediction()))
  expect_equal(pred_oob_reduced, as.vector(response_reduced$getPrediction()))


  ld = cboost$getLoggerData()
  ldat = as.data.frame(ld$logger_data)
  names(ldat) = ld$logger_names

  expect_equal(ldat$inbag, ldat$oob)
  expect_equal(ldat$inbag, cboost$getRiskVector()[-1])

  # Continuing training
  expect_output(cboost$setToIteration(700, TRUE))

  expect_silent({
    pred_oob         = as.vector(cboost$predict(newdata_full, TRUE))
    pred_oob_reduced = as.vector(cboost$predict(newdata_reduced, TRUE))
    pred             = as.vector(cboost$getPrediction(TRUE))
  })
  cf = cboost$getEstimatedParameter()
  pred_raw = as.vector(cboost$getOffset()) + rowSums(do.call(cbind, lapply(names(cf), function(pn) tensors[[pn]] %*% cf[[pn]])))

  expect_equal(pred, pred_oob)
  expect_equal(pred, pred_raw)

  expect_equal(pred_oob_reduced, pred[idx_reduced])
  expect_equal(pred_oob_reduced, pred_oob[idx_reduced])
  expect_equal(pred_oob_reduced, pred_raw[idx_reduced])

  expect_equal(pred, as.vector(response_full$getPrediction()))
  expect_equal(pred_oob_reduced, as.vector(response_reduced$getPrediction()))

  ld = cboost$getLoggerData()
  ldat = as.data.frame(ld$logger_data)
  names(ldat) = ld$logger_names

  expect_equal(ldat$inbag, ldat$oob)
  expect_equal(ldat$inbag, cboost$getRiskVector()[-1])
  expect_length(ldat$inbag, 700)
  expect_length(cboost$getSelectedBaselearner(), 700)

  # Jump back:
  expect_silent(cboost$setToIteration(200, TRUE))

  expect_silent({
    pred_oob         = as.vector(cboost$predict(newdata_full, TRUE))
    pred_oob_reduced = as.vector(cboost$predict(newdata_reduced, TRUE))
    pred             = as.vector(cboost$getPrediction(TRUE))
  })
  cf = cboost$getEstimatedParameter()
  pred_raw = as.vector(cboost$getOffset()) + rowSums(do.call(cbind, lapply(names(cf), function (pn) tensors[[pn]] %*% cf[[pn]])))

  expect_equal(pred, pred_oob)
  expect_equal(pred, pred_raw)

  expect_equal(pred_oob_reduced, pred[idx_reduced])
  expect_equal(pred_oob_reduced, pred_oob[idx_reduced])
  expect_equal(pred_oob_reduced, pred_raw[idx_reduced])

  expect_length(cboost$getSelectedBaselearner(), 200)
})
