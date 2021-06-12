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

  knots1 = compboostSplines::createKnots(x1, 20, 3)
  X1test = compboostSplines::createSplineBasis(x1, 3, knots1)
  knots2 = compboostSplines::createKnots(x2, 20, 3)
  X2test = compboostSplines::createSplineBasis(x2, 3, knots2)
  knots3 = compboostSplines::createKnots(x3, 20, 3)
  X3test = compboostSplines::createSplineBasis(x3, 3, knots3)
  knots4 = compboostSplines::createKnots(x4, 20, 3)
  X4test = compboostSplines::createSplineBasis(x4, 3, knots4)

  tensors = list(
    x1_x2_tensor = rowWiseKronecker(X1test, X2test),
    x3_x4_tensor = rowWiseKronecker(X3test, X4test)
  )

  f1  = function(x1, x2) (1 - x1) * x2^2 + x1 * sin(pi * x2)
  f2 = function(x1, x2) x1^2 + x2^2
  y  = f1(x1, x2) + f2(x3, x4) + rnorm(1000L, 0, 0.1)

  df = data.frame(x1, x2, x3, x4, y, cat = sample(LETTERS[1:3], 1000, TRUE))

  # Define data:
  ds1 = InMemoryData$new(cbind(x1), "x1")
  ds2 = InMemoryData$new(cbind(x2), "x2")
  ds3 = InMemoryData$new(cbind(x1), "x3")
  ds4 = InMemoryData$new(cbind(x2), "x4")
  ds5 = CategoricalDataRaw$new(df$cat, "ct")

  fac1 = BaselearnerPSpline$new(ds1, "spline", list(df = 4, n_knots = 10))
  fac2 = BaselearnerPSpline$new(ds2, "spline", list(df = 4))
  fac3 = BaselearnerPSpline$new(ds3, "spline", list(df = 4))
  fac4 = BaselearnerPSpline$new(ds4, "spline", list(df = 4))
  fac5 = BaselearnerCategoricalRidge$new(ds5, "category")

  expect_silent({
    tensor1 = BaselearnerTensor$new(fac1, fac2, "tensor")
    fl = BlearnerFactoryList$new()
    fl$registerFactory(tensor1)
  })
  expect_equal(fl$getRegisteredFactoryNames(), "x1_x2_tensor")
  expect_silent({
    tensor2 = BaselearnerTensor$new(fac3, fac4, "tensor")
    fl$registerFactory(tensor2)
  })
  expect_true("x3_x4_tensor" %in% fl$getRegisteredFactoryNames())

  loss = LossQuadratic$new()
  optimizer = OptimizerCoordinateDescent$new()

  newdata = list(
    x1 = InMemoryData$new(cbind(x1), "x1"),
    x2 = InMemoryData$new(cbind(x2), "x2"),
    x3 = InMemoryData$new(cbind(x3), "x3"),
    x4 = InMemoryData$new(cbind(x4), "x4")
  )
  idx_reduced = sample(seq_len(nrow(df)), 269)
  newdata_reduced = list(
    x1 = InMemoryData$new(cbind(x1[idx_reduced]), "x1"),
    x2 = InMemoryData$new(cbind(x2[idx_reduced]), "x2"),
    x3 = InMemoryData$new(cbind(x3[idx_reduced]), "x3"),
    x4 = InMemoryData$new(cbind(x4[idx_reduced]), "x4")
  )

  log_iterations = LoggerIteration$new("iter", TRUE, 500)
  log_time       = LoggerTime$new("time", FALSE, 500, "microseconds")
  log_inbag      = LoggerInbagRisk$new("inbag", FALSE, loss, 0.05, 10)
  log_oob        = LoggerOobRisk$new("oob", TRUE, loss, 0.001, 10, newdata, ResponseRegr$new("y", cbind(y)))
  log_oob_red    = LoggerOobRisk$new("oob_reduced", TRUE, loss, 0.001, 10, newdata_reduced, ResponseRegr$new("y", cbind(y[idx_reduced])))

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
    pred_oob         = as.vector(cboost$predict(newdata, TRUE))
    pred_oob_reduced = as.vector(cboost$predict(newdata_reduced, TRUE))
    pred             = as.vector(cboost$getPrediction(TRUE))
  })
  cf = cboost$getEstimatedParameter()
  pred_raw = as.vector(cboost$getOffset()) + rowSums(do.call(cbind, lapply(names(cf), function (pn) tensors[[pn]] %*% cf[[pn]])))

  expect_equal(pred_oob, pred)
  expect_equal(pred_oob_reduced, pred[idx_reduced])
  expect_equal(pred, pred_raw)

  ld = cboost$getLoggerData()
  ldat = as.data.frame(ld$logger_data)
  names(ldat) = ld$logger_names

  expect_equal(ldat$inbag, ldat$oob)
  expect_equal(ldat$inbag, cboost$getRiskVector()[-1])

  # Continuing training
  expect_output(cboost$setToIteration(700, TRUE))

  expect_silent({
    pred_oob         = as.vector(cboost$predict(newdata, TRUE))
    pred_oob_reduced = as.vector(cboost$predict(newdata_reduced, TRUE))
    pred             = as.vector(cboost$getPrediction(TRUE))
  })
  cf = cboost$getEstimatedParameter()
  pred_raw = as.vector(cboost$getOffset()) + rowSums(do.call(cbind, lapply(names(cf), function (pn) tensors[[pn]] %*% cf[[pn]])))

  expect_equal(pred_oob, pred)
  expect_equal(pred_oob_reduced, pred[idx_reduced])
  expect_equal(pred, pred_raw)

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
    pred_oob         = as.vector(cboost$predict(newdata, TRUE))
    pred_oob_reduced = as.vector(cboost$predict(newdata_reduced, TRUE))
    pred             = as.vector(cboost$getPrediction(TRUE))
  })
  cf = cboost$getEstimatedParameter()
  pred_raw = as.vector(cboost$getOffset()) + rowSums(do.call(cbind, lapply(names(cf), function (pn) tensors[[pn]] %*% cf[[pn]])))

  expect_equal(pred_oob, pred)
  expect_equal(pred_oob_reduced, pred[idx_reduced])
  expect_equal(pred, pred_raw)

  expect_length(cboost$getSelectedBaselearner(), 200)

})

test_that("Tensors with Compboost R6 wrapper", {

  q()
  R
  devtools::load_all()

  # Simulate test data:
  x1 = runif(1000L)
  x2 = runif(1000L)
  x3 = runif(1000L)
  x4 = runif(1000L)

  f1 = function(x1, x2) (1 - x1) * x2^2 + x1 * sin(pi * x2)
  f2 = function(x1, x2) x1^2 + x2^2
  y  = f1(x1, x2) + f2(x3, x4) + rnorm(1000L, 0, 0.1)

  df = data.frame(x1, x2, x3, x4, y, cat = sample(LETTERS[1:3], 1000, TRUE))

  cboost = Compboost$new(data = df, target = "y", loss = LossQuadratic$new(), learning_rate = 0.1, oob_fraction = 0.33)
  cboost$addBaselearner("x1", "spline", BaselearnerPSpline)
  cboost$addBaselearner("x2", "spline", BaselearnerPSpline)
  cboost$addBaselearner("x3", "spline", BaselearnerPSpline)
  cboost$addBaselearner("cat", "category", BaselearnerCategoricalRidge)
  cboost$train(500)


  oob_fraction = 0.3
  test_idx = seq_along(x1)[seq_len(nrow(df)) %in% sample(seq_len(nrow(df)), trunc(nrow(df) * oob_fraction))]

  cboost = Compboost$new(data = df, target = "y", loss = LossQuadratic$new(), learning_rate = 0.1, oob_fraction = 0.33,
    use_early_stopping = TRUE, stop_args = list(eps_for_break = 0.001, patience = 10L), test_idx = test_idx)
  cboost$addTensor("x1", "x2", n_knots = 10, df = 10)
  cboost$addTensor("x3", "x4", n_knots = 10, df = 10)
  cboost$train(200L)

  cboost = Compboost$new(data = df, target = "y", loss = LossQuadratic$new(), learning_rate = 0.1, oob_fraction = 0.33)
  cboost$addComponents("x1")
  cboost$addComponents("x2")
  cboost$addComponents("x3")
  cboost$train(200)

  #cboost$addLogger(LoggerOobRisk, use_as_stopper = TRUE, logger_id = "oob_risk",
    #used_loss = LossQuadratic$new(), eps_for_break = 0, patience = 5L, oob_data = cboost$prepareData(df[test_idx, ]),
    #oob_response = cboost$prepareResponse(df$y[test_idx]))


  cboost$predict()
  cboost$predict(df)

  idx_shuffle = sample(seq_len(nrow(df)), 333)
  expect_equal(cboost$predict(), cboost$predict(df))
  expect_equal(cboost$predict()[idx_shuffle], as.vector(cboost$predict(df[idx_shuffle, ])))

  cboost$train(700)
  idx_shuffle = sample(seq_len(nrow(df)), 333)
  expect_equal(cboost$predict(), cboost$predict(df))
  expect_equal(cboost$predict()[idx_shuffle], as.vector(cboost$predict(df[idx_shuffle, ])))
})
