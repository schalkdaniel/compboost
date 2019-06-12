context("Compboost parallel")

test_that("If parallel speed up the algorithm", {
  if (parallel::detectCores() > 1) {

    feats = 20
    n = 10000
    mstop = 500
    mydata = as.data.frame(do.call(cbind, lapply(seq_len(feats + 1), function (x) { rnorm(n) })))
    names(mydata) = c("target", paste0("feat", seq_len(feats)))

    expect_silent({ optimizer = OptimizerCoordinateDescent$new() })

    time1 = proc.time()

    expect_silent({
      cboost1 = Compboost$new(data = mydata, target = "target", optimizer = optimizer,
        loss = LossQuadratic$new(), learning_rate = 0.01)
    })
    nuisance = lapply(names(mydata)[-1], function (feat) cboost1$addBaselearner(feat, "spline", BaselearnerPSpline))
    cboost1$addLogger(logger = LoggerTime, use_as_stopper = FALSE, logger_id = "time",
      max_time = 0, time_unit = "seconds")

    expect_output({ cboost1$train(mstop) })

    time1 = (proc.time() - time1)[3]

    expect_silent({ optimizer = OptimizerCoordinateDescent$new(2) })

    time2 = proc.time()

    expect_silent({
      cboost2 = Compboost$new(data = mydata, target = "target", optimizer = optimizer,
        loss = LossQuadratic$new(), learning_rate = 0.01)
    })
    nuisance = lapply(names(mydata)[-1], function (feat) cboost2$addBaselearner(feat, "spline", BaselearnerPSpline))
    cboost2$addLogger(logger = LoggerTime, use_as_stopper = FALSE, logger_id = "time",
      max_time = 0, time_unit = "seconds")

    expect_output({ cboost2$train(mstop) })

    cboost2$train(mstop)
    time2 = (proc.time() - time2)[3]

    expect_true(time1 > time2)
    expect_true(tail(cboost1$getLoggerData()$time, n = 1) > tail(cboost2$getLoggerData()$time, n = 1))
    expect_equal(cboost1$getSelectedBaselearner(), cboost2$getSelectedBaselearner())
    expect_equal(cboost1$predict(), cboost2$predict())
    expect_equal(cboost1$getEstimatedCoef(), cboost2$getEstimatedCoef())
  }
})