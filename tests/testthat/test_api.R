context("API works correctly")

test_that("train works", {

  mtcars$mpg_cat = ifelse(mtcars$mpg > 15, "A", "B")  
  
  expect_silent({ cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new()) })

  expect_error(cboost$train(10))
  expect_error(
  	cboost$addBaselearner(c("hp", "wt"), "spline", PSplineBlearnerFactory, degree = 3, 
      knots = 10, penalty = 2, differences = 2)
  )

  expect_silent(
    cboost$addBaselearner("mpg_cat", "linear", PolynomialBlearnerFactory, degree = 1, 
    	intercept = FALSE)
  )
  expect_silent(
  	cboost$addBaselearner("hp", "spline", PSplineBlearnerFactory, degree = 3, 
    	knots = 10, penalty = 2, differences = 2)
  )
  expect_output(cboost$train(4000))

  expect_s4_class(cboost$model, "Rcpp_Compboost_internal")
  expect_s4_class(cboost$bl.factory.list, "Rcpp_BlearnerFactoryList")
  expect_s4_class(cboost$loss, "Rcpp_QuadraticLoss")
  expect_s4_class(cboost$optimizer, "Rcpp_GreedyOptimizer")

  expect_equal(cboost$target, "mpg")
  expect_equal(cboost$response, mtcars[["mpg"]])
  expect_equal(cboost$data, mtcars[, -which(names(mtcars) == "mpg")])
  expect_equal(cboost$bl.factory.list$getNumberOfRegisteredFactories(), 3L)
  expect_equal(sort(cboost$getFactoryNames()), sort(c("mpg_cat_A_linear", "mpg_cat_B_linear", "hp_spline")))
  expect_equal(cboost$bl.factory.list$getRegisteredFactoryNames(), sort(c("mpg_cat_A_linear", "mpg_cat_B_linear", "hp_spline")))

  expect_equal(cboost$getCurrentIteration(), 4000)
  expect_length(cboost$risk(), 4001)
  expect_length(cboost$selected(), 4000)

  expect_output(cboost$train(6000))
  expect_equal(cboost$getCurrentIteration(), 6000)
  expect_length(cboost$risk(), 6001)
	expect_length(cboost$selected(), 6000)

  expect_equal(cboost$train(100), NULL)
  expect_equal(cboost$getCurrentIteration(), 100)
  expect_length(cboost$risk(), 101)
  expect_length(cboost$selected(), 100)

  expect_true(all(unique(cboost$selected()) %in% c("hp_spline", "mpg_cat_A_linear", "mpg_cat_B_linear")))

})

test_that("predict works", {
	mtcars$mpg_cat = ifelse(mtcars$mpg > 15, "A", "B") 

  expect_silent({ 
  	cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost$addBaselearner("mpg_cat", "linear", PolynomialBlearnerFactory, degree = 1, 
    	intercept = FALSE)
    cboost$addBaselearner("hp", "spline", PSplineBlearnerFactory, degree = 3, 
    	knots = 10, penalty = 2, differences = 2)
  })

  expect_silent(cboost$train(200, trace = FALSE))

  expect_equal(cboost$predict(), cboost$predict(mtcars))
  expect_equal(as.matrix(cboost$predict()[1]), cboost$predict(mtcars[1,]))

})

test_that("plot works", {

	mtcars$mpg_cat = ifelse(mtcars$mpg > 15, "A", "B") 

  expect_silent({ 
  	cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost$addBaselearner("mpg_cat", "linear", PolynomialBlearnerFactory, degree = 1, 
    	intercept = TRUE)
    cboost$addBaselearner("hp", "spline", PSplineBlearnerFactory, degree = 3, 
    	knots = 10, penalty = 2, differences = 2)
    cboost$addBaselearner(c("hp", "wt"), "quadratic", PolynomialBlearnerFactory, degree = 2,
    	intercept = TRUE)
  })

	expect_error(cboost$plot("hp_spline"))

	expect_silent(cboost$train(2000, trace = FALSE))

	expect_error(cboost$plot("mpg_cat_A_linear"))
	expect_error(cboost$plot("i_an_no_baselearner"))
	expect_error(cboost$plot("hp_wt_quadratic"))
	expect_error(cboost$plot("hp_spline", iters = c(NA, 10)))
	expect_error({ gg = cboost$plot("hp_spline", to = 10) })
	expect_error({ gg = cboost$plot("hp_spline", from = -100) })

	expect_warning(cboost$plot("hp_spline", from = 200, to = 100))

	expect_s3_class(cboost$plot("hp_spline"), "ggplot")
	expect_s3_class(cboost$plot("hp_spline", iters = c(10, 200, 500)), "ggplot")
  expect_s3_class(cboost$plot("hp_spline", from = 150, to = 250), "ggplot")

})

test_that("multiple logger works", {

  expect_silent({ 
    cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost$addBaselearner("hp", "spline", PSplineBlearnerFactory, degree = 3, 
      knots = 10, penalty = 2, differences = 2)
    cboost$addBaselearner(c("hp", "wt"), "quadratic", PolynomialBlearnerFactory, degree = 2,
      intercept = TRUE)
  })

  expect_silent(
    cboost$addLogger(logger = TimeLogger, use.as.stopper = FALSE, logger.id = "time", 
      max.time = 0, time.unit = "microseconds"
    )
  )
  expect_silent(
    cboost$addLogger(logger = OobRiskLogger, use.as.stopper = FALSE, logger.id = "oob",
      QuadraticLoss$new(), 0.01, cboost$prepareData(mtcars), mtcars[["mpg"]])
  )
  expect_silent(
    cboost$addLogger(logger = InbagRiskLogger, use.as.stopper = FALSE, logger.id = "inbag",
      QuadraticLoss$new(), 0.01)
  )

  expect_output(cboost$train(100))

  expect_equal(cboost$risk()[-1], cboost$model$getLoggerData()$logger.data[, 3])
  expect_equal(cboost$model$getLoggerData()$logger.data[, 2], cboost$model$getLoggerData()$logger.data[, 3])
  expect_length(cboost$model$getLoggerData()$logger.names, 4)

})