context("API works correctly")

test_that("train works", {

  mtcars$mpg_cat = ifelse(mtcars$mpg > 15, "A", "B")  
  
  expect_error({ cboost = Compboost$new(mtcars, "i_am_no_feature", loss = QuadraticLoss$new()) })
  expect_silent({ cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new()) })
  expect_output(cboost$print())

  expect_equal(cboost$getCurrentIteration(), 0)
  expect_equal(cboost$risk(), NULL)
  expect_equal(cboost$selected(), NULL)
  expect_equal(cboost$coef(), NULL)

  expect_error(cboost$train(10))
  expect_error(
  	cboost$addBaselearner(c("hp", "wt"), "spline", PSplineBlearner, degree = 3, 
      n.knots = 10, penalty = 2, differences = 2)
  )

  expect_silent(
    cboost$addBaselearner("mpg_cat", "linear", PolynomialBlearner, degree = 1, 
    	intercept = FALSE)
  )
  expect_silent(
  	cboost$addBaselearner("hp", "spline", PSplineBlearner, degree = 3, 
    	n.knots = 10, penalty = 2, differences = 2)
  )
  expect_output(cboost$train(4000))
  expect_output(cboost$print())

  expect_error(
    cboost$addBaselearner("wt", "spline", PSplineBlearner, degree = 3, 
      n.knots = 10, penalty = 2, differences = 2)
  )

  expect_s4_class(cboost$model, "Rcpp_Compboost_internal")
  expect_s4_class(cboost$bl.factory.list, "Rcpp_BlearnerFactoryList")
  expect_s4_class(cboost$loss, "Rcpp_QuadraticLoss")
  expect_s4_class(cboost$optimizer, "Rcpp_CoordinateDescent")

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
    cboost$addBaselearner("mpg_cat", "linear", PolynomialBlearner, degree = 1, 
    	intercept = FALSE)
    cboost$addBaselearner("hp", "spline", PSplineBlearner, degree = 3, 
    	n.knots = 10, penalty = 2, differences = 2)
  })

  expect_silent(cboost$train(200, trace = FALSE))

  expect_equal(cboost$predict(), cboost$predict(mtcars))
  expect_equal(as.matrix(cboost$predict()[1]), cboost$predict(mtcars[1,]))
  expect_equal(cboost$predict(), cboost$predict(response = TRUE))

})

test_that("plot works", {

	mtcars$mpg_cat = ifelse(mtcars$mpg > 15, "A", "B") 

  expect_silent({ 
  	cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost$addBaselearner("mpg_cat", "linear", PolynomialBlearner, degree = 1, 
    	intercept = TRUE)
    cboost$addBaselearner("hp", "spline", PSplineBlearner, degree = 3, 
    	n.knots = 10, penalty = 2, differences = 2)
    cboost$addBaselearner(c("hp", "wt"), "quadratic", PolynomialBlearner, degree = 2,
    	intercept = TRUE)
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = TRUE) 
  })

	expect_error(cboost$plot("hp_spline"))

	expect_silent(cboost$train(2000, trace = FALSE))

	expect_error(cboost$plot())
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

  expect_warning(cboost$plot("wt_linear", iters = c(1, 10)))
  
  expect_silent(cboost$train(200, trace = FALSE))
  expect_error(cboost$plot("mpg_cat_A_linear"))
  
})

test_that("multiple logger works", {

  expect_silent({ 
    cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost$addBaselearner("hp", "spline", PSplineBlearner, degree = 3, 
      n.knots = 10, penalty = 2, differences = 2)
    cboost$addBaselearner(c("hp", "wt"), "quadratic", PolynomialBlearner, degree = 2,
      intercept = TRUE)
  })

  expect_silent(
    cboost$addLogger(logger = TimeLogger, use.as.stopper = FALSE, logger.id = "time", 
      max.time = 0, time.unit = "microseconds")
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

test_that("custom base-learner works through api", {

  expect_silent({ cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new()) })

  instantiateData = function (X) {
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

  expect_silent({ 
    cboost$addBaselearner("hp", "custom", CustomBlearner, instantiate.fun =  instantiateData, 
      train.fun = trainFun, predict.fun = predictFun, param.fun = extractParameter) 
  })
  expect_output({ cboost$train(100) })

  expect_silent({
    cboost1 = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost1$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost1$train(100, trace = FALSE)
  })

  expect_equivalent(cboost$coef(), cboost1$coef())
  expect_equal(cboost$predict(), cboost1$predict())
  expect_equal(cboost$predict(), cboost$predict(mtcars))

})


test_that("custom cpp base-learner works through api", {

  expect_silent({ cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new()) })
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(silent = TRUE)) })  
  expect_silent({ 
    cboost$addBaselearner("hp", "custom", CustomCppBlearner, instantiate.ptr =  dataFunSetter(), 
      train.ptr = trainFunSetter(), predict.ptr = predictFunSetter()) 
  })
  expect_output({ cboost$train(100) })

  expect_silent({
    cboost1 = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost1$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost1$train(100, trace = FALSE)
  })

  expect_equivalent(cboost$coef(), cboost1$coef())
  expect_equal(cboost$predict(), cboost1$predict())
  expect_equal(cboost$predict(), cboost$predict(mtcars))

})

test_that("custom loss works through api", {

  myLossFun = function (true.value, prediction) { return(0.5 * (true.value - prediction)^2) }
  myGradientFun = function (true.value, prediction) { return(prediction - true.value) }
  myConstantInitializerFun = function (true.value) { mean.default(true.value) }
  
  expect_silent({ custom.loss = CustomLoss$new(myLossFun, myGradientFun, myConstantInitializerFun) })
  expect_silent({ cboost = Compboost$new(mtcars, "mpg", loss = custom.loss) })
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(silent = TRUE)) })  
  expect_silent({ 
    cboost$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)    
    cboost$addBaselearner("qsec", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
  })
  expect_output({ cboost$train(100) })

  expect_silent({
    cboost1 = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost1$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost1$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)    
    cboost1$addBaselearner("qsec", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost1$train(100, trace = FALSE)
  })

  expect_equivalent(cboost$coef(), cboost1$coef())
  expect_equal(cboost$predict(), cboost1$predict())
  expect_equal(cboost$selected(), cboost1$selected())
  expect_equal(cboost$predict(mtcars), cboost$predict())
  expect_equal(cboost$predict(), cboost$predict(response = TRUE))
  expect_equal(cboost$predict(mtcars, response = TRUE), cboost$predict(response = TRUE))

})

test_that("custom cpp loss works through api", {

  expect_output(Rcpp::sourceCpp(code = getCustomCppExample(example = "loss")))
  
  expect_silent({ custom.loss = CustomCppLoss$new(lossFunSetter(), gradFunSetter(), constInitFunSetter()) })
  expect_silent({ cboost = Compboost$new(mtcars, "mpg", loss = custom.loss) })
  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(silent = TRUE)) })  
  expect_silent({ 
    cboost$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)    
    cboost$addBaselearner("qsec", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
  })
  expect_output({ cboost$train(100) })

  expect_silent({
    cboost1 = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
    cboost1$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost1$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)    
    cboost1$addBaselearner("qsec", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost1$train(100, trace = FALSE)
  })

  expect_equivalent(cboost$coef(), cboost1$coef())
  expect_equal(cboost$predict(), cboost1$predict())
  expect_equal(cboost$selected(), cboost1$selected())
  expect_equal(cboost$predict(mtcars), cboost$predict())
  expect_equal(cboost$predict(mtcars, response = TRUE), cboost$predict(response = TRUE))

})

test_that("training with absolute loss works", {

  expect_silent({
    cboost = Compboost$new(mtcars, "hp", loss = AbsoluteLoss$new())
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost$train(100, trace = FALSE)
  })

  expect_length(cboost$selected(), 100)
  expect_length(cboost$risk(), 101)
  expect_equal(cboost$coef()$offset, median(mtcars$hp))
  expect_equal(cboost$predict(), cboost$predict(response = TRUE))
  expect_equal(cboost$predict(mtcars), cboost$predict(mtcars, response = TRUE))

})

test_that("training throws an error with pre-defined iteration logger", {
  
  expect_silent({
    cboost = Compboost$new(mtcars, "hp", loss = AbsoluteLoss$new())
    cboost$addLogger(IterationLogger, use.as.stopper = TRUE, "iteration", max.iter = 1000)
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
  })
  
  expect_warning(cboost$train(200)) 
  expect_length(cboost$risk(), 1001)
})

test_that("training with binomial loss works", {

  mtcars$hp.cat = ifelse(mtcars$hp > 150, 1, -1)

  expect_warning({ bin.loss = BinomialLoss$new(2) })

  expect_silent({
    cboost = Compboost$new(mtcars, "hp.cat", loss = bin.loss)
    cboost$addBaselearner("hp", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
    cboost$train(100, trace = FALSE)
  })
  
  expect_output(cboost$print())

  expect_length(cboost$selected(), 100)
  expect_length(cboost$risk(), 101)
  expect_equal(cboost$coef()$offset, 0.5 * log(sum(mtcars$hp.cat > 0)/ sum(mtcars$hp.cat < 0)))
  expect_equal(1 / (1 + exp(-cboost$predict())), cboost$predict(response = TRUE))
  expect_equal(1 / (1 + exp(-cboost$predict(mtcars))), cboost$predict(mtcars, response = TRUE))

  expect_silent({
    cboost = Compboost$new(mtcars, "hp", loss = BinomialLoss$new())
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
  })
  expect_error(cboost$train(100, trace = FALSE))

  mtcars$hp.cat = ifelse(mtcars$hp > 150, 1, 0)

  expect_silent({
    cboost = Compboost$new(mtcars, "hp.cat", loss = BinomialLoss$new())
    cboost$addBaselearner("wt", "linear", PolynomialBlearner, degree = 1,
      intercept = FALSE)
  })
  expect_error(cboost$train(100, trace = FALSE))

  expect_error({ cboost = Compboost$new(iris, "Species", loss = BinomialLoss$new()) })
  expect_silent({ cboost = Compboost$new(iris[1:100, ], "Species", loss = BinomialLoss$new()) })

})

test_that("custom poisson family does the same as mboost", {

  suppressWarnings(library(mboost))

  iris$Sepal.Length = as.integer(iris$Sepal.Length)

  lossPoisson = function (truth, response) {
    return (-log(exp(response)^truth * exp(-exp(response)) / gamma(truth + 1)))
  }
  gradPoisson = function (truth, response) {
    return (exp(response) - truth)
  }
  constInitPoisson = function (truth) {
    return (log(mean.default(truth)))
  }
  expect_silent({ my.poisson.loss = CustomLoss$new(lossPoisson, gradPoisson, constInitPoisson) })
  
  expect_silent({
    cboost = Compboost$new(iris, "Sepal.Length", loss = my.poisson.loss)
    cboost$addBaselearner("Sepal.Width", "linear", PolynomialBlearner, 
      degree = 1, intercept = TRUE)
    cboost$addBaselearner("Petal.Length", "spline", PSplineBlearner, 
      degree = 3, n.knots = 10, penalty = 2, differences = 2)
    cboost$train(100, trace = FALSE)
  })
  mod = mboost(Sepal.Length ~ bols(Sepal.Width) + bbs(Petal.Length, differences = 2, lambda = 2, 
    degree = 3, knots = 10), data = iris, family = Poisson(), 
    control = boost_control(mstop = 100, nu = 0.05))
  
  expect_silent({
    coef.cboost = cboost$coef()
    coef.mboost = coef(mod)
  })

  expect_equal(coef.cboost$offset, attr(coef.mboost, "offset"))
  expect_equal(as.numeric(coef.cboost[[1]]), as.numeric(coef.mboost[[2]]))
  expect_equal(as.numeric(coef.cboost[[2]]), as.numeric(coef.mboost[[1]]))
  expect_equal(as.numeric(cboost$predict()), as.numeric(predict(mod)))
  for (i in seq_len(10)) {
    idx = sample(seq_len(nrow(iris)), sample(seq_len(nrow(iris)), 1), TRUE)
    x = iris[idx, ]
    expect_equal(cboost$predict(x), predict(mod, x))
  }
})

test_that("quadratic loss does the same as mboost", {
  suppressWarnings(library(mboost))

  expect_silent({
    cboost = Compboost$new(iris, "Sepal.Width", loss = QuadraticLoss$new())
    cboost$addBaselearner("Sepal.Length", "linear", PolynomialBlearner, 
      degree = 1, intercept = TRUE)
    cboost$addBaselearner("Petal.Length", "spline", PSplineBlearner, 
      degree = 3, n.knots = 10, penalty = 2, differences = 2)
    cboost$train(100, trace = FALSE)
  })
  mod = mboost(Sepal.Width ~ bols(Sepal.Length) + bbs(Petal.Length, differences = 2, lambda = 2, 
    degree = 3, knots = 10), data = iris, control = boost_control(mstop = 100, nu = 0.05))
  
  expect_silent({
    coef.cboost = cboost$coef()
    coef.mboost = coef(mod)
  })
  expect_equal(coef.cboost$offset, attr(coef.mboost, "offset"))
  expect_equal(as.numeric(coef.cboost[[1]]), as.numeric(coef.mboost[[2]]))
  expect_equal(as.numeric(coef.cboost[[2]]), as.numeric(coef.mboost[[1]]))
  expect_equal(as.numeric(cboost$predict()), as.numeric(predict(mod)))
  for (i in seq_len(10)) {
    idx = sample(seq_len(nrow(iris)), sample(seq_len(nrow(iris)), 1), TRUE)
    x = iris[idx, ]
    expect_equal(cboost$predict(x), predict(mod, x))
  }
})

test_that("handler throws warnings", {
  expect_silent({
    cboost = Compboost$new(iris, "Sepal.Width", loss = QuadraticLoss$new())
  })

  expect_warning(cboost$addBaselearner("Sepal.Length", "linear", PolynomialBlearner, 
      degree = 1, false.intercept = TRUE))
  
  expect_warning(cboost$addBaselearner("Petal.Length", "spline", PSplineBlearner, 
      degree = 3, n.knots = 10, penalty = 2, differences = 2, i.am.not.used = NULL))

  instantiateData = function (X) {
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

  expect_warning(cboost$addBaselearner("Sepal.Length", "custom", CustomBlearner, instantiate.fun =  instantiateData, 
      train.fun = trainFun, predict.fun = predictFun, param.fun = extractParameter, i.am.not.used = NULL))

  expect_silent({ Rcpp::sourceCpp(code = getCustomCppExample(silent = TRUE)) })  
  expect_warning(cboost$addBaselearner("Sepal.Length", "custom", CustomCppBlearner, instantiate.ptr =  dataFunSetter(), 
      train.ptr = trainFunSetter(), predict.ptr = predictFunSetter(), i.am.not.used = NULL)) 
})


test_that("default values are used by handler", {

  expect_silent({
    cboost = Compboost$new(iris, "Sepal.Width", loss = QuadraticLoss$new())
  })
  expect_silent(cboost$addBaselearner("Sepal.Length", "linear", PolynomialBlearner))
  expect_silent(cboost$addBaselearner("Petal.Length", "spline", PSplineBlearner))

})