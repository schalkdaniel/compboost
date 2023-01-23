context("JSON serialization works")

test_that("basic components", {
  file = "cboost.json"

  cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossAbsolute$new()))
  expect_silent(cboost$model$saveJson(file))
  cboost2 = expect_silent(Compboost_internal$new(file))

  testCboostJson(cboost, cboost2)
  testCboostJson(cboost, cboost2, 10)
  testCboostJson(cboost, cboost2, 1000)

  file.remove(file)
})

test_that("different losses", {
  file = "cboost.json"
  losses = c(LossQuadratic$new(), LossQuantile$new(0.2), LossAbsolute$new(), LossHuber$new())

  nn = lapply(losses, function(l) {
    cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = l))
    expect_silent(cboost$model$saveJson(file))
    cboost2 = expect_silent(Compboost_internal$new(file))

    testCboostJson(cboost, cboost2)
    testCboostJson(cboost, cboost2, 10)
    testCboostJson(cboost, cboost2, 1000)

    return(NULL)
  })

  # Test Binomial loss as categorical exception:
  cboost = expect_output(boostSplines(iris[1:100, ], "Species", loss = LossBinomial$new()))
  expect_silent(cboost$model$saveJson(file))
  cboost2 = expect_silent(Compboost_internal$new(file))

  testCboostJson(cboost, cboost2)
  testCboostJson(cboost, cboost2, 10)
  testCboostJson(cboost, cboost2, 1000)

  file.remove(file)
})

test_that("different optimizers", {
  file = "cboost.json"

  cores = 1
  if (parallel::detectCores() > 2) {
    cores = c(1, 2)
  }
  nn = lapply(cores, function(cr) {
    optimizers = c(OptimizerCoordinateDescent$new(cr),
      OptimizerCoordinateDescentLineSearch$new(cr),
      OptimizerCosineAnnealing$new(cr),
      OptimizerAGBM$new(0.1, cr))

    lapply(optimizers, function(op) {
      cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossQuadratic$new(), optimizer = op))
      expect_silent(cboost$model$saveJson(file))
      cboost2 = expect_silent(Compboost_internal$new(file))

      testCboostJson(cboost, cboost2)
      if (! grepl("OptimizerAGBM", class(op))) {
        testCboostJson(cboost, cboost2, 10)
        testCboostJson(cboost, cboost2, 1000)
      }

      return(NULL)
    })
  })

  file.remove(file)
})

test_that("complex base learner", {
  file = "cboost.json"

  # TENSOR:
  cboost = expect_silent(Compboost$new(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()))
  expect_silent(cboost$addTensor("Petal.Length", "Petal.Width"))
  expect_silent(cboost$addTensor("Sepal.Width", "Species"))
  expect_output(cboost$train(100))
  expect_silent(cboost$model$saveJson(file))
  cboost2 = expect_silent(Compboost_internal$new(file))

  testCboostJson(cboost, cboost2, blp = "Petal.Length_Petal.Width_tensor")
  testCboostJson(cboost, cboost2, 10, blp = "Petal.Length_Petal.Width_tensor")
  testCboostJson(cboost, cboost2, 1000, blp = "Petal.Length_Petal.Width_tensor")

  # CENTERED:
  cboost = expect_silent(Compboost$new(data = iris, target = "Sepal.Length", loss = LossQuadratic$new()))
  expect_silent(cboost$addComponents("Petal.Length"))
  expect_output(cboost$train(100))
  expect_silent(cboost$model$saveJson(file))
  cboost2 = expect_silent(Compboost_internal$new(file))

  testCboostJson(cboost, cboost2, blp = "Petal.Length_spline_centered")
  testCboostJson(cboost, cboost2, 40, blp = "Petal.Length_spline_centered")
  testCboostJson(cboost, cboost2, 1000, blp = "Petal.Length_spline_centered")


  file.remove(file)
})
