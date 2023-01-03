context("Response")

test_that("Regression response works correctly", {

  target = "x"
  X = as.matrix(1:10)
  loss = LossQuadratic$new()
  loss_false = LossBinomial$new()

  expect_silent({ response = ResponseRegr$new(target, X) })
  expect_equal(response$getTargetName(), target)
  expect_equal(response$getResponse(), X)
  expect_equal(response$getPrediction(), X * 0)
  expect_equal(response$getPredictionTransform(), X * 0)
  expect_equal(response$getPredictionResponse(), X * 0)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(X^2) / 2)
  expect_error(response$calculateEmpiricalRisk(loss_false))

  expect_error({ response = ResponseRegr$new(id, X, cbind(weights, weights)) })
})

test_that("Regression response with weights works correctly", {

  target = "x"
  X = as.matrix(1:10)
  weights = as.matrix(rep(c(0.5, 2), 5))
  loss = LossQuadratic$new()

  expect_silent({ response = ResponseRegr$new(target, X, weights) })
  expect_equal(response$getWeights(), weights)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(weights * X^2) / 2)
})

test_that("Binary classification response works correctly", {

  target = "x"
  threshold = 0.5
  X_false = as.matrix(1:10)
  X_correct = as.matrix(sample(c(1,-1), 10, TRUE))
  response_vec = c("neg", "pos")[(X_correct + 1) / 2 + 1]
  sigmoid = 1 / (1 + exp(-X_correct * 0))
  pred_response = ifelse(sigmoid < threshold, -1, 1)
  loss = LossBinomial$new()

  expect_error({ response = ResponseBinaryClassif$new(target, X_false) })
  expect_error({ response = ResponseBinaryClassif$new(target, "pos", c(response_vec, "test")) })
  expect_silent({ response = ResponseBinaryClassif$new(target, "pos", response_vec) })
  expect_equal(response$getTargetName(), target)
  expect_equal(response$getResponse(), X_correct)
  expect_equal(response$getPrediction(), X_correct * 0)
  expect_equal(response$getPredictionTransform(), sigmoid)
  expect_equal(response$getPredictionResponse(), pred_response)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(log(1 + exp(-2 * X_correct * response$getPrediction()))))
  expect_equal(response$getPositiveClass(), "pos")
  expect_equal(unname(response$getClassTable()), as.numeric(table(response_vec)))
  expect_error({ response = ResponseRegr$new(id, X, cbind(weights, weights)) })

  threshold = 0.8
  pred_response = ifelse(sigmoid < threshold, -1, 1)
  expect_silent({ response$setThreshold(threshold) })
  expect_equal(response$getThreshold(), threshold)
  expect_equal(response$getPredictionResponse(), pred_response)
  expect_error(response$setThreshold(1.1))
  expect_equal(response$getThreshold(), threshold)
})

test_that("Binary classification response with weights works correctly", {

  target = "x"
  threshold = 0.5
  X_correct = as.matrix(sample(c(1,-1), 10, TRUE))
  response_vec = c("neg", "pos")[(X_correct + 1) / 2 + 1]
  weights = as.matrix(rep(c(0.5, 2), 5))
  sigmoid = 1 / (1 + exp(-X_correct * 0))
  pred_response = ifelse(sigmoid < threshold, -1, 1)
  loss = LossBinomial$new()

  expect_silent({ response = ResponseBinaryClassif$new(target, "pos", response_vec, weights) })
  expect_equal(response$getWeights(), weights)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(weights * log(1 + exp(-2 * X_correct * response$getPrediction()))))
})

test_that("Loading response from JSON works", {

  ## REGRESSION:
  cboost = expect_output(boostSplines(iris, "Sepal.Length", loss = LossQuadratic$new()))
  file = here::here("temp/cboost.json")

  expect_silent(cboost$model$saveJson(file))
  expect_true(file.exists(file))
  r = expect_silent(ResponseRegr$new(file))

  expect_equal(class(r), class(cboost$response))
  expect_equal(r$getResponse(), cboost$response$getResponse())
  expect_equal(r$getPrediction(), cboost$response$getPrediction())
  expect_equal(r$getTargetName(), cboost$response$getTargetName())
  expect_equal(r$calculateEmpiricalRisk(cboost$loss), cboost$response$calculateEmpiricalRisk(cboost$loss))

  ## BINARY CLASSIFICATION:
  cboost = expect_output(boostSplines(iris[1:100, ], "Species", loss = LossBinomial$new()))
  file = here::here("temp/cboost.json")

  expect_silent(cboost$model$saveJson(file))
  expect_true(file.exists(file))
  r = expect_silent(ResponseBinaryClassif$new(file))

  expect_equal(class(r), class(cboost$response))
  expect_equal(r$getResponse(), cboost$response$getResponse())
  expect_equal(r$getPrediction(), cboost$response$getPrediction())
  expect_equal(r$getTargetName(), cboost$response$getTargetName())
  expect_equal(r$calculateEmpiricalRisk(cboost$loss), cboost$response$calculateEmpiricalRisk(cboost$loss))
  expect_equal(r$getThreshold(), cboost$response$getThreshold())
  expect_equal(r$getPositiveClass(), cboost$response$getPositiveClass())
  expect_equal(r$getClassTable(), cboost$response$getClassTable())
})
