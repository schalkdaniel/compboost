context("Response")

test_that("Regression response works correctly", {

  target = "x"
  X = as.matrix(1:10)
  loss = LossQuadratic$new()
  loss.false = LossBinomial$new()

  expect_silent({ response = ResponseRegr$new(target, X) })
  expect_equal(response$getTargetName(), target)
  expect_equal(response$getResponse(), X)
  expect_equal(response$getPrediction(), X * 0)
  expect_equal(response$getPredictionTransform(), X * 0)
  expect_equal(response$getPredictionResponse(), X * 0)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(X^2) / 2)
  expect_error(response$calculateEmpiricalRisk(loss.false))

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
  X.false = as.matrix(1:10)
  X.correct = as.matrix(sample(c(1,-1), 10, TRUE))
  sigmoid = 1 / (1 + exp(-X.correct * 0))
  pred_response = ifelse(sigmoid < threshold, -1, 1)
  loss = LossBinomial$new()

  expect_error({ response = ResponseBinaryClassif$new(target, X.false) })
  expect_silent({ response = ResponseBinaryClassif$new(target, X.correct) })
  expect_equal(response$getTargetName(), target)
  expect_equal(response$getResponse(), X.correct)
  expect_equal(response$getPrediction(), X.correct * 0)
  expect_equal(response$getPredictionTransform(), sigmoid)
  expect_equal(response$getPredictionResponse(), pred_response)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(log(1 + exp(-2 * X.correct * response$getPredictionTransform()))))

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
  X.correct = as.matrix(sample(c(1,-1), 10, TRUE))
  weights = as.matrix(rep(c(0.5, 2), 5))
  sigmoid = 1 / (1 + exp(-X.correct * 0))
  pred_response = ifelse(sigmoid < threshold, -1, 1)
  loss = LossBinomial$new()

  expect_silent({ response = ResponseBinaryClassif$new(target, X.correct, weights) })
  expect_equal(response$getWeights(), weights)
  expect_equal(response$calculateEmpiricalRisk(loss), mean(weights * log(1 + exp(-2 * X.correct * response$getPredictionTransform()))))
})