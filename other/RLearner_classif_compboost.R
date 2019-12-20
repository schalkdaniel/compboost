makeRLearner.classif.compboost = function() {
  makeRLearnerClassif(
    cl = "classif.compboost",
    package = "compboost",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "learning_rate", default = 0.01, lower = 0.00000001),
      makeNumericLearnerParam(id = "penalty", default = 2, lower = 0),
      makeIntegerLearnerParam(id = "iters", default = 100L, lower = 1),
      makeIntegerLearnerParam(id = "mstop", default = 100L, lower = 1)
    ),
    properties = c("twoclass", "numerics", "factors", "prob"),
    name = "Component-Wise Boosting",
    short.name = "compboost"
  )
}

trainLearner.classif.compboost = function(.learner, .task, .subset, .weights = NULL, ...) {

  data_temp = getTaskData(.task, .subset)

  feats = getTaskFeatureNames(.task)
  for (feat in feats) {
    if (is.factor(getTaskData(.task, .subset)[[feat]])) {
      data_temp[[feat]] = as.integer(data_temp[[feat]])
    }
  }
  cboost = Compboost$new(data = data_temp, target = getTaskTargetNames(.task), loss = LossBinomial$new())

  dots = list(...)

  for (feat in feats) {
    if (is.numeric(getTaskData(.task, .subset)[[feat]])) {
      cboost$addBaselearner(feat, "spline", BaselearnerPSpline)
    }
    if (is.factor(getTaskData(.task, .subset)[[feat]])) {
      cboost$addBaselearner(feat, "cat", BaselearnerCategorical, learning_rate = dots[["learning_rate"]], penalty = dots[["penalty"]], iters = dots[["iters"]])
    }
  }
  cboost$train(dots[["mstop"]], trace = 0)
  return (cboost)
}


predictLearner.classif.compboost = function(.learner, .model, .newdata, predict.method = "plug-in", ...) {

  temp_data = .newdata
  feats = colnames(temp_data)

  for (feat in feats) {
    if (is.factor(temp_data[[feat]])) {
      data_temp[[feat]] = as.integer(data_temp[[feat]])
    }
  }
  if (.learner$predict.type == "response") {
    out = ifelse(as.numeric(.model$learner.model$predict(temp_data, as_response = TRUE)) > 0.5, .model$task.desc$positive, .model$task.desc$negative)
    return(as.factor(out))
  } else {
    out = .model$learner.model$predict(temp_data, as_response = TRUE)
    out = cbind(out, 1 - out)
    colnames(out) = c(.model$task.desc$positive, .model$task.desc$negative)
    return(out)
  }
}


# ln = makeLearner("classif.compboost", learning_rate = 0.01, penalty = 2, iters = 10, mstop = 100, predict.type = "prob")
# test = train(ln, spam.task)
# pred = predict(test, task = spam.task)










