#' @title Wrapper to boost general additive models using components
#'
#' @description
#' This wrapper function automatically initializes the model by adding all numerical
#' features as components. This means, that for each numerical feature a linear effect
#' and non-linear spline base-learner is added. The non-linear part is constructed in way
#' that it cannot model the linear part. Hence, it is just selected if a non-linear
#' base learner is really necessary. Categorical features are dummy encoded and inserted
#' using another linear base-learners without intercept.
#'
#' The returned object is an object of the [Compboost] class. This object can be
#' used for further analyses (see `?Compboost` for details).
#'
#' @return A model of the [Compboost] class. This model is an [R6] object
#'   which can be used for retraining, predicting, plotting, and anything described in
#'   `?Compboost`.
#' @param data (`data.frame()`)\cr
#'   A data frame containing the data.
#' @param target (`character(1)` | [ResponseRegr] | [ResponseBinaryClassif])\cr
#'   Character value containing the target variable or response object. Note that the loss must match the
#'   data type of the target.
#' @template param-optimizer
#' @template param-loss
#' @param learning_rate (`numeric(1)`)\cr
#'   Learning rate to shrink the parameter in each step.
#' @param iterations (`integer(1)`)\cr
#'   Number of iterations that are trained. If `iterations == 0`, the untrained object is returned. This
#'   can be useful if other base learners (e.g. an interaction via a tensor base learner) are added.
#' @param trace (`integer(1)`)\cr
#'   Integer indicating how often a trace should be printed. Specifying \code{trace = 10}, then every
#'   10th iteration is printed. If no trace should be printed set \code{trace = 0}. Default is
#'   -1 which means that in total 40 iterations are printed.
#' @param degree (`integer(1)`)cr
#'   Polynomial degree of the splines.
#' @param n_knots (`integer(1)`)\cr
#'   Number of equidistant "inner knots". The actual number of used knots does also depend on
#'   the polynomial degree.
#' @param penalty (`numeric(1)`)\cr
#'   Penalty term for p-splines. If the penalty equals 0, then ordinary b-splines are fitted.
#'   The higher the penalty, the higher the smoothness.
#' @template param-df
#' @param differences (`integer(1)`)\cr
#'   Number of differences that are used for penalization. The higher the difference, the higher the smoothness.
#' @param data_source (`Data*`)\cr
#'   Uninitialized `Data*` object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param oob_fraction (`numeric(1)`)\cr
#'   Fraction of how much data are used to track the out of bag risk.
#' @template param-bin_root
#' @param cache_type (`character(1)`)\cr
#'   String to indicate what method should be used to estimate the parameter in each iteration.
#'   Default is \code{cache_type = "cholesky"} which computes the Cholesky decomposition,
#'   caches it, and reuses the matrix over and over again. The other option is to use
#'   \code{cache_type = "inverse"} which does the same but caches the inverse.
#' @template param-stop_args
#' @param df_cat (`numeric(1)`)\cr
#'   Degrees of freedom of the categorical base-learner.
#' @param stop_time (`character(1)`)\cr
#'   Unit of measured time.
#' @param additional_risk_logs (`list(Logger)`)\cr
#'   Additional logger passed to the `Compboost` object.
#' @examples
#' mod = boostComponents(data = iris, target = "Sepal.Length", df = 4)
#' mod$getBaselearnerNames()
#' table(mod$getSelectedBaselearner())
#' plotPEUni(mod, "Petal.Length")
#' mod$predict()
#' @export
boostComponents = function(data, target, optimizer = NULL, loss = NULL,
  learning_rate = 0.05, iterations = 100, trace = -1, degree = 3, n_knots = 20,
  penalty = 2, df = 0, differences = 2, data_source = InMemoryData,
  oob_fraction = NULL, bin_root = 0, cache_type = "inverse",
  stop_args = list(), df_cat = 1, stop_time = "microseconds", additional_risk_logs = list())
{
  if (is.null(oob_fraction) || (oob_fraction == 0)) {
    stop_args = NULL
  }
  if (checkmate::testList(stop_args)) {
    early_stop = TRUE
  } else {
    early_stop = FALSE
    stop_args = list()
  }
  model = Compboost$new(data = data, target = target, optimizer = optimizer, loss = loss,
    learning_rate = learning_rate, oob_fraction = oob_fraction, stop_args = stop_args,
    early_stop = early_stop)
  features = setdiff(colnames(data), model$response$getTargetName())

  checkmate::assertChoice(stop_time, choices = c("minuts", "seconds", "microseconds"), null.ok = TRUE)

  for (feat in features) {
    if (is.numeric(data[[feat]])) {
      model$addComponents(feat, degree = degree, n_knots = n_knots, penalty = penalty,
        df = df,  differences = differences, bin_root = bin_root, cache_type = cache_type)
    } else {
      checkmate::assertNumeric(df_cat, len = 1L, lower = 1)
      if (length(unique(feat)) > df_cat) stop("Categorical degree of freedom must be smaller than the number of classes (here <", length(unique(feat)), ")")
      model$addBaselearner(feat, "ridge", BaselearnerCategoricalRidge, df = df_cat)
    }
  }
  if (! is.null(stop_time)) {
    model$addLogger(LoggerTime, FALSE, "time", 0, stop_time)
  }
  if (length(additional_risk_logs) > 0) {
    for (i in seq_along(additional_risk_logs)) {
      if (! is.null(additional_risk_logs[[i]]$data))
        ndat = additional_risk_logs[[i]]$data
      else
        ndat = model$data

      model$addLogger(logger = LoggerOobRisk, use_as_stopper = FALSE, logger_id = paste0("risk", names(additional_risk_logs)[i]),
        used_loss = additional_risk_logs[[i]]$loss, esp_for_break = 0, patience = 1, oob_data = model$prepareData(ndat),
        oob_response = model$prepareResponse(ndat[[target]]))
    }
  }
  if (iterations == 0) return(model)

  model$train(iterations, trace)
  return(model)
}
