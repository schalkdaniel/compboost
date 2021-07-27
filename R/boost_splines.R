#' Wrapper to boost general additive models for each feature.
#'
#' This wrapper function automatically initialize the model by adding all numerical
#' features as spline base-learner. Categorical features are dummy encoded and inserted
#' using another linear base-learners without intercept. The function \code{boostSplines}
#' does also train the model.
#'
#' The returned object is an object of the \code{Compboost} class. This object can be
#' used for further analyses (see \code{?Compboost} for details).
#'
#' @return A model of the \code{Compboost} class. This model is an \code{R6} object
#'   which can be used for retraining, predicting, plotting, and anything described in
#'   \code{?Compboost}.
#' @param data [\code{data.frame}]\cr
#'   A data frame containing the data.
#' @param target [\code{character(1)} or \code{Response} class]\cr
#'   Character value containing the target variable or Response object. Note that the loss must match the
#'   data type of the target.
#' @param optimizer [\code{S4 Optimizer}]\cr
#'   An initialized \code{S4 Optimizer} object exposed by Rcpp (e.g. \code{OptimizerCoordinateDescent$new()})
#'   to select features at each iteration.
#' @param loss [\code{S4 Loss}]\cr
#'   Initialized \code{S4 Loss} object exposed by Rcpp that is used to calculate the risk and pseudo
#'   residuals (e.g. \code{LossQuadratic$new()}).
#' @param learning_rate [\code{numeric(1)}]\cr
#'   Learning rate to shrink the parameter in each step.
#' @param iterations [\code{integer(1)}]\cr
#'   Number of iterations that are trained.
#' @param trace [\code{integer(1)}]\cr
#'   Integer indicating how often a trace should be printed. Specifying \code{trace = 10}, then every
#'   10th iteration is printed. If no trace should be printed set \code{trace = 0}. Default is
#'   -1 which means that in total 40 iterations are printed.
#' @param degree [\code{integer(1)}]\cr
#'   Polynomial degree of the splines.
#' @param n_knots [\code{integer(1)}]\cr
#'   Number of equidistant "inner knots". The actual number of used knots does also depend on
#'   the polynomial degree.
#' @param penalty [\code{numeric(1)}]\cr
#'   Penalty term for p-splines. If the penalty equals 0, then ordinary b-splines are fitted.
#'   The higher the penalty, the higher the smoothness.
#' @param df [\code{numeric(1)}]\cr
#'   Degrees of freedom of the whole spline. It is important to set the same amount of degrees of freedom to be able to compare different base-learner.
#' @param differences [\code{integer(1)}]\cr
#'   Number of differences that are used for penalization. The higher the difference, the higher the smoothness.
#' @param data_source [\code{S4 Data}]\cr
#'   Uninitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param oob_fraction [\code{numeric(1)}]\cr
#'   Fraction of how much data we want to use to track the out of bag risk.
#' @param bin_root [\code{integer(1)}]\cr
#'   If set to a value greater than zero, binning is applied and reduces the number of used
#'   x values to n^(1/bin_root) equidistant points. If you want to use binning we suggest
#'   to set \code{bin_root = 2}.
#' @param bin_method [\code{character(1)}]\cr
#'   Method used to calculate knots for binning. Possible choices are "quantile" and "linear".
#' @param cache_type [\code{character(1)}+]\cr
#'   String to indicate what method should be used to estimate the parameter in each iteration.
#'   Default is \code{cache_type = "cholesky"} which computes the Cholesky decomposition,
#'   caches it, and reuses the matrix over and over again. The other option is to use
#'   \code{cache_type = "inverse"} which does the same but caches the inverse.
#' @param stop_args [\code{list(2)}]\cr
#'   List containing two elements `patience` and `eps_for_break` which can be set to use early stopping on the left out data
#'   from setting `oob_fraction`.
#' @param df_cat [\code{numeric(1)}]\cr
#'   Degrees of freedom of the categorical base-learner.
#' @examples
#' mod = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(),
#'   oob_fraction = 0.3)
#' mod$getBaselearnerNames()
#' mod$getEstimatedCoef()
#' table(mod$getSelectedBaselearner())
#' mod$predict()
#' mod$plot("Sepal.Width_spline")
#' mod$plotInbagVsOobRisk()
#' @export
boostSplines = function(data, target, optimizer = OptimizerCoordinateDescent$new(), loss,
  learning_rate = 0.05, iterations = 100, trace = -1, degree = 3, n_knots = 20,
  penalty = 2, df = 0, differences = 2, data_source = InMemoryData,
  oob_fraction = NULL, bin_root = 0, bin_method = "linear", cache_type = "inverse",
  stop_args = list(), df_cat = 1, stop_time = "microseconds", additional_risk_logs = list())
{
  model = Compboost$new(data = data, target = target, optimizer = optimizer, loss = loss,
    learning_rate = learning_rate, oob_fraction = oob_fraction, stop_args)
  features = setdiff(colnames(data), model$response$getTargetName())

  checkmate::assertChoice(stop_time, choices = c("minuts", "seconds", "microseconds"), null.ok = TRUE)

  # This loop could be replaced with foreach???
  # Issue:
  for (feat in features) {
    if (is.numeric(data[[feat]])) {
      model$addBaselearner(feat, "spline", BaselearnerPSpline, data_source,
        degree = degree, n_knots = n_knots, penalty = penalty, df = df,  differences = differences,
        bin_root = bin_root, bin_method = bin_method, cache_type = cache_type)
    } else {
      checkmate::assertNumeric(df_cat, len = 1L, lower = 1)
      if (length(unique(feat)) > df_cat) stop("Categorical degree of freedom must be smaller than the number of classes (here <", length(unique(feat)), ")")
      model$addBaselearner(feat, "ridge", BaselearnerCategoricalRidge, data_source,
        df = df_cat)
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
  model$train(iterations, trace)
  return(model)
}
