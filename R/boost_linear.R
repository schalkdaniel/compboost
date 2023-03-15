#' @title Wrapper to boost linear models for each feature.
#'
#' @description
#' This wrapper function automatically initializes the model by adding all numerical
#' features as linear base-learner. Categorical features are dummy encoded and inserted
#' using another linear base-learners without intercept. The function `boostLinear`
#' does also train the model.
#'
#' The returned object is an object of the [Compboost] class. This object can be
#' used for further analyses (see `?Compboost` for details).
#'
#' @return A model of the [Compboost] class. This model is an [R6] object
#'   which can be used for retraining, predicting, plotting, and anything described in
#'   `?Compboost`.
#' @param data (`data.frame`)\cr
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
#'   Integer indicating how often a trace should be printed. Specifying `trace = 10`, then every
#'   10th iteration is printed. If no trace should be printed set `trace = 0`. Default is
#'   -1 which means that in total 40 iterations are printed.
#' @param intercept (`logical(1)`)\cr
#'   Internally used by [BaselearnerPolynomial]. This logical value indicates if
#'   each feature should get an intercept or not (default is `TRUE`).
#' @param df_cat (`numeric(1)`)\cr
#'   Degrees of freedom of the categorical base-learner.
#' @param data_source (`Data*`)\cr
#'   Uninitialized `Data*` object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param oob_fraction (`numeric(1)`)\cr
#'   Fraction of how much data are used to track the out of bag risk.
#' @template param-stop_args
#' @examples
#' mod = boostLinear(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(),
#'   oob_fraction = 0.3)
#' mod$getBaselearnerNames()
#' mod$getEstimatedCoef()
#' table(mod$getSelectedBaselearner())
#' mod$predict()
#' @export
boostLinear = function(data, target, optimizer = NULL, loss = NULL, learning_rate = 0.05,
  iterations = 100, trace = -1, intercept = TRUE, data_source = InMemoryData,
  df_cat = 2, oob_fraction = NULL, stop_args = NULL)
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

	for (feat in features) {
		if (is.numeric(data[[feat]])) {
			model$addBaselearner(feat, "linear", BaselearnerPolynomial, data_source,
				degree = 1, intercept = intercept)
		} else {
			model$addBaselearner(feat, "ridge", BaselearnerCategoricalRidge, data_source, df = df_cat)
		}
	}
  if (iterations == 0) return(model)

	model$train(iterations, trace)
	return(model)
}
