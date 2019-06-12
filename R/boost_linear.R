#' Wrapper to boost linear models for each feature.
#'
#' This wrapper function automatically initialize the model by adding all numerical
#' features as linear base-learner. Categorical features are dummy encoded and inserted 
#' using another linear base-learners without intercept. The function \code{boostLinear} 
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
#' @param target [\code{character(1)}]\cr
#'   Character value containing the target variable. Note that the loss must match the 
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
#' @param intercept [\code{logical(1)}]\cr
#'   Internally used by \code{BaselearnerPolynomial}. This logical value indicates if
#'   each feature should get an intercept or not (default is \code{TRUE}).
#' @param data_source [\code{S4 Data}]\cr
#'   Uninitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param data_target [\code{S4 Data}]\cr
#'   Uninitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param oob_fraction [\code{numeric(1)}]\cr
#'   Fraction of how much data are used to track the out of bag risk.
#' @examples
#' mod = boostLinear(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(), 
#'   oob_fraction = 0.3)
#' mod$getBaselearnerNames()
#' mod$getEstimatedCoef()
#' table(mod$getSelectedBaselearner())
#' mod$predict()
#' mod$plot("Sepal.Width_linear")
#' mod$plotInbagVsOobRisk()
#' @export
boostLinear = function(data, target, optimizer = OptimizerCoordinateDescent$new(), loss, 
	learning_rate = 0.05, iterations = 100, trace = -1, intercept = TRUE, 
	data_source = InMemoryData, data_target = InMemoryData, oob_fraction = NULL) 
{
	model = Compboost$new(data = data, target = target, optimizer = optimizer, loss = loss, 
		learning_rate = learning_rate, oob_fraction = oob_fraction, time_spline_pars)
  
  if(class(model$response)[1] %in% c("Rcpp_ResponseFDA","Rcpp_ResponseFDALong")){
    features = names(data)
  } else {
     features = setdiff(colnames(data), target)
  }
 
	for (feat in features) {
		if (is.numeric(data[[feat]])) {
			model$addBaselearner(feat, "linear", BaselearnerPolynomial, data_source, data_target,
				degree = 1, intercept = intercept)
		} else {
			model$addBaselearner(feat, "category", BaselearnerPolynomial, data_source, data_target,
				degree = 1, intercept = FALSE)
		}
	}
	model$train(iterations, trace)
	return (model)
}
