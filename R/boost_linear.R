#' Wrapper to boost linear models for each feature.
#'
#' This wrapper function automatically initializes the model by adding all numerical
#' features of a dataset within a linear base-learner. Categorical features are
#' dummy encoded and inserted using linear base-learners without intercept. After 
#' initializing the model \code{boostLinear} also fits as many iterations as given 
#' by the user through \code{iters}. 
#' 
#' The returned object is an object of the \code{Compboost} class which then can be 
#' used for further analyses (see \code{?Compboost} for details). 
#'
#' @return Usually a model of class \code{Compboost}. This model is an \code{R6} object
#'   which can be used for retraining, predicting, plotting, and anything described in 
#'   \code{?Compboost}.
#' @param data [\code{data.frame}]\cr
#'   A data frame containing the data on which the model should be built. 
#' @param target [\code{character(1)}]\cr
#'   Character indicating the target variable. Note that the loss must match the 
#'   data type of the target.
#' @param optimizer [\code{S4 Optimizer}]\cr
#'   Optimizer to select features. This should be an initialized \code{S4 Optimizer} object
#'   exposed by Rcpp (for instance \code{OptimizerCoordinateDescent$new()}).
#' @param loss [\code{S4 Loss}]\cr
#'   Loss used to calculate the risk and pseudo residuals. This object must be an initialized
#'   \code{S4 Loss} object exposed by Rcpp (for instance \code{LossQuadratic$new()}).
#' @param learning.rate [\code{numeric(1)}]\cr
#'   Learning rate which is used to shrink the parameter in each step.
#' @param iterations [\code{integer(1)}]\cr
#'   Number of iterations that are trained.
#' @param trace [\code{integer(1)}]\cr
#'   Integer indicating how often a trace should be printed. Specifying \code{trace = 10}, then every
#'   10th iteration is printed. If no trace should be printed set \code{trace = 0}. Default is
#'   -1 which means that we set \code{trace} at a value that 40 iterations are printed.
#' @param intercept [\code{logical(1)}]\cr
#'   Internally used by \code{BaselearnerPolynomial}. This logical value indicates if
#'   each feature should get an intercept or not (default is \code{TRUE}).
#' @param data.source [\code{S4 Data}]\cr
#'   Uninitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param data.target [\code{S4 Data}]\cr
#'   Uninitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @examples
#' mod = boostLinear(data = iris, target = "Sepal.Length", loss = LossQuadratic$new())
#' mod$getBaselearnerNames()
#' mod$getEstimatedCoef()
#' table(mod$getSelectedBaselearner())
#' mod$predict()
#' mod$plot("Sepal.Width_linear")
#' @export
boostLinear = function(data, target, optimizer = OptimizerCoordinateDescent$new(), loss, 
	learning.rate = 0.05, iterations = 100, trace = -1, intercept = TRUE, 
	data.source = InMemoryData, data.target = InMemoryData) 
{
	model = Compboost$new(data = data, target = target, loss = loss, learning.rate = learning.rate)
	features = setdiff(colnames(data), target)

	# This loop could be replaced with foreach???
	# Issue: 
	for(feat in features) {
		if (is.numeric(data[[feat]])) {
			model$addBaselearner(feat, "linear", BaselearnerPolynomial, data.source, data.target,
				degree = 1, intercept = intercept)
		} else {
			model$addBaselearner(feat, "category", BaselearnerPolynomial, data.source, data.target,
				degree = 1, intercept = FALSE)
		}
	}
	model$train(iterations, trace)
	return(model)
}
