#' Wrapper to boost linear models for each feature.
#'
#' This wrapper function automatically initializes the model by adding all numerical
#' features of a dataset within a linear base-learner. Categorical features are
#' dummy encoded and inserted using linear base-learners without intercept. After 
#' initializing the model \code{boostLinear} also fits as many iterations as given 
#' by the user through \code{iters}. 
#' 
#' The returend object is an object of the \code{Compboost} class which then can be 
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
#'   exposed by Rcpp (for instance \code{CoordinateDescent$new()}).
#' @param loss [\code{S4 Loss}]\cr
#'   Loss used to calculate the risk and pseudo residuals. This object must be an initialized
#'   \code{S4 Loss} object exposed by Rcpp (for instance \code{QuadraticLoss$new()}).
#' @param learning.rate [\code{numeric(1)}]\cr
#'   Learning rate which is used to shrink the parameter in each step.
#' @param iterations [\code{integer(1)}]\cr
#'   Number of iterations that are trained.
#' @param trace [\code{logical(1)}]\cr
#'   Logical to indicate whether the trace should be printed or not.
#' @param intercept [\code{logical(1)}]\cr
#'   Internally used by \code{PolynomialBlearner}. This logical value indicates if
#'   each feature should get an intercept or not (default is \code{TRUE}).
#' @param data.source [\code{S4 Data}]\cr
#'   Unitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param data.target [\code{S4 Data}]\cr
#'   Unitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @examples
#' mod = boostLinear(data = iris, target = "Sepal.Length", loss = QuadraticLoss$new())
#' mod$getBaselearnerNames()
#' mod$coef()
#' table(mod$selected())
#' mod$predict()
#' mod$plot("Sepal.Width_linear")
#' @export
boostLinear = function(data, target, optimizer = CoordinateDescent$new(), loss, 
	learning.rate = 0.05, iterations = 100, trace = TRUE, intercept = TRUE, 
	data.source = InMemoryData, data.target = InMemoryData) 
{
	model = Compboost$new(data = data, target = target, loss = loss, learning.rate = learning.rate)
	features = setdiff(colnames(data), target)

	# This loop could be replaced with foreach???
	# Issue: 
	for(feat in features) {
		if (is.numeric(data[[feat]])) {
			model$addBaselearner(feat, "linear", PolynomialBlearner, data.source, data.target,
				degree = 1, intercept = intercept)
		} else {
			model$addBaselearner(feat, "category", PolynomialBlearner, data.source, data.target,
				degree = 1, intercept = FALSE)
		}
	}
	model$train(iterations, trace)
	return(model)
}

if (FALSE) {
	mod = boostLinear(data = iris, target = "Sepal.Length", loss = QuadraticLoss$new())
}