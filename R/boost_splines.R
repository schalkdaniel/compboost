#' Wrapper to boost p spline models for each feature.
#'
#' This wrapper function automatically initializes the model by adding all numerical
#' features of a dataset within a spline base-learner. Categorical features are
#' dummy encoded and inserted using linear base-learners without intercept. After 
#' initializing the model \code{boostSpline} also fits as many iterations as given 
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
#' @param degree [\code{integer(1)}]\cr
#'   Polynomial degree of the splines used for modelling. Note that the number of parameter
#'   increases with the degrees.
#' @param n.knots [\code{integer(1)}]\cr
#'   Number of equidistant "inner knots". The real number of used knots also depends on
#'   the polynomial degree.
#' @param penalty [\code{numeric(1)}]\cr
#'   Penalty term for p-splines. If penalty equals 0, then ordinary b-splines are fitted.
#'   The higher penalty, the higher the smoothness.
#' @param differences [\code{integer(1)}]\cr
#'   Number of differences that are used for penalization. The higher this value is, the
#'   more function values of neighbour knots are forced to be more similar which results
#'   in a smoother courve.
#' @param data.source [\code{S4 Data}]\cr
#'   Unitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @param data.target [\code{S4 Data}]\cr
#'   Unitialized \code{S4 Data} object which is used to store the data. At the moment
#'   just in memory training is supported.
#' @examples
#' mod = boostSplines(data = iris, target = "Sepal.Length", loss = QuadraticLoss$new())
#' mod$getBaselearnerNames()
#' mod$coef()
#' table(mod$selected())
#' mod$predict()
#' mod$plot("Sepal.Width_spline")
#' @export
boostSplines = function(data, target, optimizer = CoordinateDescent$new(), loss, 
  learning.rate = 0.05, iterations = 100, trace = TRUE, degree = 3, n.knots = 20, 
  penalty = 2, differences = 2, data.source = InMemoryData, data.target = InMemoryData) 
{
  model = Compboost$new(data = data, target = target, loss = loss, learning.rate = learning.rate)
  features = setdiff(colnames(data), target)

  # This loop could be replaced with foreach???
  # Issue: 
  for(feat in features) {
    if (is.numeric(data[[feat]])) {
      model$addBaselearner(feat, "spline", PSplineBlearner, data.source, data.target,
        degree = degree, n.knots = n.knots, penalty = penalty, differences = differences)
    } else {
      model$addBaselearner(feat, "category", PolynomialBlearner, data.source, data.target,
        degree = 1, intercept = FALSE)
    }
  }
  model$train(iterations, trace)
  return(model)
}

if (FALSE) {
  mod = boostSplines(data = iris, target = "Sepal.Length", loss = QuadraticLoss$new())
}