#' Load Compboost model from JSON
#'
#' @description
#' Load a [Compboost] object from a JSON file. Because of the underlying \code{C++} objects,
#' it is not possible to use \code{R}'s native load and save methods.
#'
#' @return A model of the \code{Compboost} class. This model is an \code{R6} object
#'   which can be used for retraining, predicting, plotting, and anything described in
#'   \code{?Compboost}.
#' @param file (`character(1)`)\cr
#'   A data frame containing the data.
#' @examples
#' cboost = boostLinear(data = iris, target = "Sepal.Length", loss = LossQuadratic$new(),
#'   oob_fraction = 0.3)
#' cboost$model$saveJson("cboost.json")
#' cboost2 = loadFromJson("cboost.json")
#' file.remove("cboost.json")
#' @export
loadFromJson = function(file)
{
  checkmate::assertFile(file, extension = c("json", "JSON", "Json"))
  return(Compboost$new(file = file))
}

extractResponse = function(r) {
  if (r$getResponseType() == "regression") {
    rn = ResponseRegr$new(r)
    attr(rn, "positive") = NULL
    return(rn)
  }
  if (r$getResponseType() == "binary_classif") {
    rn = ResponseBinaryClassif$new(r)
    attr(rn, "positive") = rn$getPositiveClass()
    return(rn)
  }
  stop("Was not able to load response.")
}

extractOptimizer = function(op) {
  if (op$getOptimizerType() == "coo_descent") {
    return(OptimizerCoordinateDescent$new(op, TRUE))
  }
  if (op$getOptimizerType() == "coo_descent_ls") {
    return(OptimizerCoordinateDescentLineSearch$new(op, TRUE))
  }
  if (op$getOptimizerType() == "cosine_ann") {
    return(OptimizerCosineAnnealing$new(op, TRUE))
  }
  if (op$getOptimizerType() == "agbm") {
    return(OptimizerAGBM$new(op, TRUE, TRUE, TRUE))
  }
  stop("Was not able to load optimizer.")
}

extractLoss = function(l) {
  if (l$getLossType() == "quadratic") {
    return(LossQuadratic$new(l, TRUE, TRUE))
  }
  if (l$getLossType() == "absolute") {
    return(LossQuadratic$new(l, TRUE, TRUE))
  }
  if (l$getLossType() == "quantile") {
    return(LossQuadratic$new(l, TRUE, TRUE))
  }
  if (l$getLossType() == "huber") {
    return(LossQuadratic$new(l, TRUE, TRUE))
  }
  if (l$getLossType() == "binomial") {
    return(LossQuadratic$new(l, TRUE, TRUE))
  }
  stop("Was not able to load optimizer.")
}

extractBaselearnerFactory = function(blf) {
  if (blf$getModelName() == "polynomial") {
    return(BaselearnerPolynomial$new(blf))
  }
  if (blf$getModelName() == "pspline") {
    return(BaselearnerPSpline$new(blf))
  }
  if (blf$getModelName() == "tensor") {
    return(BaselearnerTensor$new(blf))
  }
  if (blf$getModelName() == "centered") {
    return(BaselearnerCentered$new(blf))
  }
  if (blf$getModelName() == "cridge") {
    return(BaselearnerCategoricalRidge$new(blf))
  }
  if (blf$getModelName() == "cbinary") {
    return(BaselearnerCategoricalBinary$new(blf))
  }
  stop("Was not able to load factory.")
}

extractData = function(d) {
  if (d$getDataType() == "in_memory") {
    return(InMemoryData$new(d))
  }
  if (d$getDataType() == "categorical") {
    return(CategoricalDataRaw$new(d))
  }
  stop("Was not able to load data object.")
}
