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
#' cboost$saveJson("cboost.json")
#' cboost2 = loadFromJson("cboost.json")
#' @export
loadFromJson = function(file)
{
  checkmate::assertFile(file, extension = c("json", "JSON", "Json"))

  cboost = Compboost$loadFromJson(file)
}
