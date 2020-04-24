#' @export
testBoostSplineIris = function () {
  boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new())
}
