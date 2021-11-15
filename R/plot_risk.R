#' Visualize the risk
#'
#' This function visualizes the risk during training. If validation data are given, then
#' the train risk is plotted against the validation risk.
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @examples
#' cboost_no_valdat = boostSplines(data = iris, target = "Sepal.Length",
#'   loss = LossQuadratic$new())
#' plotRisk(cboost_no_valdat)
#'
#' cboost_valdat = boostSplines(data = iris, target = "Sepal.Length",
#'   loss = LossQuadratic$new(), oob_fraction = 0.3)
#' plotRisk(cboost_valdat)
#' @export
plotRisk = function(cboost) {
  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")
  checkmate::assertClass(cboost, "Compboost")

  if (is.null(cboost$model))
    stop("Model has not been trained!")

  if (! cboost$model$isTrained())
    stop("Model has not been trained!")

  inbag_trace = cboost$getInbagRisk()
  oob_data = cboost$getLoggerData()

  if ("oob_risk" %in% names(oob_data)) {
    oob_trace = oob_data[["oob_risk"]]

    df_risk = data.frame(
      risk = c(inbag_trace, oob_trace),
      type = rep(c("inbag", "oob"), times = c(length(inbag_trace), length(oob_trace))),
      iter = c(seq_along(inbag_trace), seq_along(oob_trace))
    )

    gg = ggplot2::ggplot(df_risk, ggplot2::aes_string(x = "iter", y = "risk", color = "type"))
  } else {
    df_risk = data.frame(iter = seq_along(inbag_trace), risk = inbag_trace)
    gg = ggplot2::ggplot(df_risk, ggplot2::aes_string(x = "iter", y = "risk"))
  }
  gg = gg + ggplot2::geom_line(size = 1.1) +
    ggplot2::xlab("Iteration") +
    ggplot2::ylab("Risk")

  return(gg)
}
