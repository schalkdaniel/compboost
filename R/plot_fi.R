#' Visualize the feature importance
#'
#' This function visualizes the feature importance as horizontal bar plot.
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @param num_feats [\code{integer(1L)}]\cr
#'   Number of features that are visualized. All features are added if set to \code{NULL}.
#' @param aggregate [\code{logical(1L)}]\cr
#'   Flag whether the feature importance is aggregated by feature. Otherwise it is
#'   visualized per base learner.
#' @examples
#' cboost = boostSplines(data = iris, target = "Sepal.Length", loss = LossQuadratic$new())
#' plotFeatureImportance(cboost)
#' plotFeatureImportance(cboost, num_feats = 2)
#' plotFeatureImportance(cboost, num_feats = 2, aggregate = FALSE)
#' @export
plotFeatureImportance = function(cboost, num_feats = NULL, aggregate = TRUE) {
  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")

  checkmate::assertClass(cboost, "Compboost")
  checkmate::assertIntegerish(num_feats, len = 1L, null.ok = TRUE)
  checkmate::assertLogical(aggregate, len = 1L)

  if (is.null(cboost$model))
    stop("Model has not been trained!")

  if (! cboost$model$isTrained())
    stop("Model has not been trained!")

  if (is.null(num_feats)) {
    df_tmp = data.frame(
      feat = cboost$bl_factory_list$getDataNames(),
      bl   = cboost$bl_factory_list$getRegisteredFactoryNames())

    bl_sel = unique(cboost$getSelectedBaselearner())

    df_tmp = df_tmp[df_tmp$bl %in% bl_sel, ]
    num_feats = length(unique(df_tmp$feat))
  }
  df_vip = cboost$calculateFeatureImportance(num_feats, aggregate)

  ## First column containing the names contains the base learner or the feature depending on the aggregation.
  ## Therefore, set a general baselearner column used for ggplot:
  df_vip$baselearner = df_vip[[1]]
  gg = ggplot2::ggplot(df_vip, ggplot2::aes(x = reorder(ggplot2::.data$baselearner, ggplot2::.data$risk_reduction),
      y = ggplot2::.data$risk_reduction)) +
    ggplot2::geom_bar(stat = "identity") + ggplot2::coord_flip() + ggplot2::ylab("Importance") + ggplot2::xlab("")

  return(gg)
}
