#' Visualize partial effect of a feature
#'
#' This function visualizes the contribution of a specific feature to the overall
#' prediction score. If multiple base learner of the same features are included,
#' they are all added to the graphic as well as the aggregated contribution. The
#' difference to \code{plotBaselearner} is that all base learners are visualized while
#' \code{plotBaselearner} only visualizes one specific base learner. The function
#' also automatically decides whether the given feature is numeric or categorical and
#' chooses an appropriate technique (lines for numeric and horizontal lines for categorical).
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @param feat [\code{character(1L)}]\cr
#'   Name of the feature.
#' @param npoints [\code{integer(1L)}]\cr
#'   Number of points which are predicted for the lines (only applies to numerical features).
#' @param individual [\code{logical(1L)}]\cr
#'   Flag whether individual base learners should be added to the graphic or not.
#' @examples
#' cboost = Compboost$new(data = iris, target = "Petal.Length",
#'   loss = LossQuadratic$new())
#' cboost$addComponents("Sepal.Width")
#' cboost$train(500L)
#' plotPEUni(cboost, "Sepal.Width")
#' @export
plotPEUni = function(cboost, feat, npoints = 100L, individual = TRUE) {
  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")
  checkmate::assertClass(cboost, "Compboost")
  if (is.null(cboost$model))
    stop("Model has not been trained!")

  if (! cboost$model$isTrained())
    stop("Model has not been trained!")

  feats = cboost$bl_factory_list$getDataNames()
  checkmate::assertChoice(x = feat, choices = feats)
  checkmate::assertIntegerish(x = npoints, len = 1L, lower = 10L)
  checkmate::assertLogical(x = individual, len = 1L)

  x = cboost$data[[feat]]
  if (! is.numeric(x))
    x = unique(x)
  else
    x = seq(min(x), max(x), length.out = npoints)

  df_plt = data.frame(x = x)
  names(df_plt) = feat

  newdat  = suppressWarnings(cboost$prepareData(df_plt))
  blnames = cboost$bl_factory_list$getRegisteredFactoryNames()
  blnames = blnames[feats == feat]

  blsel   = unique(cboost$getSelectedBaselearner())
  blnames = blnames[blnames %in% blsel]

  ll_plt = lapply(blnames, function(bln) {
    data.frame(
      x  = x,
      y  = cboost$model$predictFactoryNewData(bln, newdat),
      bl = bln)
  })
  df_ind = do.call(rbind, ll_plt)
  df_agg = data.frame(
    x  = x,
    y  = Reduce("+", lapply(ll_plt, function(ll) ll$y)))

  if (length(blnames) == 1) individual = FALSE

  gg = ggplot2::ggplot()
  if (individual) {
    if (is.numeric(x)) {
      gg = gg +
        ggplot2::geom_line(data = df_ind, mapping = ggplot2::aes(x = x, y = y, color = bl), linewidth = 0.6) +
        ggplot2::geom_line(data = df_agg, mapping = ggplot2::aes(x = x, y = y, color = 'Aggregated Contribution'), linewidth = 1.2)
    } else {
      gg = gg +
        ggplot2::geom_boxplot(data = df_ind, mapping = ggplot2::aes(x = x, y = y, color = bl), alpha = 0.6) +
        ggplot2::geom_boxplot(data = df_agg, mapping = ggplot2::aes(x = x, y = y, color = 'Aggregated Contribution'), size = 1.2)
    }
      gg = gg + ggplot2::labs(color = "Baselearner")
  } else {
    if (is.numeric(x)) {
      gg = gg + ggplo2::geom_line(data = df_agg, mapping = ggplot2::aes(x = x, y = y))
    } else {
      gg = gg + ggplot2::geom_boxplot(data = df_agg, mapping = ggplot2::aes(x = x, y = y))
    }
  }
  gg = gg +
    ggplot2::labs(color = "Baselearner") +
    ggplot2::xlab(feat) +
    ggplot2::ylab("Contribution to\nprediction scroes")

  return(gg)
}

#' Visualize contribution of one base learner
#'
#' This function visualizes the contribution of a base learner to the overall
#' prediction score. For visualization of partial effects see \code{plotPEUni}.
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @param blname [\code{character(1L)}]\cr
#'   Name of the base learner. Must be one of \code{cboost$getBaselearnerNames()}.
#' @param npoints [\code{integer(1L)}]\cr
#'   Number of points which are predicted for the lines (only applies to numerical features).
#' @examples
#' cboost = Compboost$new(data = iris, target = "Petal.Length",
#'   loss = LossQuadratic$new())
#' cboost$addComponents("Sepal.Width")
#' cboost$train(500L)
#' plotBaselearner(cboost, "Sepal.Width_linear")
#' plotBaselearner(cboost, "Sepal.Width_spline_centered")
#' @export
plotBaselearner = function(cboost, blname, npoints = 100L) {
  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")
  checkmate::assertClass(cboost, "Compboost")
  if (is.null(cboost$model))
    stop("Model has not been trained!")

  if (! cboost$model$isTrained())
    stop("Model has not been trained!")

  checkmate::assertChoice(x = blname, choices = cboost$getBaselearnerNames())
  checkmate::assertIntegerish(x = npoints, len = 1L, lower = 10L)

  feats = unique(cboost$bl_factory_list$getDataNames())
  feat  = feats[vapply(feats, FUN.VALUE = logical(1L), FUN = function(feat) grepl(feat, blname))]

  x = cboost$data[[feat]]
  if (! is.numeric(x))
    x = unique(x)
  else
    x = seq(min(x), max(x), length.out = npoints)

  df_plt = data.frame(x = x)
  names(df_plt) = feat

  newdat = suppressWarnings(cboost$prepareData(df_plt))
  df_plt = data.frame(x = x, y = cboost$model$predictFactoryNewData(blname, newdat))

  gg = ggplot2::ggplot(data = df_plt, mapping = ggplot2::aes(x = x, y = y)) +
    ggplot2::xlab(feat) +
    ggplot2::ylab("Contribution to\nprediction scores")

  if (! is.numeric(x)) {
    gg = gg + ggplot2::geom_boxplot()
  } else {
    gg = gg + ggplot2::geom_line()
  }
  return(gg)
}
