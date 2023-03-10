#' @title Visualize partial effect of a feature
#'
#' @description
#' This function visualizes the contribution of a specific feature to the overall
#' prediction score. If multiple base learner of the same features are included,
#' they are all added to the graphic as well as the aggregated contribution. The
#' difference to [plotBaselearner()] is that potentially multiple base learners
#' that are based on `feat` are aggregated and visualized while [plotBaselearner()]
#' only visualizes the contribution of one specific base learner. The function
#' also automatically decides whether the given feature is numeric or categorical and
#' chooses an appropriate technique (lines for numeric and horizontal lines for categorical).
#'
#' @return `ggplot` object containing the graphic.
#' @param cboost ([Compboost])\cr
#'   A trained [Compboost] object.
#' @param feat (`character(1L)`)\cr
#'   Name of the feature.
#' @param npoints (`integer(1L)`)\cr
#'   Number of points which are predicted for the lines (only applies to numerical features).
#' @param individual (`logical(1L)`)\cr
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

  feats = unique(cboost$bl_factory_list$getDataNames())
  checkmate::assertChoice(x = feat, choices = feats)
  checkmate::assertIntegerish(x = npoints, len = 1L, lower = 10L)
  checkmate::assertLogical(x = individual, len = 1L)

  blnames = lapply(cboost$model$getFactoryMap(), function(blf) {
    feat %in% blf$getFeatureName()
  })
  blnames = names(blnames)[unlist(blnames)]

  f = cboost$baselearner_list[[blnames[1]]]$factory
  if (getBaselearnerFeatureType(f) == "numeric") {
    minmax = f$getMinMax()
    x = seq(minmax[1], minmax[2], length.out = npoints)
  } else {
    vals = do.call(c, lapply(cboost$baselearner_list, function(bl) bl$factory$getValueNames()[[1]]))
    x = unique(vals)
  }

  df_plt = data.frame(x = x)
  names(df_plt) = feat

  newdat  = suppressWarnings(cboost$prepareData(df_plt))

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

  .data = ggplot2::.data
  gg = ggplot2::ggplot()
  if (individual) {
    if (is.numeric(x)) {
      gg = gg +
        ggplot2::geom_line(data = df_ind, mapping = ggplot2::aes(x = .data$x, y = .data$y,
          color = .data$bl),  linewidth = 0.6) +
        ggplot2::geom_line(data = df_agg, mapping = ggplot2::aes(x = .data$x, y = .data$y,
          color = 'Aggregated Contribution'), linewidth = 1.2)
    } else {
      gg = gg +
        ggplot2::geom_boxplot(data = df_ind, mapping = ggplot2::aes(x = .data$x, y = .data$y,
          color = .data$bl), alpha = 0.6) +
        ggplot2::geom_boxplot(data = df_agg, mapping = ggplot2::aes(x = .data$x, y = .data$y,
          color = 'Aggregated Contribution'), size = 1.2)
    }
      gg = gg + ggplot2::labs(color = "Baselearner")
  } else {
    if (is.numeric(x)) {
      gg = gg + ggplot2::geom_line(data = df_agg, mapping = ggplot2::aes(x = .data$x, y = .data$y))
    } else {
      gg = gg + ggplot2::geom_boxplot(data = df_agg, mapping = ggplot2::aes(x = .data$x, y = .data$y))
    }
  }
  gg = gg +
    ggplot2::labs(color = "Baselearner") +
    ggplot2::xlab(feat) +
    ggplot2::ylab("Contribution to\nprediction scores")

  return(gg)
}

#' @title Visualize contribution of one base learner
#'
#' @description
#' This function visualizes the contribution of a base learner to the overall
#' prediction score. For visualization of partial effects see [plotPEUni()].
#'
#' @return `ggplot` object containing the graphic.
#' @param cboost ([Compboost])\cr
#'   A trained [Compboost] object.
#' @param blname (`character(1L)`)\cr
#'   Name of the base learner. Must be one of `cboost$getBaselearnerNames()`.
#' @param npoints (`integer(1L)`)\cr
#'   Number of points which are predicted for the lines (only applies to numerical features).
#' @examples
#' cboost = Compboost$new(data = iris, target = "Petal.Length",
#'   loss = LossQuadratic$new())
#' cboost$addComponents("Sepal.Width")
#' cboost$train(500L)
#' plotBaselearner(cboost, "Sepal.Width_linear")
#' plotBaselearner(cboost, "Sepal.Width_Sepal.Width_spline_centered")
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
  if (length(unique(cboost$baselearner_list[[blname]]$factory$getFeatureName())) > 1) {
    stop("`$plotBaselearner()` only works on univariate base learner")
  }

  feats = unique(cboost$bl_factory_list$getDataNames())
  feat  = feats[vapply(feats, FUN.VALUE = logical(1L), FUN = function(feat) grepl(feat, blname))]

  f = cboost$baselearner_list[[blname]]$factory
  if (getBaselearnerFeatureType(f) == "numeric") {
    minmax = f$getMinMax()
    x = seq(minmax[1], minmax[2], length.out = npoints)
  } else {
    x = names(f$getDictionary())
  }

  df_plt = data.frame(x = x)
  names(df_plt) = feat

  newdat = suppressWarnings(cboost$prepareData(df_plt))
  df_plt = data.frame(x = x, y = cboost$model$predictFactoryNewData(blname, newdat))

  .data = ggplot2::.data
  gg = ggplot2::ggplot(data = df_plt, mapping = ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::xlab(feat) +
    ggplot2::ylab("Contribution to\nprediction scores")

  if (! is.numeric(x)) {
    gg = gg + ggplot2::geom_boxplot()
  } else {
    gg = gg + ggplot2::geom_line()
  }
  return(gg)
}
