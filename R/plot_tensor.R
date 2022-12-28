#' Visualize bivariate tensor products
#'
#' This function visualizes the contribution of a bivariate tensor product.
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @param tname [\code{character(2L)}]\cr
#'   Name of the tensor base learner.
#' @param npoints [\code{integer(1L)}]\cr
#'   Number of grid points per numerical feature. Note: For two numerical features
#'   the overall number of grid points is \code{npoints^2}. For a numerical and
#'   categorical feature it is \code{npoints * ncat} with \code{ncat} the number
#'   of categories. For two categorical features \code{ncat^2} grid points are
#'   drawn.
#' @param nbins [\code{logical(1L)}]\cr
#'   Number of bins for the surface. Only applies in the case of two numerical features.
#'   A smooth surface is drawn if \code{nbins = NULL}.
#' @examples
#' cboost = Compboost$new(data = iris, target = "Petal.Length",
#'       loss = LossQuadratic$new())
#'
#' cboost$addBaselearner("Sepal.Width", "spline", BaselearnerPSpline, df = 4)
#' cboost$addBaselearner("Sepal.Length", "spline", BaselearnerPSpline, df = 4)
#' cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge)
#' cboost$addTensor("Sepal.Width", "Sepal.Length", df1 = 4, df2 = 4)
#' cboost$addTensor("Sepal.Width", "Species", df1 = 4, df2 = 2)
#'
#' cboost$train(1000L)
#'
#' plotTensor(cboost, "Sepal.Width_Species_tensor")
#' plotTensor(cboost, "Sepal.Width_Sepal.Length_tensor")
#' plotTensor(cboost, "Sepal.Width_Sepal.Length_tensor", nbins = NULL)
#' @export
plotTensor = function(cboost, tname, npoints = 100L, nbins = 15L) {
  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")
  checkmate::assertClass(cboost, "Compboost")

  if (is.null(cboost$model))
    stop("Model has not been trained!")

  if (! cboost$model$isTrained())
    stop("Model has not been trained!")

  blsel = unique(cboost$getSelectedBaselearner())
  if (! checkmate::testChoice(x = tname, choices = blsel)) {
    stop("Tensor base learner '", tname, "' was not selected. The selected base learner are {",
      paste(paste0("'", blsel, "'"), collapse = ","), "}. Maybe you misspelled the base learner",
      "or did not train long enough.")
  }

  checkmate::assertIntegerish(x = npoints, len = 1L, lower = 10L)
  checkmate::assertIntegerish(x = nbins, len = 1L, lower = 5L, null.ok = TRUE)

  feats = colnames(cboost$data)
  feats = feats[sapply(feats, function(fn) grepl(fn, tname))]

  df_raw   = cboost$data[, feats]
  fclasses = sapply(df_raw, is.numeric)

  if (sum(fclasses) == 0) {
    x1unique = unique(df_raw[[1]])
    x2unique = unique(df_raw[[2]])
    df_prep = expand.grid(x1unique, x2unique)
    colnames(df_prep) = feats

    return(plotTensorCatCat(cboost, tname, df_prep))
  }
  if (sum(fclasses) == 1) {
    idxnum = which(fclasses)
    idxcat = which(!fclasses)

    xnum = df_raw[[idxnum]]
    xnum = seq(min(xnum), max(xnum), length.out = npoints)

    xcat = unique(df_raw[[idxcat]])
    df_prep = expand.grid(xnum, xcat)
    colnames(df_prep) = feats[c(idxnum, idxcat)]

    return(plotTensorNumCat(cboost, tname, df_prep))
  }
  if (sum(fclasses) == 2) {
    x1 = df_raw[[1]]
    x2 = df_raw[[2]]

    x1 = seq(min(x1), max(x1), length.out = npoints)
    x2 = seq(min(x2), max(x2), length.out = npoints)

    df_prep = expand.grid(x1, x2)
    colnames(df_prep) = feats

    return(plotTensorNumNum(cboost, tname, df_prep, nbins))
  }
  return(NULL)
}

plotTensorNumNum = function(cboost, tname, df, nbins) {
  ll_ds = suppressWarnings(cboost$prepareData(df))
  feats = colnames(df)

  df$y = cboost$model$predictFactoryNewData(tname, ll_ds)

  gg = ggplot2::ggplot()
  if (is.null(nbins)) {
    gg = gg +
      ggplot2::geom_raster(data = df, ggplot2::aes(x = ggplot2::.data[[feats[1]]],
        y = ggplot2::.data[[feats[2]]], fill = y)) +
      ggplot2::labs(fill = "")
  } else {
    gg = gg +
      ggplot2::geom_contour_filled(data = df, ggplot2::aes(x = ggplot2::.data[[feats[1]]],
        y = ggplot2::.data[[feats[2]]], z = y), bins = nbins) +
      ggplot2::labs(fill = "")
  }
  return(gg)
}

plotTensorNumCat = function(cboost, tname, df) {
  ll_ds = suppressWarnings(cboost$prepareData(df))
  feats = colnames(df)

  df$y = cboost$model$predictFactoryNewData(tname, ll_ds)

  ggplot2::ggplot(data = df, ggplot2::aes(x = ggplot2::.data[[feats[1]]], y = y, color = ggplot2::.data[[feats[2]]])) +
    ggplot2::geom_line() +
    ggplot2::ylab("Contribution to prediction")
}

plotTensorCatCat = function(cboost, tname, df) {
  ll_ds = suppressWarnings(cboost$prepareData(df))
  feats = colnames(df)

  df$y = cboost$model$predictFactoryNewData(tname, ll_ds)
  ggplot2::ggplot(data = df, ggplot2::aes(x = ggplot2::.data[[feats[1]]], y = ggplot2::.data[[feats[2]]], fill = y)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::labs(fill = "")
}
