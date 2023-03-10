#' @title Visualize bivariate tensor products
#'
#' @description
#' This function visualizes the contribution of a bivariate tensor product.
#'
#' @return `ggplot` object containing the graphic.
#' @param cboost ([Compboost])\cr
#'   A trained \code{Compboost} object.
#' @param tname (`character(2L)`)\cr
#'   Name of the tensor base learner.
#' @param npoints (`integer(1L)`)\cr
#'   Number of grid points per numerical feature. Note: For two numerical features
#'   the overall number of grid points is `npoints^2`. For a numerical and
#'   categorical feature it is `npoints * ncat` with `ncat` the number
#'   of categories. For two categorical features `ncat^2` grid points are
#'   drawn.
#' @param nbins (`logical(1L)`)\cr
#'   Number of bins for the surface. Only applies in the case of two numerical features.
#'   A smooth surface is drawn if `nbins = NULL`.
#' @examples
#' cboost = Compboost$new(data = iris, target = "Petal.Length",
#'   loss = LossQuadratic$new())
#'
#' cboost$addTensor("Sepal.Width", "Sepal.Length", df1 = 4, df2 = 4)
#' cboost$addTensor("Sepal.Width", "Species", df1 = 4, df2 = 2)
#'
#' cboost$train(150L)
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

  #### REPLACE
  f = cboost$baselearner_list[[tname]]$factory
  feats = f$getFeatureName()
  df_raw = cboost$data[, feats]
  fclasses = sapply(df_raw, is.numeric)
  fvals = f$getValueNames()

  if (sum(fclasses) == 0) {
    x1unique = fvals[[feats[1]]]
    x2unique = fvals[[feats[2]]]
    df_prep = expand.grid(x1unique, x2unique)
    colnames(df_prep) = feats

    return(plotTensorCatCat(cboost, tname, df_prep))
  }
  if (sum(fclasses) == 1) {
    minmax = f$getMinMax()
    idxnum = which(fclasses)
    idxcat = which(!fclasses)
    if (idxnum == 1) {
      imm = c(1, 2)
    } else {
      imm = c(3, 4)
    }

    xnum = seq(minmax[imm[1]], minmax[imm[2]], length.out = npoints)

    xcat = fvals[[feats[idxcat]]]
    df_prep = expand.grid(xnum, xcat)
    colnames(df_prep) = feats[c(idxnum, idxcat)]

    return(plotTensorNumCat(cboost, tname, df_prep))
  }
  if (sum(fclasses) == 2) {
    minmax = f$getMinMax()

    x1 = seq(minmax[1], minmax[2], length.out = npoints)
    x2 = seq(minmax[3], minmax[4], length.out = npoints)

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

  .data = ggplot2::.data
  gg = ggplot2::ggplot()
  if (is.null(nbins)) {
    gg = gg +
      ggplot2::geom_raster(data = df, ggplot2::aes(x = .data[[feats[1]]], y = .data[[feats[2]]], fill = .data$y)) +
      ggplot2::labs(fill = "")
  } else {
    gg = gg +
      ggplot2::geom_contour_filled(data = df, ggplot2::aes(x = .data[[feats[1]]], y = .data[[feats[2]]], z = .data$y),
        bins = nbins) +
      ggplot2::labs(fill = "")
  }
  return(gg)
}

plotTensorNumCat = function(cboost, tname, df) {
  ll_ds = suppressWarnings(cboost$prepareData(df))
  feats = colnames(df)

  df$y = cboost$model$predictFactoryNewData(tname, ll_ds)

  .data = ggplot2::.data
  ggplot2::ggplot(data = df, ggplot2::aes(x = .data[[feats[1]]], y = .data$y, color = .data[[feats[2]]])) +
    ggplot2::geom_line() +
    ggplot2::ylab("Contribution to prediction")
}

plotTensorCatCat = function(cboost, tname, df) {
  ll_ds = suppressWarnings(cboost$prepareData(df))
  feats = colnames(df)

  df$y = cboost$model$predictFactoryNewData(tname, ll_ds)
  .data = ggplot2::.data
  ggplot2::ggplot(data = df, ggplot2::aes(x = .data[[feats[1]]], y = .data[[feats[2]]], fill = .data$y)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::labs(fill = "")
}
