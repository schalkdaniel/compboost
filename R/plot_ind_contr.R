#' Decompose the predicted value based on the given features
#'
#' This function visualizes the contribution of each feature regarding the predicted value.
#' By default, multiple base learners defined on one feature are aggregated. If you
#' want to show the contribution of single base learner, then set \code{aggregate = FALSE}.
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @param newdata [\code{data.frame}]\cr
#'   Data frame containing exactly one row holding the new observations.
#' @param aggregate [\code{logical(1L)}]\cr
#'   Number of colored base learners added to the legend.
#' @param colbreaks [\code{numeric()}]\cr
#'   Breaks to visualize/highlight different predicted values. Default creates different
#'   colors for positive and negative score values. If set to \code{NULL} no coloring
#'   is applied.
#' @param collabels [\code{character(length(colbreaks) - 1)}]\cr
#'   Labels for the color breaks. If set to \code{NULL} intervals are used as labels.
#' @param nround [\code{integer(1L)}]\cr
#'   Digit passed to \code{round} for labels (default is \code{nround = 2L}).
#' @param offset [\code{logical(1L)}]\cr
#'   Flag to indicate whether the offset should be added to the figure or not.
#' @examples
#' dat = mtcars
#' fnum = c("cyl", "disp", "hp", "drat", "wt", "qsec")
#' fcat = c("vs", "am", "gear", "carb")
#' for (fn in fcat) dat[[fn]] = as.factor(dat[[fn]])
#'
#' cboost = Compboost$new(data = dat, target = "mpg",
#'   loss = LossQuadratic$new())
#'
#' for (fn in fnum) cboost$addComponents(fn, df = 3)
#' for (fn in fcat) cboost$addBaselearner(fn, "ridge", BaselearnerCategoricalRidge)
#' cboost$train(500L)
#' cbreaks = c(-Inf, -0.1, 0.1, Inf)
#' clabs   = c("bad", "middle", "good")
#' plotIndividualContribution(cboost, dat[10, ], colbreaks = cbreaks, collabels = clabs)
#' plotIndividualContribution(cboost, dat[10, ], offset = FALSE, colbreaks = cbreaks, collabels = clabs)
#' @export
plotIndividualContribution = function(cboost, newdata, aggregate = TRUE, colbreaks = c(-Inf, 0, Inf),
  collabels = c("negative", "positive"), nround = 2L, offset = TRUE) {

  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")
  checkmate::assertClass(cboost, "Compboost")
  checkmate::assertLogical(aggregate, len = 1L)

  if (! is.null(colbreaks))
    checkmate::assertNumeric(colbreaks, finite = FALSE, min.len = 2L)

  if (! is.null(collabels)) {
    if (is.null(colbreaks))
      checkmate::assertCharacter(collabels)
    else
      checkmate::assertCharacter(collabels, len = length(colbreaks) - 1)
  }

  checkmate::assertIntegerish(nround, len = 1L)
  checkmate::assertLogical(offset, len = 1L)

  if (is.null(cboost$model))
    stop("Model has not been trained!")

  if (! cboost$model$isTrained())
    stop("Model has not been trained!")

  feats   = cboost$bl_factory_list$getDataNames()
  blnames = cboost$bl_factory_list$getRegisteredFactoryNames()
  if (offset)
    df_bls  = data.frame(bl = c(blnames, "offset"), feat = c(feats, "offset"))
  else
    df_bls  = data.frame(bl = blnames, feat = feats)

  nuisance = lapply(unique(feats), function(fn) {
    checkmate::assertChoice(fn, choices = colnames(newdata))
  })
  checkmate::assertDataFrame(newdata, nrows = 1L)
  ll_ds    = cboost$prepareData(newdata)
  ll_preds = c(cboost$model$predictIndividual(ll_ds), offset = cboost$model$getOffset())
  df_preds = data.frame(bl = names(ll_preds), value = unname(unlist(ll_preds)))

  df_plt = merge(df_bls, df_preds, by = "bl", all.x = TRUE)
  df_plt$value[is.na(df_plt$value)] = 0

  fval = paste0("(", vapply(X = newdata, FUN.VALUE = character(1L), FUN = function(x) {
    if (is.numeric(x)) return(as.character(round(x, nround)))
    return(as.character(x))
  }), ")")
  if (offset)
    df_fval = data.frame(feat = c(colnames(newdata), "offset"), fval = c(fval, ""))
  else
    df_fval = data.frame(feat = colnames(newdata), fval = fval)

  if (aggregate) {
    df_plt = aggregate(x = df_plt$value, by = list(df_plt$feat), FUN = sum)
    names(df_plt) = c("feat", "value")
  } else {
    df_plt$feat = df_plt$bl
  }
  df_plt        = merge(df_plt, df_fval, by = "feat")
  df_plt$label  = paste0(df_plt$feat, " ", df_plt$fval)
  df_plt        = df_plt[order(df_plt$value, decreasing = TRUE), ]
  df_plt$bl_num = rev(seq_len(nrow(df_plt)))

  if (! is.null(colbreaks)) {
    if (! is.null(collabels))
      df_plt$cbreak = cut(df_plt$value, breaks = colbreaks, labels = collabels)
    else
      df_plt$cbreak = cut(df_plt$value, breaks = colbreaks)
  }

  pred = round(sum(df_plt$value), nround)
  subtitle = paste0("Score: ", pred)

  if (is.null(colbreaks[1]))
    gg = ggplot2::ggplot(df_plt, ggplot2::aes(x = value, y = bl_num))
  else
    gg = ggplot2::ggplot(df_plt, ggplot2::aes(x = value, y = bl_num, color = cbreak, fill = cbreak))

  gg = gg +
    ggplot2::geom_vline(xintercept = 0, color = "dark grey", alpha = 0.6) +
    ggplot2::geom_segment(ggplot2::aes(xend = 0, yend = bl_num)) +
    ggplot2::geom_point() +
    ggplot2::ylab("") +
    ggplot2::xlab("Contribution to predicted value") +
    ggplot2::labs(color = "", fill = "") +
    ggplot2::scale_y_continuous(labels = df_plt$label, breaks = df_plt$bl_num) +
    ggplot2::ggtitle("Prediction", subtitle)
  return(gg)
}
