#' Visualize base learner traces
#'
#' This function shows how the base learners evolves over the fitting process.
#' The default is to show how the frequency of a single base learner included
#' into the model evolves. Additionally, with the \code{value} argument, vectors
#' (e.g. the risk) can be used to show how the base learner specific risk reduction
#' evolves during the fitting process.
#'
#' @return \code{ggplot} object containing the graphic.
#' @param cboost [\code{Compboost} class]\cr
#'   A trained \code{Compboost} object.
#' @param value [\code{numeric(1L) | numeric(length(cboost$getSelectedBaselearner()))}]\cr
#'   Value used to show the base learner development w.r.t. to the value.
#' @param n_legend [\code{integer(1L)}]\cr
#'   Number of colored base learners added to the legend.
#' @examples
#' cboost = Compboost$new(data = iris, target = "Petal.Length",
#'  loss = LossQuadratic$new())
#' cboost$addComponents("Sepal.Width")
#' cboost$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge)
#' cboost$train(500L)
#' plotBaselearnerTraces(cboost)
#' @export
plotBaselearnerTraces = function(cboost, value = 1, n_legend = 5L) {
  if (! requireNamespace("ggplot2", quietly = TRUE)) stop("Please install ggplot2 to create plots.")
  if (! requireNamespace("ggrepel", quietly = TRUE)) stop("Please install ggrepel to create plots.")

  if (is.null(cboost$model)) stop("Model needs to be trained first.")

  # Creating the base dataframe which is used to calculate the traces for the selected base-learner:
  bl       = as.factor(cboost$getSelectedBaselearner())
  df_plot  = data.frame(iters = seq_along(bl), blearner = bl, value = value)

  if (length(value) %in% c(1L, length(bl))) {
    checkmate::assertNumeric(value)
  } else {
    stop("Assertion on 'value' failed: Must have length 1 or ", length(bl), ".")
  }
  checkmate::assertCount(n_legend, positive = TRUE)

  # Aggregate value by calculating the cumulative sum grouped by base-learner:
  df_plot = do.call(rbind, lapply(X = levels(bl), FUN = function(lab) {
    df_temp = df_plot[df_plot$blearner == lab, ]
    df_temp = df_temp[order(df_temp$iters), ]
    df_temp$value = cumsum(df_temp$value) / length(bl)

    return(df_temp)
  }))

  # Get top 'n_legend' base-learner that are highlighted:
  top_values = vapply(X = levels(bl), FUN.VALUE = numeric(1L), FUN = function(lab) {
    df_temp = df_plot[df_plot$blearner == lab, ]
    return(max(df_temp$value))
  })
  top_labs = as.factor(names(sort(top_values, decreasing = TRUE)))[seq_len(n_legend)]

  idx_top_lab = df_plot$blearner %in% top_labs

  df_plot_top    = df_plot[idx_top_lab, ]
  df_plot_nottop = df_plot[! idx_top_lab, ]

  df_label = do.call(rbind, lapply(X = top_labs, FUN = function(lab) {
    df_temp = df_plot[df_plot$blearner == lab, ]
    df_temp[which.max(df_temp$iters), ]
  }))

  gg = ggplot2::ggplot() +
    ggplot2::geom_line(data = df_plot_top, ggplot2::aes(x = iters, y = value, color = blearner),
      show.legend = FALSE) +
    ggplot2::geom_line(data = df_plot_nottop, ggplot2::aes(x = iters, y = value, group = blearner),
      alpha = 0.2, show.legend = FALSE) +
    ggrepel::geom_label_repel(data = df_label, ggplot2::aes(x = iters, y = value, label = round(value, 4),
      fill = blearner), colour = "white", fontface = "bold", show.legend = TRUE) +
    ggplot2::xlab("Iteration") +
    ggplot2::ylab("Cumulated Value\nof Included Base-Learner") +
    ggplot2::scale_fill_discrete(name = paste0("Top ", n_legend, " Base-Learner")) +
    ggplot2::guides(color = "none")

  return(gg)
}
