calculateFeatEffectData = function (cboost_obj, bl_list, blearner_name, iters, from, to, length_out)
{
  if (is.null(cboost_obj$model)) {
    stop("Model needs to be trained first.")
  }
  lapply(iters, function (i) checkmate::assertCount(i, null.ok = TRUE))
  checkmate::assertCount(length_out, positive = TRUE)
  checkmate::assertCharacter(blearner_name, len = 1, null.ok = TRUE)

  if (is.null(blearner_name)) {
    stop("Please specify a valid base-learner plus feature.")
  }
  if (! blearner_name %in% cboost_obj$getBaselearnerNames()) {
    stop("Your requested feature plus learner is not available. Check 'getBaselearnerNames()' for available learners.")
  }
  if (length(bl_list[[blearner_name]]$feature) > 1) {
    stop("Only univariate plotting is supported.")
  }
  # Check if selected base-learner includes the proposed one + check if iters is big enough:
  iter_min = which(cboost_obj$getSelectedBaselearner() == blearner_name)[1]
  if (! blearner_name %in% unique(cboost_obj$getSelectedBaselearner())) {
    stop("Requested base-learner plus feature was not selected.")
  } else {
    if (any(iters < iter_min)) {
      warning("Requested base-learner plus feature was first selected at iteration ", iter_min)
    }
  }
  feat_name = bl_list[[blearner_name]]$factory$getFeatureName()

  checkmate::assertNumeric(x = cboost_obj$data[[feat_name]], min.len = 2, null.ok = FALSE)
  checkmate::assertNumeric(from, lower =  min(cboost_obj$data[[feat_name]]), upper = max(cboost_obj$data[[feat_name]]), len = 1, null.ok = TRUE)
  checkmate::assertNumeric(to, lower =  min(cboost_obj$data[[feat_name]]), upper = max(cboost_obj$data[[feat_name]]), len = 1, null.ok = TRUE)

  if (is.null(from)) {
    from = min(cboost_obj$data[[feat_name]])
  }
  if (is.null(to)) {
    to = max(cboost_obj$data[[feat_name]])
  }
  plot_data = as.matrix(seq(from = from, to = to, length.out = length_out))
  feat_map  = bl_list[[blearner_name]]$factory$transformData(plot_data)

  # Create data.frame for plotting depending if iters is specified:
  if (! is.null(iters[1])) {
    preds = lapply(iters, function (x) {
      if (x >= iter_min) {
        return(feat_map %*% cboost_obj$model$getParameterAtIteration(x)[[blearner_name]])
      } else {
        return(rep(0, length_out))
      }
    })
    names(preds) = iters

    df_plot = data.frame(
      effect    = unlist(preds),
      iteration = as.factor(rep(iters, each = length_out)),
      feature   = plot_data
    )
  } else {
    df_plot = data.frame(
      effect  = feat_map %*% cboost_obj$getEstimatedCoef()[[blearner_name]],
      feature = plot_data
    )
  }
  return(df_plot)
}

plotFeatEffect = function (cboost_obj, bl_list, blearner_name, iters, from, to, length_out)
{
  df_plot = calculateFeatEffectData(cboost_obj = cboost_obj, bl_list = bl_list, blearner_name = blearner_name,
    iters = iters, from = from, to = to, length_out = length_out)

  # Use aes_string to avoid check note:
  # > checking R code for possible problems ... NOTE
  # >   plotFeatEffect: no visible binding for global variable ‘feature’
  # >   plotFeatEffect: no visible binding for global variable ‘effect’
  # >   plotFeatEffect: no visible binding for global variable ‘iteration’
  if (! is.null(iters[1])) {
    gg = ggplot2::ggplot(df_plot, ggplot2::aes_string("feature", "effect", color = "iteration"))
  } else {
    gg = ggplot2::ggplot(df_plot, ggplot2::aes_string("feature", "effect"))
  }
  # If there are too much rows we need to take just a sample or completely remove rugs:
  if (nrow(cboost_obj$data) > 1000) {
    idx_rugs = sample(seq_len(nrow(cboost_obj$data)), 1000, FALSE)
  } else {
    idx_rugs = seq_len(nrow(cboost_obj$data))
  }

  feat_name = bl_list[[blearner_name]]$factory$getFeatureName()
  from = min(df_plot$feature)
  to = max(df_plot$feature)

  gg = gg +
    ggplot2::geom_line() +
    ggplot2::geom_rug(data = cboost_obj$data[idx_rugs,,drop=FALSE], ggplot2::aes_string(x = feat_name), inherit.aes = FALSE,
      alpha = 0.8) +
    ggplot2::xlab(feat_name) +
    ggplot2::xlim(from, to) +
    ggplot2::ylab("Additive Contribution") +
    ggplot2::labs(title = paste0("Effect of ", blearner_name),
      subtitle = "Additive contribution of predictor")

  return(gg)
}

plotBlearnerTraces = function (cboost_obj, value = 1, n_legend = 5L)
{
  if (! requireNamespace("ggplot2", quietly = TRUE)) { stop("Please install ggplot2 to create plots.") }
  if (! requireNamespace("ggrepel", quietly = TRUE)) { stop("Please install ggrepel to create plots.") }

  if (is.null(cboost_obj$model)) stop("Model needs to be trained first.")

  # Creating the base dataframe which is used to calculate the traces for the selected base-learner:
  bl       = as.factor(cboost_obj$getSelectedBaselearner())
  df_plot  = data.frame(iters = seq_along(bl), blearner = bl, value = value)

  if (length(value) %in% c(1L, length(bl))) {
    checkmate::assertNumeric(value)
  } else {
    stop("Assertion on 'value' failed: Must have length 1 or ", length(bl), ".")
  }
  checkmate::assertCount(n_legend, positive = TRUE)

  # Aggregate value by calculating the cumulative sum grouped by base-learner:
  df_plot = do.call(rbind, lapply(X = levels(bl), FUN = function (lab) {
    df_temp = df_plot[df_plot$blearner == lab, ]
    df_temp = df_temp[order(df_temp$iters), ]
    df_temp$value = cumsum(df_temp$value) / length(bl)

    return(df_temp)
  }))

  # Get top 'n_legend' base-learner that are highlighted:
  top_values = vapply(X = levels(bl), FUN.VALUE = numeric(1L), FUN = function (lab) {
    df_temp = df_plot[df_plot$blearner == lab, ]
    return (max(df_temp$value))
  })
  top_labs = as.factor(names(sort(top_values, decreasing = TRUE)))[seq_len(n_legend)]

  idx_top_lab = df_plot$blearner %in% top_labs

  df_plot_top    = df_plot[idx_top_lab, ]
  df_plot_nottop = df_plot[! idx_top_lab, ]

  df_label = do.call(rbind, lapply(X = top_labs, FUN = function (lab) {
    df_temp = df_plot[df_plot$blearner == lab, ]
    df_temp[which.max(df_temp$iters), ]
  }))

  gg = ggplot2::ggplot() +
    ggplot2::geom_line(data = df_plot_top, ggplot2::aes(x = iters, y = value, color = blearner), show.legend = FALSE) +
    ggplot2::geom_line(data = df_plot_nottop, ggplot2::aes(x = iters, y = value, group = blearner), alpha = 0.2, show.legend = FALSE) +
    ggrepel::geom_label_repel(data = df_label, ggplot2::aes(x = iters, y = value, label = round(value, 4), fill = blearner),
      colour = "white", fontface = "bold", show.legend = TRUE) +
    ggplot2::xlab("Iteration") +
    ggplot2::ylab("Cumulated Value\nof Included Base-Learner") +
    ggplot2::scale_fill_discrete(name = paste0("Top ", n_legend, " Base-Learner")) +
    ggplot2::guides(color = FALSE)

  return(gg)
}
