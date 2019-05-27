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
  feat_name = bl_list[[blearner_name]]$target$getIdentifier()

  checkmate::assertNumeric(x = cboost_obj$data[[feat_name]], min.len = 2, null.ok = FALSE)
  checkmate::assertNumeric(from, lower =  min(cboost_obj$data[[feat_name]]), upper = max(cboost_obj$data[[feat_name]]), len = 1, null.ok = TRUE)
  checkmate::assertNumeric(to, lower =  min(cboost_obj$data[[feat_name]]), upper = max(cboost_obj$data[[feat_name]]), len = 1, null.ok = TRUE)

  if (is.null(from)) {
    from = min(cboost_obj$data[[feat_name]])
  }
  if (is.null(to)) {
    to = max(cboost_obj$data[[feat_name]])
  }
  

  # here we will need to loop through the baselearners
  plot_data = as.matrix(seq(from = from, to = to, length.out = 100))
  feat_map  = bl_list[[blearner_name]]$factory$transformData(plot_data)
  

  if(class(cboost_obj$response)[1] %in% c("Rcpp_ResponseFDA","Rcpp_ResponseFDALong")){
      if(class(cboost_obj$response)[1] == "Rcpp_ResponseFDA"){
      plot_data = as.matrix(seq(from = from, to = to, length.out = nrow(cboost_obj$grid_mat[[1]])))
      time_grid = seq(min(cboost_obj$response$getGrid()), max(cboost_obj$response$getGrid()),length.out = nrow(plot_data))
      
      coefs_extract = cboost_obj$getEstimatedCoef()[[blearner_name]]
      
      coef_mat = matrix(0, nrow = nrow(plot_data), ncol = nrow(plot_data))
      
      for(i in 1:length(time_grid)){
        for(j in 1:length(plot_data)){
          transformed_time = cboost_obj$time_spline$transformData(as.matrix(time_grid[i]))
          coef_mat[i,j] = bl_list[[blearner_name]]$factory$transformDataTime(as.matrix(plot_data[j]), transformed_time) %*% coefs_extract
          }
      }
      
      df_plot = list(coef_mat = coef_mat, time_grid = time_grid, plot_data = plot_data) 
      } 
    
    if(class(cboost_obj$response)[1] == "Rcpp_ResponseFDALong"){
    plot_data = as.matrix(seq(from = from, to = to, length.out = nrow(cboost_obj$grid_mat[[1]])))
    time_grid = seq(min(unlist(cboost_obj$response$getGrid_field())), max(unlist(cboost_obj$response$getGrid_field())),length.out = nrow(plot_data))
    
    coefs_extract = cboost_obj$getEstimatedCoef()[[blearner_name]]
    
    coef_mat = matrix(0, nrow = nrow(plot_data), ncol = nrow(plot_data))
    browser()
    for(i in 1:length(time_grid)){
      for(j in 1:length(plot_data)){
        transformed_time = cboost_obj$time_spline$transformData(as.matrix(time_grid[i]))
        coef_mat[i,j] = bl_list[[blearner_name]]$factory$transformDataTime(as.matrix(plot_data[j]), transformed_time) %*% coefs_extract
      }
    }
    df_plot = list(coef_mat = coef_mat, time_grid = time_grid, plot_data = plot_data) 
    
    }
  } else{
      # Create data.frame for plotting depending if iters is specified:
      if (! is.null(iters[1])) {
        preds = lapply(iters, function (x) {
          if (x >= iter.min) {
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
        coefs_extract = cboost_obj$getEstimatedCoef()
        df_plot = data.frame(
          effect  = feat_map %*% coefs_extract[[blearner_name]],
          feature = plot_data
        )
      }
        
  }
  return(df_plot)
}

plotFeatEffect = function (cboost_obj, bl_list, blearner_name, iters, from, to, length_out)
{

  df_plot = calculateFeatEffectData(cboost_obj = cboost_obj, bl_list = bl_list, blearner_name = blearner_name,
    iters = iters, from = from, to = to, length_out = length_out)

  if(!is.list(df_plot)){
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
    
    feat_name = bl_list[[blearner_name]]$target$getIdentifier()
    from = min(df_plot$feature)
    to = max(df_plot$feature)
    
    gg = gg +
      ggplot2::geom_line() +
      ggplot2::geom_rug(data = cboost_obj$data[idx_rugs,], ggplot2::aes_string(x = feat_name), inherit.aes = FALSE,
        alpha = 0.8) +
      ggplot2::xlab(feat_name) +
      ggplot2::xlim(from, to) +
      ggplot2::ylab("Additive Contribution") +
      ggplot2::labs(title = paste0("Effect of ", blearner_name),
        subtitle = "Additive contribution of predictor")
  } else{

    graphics::image(x = df_plot$time_grid, y = df_plot$plot_data, z = df_plot$coef_mat, col = heat.colors(100),
      main = paste("Effect of",blearner_name), xlab = "t", ylab = "Variable")
    contour.default(x = df_plot$time_grid, y = df_plot$plot_data, z = df_plot$coef_mat, add = TRUE)
    return()
  }


  return(gg)
}

