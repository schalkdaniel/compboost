## Paramsets:
## ----------------------

ps_interpretML = function(task, id) {
  ParamSet$new(
    params = list(
      ParamDbl$new(paste0(id, ".learning_rate"), lower = 0.001, upper = 0.5),
      ParamInt$new(paste0(id, ".max_rounds"), lower = 200L, upper = 5000L)
    )
  )
}

ps_xgboost = function(task, id) {
  ParamSet$new(
    params = list(
      ParamDbl$new(paste0(id, ".ps_xgboost.eta"), lower = 0.001, upper = 0.5),
      ParamInt$new(paste0(id, ".ps_xgboost.max_depth"), lower = 3L, upper = 20L),
      ParamInt$new(paste0(id, ".ps_xgboost.nrounds"), lower = 200L, upper = 5000L),
      ParamDbl$new(paste0(id, ".ps_xgboost.colsample_bytree"), lower = 0.5, upper = 1),
      ParamDbl$new(paste0(id, ".ps_xgboost.colsample_bylevel"), lower = 0.5, upper = 1),
      ParamDbl$new(paste0(id, ".ps_xgboost.subsample"), lower = 0.5, upper = 1),
      ParamDbl$new(paste0(id, ".ps_xgboost.gamma"), lower = 2^(-7), upper = 2^6)
      #ParamDbl$new(paste0(id, ".ps_xgboost.lambda"), lower = 0, upper = 1e4),
      #ParamDbl$new(paste0(id, ".ps_xgboost.alpha"), lower = 0, upper = 1e4)
    )
  )
}
ps_cboost = function(task, id) {
  ParamSet$new(
    params = list(
      ParamDbl$new(id = paste0(id, ".df"), lower = 2, upper = 10),
      ParamDbl$new(id = paste0(id, ".df_cat"), lower = 2, upper = 10),
      ParamDbl$new(id = paste0(id, ".learning_rate"), lower = 0.001, upper = 0.5),
      ParamInt$new(id = paste0(id, ".mstop"), lower = 200L, upper = 5000L)
      #ParamFct$new(id = "classif.compboost.optimizer", levels = c("cod", "nesterov")),
      #ParamDbl$new(id = "classif.compboost.momentum", lower = 0.000001, upper = 0.01)
    )
  )
}
ps_cboost_nesterov = function(task, id) {
  ParamSet$new(
    params = list(
      ParamDbl$new(id = paste0(id, ".df"), lower = 2, upper = 10),
      ParamDbl$new(id = paste0(id, ".df_cat"), lower = 2, upper = 10),
      ParamDbl$new(id = paste0(id, ".learning_rate"), lower = 0.001, upper = 0.5),
      ParamDbl$new(id = paste0(id, ".oob_fraction"), lower = 0.2, upper = 0.5),
      ParamInt$new(id = paste0(id, ".mstop"), lower = 200L, upper = 5000L),
      #ParamFct$new(id = paste0(id, ".optimizer"), levels = c("cod", "nesterov")),
      ParamDbl$new(id = paste0(id, ".momentum"), lower = 0.000001, upper = 0.1)
    )
  )
}

ps_gamboost = function(task, id) {
  ParamSet$new(
    params = list(
      ParamInt$new(id = paste0(id, ".dfbase"), lower = 3, upper = 10),
      ParamInt$new(id = paste0(id, ".mstop"), lower = 200, upper = 5000),
      ParamDbl$new(id = paste0(id, ".nu"), lower = 0.001, 0.5)
    )
  )
}

ps_ranger = function(task, id) {
  nfeat = length(task$feature_names)
  ncols = task$ncol
  if (nfeat > 0) {
    mtry_upper = nfeat
  } else {
    if (ncols > 0) {
      mtry_upper = ncols
    } else {
      stop("Either length(feature_names) or ncols must be at least 1")
    }
  }
  ParamSet$new(
    params = list(
      ParamInt$new(id = paste0(id, ".mtry"), lower = 1L, upper = mtry_upper),
      ParamInt$new(id = paste0(id, ".max.depth"), lower = 0, upper = 50),
      ParamInt$new(id = paste0(id, ".min.node.size"), lower = 1L, upper = 100L),
      ParamInt$new(id = paste0(id, ".num.trees"), lower = 200L, upper = 2000L)
    )
  )
}

ps_rpart = function(task, id) {
  ParamSet$new(
    params = list(
      ParamInt$new(id = paste0(id, ".minsplit"), lower = 1, upper = 100),
      ParamDbl$new(id = paste0(id, ".cp"), lower = 0, upper = 1),
      ParamInt$new(id = paste0(id, ".maxdepth"), lower = 1, upper = 30)
    )
  )
}
