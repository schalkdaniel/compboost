## Learners:
## ---------------------
updatePars = function(lrn, hps_add) {
  hps_old = lrn$param_set$values
  for (hp in names(hps_add)) {
    hps_old[[hp]] = hps_add[[hp]]
  }
  return(hps_old)
}
ncores = parallel::detectCores()
cwb_pars = list(
  patience = 10L,
  oob_fraction = 0.67,
  mstop = 5000L,
  oob_seed = 1618,
  eps_for_break = 0.00001,
  use_stopper = TRUE,
  ncores = ncores,
  stop_both = TRUE)


### Classification

### CWB tune all:
classif_lrn_cboost1 = lrn("classif.compboost", id = "ps_cboost1",  ncores = ncores, predict_type = "prob")

classif_lrn_cboost_bin1 = lrn("classif.compboost", id = "ps_cboost2",
  ncores = ncores, predict_type = "prob", bin_root = 2L)


### CWB CA  tune all
classif_lrn_cboost4 = lrn("classif.compboost", id = "ps_cboost_anneal1",
  ncores = ncores,  predict_type = "prob", optimizer = "cos-anneal")

classif_lrn_cboost_bin4 = lrn("classif.compboost", id = "ps_cboost_anneal2",
  ncores = ncores, predict_type = "prob", bin_root = 2L, optimizer = "cos-anneal")


### ACWB tune all:
classif_lrn_cboost2 = lrn("classif.compboost", id = "ps_cboost_nesterov1",
  use_stopper = TRUE, ncores = ncores,  predict_type = "prob", patience = 5L,
  optimizer = "nesterov", momentum = 0.05, oob_fraction = 0.67)

classif_lrn_cboost_bin2 = lrn("classif.compboost", id = "ps_cboost_nesterov2",
  use_stopper = TRUE, ncores = ncores, predict_type = "prob", patience = 5L,
  bin_root = 2L, optimizer = "nesterov", momentum = 0.05, oob_fraction = 0.67)


### ACWB  tune all
classif_lrn_cboost3 = lrn("classif.compboost", id = "ps_cboost_nesterov1_norestart",
  ncores = ncores,  predict_type = "prob", optimizer = "nesterov", restart = FALSE,
  momentum = 0.001)

classif_lrn_cboost_bin3 = lrn("classif.compboost", id = "ps_cboost_nesterov2_norestart",
  ncores = ncores, predict_type = "prob", bin_root = 2L, optimizer = "nesterov", restart = FALSE,
  momentum = 0.001)


### CWB - tuning
classif_lrn_cwb = lrn("classif.compboost", id = "ps_cwb1", predict_type = "prob",
  optimizer = "cod", restart = FALSE)
classif_lrn_cwb$param_set$values = updatePars(classif_lrn_cwb, cwb_pars)

classif_lrn_cwb_bin = lrn("classif.compboost", id = "ps_cwb1_bin", predict_type = "prob",
  optimizer = "cod", restart = FALSE, bin_root = 2L)
classif_lrn_cwb_bin$param_set$values = updatePars(classif_lrn_cwb_bin, cwb_pars)

### ACWB - tuning
classif_lrn_acwb = lrn("classif.compboost", id = "ps_cwb2", predict_type = "prob",
  optimizer = "nesterov", restart = FALSE, momentum = 0.0034)
classif_lrn_acwb$param_set$values = updatePars(classif_lrn_acwb, cwb_pars)

classif_lrn_acwb_bin = lrn("classif.compboost", id = "ps_cwb2_bin", predict_type = "prob",
  optimizer = "nesterov", restart = FALSE, momentum = 0.0034, bin_root = 2L)
classif_lrn_acwb_bin$param_set$values = updatePars(classif_lrn_acwb_bin, cwb_pars)

### HCWB - tuning
classif_lrn_hcwb = lrn("classif.compboost", id = "ps_cwb3", predict_type = "prob",
  optimizer = "nesterov", restart = TRUE, momentum = 0.03)
classif_lrn_hcwb$param_set$values = updatePars(classif_lrn_hcwb, cwb_pars)

classif_lrn_hcwb_bin = lrn("classif.compboost", id = "ps_cwb3_bin", predict_type = "prob",
  optimizer = "nesterov", restart = TRUE, momentum = 0.03, bin_root = 2L)
classif_lrn_hcwb_bin$param_set$values = updatePars(classif_lrn_hcwb_bin, cwb_pars)

### CWB - no tuning
classif_lrn_cwb_notune = lrn("classif.compboost", id = "ps_cwb4_notune", predict_type = "prob",
  optimizer = "cod", restart = FALSE, learning_rate = 0.1, df_autoselect = TRUE)
classif_lrn_cwb_notune$param_set$values = updatePars(classif_lrn_cwb_notune, cwb_pars)

classif_lrn_cwb_notune_bin = lrn("classif.compboost", id = "ps_cwb4_notune_bin", predict_type = "prob",
  optimizer = "cod", restart = FALSE, learning_rate = 0.1, df_autoselect = TRUE, bin_root = 2L)
classif_lrn_cwb_notune_bin$param_set$values = updatePars(classif_lrn_cwb_notune_bin, cwb_pars)

### ACWB - no tuning
classif_lrn_acwb_notune = lrn("classif.compboost", id = "ps_cwb5_notune", predict_type = "prob",
  optimizer = "nesterov", restart = FALSE, learning_rate = 0.01, momentum = 0.0034, df_autoselect = TRUE)
classif_lrn_acwb_notune$param_set$values = updatePars(classif_lrn_acwb_notune, cwb_pars)

classif_lrn_acwb_notune_bin = lrn("classif.compboost", id = "ps_cwb5_notune_bin", predict_type = "prob",
  optimizer = "nesterov", restart = FALSE, learning_rate = 0.01, momentum = 0.0034, df_autoselect = TRUE,
  bin_root = 2L)
classif_lrn_acwb_notune_bin$param_set$values = updatePars(classif_lrn_acwb_notune_bin, cwb_pars)

### HCWB - no tuning
classif_lrn_hcwb_notune = lrn("classif.compboost", id = "ps_cwb6_notune", predict_type = "prob",
  optimizer = "nesterov", restart = TRUE, learning_rate = 0.01, momentum = 0.03, df_autoselect = TRUE)
classif_lrn_hcwb_notune$param_set$values = updatePars(classif_lrn_hcwb_notune, cwb_pars)

classif_lrn_hcwb_notune_bin = lrn("classif.compboost", id = "ps_cwb6_notune_bin", predict_type = "prob",
  optimizer = "nesterov", restart = TRUE, learning_rate = 0.01, momentum = 0.03, df_autoselect = TRUE,
  bin_root = 2L)
classif_lrn_hcwb_notune_bin$param_set$values = updatePars(classif_lrn_hcwb_notune_bin, cwb_pars)



classif_lrn_xgboost = lrn("classif.xgboost", id = "ps_xgboost", predict_type = "prob", nthread = ncores)

classif_lrn_gamboost = lrn("classif.gamboost", id = "ps_gamboost", predict_type = "prob")

classif_lrn_rpart = lrn("classif.rpart", id = "ps_rpart", predict_type = "prob")

classif_lrn_ranger = lrn("classif.ranger", id = "ps_ranger", predict_type = "prob", num.threads = ncores)

classif_lrn_interpretML = lrn("classif.interpretML_reticulate", id = "ps_interpretML",
  predict_type = "prob", n_jobs = ncores)

learners_classif = list(
  classif_lrn_cboost1 = classif_lrn_cboost1,
  classif_lrn_cboost2 = classif_lrn_cboost2,
  classif_lrn_cboost3 = classif_lrn_cboost3,
  classif_lrn_cboost4 = classif_lrn_cboost4,
  classif_lrn_cboost_bin1 = classif_lrn_cboost_bin1,
  classif_lrn_cboost_bin2 = classif_lrn_cboost_bin2,
  classif_lrn_cboost_bin3 = classif_lrn_cboost_bin3,
  classif_lrn_cboost_bin4 = classif_lrn_cboost_bin4,

  classif_lrn_cwb = classif_lrn_cwb,
  classif_lrn_cwb_bin = classif_lrn_cwb_bin,

  classif_lrn_acwb = classif_lrn_acwb,
  classif_lrn_acwb_bin = classif_lrn_acwb_bin,

  classif_lrn_hcwb = classif_lrn_hcwb,
  classif_lrn_hcwb_bin = classif_lrn_hcwb_bin,

  classif_lrn_cwb_notune = classif_lrn_cwb_notune,
  classif_lrn_cwb_notune_bin = classif_lrn_cwb_notune_bin,

  classif_lrn_acwb_notune = classif_lrn_acwb_notune,
  classif_lrn_acwb_notune_bin = classif_lrn_acwb_notune_bin,

  classif_lrn_hcwb_notune = classif_lrn_hcwb_notune,
  classif_lrn_hcwb_notune_bin = classif_lrn_hcwb_notune_bin,

  classif_lrn_xgboost = classif_lrn_xgboost,
  classif_lrn_gamboost = classif_lrn_gamboost,
  classif_lrn_rpart = classif_lrn_rpart,
  classif_lrn_ranger = classif_lrn_ranger,
  classif_lrn_interpretML = classif_lrn_interpretML)

if ("config" %in% ls()) {
  learners_classif = learners_classif[config$learner]
}

learners_classif = lapply(learners_classif, function(l) {
  l$encapsulate = c(train = "evaluate", predict = "evaluate")
  #l$encapsulate = "none"
  l$fallback = lrn("classif.featureless")

  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l
  return(l)
})

