## Learners:
## ---------------------

ncores = parallel::detectCores()

### Classification

classif_lrn_cboost1 = lrn("classif.compboost", id = "ps_cboost1",  ncores = ncores, predict_type = "prob")

classif_lrn_cboost2 = lrn("classif.compboost", id = "ps_cboost_nesterov1",
  use_stopper = TRUE, ncores = ncores,  predict_type = "prob", patience = 5L,
  optimizer = "nesterov")

classif_lrn_cboost3 = lrn("classif.compboost", id = "ps_cboost_nesterov1_norestart",
  ncores = ncores,  predict_type = "prob", optimizer = "nesterov", restart = FALSE)

classif_lrn_cboost4 = lrn("classif.compboost", id = "ps_cboost_anneal1",
  ncores = ncores,  predict_type = "prob", optimizer = "cos-anneal")

classif_lrn_cboost_bin1 = lrn("classif.compboost", id = "ps_cboost2",
  ncores = ncores, predict_type = "prob", bin_root = 2L)

classif_lrn_cboost_bin2 = lrn("classif.compboost", id = "ps_cboost_nesterov2",
  use_stopper = TRUE, ncores = ncores, predict_type = "prob", patience = 5L,
  bin_root = 2L, optimizer = "nesterov")

classif_lrn_cboost_bin3 = lrn("classif.compboost", id = "ps_cboost_nesterov2_norestart",
  ncores = ncores, predict_type = "prob", bin_root = 2L, optimizer = "nesterov", restart = FALSE)

classif_lrn_cboost_bin4 = lrn("classif.compboost", id = "ps_cboost_anneal2",
  ncores = ncores, predict_type = "prob", bin_root = 2L, optimizer = "cos-anneal")

classif_lrn_xgboost = lrn("classif.xgboost", id = "ps_xgboost", predict_type = "prob", nthread = ncores)

classif_lrn_gamboost = lrn("classif.gamboost", id = "ps_gamboost", predict_type = "prob")

classif_lrn_rpart = lrn("classif.rpart", id = "ps_rpart", predict_type = "prob")

classif_lrn_ranger = lrn("classif.ranger", id = "ps_ranger", predict_type = "prob")

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
  l$fallback = lrn("classif.featureless")

  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l
  return(l)
})

