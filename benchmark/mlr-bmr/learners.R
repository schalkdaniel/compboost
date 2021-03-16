## Learners:
## ---------------------

ncores = parallel::detectCores() - 2L
#load("config.Rda")

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

classif_lrn_xgboost = lrn("classif.xgboost", id = "ps_xgboost", predict_type = "prob")

classif_lrn_gamboost = lrn("classif.gamboost", id = "ps_gamboost", predict_type = "prob")

classif_lrn_rpart = lrn("classif.rpart", id = "ps_rpart", predict_type = "prob")

classif_lrn_ranger = lrn("classif.ranger", id = "ps_ranger", predict_type = "prob")

classif_lrn_interpretML = lrn("classif.interpretML", id = "ps_interpretML", predict_type = "prob")

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

learners_classif = learners_classif[config$learner]

learners_classif = lapply(learners_classif, function(l) {
  if (grepl(pattern = "cboost", x = l$id)) {
    l$encapsulate = c(train = "evaluate", predict = "evaluate")
    l$fallback = lrn("classif.featureless")
  }
  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l
  return(l)
})


### Regression

if (FALSE) {
regr_lrn_cboost1 = lrn("regr.compboost", id = "ps_cboost1", ncores = ncores)

regr_lrn_cboost2 = lrn("regr.compboost", id = "ps_cboost_nesterov1",
  use_stopper = TRUE, ncores = ncores, patience = 5L, optimizer = "nesterov")

regr_lrn_cboost_bin1 = lrn("regr.compboost", id = "ps_cboost2",  ncores = ncores, bin_root = 2L)

regr_lrn_cboost_bin2 = lrn("regr.compboost", id = "ps_cboost_nesterov2",
  use_stopper = TRUE, ncores = ncores, patience = 5L, bin_root = 2L,
  optimizer = "nesterov")

regr_lrn_xgboost = lrn("regr.xgboost", id = "ps_xgboost")

regr_lrn_gamboost = lrn("regr.gamboost", id = "ps_gamboost")

regr_lrn_rpart = lrn("regr.rpart", id = "ps_rpart")

regr_lrn_ranger = lrn("regr.ranger", id = "ps_ranger")

learners_regr = list(
  regr_lrn_cboost1,
  regr_lrn_cboost2,
  regr_lrn_cboost_bin1,
  regr_lrn_cboost_bin2,
  regr_lrn_xgboost,
  regr_lrn_gamboost,
  regr_lrn_rpart,
  regr_lrn_ranger)

learners_regr = lapply(learners_regr, function(l) {
  l$encapsulate = c(train = "evaluate", predict = "evaluate")
  if ("twoclass" %in% l$properties) {
    l$fallback = lrn("classif.featureless")
  } else {
    l$fallback = lrn("regr.featureless")
  }

  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l

  l
})
}
