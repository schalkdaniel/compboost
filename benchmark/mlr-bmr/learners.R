## Learners:
## ---------------------

ncores = 5L
oob_frac = 0.4
mstop = 20000L

### Classification

classif_lrn_cboost1 = lrn("classif.compboost", id = "ps_cboost1", oob_fraction = oob_frac,
  use_stopper = TRUE, silent = TRUE, mstop = mstop, ncores = ncores,
  predict_type = "prob", patience = 5L)

classif_lrn_cboost2 = lrn("classif.compboost", id = "ps_cboost_nesterov1",
  oob_fraction = oob_frac, use_stopper = TRUE, silent = TRUE, mstop = mstop,
  ncores = ncores,  predict_type = "prob", patience = 5L, optimizer = "nesterov")

classif_lrn_cboost_bin1 = lrn("classif.compboost", id = "ps_cboost2", oob_fraction = oob_frac,
  use_stopper = TRUE, silent = TRUE, mstop = mstop, ncores = ncores,
  predict_type = "prob", patience = 5L, bin_root = 2L)

classif_lrn_cboost_bin2 = lrn("classif.compboost", id = "ps_cboost_nesterov2",
  oob_fraction = 0.4, use_stopper = TRUE, silent = TRUE, mstop = mstop, ncores = ncores,
  predict_type = "prob", patience = 5L, bin_root = 2L, optimizer = "nesterov")

classif_lrn_xgboost = lrn("classif.xgboost", id = "ps_xgboost", predict_type = "prob")

classif_lrn_gamboost = lrn("classif.gamboost", id = "ps_gamboost", predict_type = "prob")

classif_lrn_rpart = lrn("classif.rpart", id = "ps_rpart", predict_type = "prob")

classif_lrn_ranger = lrn("classif.ranger", id = "ps_ranger", predict_type = "prob")

learners_classif = list(
  classif_lrn_cboost1,
  classif_lrn_cboost2,
  classif_lrn_cboost_bin1,
  classif_lrn_cboost_bin2,
  classif_lrn_xgboost,
  #classif_lrn_gamboost,
  classif_lrn_rpart,
  classif_lrn_ranger)

learners_classif = lapply(learners_classif, function (l) {
  l$encapsulate = c(train = "evaluate", predict = "evaluate")
  l$fallback = lrn("classif.featureless")

  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l

  l
})


### Regression

regr_lrn_cboost1 = lrn("regr.compboost", id = "ps_cboost1", oob_fraction = oob_frac,
  use_stopper = TRUE, silent = TRUE, mstop = 5000, ncores = ncores, patience = 5L)

regr_lrn_cboost2 = lrn("regr.compboost", id = "ps_cboost_nesterov1",
  oob_fraction = oob_frac, use_stopper = TRUE, silent = TRUE, mstop = 5000,
  ncores = ncores, patience = 5L, optimizer = "nesterov")

regr_lrn_cboost_bin1 = lrn("regr.compboost", id = "ps_cboost2", oob_fraction = oob_frac,
  use_stopper = TRUE, silent = TRUE, mstop = 5000, ncores = ncores,
  patience = 5L, bin_root = 2L)

regr_lrn_cboost_bin2 = lrn("regr.compboost", id = "ps_cboost_nesterov2",
  oob_fraction = oob_frac, use_stopper = TRUE, silent = TRUE, mstop = 5000, ncores = ncores,
  patience = 5L, bin_root = 2L, optimizer = "nesterov")

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
  #regr_lrn_gamboost,
  regr_lrn_rpart,
  regr_lrn_ranger)

learners_regr = lapply(learners_regr, function (l) {
  l$encapsulate = c(train = "evaluate", predict = "evaluate")
  if ("twoclass" %in% l$properties) { l$fallback = lrn("classif.featureless") } else { l$fallback = lrn("regr.featureless") }

  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l

  l
})
