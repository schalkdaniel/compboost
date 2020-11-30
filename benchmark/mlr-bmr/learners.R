## Learners:
## ---------------------

lrn_cboost1 = lrn("classif.compboost", id = "ps_cboost1", oob_fraction = 0.5,
  use_stopper = TRUE, silent = TRUE, mstop = 5000, ncores = 4L,
  predict_type = "prob", patience = 5L)

lrn_cboost2 = lrn("classif.compboost", id = "ps_cboost_nesterov1",
  oob_fraction = 0.5, use_stopper = TRUE, silent = TRUE, mstop = 5000,
  ncores = 4L,  predict_type = "prob", patience = 5L)

lrn_cboost_bin1 = lrn("classif.compboost", id = "ps_cboost2", oob_fraction = 0.5,
  use_stopper = TRUE, silent = TRUE, mstop = 5000, ncores = 4L,
  predict_type = "prob", patience = 5L, bin_root = 2L)

lrn_cboost_bin2 = lrn("classif.compboost", id = "ps_cboost_nesterov2",
  oob_fraction = 0.5, use_stopper = TRUE, silent = TRUE, mstop = 5000, ncores = 4L,
  predict_type = "prob", patience = 5L, bin_root = 2L)

lrn_xgboost = lrn("classif.xgboost", id = "ps_xgboost", predict_type = "prob")

lrn_gamboost = lrn("classif.gamboost", id = "ps_gamboost", predict_type = "prob")

lrn_rpart = lrn("classif.rpart", id = "ps_rpart", predict_type = "prob")

lrn_ranger = lrn("classif.ranger", id = "ps_ranger", predict_type = "prob")

learners = list(
  lrn_cboost1,
  lrn_cboost2,
  lrn_cboost_bin1,
  lrn_cboost_bin2,
  lrn_xgboost,
  lrn_gamboost,
  lrn_rpart,
  lrn_ranger)

learners = lapply(learners, function (l) {
  l$encapsulate = c(train = "evaluate")
  l$fallback = lrn("classif.featureless")

  if (l$id == "ps_xgboost") l = po("encode", method = "one-hot") %>>% l

  l
})
