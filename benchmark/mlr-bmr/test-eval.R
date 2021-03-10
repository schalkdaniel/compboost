library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3pipelines)
library(paradox)

base_dir = here::here()
bm_dir = paste0(base_dir, "/benchmark/mlr-bmr/")

library(R6)
source(paste0(bm_dir, "classifCompboost.R"))
source(paste0(bm_dir, "regrCompboost.R"))

resampling_outer = rsmp("cv", folds = 10)
measure_classif = msr("classif.auc")

requireNamespace("mlr3oml")
ts = tsk("oml", task_id = 31L)

classif_lrn_xgboost = po("encode", method = "one-hot") %>>%
  lrn("classif.xgboost", id = "ps_xgboost", predict_type = "prob",
    # OpenML values:
    eta	= 0.00141919552592143,
    max_depth	= 13,
    min_child_weight	= 1.43663223506804,
    subsample	= 0.491642977297306,
    colsample_bytree	= 0.391775049036369,
    colsample_bylevel	= 0.238421217072755,
    lambda	= 0.0888615471667706,
    alpha	= 0.0088027484174215,
    nthread	= 1,
    nrounds	= 3332)

classif_lrn_ranger = lrn("classif.ranger", id = "ps_xgboost", predict_type = "prob",
    # OpenML values:
    num.trees	= 554,
    mtry = 1,
    min.node.size	= 2,
    sample.fraction	= 0.517814406077377)

classif_lrn_cboost = lrn("classif.compboost", id = "ps_cboost1",  ncores = 4, predict_type = "prob",
  df = 5, df_cat = 5, mstop = 2000, learning_rate = 0.01)


robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")

lrn_final = GraphLearner$new(robustify %>>% classif_lrn_xgboost)
lrn_final = GraphLearner$new(robustify %>>% classif_lrn_ranger)
lrn_final = GraphLearner$new(robustify %>>% classif_lrn_cboost)

rr = resample(task = ts, learner = lrn_final, resampling = resampling_outer)
rr$score(measure_classif)
rr$aggregate(measure_classif)


