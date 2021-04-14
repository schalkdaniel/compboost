tsks_classif = rbind(
  data.frame(type = "oml", name = "54"),           # Hepatitis
  data.frame(type = "oml", name = "37"),           # Diabetes
  data.frame(type = "oml", name = "4534"),         # Analcat Halloffame
  data.frame(type = "mlr", name = "spam"),         # Spam
  data.frame(type = "oml", name = "7592"),         # Adult
  data.frame(type = "oml", name = "168335"),       # MiniBooNE
  data.frame(type = "script", name = "albert"),    # Albert
  data.frame(type = "oml", name = "168337"),       # Guillermo
  data.frame(type = "oml", name = "359994")        # SF Police Incidents
)

learners = c(
  "classif_lrn_cboost1",            # CWB (without binning)
  "classif_lrn_cboost_bin1",        #     (with binning)
  "classif_lrn_cboost4",            # CWB cosine annealing (without binning)
  "classif_lrn_cboost_bin4",        #                      (with binning)
  "classif_lrn_cboost3",            # ACWB (without binning)
  "classif_lrn_cboost_bin3",        #      (with binning)
  "classif_lrn_cboost2",            # hCWB (without binning)
  "classif_lrn_cboost_bin2",        #      (with binning)
  "classif_lrn_xgboost",            # Boosted trees
  "classif_lrn_gamboost",           # CWB (mboost variant)
  "classif_lrn_ranger",             # Random forest
  "classif_lrn_interpretML"         # Interpret
)

extractStringBetween = function (str, left, right) {
  tmp = sapply(strsplit(str, left), function (x) x[2])
  sapply(strsplit(tmp, right), function (x) x[1])
}

getResampleInstance = function(task) {
  if (task$nrow <= 2000) {
    resampling_inner = rsmp("cv", folds = 3)
    resampling_outer = rsmp("repeated_cv", folds = 5, repeats = 10L)
  }
  if ((task$nrow <= 100000) && (task$nrow > 2000)) {
    resampling_inner = rsmp("cv", folds = 3)
    resampling_outer = rsmp("cv", folds = 5)
  }
  if ((task$nrow > 100000)) {
    resampling_inner = rsmp("holdout", ratio = 0.33)
    resampling_outer = rsmp("holdout", ratio = 0.33)
  }
  resampling_outer$instantiate(task)
  return(list(inner = resampling_inner, outer = resampling_outer))
}

suppressMessages(library(mlr3))
suppressMessages(library(mlr3tuning))
suppressMessages(library(mlrintermbo))
suppressMessages(library(mlr3learners))
suppressMessages(library(mlr3extralearners))
suppressMessages(library(mlr3pipelines))
suppressMessages(library(paradox))
suppressMessages(library(R6))

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")

source("learner-src/classifCompboost.R")
source("learner-src/classifInterpretML_reticulate.R")

base_dir = "~/repos/compboost/benchmark/mlr-bmr/"

pars = list(
  "classif_lrn_cboost1" = list(mstop = 3500L),
  "classif_lrn_cboost_bin1" = list(mstop = 3500L),
  "classif_lrn_cboost4" = list(mstop = 3500L),
  "classif_lrn_cboost_bin4" = list(mstop = 3500L),
  "classif_lrn_cboost3" = list(mstop = 3500L),
  "classif_lrn_cboost_bin3" = list(mstop = 3500L),
  "classif_lrn_cboost2" = list(mstop = 3500L),
  "classif_lrn_cboost_bin2" = list(mstop = 3500L),
  "classif_lrn_xgboost" = list(ps_xgboost.nrounds = 3500L, ps_xgboost.max_depth = 15L),
  "classif_lrn_gamboost" = list(mstop = 3500L),
  "classif_lrn_ranger" = list(num.trees = 1400L, max.depth = 0),
  "classif_lrn_interpretML" = list(max_rounds = 3500L)
)
pars_dim = list(
  "classif_lrn_cboost1" = 4,
  "classif_lrn_cboost_bin1" = 4,
  "classif_lrn_cboost4" = 4,
  "classif_lrn_cboost_bin4" = 4,
  "classif_lrn_cboost3" = 6,
  "classif_lrn_cboost_bin3" = 6,
  "classif_lrn_cboost2" = 6,
  "classif_lrn_cboost_bin2" = 6,
  "classif_lrn_xgboost" = 8,
  "classif_lrn_gamboost" = 3,
  "classif_lrn_ranger" = 4,
  "classif_lrn_interpretML" = 2
)


ll_est = list()
for (its in seq_len(nrow(tsks_classif))) {

  cat(its, "/", nrow(tsks_classif), "\n", sep = "")

  config = list(task = tsks_classif$name[its], type = tsks_classif$type[its], learner = "dummy")
  source("tasks.R")
  ts = tasks_classif[[1]]

  set.seed(31415L)
  res = getResampleInstance(ts)

  k = 1L
  train_times = numeric(length(learners))
  for (learner in learners) {
    cat("  Learner: ", learner, " ", sep = "")

    config = list(task = tsks_classif$name[its], type = tsks_classif$type[its], learner = learner)
    source("learners.R")

    lrn = learners_classif[[1]]$clone(deep = TRUE)
    if (learner == "classif_lrn_xgboost") {
      lrn$param_set$values = c(pars[[learner]], encode.method = "one-hot")
    } else {
      lrn$param_set$values = pars[[learner]]
    }
    lrn = GraphLearner$new(robustify %>>% lrn)

    lrn$train(ts)

    train_times[k] = res$outer$iters * (lrn$state$train_time * res$inner$iters * 50 * pars_dim[[learner]] + 1)
    cat(round(train_times[k] / 60 / 60, 4), "hours\n")
    k = k + 1L
  }
  ll_est[[as.character(tsks_classif$name[its])]] = data.frame(
    task = as.character(tsks_classif$name[its]),
    learner = learners,
    train_time = train_times)
}


