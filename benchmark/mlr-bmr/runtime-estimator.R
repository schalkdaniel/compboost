load("config-runtime-estimator.Rda")

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

tsks_classif = tsks_classif[config_runtime$tidx, ]
learners = learners[config_runtime$lidx]

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

  cat(as.character(tsks_classif$name[its]), "\n", sep = "")

  config = list(task = tsks_classif$name[its], type = tsks_classif$type[its], learner = "dummy")
  temp = capture.output(source("tasks.R"))
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
    }
    for (pj in seq_along(pars[[learner]])) {
      lrn$param_set$values[[names(pars[[learner]])[pj]]] = pars[[learner]][[pj]]
    }
    lrn = GraphLearner$new(robustify %>>% lrn)

    tts = numeric(3L)
    for (j in seq_len(3L)) {
      lrn$train(ts$filter(res$outer$train_set(1L)))
      tts[j] = lrn$state$train_time
    }

    train_times[k] = res$outer$iters * (mean(tts) * res$inner$iters * 50 * pars_dim[[learner]] + 1)
    cat(round(train_times[k] / 60 / 60, 4), "hours\n")
    k = k + 1L
  }
}

if (FALSE) {

  nevals = list(
    "54" = list(outer = 50L, inner = 3L),
    "37" = list(outer = 50L, inner = 3L),
    "4534" = list(outer = 50L, inner = 3L),
    "spam" = list(outer = 5L, inner = 3L),
    "7592" = list(outer = 5L, inner = 3L),
    "168335" = list(outer = 1L, inner = 1L),
    "albert" = list(outer = 1L, inner = 1L),
    "168337" = list(outer = 5L, inner = 3L),
    "359994" = list(outer = 1L, inner = 1L)
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


  lines = readLines("log-runtime-2021-04-20-new.txt")
  ll = list()
  for (tsn in as.character(tsks_classif$name)) {
    tidx = tsn == lines
    times = regmatches(lines[which(tidx) + 1L], gregexpr("[[:digit:]]+", lines[which(tidx) + 1L]))
    times = as.numeric(sapply(times, function(x) {
      if (length(x) == 3) {
        paste0(x[2], ".", x[3])
      } else {
        paste0(x[1], ".", x[2])
      }
    }))
    times = times * 60^2
    #llrns = extractStringBetween(lines[which(tidx) + 1L], "Learner: ", " ")
    llrns = learners
    out = data.frame(task = tsn, learner = llrns, time_train = times,
      nevals_outer = nevals[[tsn]]$outer, nevals_inner = nevals[[tsn]]$inner,
      par_dim = unlist(pars_dim[llrns]), budget_per_dim = 50L)
    out$time_train = out$time_train / (out$nevals_outer * out$nevals_inner * out$par_dim * out$budget_per_dim + out$nevals_outer)
    rownames(out) = NULL
    ll[[tsn]] = out
  }
  df_run = do.call(rbind, unname(ll))
  attr(df_run, "estRuntime") = function(out) {
    out$time_train * (out$nevals_outer * out$nevals_inner * out$par_dim * out$budget_per_dim + out$nevals_outer) / 60^2
  }
  df_run$runtime_extrapolated = attr(df_run, "estRuntime")(df_run)
  save(df_run, file = "df-runtime-est.Rda")

  library(dplyr)

  day_machines = 24 * 7

  (df_run %>%
    filter(task != "168337"))$runtime_extrapolated %>%
    sum(na.rm = TRUE) / day_machines

  df_run %>%
    group_by(task) %>%
    summarize(time_total = sum(runtime_extrapolated, na.rm = TRUE) / 24)

  df_run %>%
    group_by(learner) %>%
    summarize(time_total = sum(runtime_extrapolated, na.rm = TRUE) / 24)
  df_run %>%
    filter(task != "168337") %>%
    group_by(learner) %>%
    summarize(time_total = sum(runtime_extrapolated, na.rm = TRUE) / 24)

}
