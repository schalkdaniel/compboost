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

# Each sever gets just a few tasks to efficiently distribute
# over several machines (name of the server is saved in '/etc/hostname'
#
# @param on_host logical(1) Indicate if host selector should be applied.
#   If `on_host = FALSE` all tasks are selected.
serverSelector = function(on_host = FALSE) {
  if (on_host) {
    host = readLines("/etc/hostname")
    host_tasks = list(
      "bigger-benchmarks2" = c(1, 2),
      "cacb1" = c(3, 4),
      "cacb2" = 5,
      "cacb3" = 6,
      "cacb4" = 7,
      "cacb5" = 8,
      "cacb6" = 9)
    idx = host_tasks[[host]]
    if (is.null(idx[1])) stop("Server is not one of {", paste(names(host_tasks), collapse = ", "), "}")
    return(idx)
  } else {
    return(seq_len(nrow(tsks_classif)))
  }
}
# Apply selector:
tsks_classif = tsks_classif[serverSelector(TRUE), ]


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

df_res_file = paste0(base_dir, "/ll_best.Rda")
if (any(grepl(pattern = "ll_best.Rda", x = list.files(paste0(base_dir, "res_results"), full.names = TRUE)))) {
  load(df_res_file)
} else {
  ll_best = NULL
}
ll_best_old = ll_best

ll_best = list()
files = list.files(paste0(base_dir, "res-results"), full.names = TRUE)
for (fn in files) {
  is_done = fn %in% names(ll_best_old)

  if (! is_done) {
    load(fn)
    pars = lapply(bmr_res[[1]], function (bm) bm$learner_param_vals)
    learner = paste0("classif_", extractStringBetween(fn, left = "classif_", right = ".Rda"))
    tsk_name = extractStringBetween(fn, left = "-task", right = "-classif")
    tsk_idx = which(tsks_classif$name == tsk_name)

    config = list(task = tsks_classif$name[tsk_idx], type = tsks_classif$type[tsk_idx], learner = learner)

    source("tasks.R")
    source("learners.R")

    ts = tasks_classif[[1]]

    lrn = GraphLearner$new(robustify %>>% learners_classif[[1]])
    lrns = lapply(pars, function (xs) {
      lrn0 = lrn$clone(deep = TRUE)
      lrn0$param_set$values = unlist(xs)
      lrn0
    })

    set.seed(31415L)
    res = getResampleInstance(ts)

    ll_res = list()
    if (res$outer$iters > 1) pb = txtProgressBar(min = 1, max = res$outer$iters, style = 3)
    for(j in seq_len(res$outer$iters)) {
      if (res$outer$iters > 1) setTxtProgressBar(pb, j)

      train_times = numeric(5L)
      for (k in seq_len(5L)) {
        lrns[[j]]$train(ts, res$outer$train_set(j))
        pred = lrns[[j]]$predict(ts, res$outer$test_set(j))
        auc = unname(pred$score(msrs("classif.auc"))["classif.auc"])
        train_times[k] = lrns[[j]]$state$train_time
      }
      ll_res[[j]] = data.frame(auc = auc, time_train = mean(train_times), iter = j,
        learner = learner, task = ts$id)
    }
    df_res = do.call(rbind, ll_res)
    df_pars = do.call(rbind, lapply(pars, function (xs) {
      xs = xs[[1]]
      xs = xs[grepl(pattern = "ps_", x = names(xs))]
      names(xs) = paste0("param.", sapply(strsplit(names(xs), split = "[.]"), function(x) x[length(x)]))
      as.data.frame(xs)
    }))
    df_res = cbind(df_res, df_pars)
    ll_best[[fn]] = df_res
  }
}
ll_best = c(ll_best_old, ll_best)
save(ll_best, file = paste0(base_dir, "/ll_best.Rda"))

if (FALSE) {
  library(dplyr)

  df = do.call(rbind, lapply(ll_best, function (bmr) bmr %>% select(learner, task, auc, time_train, iter, param.mstop)))
  df %>% group_by(learner, task) %>%
    mutate(auc0 = auc) %>%
    summarize(time = median(time_train), time_sd = sd(time_train), auc = median(auc0), auc_sd = sd(auc0))
}




