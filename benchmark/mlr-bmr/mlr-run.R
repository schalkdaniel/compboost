if (FALSE) {
  devtools::install("~/repos/compboost")
  install.packages(c("mlr3", "mlr3tuning", "mlr3learners", "mlr3pipelines",
    "paradox", "xgboost", "ranger", "mboost", "mlr3oml", "interpret", "mlrMBO",
    "mlrintermbo"))
  remotes::install_github("mlr-org/mlr3extralearners")
}

load("config.Rda")
if (FALSE) {
  config$task = "31"
  config$learner = "classif_lrn_interpretML"
  tt = "CLASSIFICATION"
  i = 1
}

suppressMessages(library(mlr3))
suppressMessages(library(mlr3tuning))
suppressMessages(library(mlrintermbo))
suppressMessages(library(mlr3learners))
suppressMessages(library(mlr3extralearners))
suppressMessages(library(mlr3pipelines))
suppressMessages(library(paradox))

base_dir = here::here()
bm_dir = paste0(base_dir, "/benchmark/mlr-bmr/")

library(R6)
source(paste0(bm_dir, "learner-src/classifCompboost.R"))
source(paste0(bm_dir, "learner-src/classifInterpretML.R"))
source(paste0(bm_dir, "learner-src/regrCompboost.R"))


### Benchmark:
### ==========================================

bm_test = FALSE
bm_small = FALSE
bm_full = TRUE

if (bm_test) {
  n_evals = 2L
  n_evals_mbo = 3L
  resampling_inner = rsmp("holdout")
  resampling_outer = rsmp("holdout", ratio = 0.2)
}
if (bm_small) {
  n_evals = 20L
  n_evals_mbo = 50L
  resampling_inner = rsmp("cv", folds = 2)
  resampling_outer = rsmp("cv", folds = 2)
}
if (bm_full) {
  n_evals = 100L
  n_evals_mbo = 60L
  resampling_inner = rsmp("cv", folds = 3)
  resampling_outer = rsmp("cv", folds = 5)
  #resampling_outer = rsmp("repeated_cv", folds = 5, repeats = 10L)
}


measure_classif = msr("classif.auc")
#measure_regr = msr("regr.mse")

source(paste0(bm_dir, "tasks.R"))
source(paste0(bm_dir, "param-sets.R"))
source(paste0(bm_dir, "learners.R"))
source(paste0(bm_dir, "design.R"))


## Run benchmark:
## -----------------------

#msrs_types = list(
  #CLASSIFICATION = c("time_train", "time_predict", "time_both",
    #"classif.auc", "classif.ce", "classif.bbrier"),
  #REGRESSION = c("time_train", "time_predict", "time_both",
    #"regr.mse", "regr.mae", "regr.rsq"))

msrs_classif = c("time_train", "time_predict", "time_both",
  "classif.auc", "classif.ce", "classif.bbrier")

cat("\n>> [", as.character(Sys.time()), "]  BENCHMARK:\n", sep = "")
logfile = paste0(bm_dir, "log-files/mlr3log-", format(Sys.Date(), "%Y-%m-%d"),
  "-task", config$task, "-", config$learner, ".txt")

cat("\t>> [", as.character(Sys.time()), "] Starting benchmark\n", sep = "")

e = try({

  sink(logfile)
  time = proc.time()
  bmr = benchmark(design_classif, store_models = TRUE)
  time = proc.time() - time
  sink()

  cat("    >> [", as.character(Sys.time()), "] Finish benchmark in ", time[3], " seconds\n", sep = "")
  cat("    >> [", as.character(Sys.time()), "] Aggregate results and store data\n", sep = "")

  lrners       = as.data.table(bmr)$learner
  bmr_tune_res = lapply(lrners, function(b) b$tuning_result)

  bmr_aggr     = bmr$aggregate(msrs(msrs_classif))
  idx_aggr_rmv = which(names(bmr_aggr) %in% "resample_result")
  bmr_aggr     = as.data.frame(bmr_aggr)[, -idx_aggr_rmv]

  bmr_score     = bmr$score(msrs(msrs_classif))
  idx_score_rmv = which(names(bmr_score) %in% c("task", "resampling", "learner", "prediction"))
  bmr_score     = as.data.frame(bmr_score)[, -idx_score_rmv]

  bmr_res = list(bmr_tune_res, bmr_aggr, bmr_score)

  bm_file = paste0(bm_dir, "res-results/bmr-", format(Sys.Date(),
    "%Y-%m-%d"), "-task", config$task, "-", config$learner, ".Rda")

  save(bmr_res, file = bm_file)
  cat("    >> [", as.character(Sys.time()), "] Save ", bm_file, "\n", sep = "")

  rm(bmr, bmr_tune_res, bmr_aggr, bmr_res, bmr_score)
})
if ("try-error" %in% class(e)) {
  cat(e)
}
