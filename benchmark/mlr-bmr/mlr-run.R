load("config.Rda")
if (FALSE) {
  #config = list(task = "4534", type = "oml", learner = "classif_lrn_cboost2")
  #config = list(task = "4534", type = "oml", learner = "classif_lrn_xgboost")
  #config = list(task = "4534", type = "oml", learner = "classif_lrn_acwb_bin")
  config = list(task = "7592", type = "oml", learner = "classif_lrn_acwb_notune_bin")
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
source(paste0(bm_dir, "learner-src/classifInterpretML_reticulate.R"))


### Benchmark:
### ==========================================

seed = 31415L

bm_test = FALSE
bm_small = FALSE
bm_full = TRUE

if (bm_test) {
  n_evals_per_dim = 15L
  getResampleInstance = function(task) {
    resampling_inner = rsmp("holdout")
    resampling_outer = rsmp("holdout", ratio = 0.2)
    resampling_outer$instantiate(task)
    return(list(inner = resampling_inner, outer = resampling_outer))
  }
}
if (bm_small) {
  n_evals_per_dim = 10L
  getResampleInstance = function(task) {
    resampling_inner = rsmp("cv", folds = 2)
    resampling_outer = rsmp("cv", folds = 2)
    resampling_outer$instantiate(task)
    return(list(inner = resampling_inner, outer = resampling_outer))
  }
}
if (bm_full) {
  n_evals_per_dim = 50L

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
}

measure_classif = msr("classif.auc")

source(paste0(bm_dir, "extract-archive.R"))
source(paste0(bm_dir, "tasks.R"))
source(paste0(bm_dir, "param-sets.R"))
source(paste0(bm_dir, "learners.R"))
source(paste0(bm_dir, "design.R"))

## Run benchmark:
## -----------------------

# Measure which are tracked:
msrs_classif = c("time_train", "time_predict", "time_both",
  "classif.auc", "classif.ce", "classif.bbrier")

cat("\n>> [", as.character(Sys.time()), "]  BENCHMARK:\n", sep = "")
logfile = paste0(bm_dir, "log-files/mlr3log-", format(Sys.Date(), "%Y-%m-%d"),
  "-task", config$task, "-", config$learner, ".txt")

cat("\t>> [", as.character(Sys.time()), "] Starting benchmark\n", sep = "")

e = try({

  options(mlr3.debug = TRUE)
  lgr::get_logger("mlr3")$set_threshold("trace")
  lgr::get_logger("mlr3tuning")$set_threshold("trace")
  lgr::get_logger("bbotk")$set_threshold("trace")

  sink(logfile)
  time = proc.time()
  bmr = benchmark(design_classif, store_models = TRUE)
  time = proc.time() - time
  sink()

  cat("    >> [", as.character(Sys.time()), "] Finish benchmark in ", time[3], " seconds\n", sep = "")
  cat("    >> [", as.character(Sys.time()), "] Aggregate results and store data\n", sep = "")

  bmr_arx = extractArchive(bmr)
  bmr_nest_arx = NULL
  if (! grepl("notune", config$learner)) bmr_nest_arx = extractNestedArchive(bmr)

  lrners       = as.data.table(bmr)$learner
  bmr_tune_res = lapply(lrners, function(b) b$tuning_result)

  bmr_aggr     = bmr$aggregate(msrs(msrs_classif))
  idx_aggr_rmv = which(names(bmr_aggr) %in% "resample_result")
  bmr_aggr     = as.data.frame(bmr_aggr)[, -idx_aggr_rmv]

  bmr_score     = bmr$score(msrs(msrs_classif))
  idx_score_rmv = which(names(bmr_score) %in% c("task", "resampling", "learner", "prediction"))
  bmr_score     = as.data.frame(bmr_score)[, -idx_score_rmv]
  if (nrow(design_classif) == 1) {
    bmr_score$n_evals = design_classif$learner[[1]]$instance_args$terminator$param_set$values$n_evals
  }

  bmr_res = list(bmr_tune_res, bmr_aggr, bmr_score, archive = bmr_arx,
    nested_archive = bmr_nest_arx)

  bm_file = paste0(bm_dir, "res-results/bmr-", format(Sys.Date(),
    "%Y-%m-%d"), "-task", config$task, "-", config$learner, ".Rda")

  save(bmr_res, file = bm_file)
  cat("    >> [", as.character(Sys.time()), "] Save ", bm_file, "\n", sep = "")

  rm(bmr, bmr_tune_res, bmr_aggr, bmr_res, bmr_score)
})
if ("try-error" %in% class(e)) {
  cat(e)
}
