if (FALSE) {
  devtools::install("~/repos/compboost")
  install.packages(c("mlr3", "mlr3tuning", "mlr3learners", "mlr3pipelines",
    "paradox", "xgboost", "ranger", "mboost", "mlr3oml"))
  remotes::install_github("mlr-org/mlr3extralearners")
}

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


### Benchmark:
### ==========================================

test = FALSE
if (test) {
  n_evals = 1L
  resampling_inner = rsmp("holdout")
  resampling_outer = rsmp("holdout")
} else {
  n_evals = 100L
  resampling_inner = rsmp("cv", folds = 3)
  resampling_outer = rsmp("cv", folds = 5)
}
measure_classif = msr("classif.auc")
measure_regr = msr("regr.mse")

source(paste0(bm_dir, "tasks.R"))
if (FALSE) {
  tasks_classif = tasks_classif[[1]]
  tasks_regr = tasks_regr[[1]]
}
source(paste0(bm_dir, "param-sets.R"))
source(paste0(bm_dir, "learners.R"))
source(paste0(bm_dir, "design.R"))


## Run benchmark:
## -----------------------

# split design to have multiple smaller ones per task:
nm_tasks_classif_all = sapply(design_classif$task, function(t) t$id)
nm_tasks_classif = unique(nm_tasks_classif_all)

nm_tasks_regr_all = sapply(design_regr$task, function(t) t$id)
nm_tasks_regr = unique(nm_tasks_regr_all)

designs_classif = lapply(nm_tasks_classif, function(nm) {
  design_classif[nm_tasks_classif_all == nm, ]
})
designs_regr = lapply(nm_tasks_regr, function(nm) {
  design_regr[nm_tasks_regr_all == nm, ]
})

dt = format(Sys.time(), "%Y-%b-%d")

bmr_task_types = c("CLASSIFICATION", "REGRESSION")
designs = list(
  CLASIFICATION = designs_classif,
  REGRESSION    = designs_regr)
msrs_types = list(
  CLASSIFICATION = c("time_train", "time_predict", "time_both",
    "classif.auc", "classif.ce", "classif.bbrier"),
  REGRESSION = c("time_train", "time_predict", "time_both",
    "regr.mse", "regr.mae", "regr.rsq"))

for (tt in bmr_task_types) {
  cat("\n>> ", as.character(Sys.time()), ": ", tt, " BENCHMARK:\n", sep = "")
  for (i in seq_along(designs[[tt]])) {
    if (tt == "CLASSIFICATION") logfile = paste0(bm_dir, "res-results/mlr3log-classif-", i, ".txt")
    if (tt == "REGRESSION") logfile = paste0(bm_dir, "res-results/mlr3log-regr-", i, ".txt")

    cat("  >> ", as.character(Sys.time()), ": Batch ", i, "/", length(designs[[tt]]), "\n", sep = "")

    sink(logfile)
    time = proc.time()
    bmr = benchmark(designs[[tt]][[i]], store_models = TRUE)
    time = proc.time() - time
    sink()

    cat("    >> ", as.character(Sys.time()), ": Finish benchmark in ", time[3], " seconds\n", sep = "")
    cat("    >> ", as.character(Sys.time()), ": Aggregate results and store data\n", sep = "")

    lrners       = as.data.table(bmr)$learner
    bmr_tune_res = lapply(lrners, function(b) b$tuning_result)
    bmr_aggr     = bmr$aggregate(msrs(msrs_types[[tt]]))[, -2]

    iters = sapply(lrners, function(l) {
      iters = NA
      if (grepl(pattern = "cboost", l$id)) {
        b = l$model$learner$model
        nms = names(b)
        b = b[[nms[grepl(pattern = "ps_cboost", x = nms)]]]$model
        if ("cboost_restart" %in% names(b)) iters = length(b$cboost$getSelectedBaselearner())
      }
      return(c(iters_momentum = iters))
    })
    bmr_aggr = cbind(bmr_aggr, t(iters))

    bmr_score     = bmr$score(msrs(msrs_types[[tt]]))
    idx_score_rmv = which(names(bmr_score) %in% c("task", "resampling", "learner", "prediction"))
    bmr_score     = as.data.frame(bmr_score)[, -idx_score_rmv]

    bmr_res = list(bmr_tune_res, bmr_aggr, bmr_score)

    if (tt == "CLASSIFICATION") bm_file = paste0(bm_dir, "res-results/bmr-classif-", i, "-", dt, ".Rda")
    if (tt == "REGRESSION")     bm_file = paste0(bm_dir, "res-results/bmr-regr-", i, "-", dt, ".Rda")

    save(bmr_res, file = bm_file)
    cat("    >> ", Sys.time(), ": Save ", bm_file, "\n", sep = "")

    rm(bmr, bmr_tune_res, bmr_aggr, bmr_res, bmr_score)
  }
}





#for (i in seq_along(designs_classif)) {
  #cat(paste0(">> ", Sys.time(), ": Batch ", i, "/", length(designs_classif), "\n"))

  #sink(logfile)
  #time = proc.time()
  #bmr_classif  = benchmark(designs_classif[[i]], store_models = TRUE)
  #time = proc.time() - time
  #sink()
  #cat("\n\n>> Finish benchmark")
  #print(time)

  #cat("\n>> Aggregate results and store:")
  #lrners = as.data.table(bmr_classif)$learner
  #bmr_tune_res = lapply(lrners, function (b) b$tuning_result)
  #bmr_aggr = bmr_classif$aggregate(msrs(c("classif.auc", "classif.ce", "time_train", "classif.bbrier")))[,-2]

  #iters = sapply(lrners, function (l) {
    #iters = NA
    #iters_restart = NA
    #if(grepl(pattern = "cboost", l$id)) {
      #b = l$model$learner$model
      #nms = names(b)
      #b = b[[nms[grepl(pattern = "ps_cboost", x = nms)]]]$model
      #if ("cboost" %in% names(b)) {
        #iters = length(b$cboost$getSelectedBaselearner())
      #}
      #if ("cboost_restart" %in% names(b)) {
        #iters_restart = iters + length(b$cboost_restart$getSelectedBaselearner())
      #}
    #}
    #return (c(iters = iters, iters_restart = iters_restart))
  #})
  #bmr_aggr = cbind(bmr_aggr, t(iters))

  #bmr_score = bmr_classif$score(msrs(c("time_train", "time_predict", "time_both", "classif.auc", "classif.ce", "classif.bbrier")))
  #bmr_score = as.data.frame(bmr_score)[,-which(names(bmr_score) %in% c("task", "resampling", "learner", "prediction"))]

  #bmr_res = list(bmr_tune_res, bmr_aggr, bmr_score)


  #bm_file = paste0(bm_dir, "res-results/bmr-classif-", i, "-", dt, ".Rda")
  #save(bmr_res, file = bm_file)
  #cat(paste0("\n>> ", Sys.time(), ": Save ", bm_file, "\n\n"))

  #rm(bmr_classif, bmr_tune_res, bmr_aggr, bmr_res, bmr_score)
#}

#cat(paste0("\n>> ", Sys.time(), ": REGRESSION BENCHMARK:\n"))

#for (i in seq_along(designs_regr)) {
  #logfile = paste0(bm_dir, "res-results/mlr3log-regr-", i, ".txt")

  #cat(paste0(">> ", Sys.time(), ": Batch ", i, "/", length(designs_regr), "\n"))

  #sink(logfile)
  #time = proc.time()
  #time = proc.time() - time
  #bmr_regr  = benchmark(designs_regr[[1]], store_models = TRUE)
  #sink()
  #print(time)
  #cat("\n\n>> ", Sys.time(), ": Finish benchmark")

  #cat("\n>> Aggregate results and store:")
  #lrners = as.data.table(bmr_regr)$learner
  #bmr_tune_res = lapply(lrners, function (b) b$tuning_result)
  #bmr_aggr = bmr_regr$aggregate(msrs(c("regr.mse", "regr.mae", "time_train", "regr.rsq")))[,-2]

  #iters = sapply(lrners, function (l) {
    #iters = NA
    #iters_restart = NA
    #if(grepl(pattern = "cboost", l$id)) {
      #b = l$model$learner$model
      #nms = names(b)
      #b = b[[nms[grepl(pattern = "ps_cboost", x = nms)]]]$model
      #if ("cboost" %in% names(b)) {
        #iters = length(b$cboost$getSelectedBaselearner())
      #}
      #if ("cboost_restart" %in% names(b)) {
        #iters_restart = iters + length(b$cboost_restart$getSelectedBaselearner())
      #}
    #}
    #return (c(iters = iters, iters_restart = iters_restart))
  #})
  #bmr_aggr = cbind(bmr_aggr, t(iters))

  #bmr_score = bmr_regr$score(msrs(c("time_train", "time_predict", "time_both", "regr.mse", "regr.mae", "regr.rsq")))
  #bmr_score = as.data.frame(bmr_score)[,-which(names(bmr_score) %in% c("task", "resampling", "learner", "prediction"))]

  #bmr_res = list(bmr_tune_res, bmr_aggr, bmr_score)

  #print(time)
  #bm_file = paste0(bm_dir, "res-results/bmr-regr-", i, "-", dt, ".Rda")
  #save(bmr_res, file = bm_file)
  #cat("\n\n>> ", Sys.time(), ": Save ", bm_file, "\n\n")

  #rm(bmr_regr, bmr_tune_res, bmr_aggr, bmr_res, bmr_score)
#}

