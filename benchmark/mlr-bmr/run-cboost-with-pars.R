config = list(task = "168335", type = "oml", learner = "classif_lrn_cboost2")
files = list.files("~/repos/compboost/benchmark/mlr-bmr/res-results", full.names = TRUE)
file = files[grepl(pattern = config$learner, x = files) & grepl(pattern = config$task, x = files)]

suppressMessages(library(mlr3))
suppressMessages(library(mlr3pipelines))

load(file)
source("tasks.R")

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")

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

pars = lapply(bmr_res[[1]], function(rs) rs$learner_param_vals)

iter = 1L
xs = unlist(pars[[iter]])
ts = tasks_classif[[1]]$clone(deep = TRUE)

set.seed(31415L)
res = getResampleInstance(ts)
ts$filter(res$outer$train_set(iter))

robustify_pp = robustify$clone(deep = TRUE)
robustify_pp$train(ts)
ts_pp = robustify_pp$predict(ts)$removeconstants_after.output
dat = ts_pp$data()

library(compboost)

xs_cboost = xs[grepl(pattern = "ps_cboost", x = names(xs))]
names(xs_cboost) = sapply(names(xs_cboost), function(nm) strsplit(x = nm, split = "[.]")[[1]][2])
xs_cboost$ncores = parallel::detectCores()

if (xs_cboost$optimizer == "cod") {
  optimizer = OptimizerCoordinateDescent$new(xs_cboost$ncores)
}
if (xs_cboost$optimizer == "nesterov") {
  optimizer = OptimizerAGBM$new(xs_cboost$momentum, xs_cboost$ncores)
}
if (xs_cboost$optimizer == "cos-anneal") {
  optimizer = OptimizerCosineAnnealing$new(0.001, 0.3, 4, xs_cboost$mstop,
    xs_cboost$ncores)
}

if (xs_cboost$use_stopper) {
  stop_args = list(patience = xs_cboost$patience, eps_for_break = 0)
} else {
  stop_args = list()
}

seed = sample(seq_len(1e6), 1)

set.seed(seed)
cboost = compboost::boostSplines(
  data = dat,
  target = ts_pp$target_names,
  iterations = xs_cboost$mstop,
  optimizer = optimizer,
  loss = compboost::LossBinomial$new(),
  df = xs_cboost$df,
  learning_rate = xs_cboost$learning_rate,
  oob_fraction = xs_cboost$oob_fraction,
  stop_args = stop_args,
  bin_root = if(grepl(pattern = "bin", x = config$learner)) { 2 } else { 0 },
  bin_method = "linear",
  df_cat = xs_cboost$df_cat)

out$cboost = cboost
iters = length(cboost$getSelectedBaselearner())

### Restart:
if ((xs_cboost$optimizer == "nesterov") && xs_cboost$restart) {
  iters_remaining = xs_cboost$mstop - iters

  if (iters_remaining > 0) {
    set.seed(seed)
    cboost_restart = compboost::boostSplines(
      data = task$data(),
      target = task$target_names,
      iterations = iters_remaining,
      optimizer = compboost::OptimizerCoordinateDescent$new(xs_cboost$ncores),
      loss = compboost::LossBinomial$new(cboost$predict(task$data()), TRUE),
      df = xs_cboost$df,
      learning_rate = xs_cboost$learning_rate,
      bin_root = xs_cboost$bin_root,
      bin_method = xs_cboost$bin_method,
      df_cat = xs_cboost$df_cat)

    out$cboost_restart = cboost_restart
  }
}
