suppressMessages(library(mlr3))
suppressMessages(library(mlr3tuning))
suppressMessages(library(mlrintermbo))
suppressMessages(library(mlr3learners))
suppressMessages(library(mlr3extralearners))
suppressMessages(library(mlr3oml))
suppressMessages(library(mlr3pipelines))

library(R6)
source("classifCompboost.R")

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")

tasks = list(tsk("oml", task_id = 7592L), tsk("oml", task_id = 168335L))

#source("../load-albert.R")
#task = ts_file

# CWB:
lrn01 = lrn("classif.compboost", mstop = 2000, optimizer = "cod", ncores = parallel::detectCores()-1,
  df = 5, df_cat = 5, predict_type = "prob")
lrn02 = lrn("classif.compboost", mstop = 2000, optimizer = "cod", ncores = parallel::detectCores()-1,
  bin_root = 2L, bin_method = "quantile", df = 5, df_cat = 5, predict_type = "prob")
lrn03 = lrn("classif.compboost", mstop = 2000, optimizer = "cod", ncores = parallel::detectCores()-1,
  bin_root = 1.5, bin_method = "quantile", df = 5, df_cat = 5, predict_type = "prob")
lrn04 = lrn("classif.compboost", mstop = 2000, optimizer = "cod", ncores = parallel::detectCores()-1,
  bin_root = 2L, bin_method = "linear", df = 5, df_cat = 5, predict_type = "prob")
lrn05 = lrn("classif.compboost", mstop = 2000, optimizer = "cod", ncores = parallel::detectCores()-1,
  bin_root = 1.5, bin_method = "linear", df = 5, df_cat = 5, predict_type = "prob")

# hCWB:
lrn11 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  df = 5, df_cat = 5, predict_type = "prob")
lrn12 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 2L, bin_method = "quantile", df = 5, df_cat = 5, predict_type = "prob")
lrn13 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 1.5, bin_method = "quantile", df = 5, df_cat = 5, predict_type = "prob")
lrn14 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 2L, bin_method = "linear", df = 5, df_cat = 5, predict_type = "prob")
lrn15 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 1.5, bin_method = "linear", df = 5, df_cat = 5, predict_type = "prob")

# ACWB:
lrn21 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  df = 5, df_cat = 5, restart = FALSE, predict_type = "prob")
lrn22 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 2L, bin_method = "quantile", df = 5, df_cat = 5, restart = FALSE, predict_type = "prob")
lrn23 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 1.5, bin_method = "quantile", df = 5, df_cat = 5, restart = FALSE, predict_type = "prob")
lrn24 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 2L, bin_method = "linear", df = 5, df_cat = 5, restart = FALSE, predict_type = "prob")
lrn25 = lrn("classif.compboost", mstop = 2000, optimizer = "nesterov", ncores = parallel::detectCores()-1,
  bin_root = 1.5, bin_method = "linear", df = 5, df_cat = 5, restart = FALSE, predict_type = "prob")

lrns = list(lrn01, lrn02, lrn03, lrn04, lrn05, lrn11, lrn12, lrn13, lrn14, lrn15, lrn21, lrn22, lrn23, lrn24, lrn25)

bmr = benchmark(benchmark_grid(
  tasks = tasks,
  learners = lapply(lrns, function (l) robustify %>>% l),
  resamplings = rsmp("cv", folds = 3)))

bmr$aggregate(msrs(c("classif.ce", "time_train")))
bmr$score(msrs(c("classif.ce", "time_train")))

