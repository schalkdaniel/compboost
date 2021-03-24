suppressMessages(library(mlr3))
suppressMessages(library(mlr3tuning))
suppressMessages(library(mlr3learners))
suppressMessages(library(mlr3extralearners))
suppressMessages(library(mlr3pipelines))
suppressMessages(library(mlr3pipelines))
suppressMessages(requireNamespace("mlr3oml"))

base_dir = here::here()
bm_dir = paste0(base_dir, "/benchmark/mlr-bmr/")

load("config.Rda")

library(R6)
source(paste0(bm_dir, "learner-src/classifCompboost.R"))
source(paste0(bm_dir, "learner-src/classifInterpretML.R"))
source(paste0(bm_dir, "learners.R"))

if (config$type == "oml") {
  ts = tsk("oml", task_id = as.integer(config$task))
}

if (config$type == "script") {
  source(paste0("load-", config$name, ".R"))
  ts = ts_file
}

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")


# Trigger compboostSplines as flag when the fitting starts. compboostSplines
# can be extracted by valgrind.
tmp = compboostSplines::createKnots(1:10, 3, 2)

lrn = learners_classif[[config$learner]]
lrn = GraphLearner$new(robustify %>>% lrn)
lrn$train(ts)


