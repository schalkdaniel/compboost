suppressMessages(library(mlr3))
suppressMessages(library(mlr3tuning))
suppressMessages(library(mlrintermbo))
suppressMessages(library(mlr3learners))
suppressMessages(library(mlr3extralearners))
suppressMessages(library(mlr3pipelines))
suppressMessages(library(paradox))
suppressMessages(library(mlr3oml))

library(R6)
source("classifInterpretML.R")
source("classifInterpretML_reticulate.R")

#source("../load-albert.R")
#task = ts_file

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")


task = tsk("spam")
#task = tsk("oml", task_id = 31)


lrn1 = lrn("classif.interpretML_reticulate", max_rounds = 500L, predict_type = "prob", learning_rate = 0.01,
  validation_size = 0, random_state = 1337, n_jobs = parallel::detectCores())

#lrn1$train(task)
#lrn1$predict(task)

#lrn2 = lrn("classif.interpretML", max_rounds = 500L, predict_type = "prob", learning_rate = 0.01,
#  validation_size = 0, random_state = 1337)

bmr = benchmark(benchmark_grid(
  tasks = list(
               #tsk("oml", task_id = 168337),
               tsk("oml", task_id = 7592),
               tsk("oml", task_id = 168335)),
  learners = list(robustify %>>% lrn1),
  resamplings = rsmp("cv", folds = 3)))
bmr$score(msrs(c("classif.auc", "time_train")))

