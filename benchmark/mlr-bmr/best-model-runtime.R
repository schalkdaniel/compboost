load("~/repos/compboost/benchmark/mlr-bmr/config-best-model.Rda")

tsks_classif = rbind(
  data.frame(type = "oml", name = "54"),           # Hepatitis
  data.frame(type = "oml", name = "37"),           # Diabetes
  data.frame(type = "oml", name = "4534"),         # Analcat Halloffame
  data.frame(type = "mlr", name = "spam"),         # Spam
  data.frame(type = "oml", name = "7592"),         # Adult
  data.frame(type = "oml", name = "168335"),       # MiniBooNE
  data.frame(type = "script", name = "albert"),    # Albert
  # data.frame(type = "oml", name = "168337"),       # Guillermo
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

tsks_classif = tsks_classif[tsks_classif$name == config_runtime$ts, ]
learner = config_runtime$ln

extractStringBetween = function(str, left, right) {
  tmp = sapply(strsplit(str, left), function(x) x[2])
  sapply(strsplit(tmp, right), function(x) x[1])
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

config = list(task = tsks_classif$name, type = tsks_classif$type, learner = learner)

temp = capture.output(source("tasks.R"))
ts = tasks_classif[[1]]

set.seed(31415L)
res = getResampleInstance(ts)

source("learners.R")

load(config_runtime$file)
pars = bmr_res[[1]]

ll_run = list()
cat("  Fitting", length(pars), "best parameter combos")
for (i in seq_along(pars)) {
  cat(i, "/", length(pars), "\n", sep = "")
  lrn = learners_classif[[1]]$clone(deep = TRUE)
  lrn = GraphLearner$new(robustify %>>% lrn)
  xs = unlist(pars[[i]]$learner_param_vals)
  idx_nc = grepl("ncores", names(xs))
  if (any(idx_nc)) {
    xs[[which(idx_nc)]] = parallel::detectCores()
  }
  reps = 3L
  tts = numeric(reps)
  for (j in seq_len(reps)) {
    lrn0 = lrn$clone(deep = TRUE)
    lrn0$param_set$values = xs
    ts0 = ts$clone(deep = TRUE)
    ts0$filter(res$outer$train_set(i))

    lrn$train(ts0)
    tts[j] = lrn0$state$train_time
  }
  ll_run[[i]] = data.frame(learner = learner, task = config_runtime$ts,
    iteration = i, time_train = mean(tts))
}
df_best = do.call(rbind, ll_run)
save(df_best, file = paste0("best-runs/", config_runtime$file))
