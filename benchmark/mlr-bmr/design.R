## Build design:
## -----------------------

design_classif = benchmark_grid(
  learner = learners_classif,
  task = tasks_classif,
  # Dummy! The resampling strategy is overwritten when updating
  # the design because it is dependent on the task.
  resampling = rsmp("holdout")
)

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")

getAT = function(lrn, ps, res, add_pipe = NULL) {
  glearner = robustify
  if (! is.null(add_pipe)) glearner = glearner %>>% add_pipe
  glearner = glearner %>>% lrn

  if (grepl("notune", lrn$id)) {
   return(glearner)
  }
  measure    = measure_classif$clone(deep = TRUE)
  terminator = trm("evals", n_evals = n_evals_per_dim * ps$length)

  # Uses Kriging as default learner with just numerical parameter
  base = mlr3learners::LearnerRegrKM$new()
  base$param_set$values = list(covtype = "matern3_2", optim.method = "BFGS",
    jitter = 1e12, nugget.stability = 1e-8)
  base$predict_type = "se"
  tuner = tnr("intermbo", surrogate.learner = base, initial.design.size = 8L * ps$length,
    on.surrogate.error = "warn") # Inital design default: 4L * ps$length

  set.seed(31415)
  dsgn = paradox::generate_design_lhs(ps, 8L * ps$length)$data
  tuner$param_set$values$initial.design.size = 0
  tuner$param_set$values$fixed.initial.design = dsgn

  at = AutoTuner$new(
    learner      = GraphLearner$new(glearner),
    resampling   = res$clone(deep = TRUE),
    measure      = measure,
    search_space = ps$clone(deep = TRUE),
    terminator   = terminator$clone(deep = TRUE),
    tuner        = tuner$clone(deep = TRUE),
    store_models = FALSE,
    store_tuning_instance = TRUE)
  return(at)
}

updateDesign = function(design) {
  ats = list()
  for (i in seq_len(nrow(design))) {

    cat(i, "/", nrow(design), "\n")
    cl = stringr::str_remove(design$learner[[i]]$id, "encode.")
    if (grepl("ps_cboost_nesterov", cl)) {
      cl = "ps_cboost_nesterov"
    } else {
      if (grepl("ps_cboost", cl)) cl = "ps_cboost"
    }
    if (grepl("ps_cwb", cl)) cl = "ps_cwb"

    notune = grepl("notune", design$learner[[i]]$id)
    if (notune) {
      ps = NULL
    } else {
      ps = do.call(cl, list(task = robustify$train(design$task[[i]])[[1]], id = design$learner[[i]]$id))
    }

    res = getResampleInstance(design$task[[i]])$inner$clone(deep = TRUE)
    design$task[[i]]$col_roles$stratum = design$task[[i]]$target_names
    at = getAT(design$learner[[i]], ps, res)

    set.seed(seed)
    design$resampling[[i]] = getResampleInstance(design$task[[i]])$outer$clone(deep = TRUE)

    if (!notune) {
      # Check if params do match:
      if (! all(ps$ids() %in% at$learner$param_set$ids())) stop("Arguments does not match for ", at$id)
    }

    ats = c(ats, list(at))
  }
  design$learner = ats
  return (design)
}

design_classif = updateDesign(design_classif)
