## Build design:
## -----------------------

design_classif = benchmark_grid(
  learner = learners_classif,
  task = tasks_classif,
  resampling = resampling_outer$clone(deep = TRUE)
)
design_regr = benchmark_grid(
  learner = learners_regr,
  task = tasks_regr,
  resampling = resampling_outer$clone(deep = TRUE)
)

robustify = po("removeconstants", id = "removeconstants_before") %>>%
  po("imputemedian", id = "imputemedian_num", affect_columns = selector_type(c("integer", "numeric"))) %>>%
  po("imputemode", id = "imputemode_fct", affect_columns = selector_type(c("character", "factor", "ordered"))) %>>%
  po("collapsefactors", target_level_count = 10) %>>%
  po("removeconstants", id = "removeconstants_after")

getAT = function (lrn, ps, res, add_pipe = NULL) {
  glearner = robustify
  if (! is.null(add_pipe)) glearner = glearner %>>% add_pipe
  glearner = glearner %>>% lrn

  if ("twoclass" %in% lrn$properties) { measure = measure_classif$clone(deep = TRUE) } else { measure = measure_regr$clone(deep = TRUE) }

  AutoTuner$new(
    learner      = GraphLearner$new(glearner),
    resampling   = res$clone(deep = TRUE),
    measure      = measure,
    search_space = ps$clone(deep = TRUE),
    terminator   = trm("evals", n_evals = n_evals),
    tuner        = tnr("random_search"),
    store_tuning_instance = TRUE)
}

updateDesign = function (design) {
  ats = list()
  for (i in seq_len(nrow(design))) {
    cat(i, "/", nrow(design), "\n")
    cl = stringr::str_remove(design$learner[[i]]$id, "encode.")
    if (grepl("ps_cboost_nesterov", cl)) {
      cl = "ps_cboost_nesterov"
    } else {
      if (grepl("ps_cboost", cl)) cl = "ps_cboost"
    }

    ps = do.call(cl, list(task = robustify$train(design$task[[i]])[[1]], id = design$learner[[i]]$id))

    res = resampling_inner$clone(deep = TRUE)
    design$task[[i]]$col_roles$stratum = design$task[[i]]$target_names
    #res$instantiate(design$task[[i]])

    at = getAT(design$learner[[i]], ps, res)

    # Check if params do match:
    if (! all(ps$ids() %in% at$learner$param_set$ids())) stop("Arguments does not match for ", at$id)

    ats = c(ats, list(at))
  }
  design$learner = ats
  return (design)
}

design_classif = updateDesign(design_classif)
design_regr = updateDesign(design_regr)
