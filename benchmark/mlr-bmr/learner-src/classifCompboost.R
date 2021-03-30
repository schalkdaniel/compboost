LearnerClassifCompboost = R6Class("LearnerClassifCompboost",
  inherit = LearnerClassif,
  public = list(

    #' @description
    #' Create a `LearnerClassifCompboost` object.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamDbl$new(id = "df", default = 5, lower = 1),
          ParamInt$new(id = "mstop", default = 100L, lower = 1L),
          ParamDbl$new(id = "learning_rate", default = 0.05, lower = 0),
          ParamDbl$new(id = "n_knots", default = 20L, lower = 4),
          ParamFct$new(id = "optimizer", default = "cod", levels = c("cod", "nesterov", "cos-anneal")),
          ParamInt$new(id = "ncores", default = 1L, lower = 1L, upper = parallel::detectCores()),
          ParamDbl$new(id = "momentum", default = 0.0005, lower = 0),
          ParamDbl$new(id = "oob_fraction", default = 0, lower = 0, upper = 0.9),
          ParamLgl$new(id = "use_stopper", default = FALSE),
          ParamInt$new(id = "patience", default = 5, lower = 1),
          ParamDbl$new(id = "eps_for_break", default = 0),
          ParamDbl$new(id = "bin_root", default = 0, lower = 0, upper = 4),
          ParamFct$new(id = "bin_method", default = "quantile", levels = c("linear", "quantile")),
          ParamDbl$new(id = "df_cat", default = 1, lower = 1),
          ParamLgl$new(id = "restart", default = TRUE)
        )
      )
      #ps$add_dep("momentum", "optimizer", CondEqual$new("nesterov"))
      #ps$add_dep("use_stopper", "oob_fraction", Condition$new(oob_fraction > 0))
      #ps$add_dep("patience", "use_stopper", CondEqual$new(TRUE))
      #ps$add_dep("eps_for_break", "use_stopper", CondEqual$new(TRUE))

      super$initialize(
        id = "classif.compboost",
        packages = "compboost",
        feature_types = c("numeric", "factor", "integer", "character"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("twoclass")
      )
    }
  ),

  private = list(
    .train = function(task) {

      #browser()

      pdefaults = self$param_set$default
      pars = self$param_set$values
      for (id in self$param_set$ids()) {
        if (is.null(pars[[id]])) pars[[id]] = pdefaults[[id]]
      }
      self$param_set$values = pars

      if (self$param_set$values$optimizer == "cod") {
        optimizer = compboost::OptimizerCoordinateDescent$new(self$param_set$values$ncores)
      }
      if (self$param_set$values$optimizer == "nesterov") {
        optimizer = compboost::OptimizerAGBM$new(self$param_set$values$momentum, self$param_set$values$ncores)
      }
      if (self$param_set$values$optimizer == "cos-anneal") {
        optimizer = compboost::OptimizerCosineAnnealing$new(0.001, 0.3, 4, self$param_set$values$mstop,
          self$param_set$values$ncores)
      }


      if (self$param_set$values$use_stopper) {
        stop_args = list(patience = self$param_set$values$patience, eps_for_break = self$param_set$values$eps_for_break)
      } else {
        stop_args = list()
      }

      out = list()
      seed = sample(seq_len(1e6), 1)
      #browser()

      nuisance = capture.output({
      set.seed(seed)
      cboost = compboost::boostSplines(
        data = task$data(),
        target = task$target_names,
        iterations = self$param_set$values$mstop,
        optimizer = optimizer,
        loss = compboost::LossBinomial$new(),
        df = self$param_set$values$df,
        learning_rate = self$param_set$values$learning_rate,
        oob_fraction = self$param_set$values$oob_fraction,
        stop_args = stop_args,
        bin_root = self$param_set$values$bin_root,
        bin_method = self$param_set$values$bin_method,
        df_cat = self$param_set$values$df_cat)

      out$cboost = cboost
      iters = length(cboost$getSelectedBaselearner())

      ### Restart:
      if ((self$param_set$values$optimizer == "nesterov") && self$param_set$values$restart) {
        iters_remaining = self$param_set$values$mstop - iters

        if (iters_remaining > 0) {
          set.seed(seed)
          cboost_restart = compboost::boostSplines(
            data = task$data(),
            target = task$target_names,
            iterations = iters_remaining,
            optimizer = compboost::OptimizerCoordinateDescent$new(self$param_set$values$ncores),
            loss = compboost::LossBinomial$new(cboost$predict(task$data()), TRUE),
            df = self$param_set$values$df,
            learning_rate = self$param_set$values$learning_rate,
            bin_root = self$param_set$values$bin_root,
            bin_method = self$param_set$values$bin_method,
            df_cat = self$param_set$values$df_cat)

          out$cboost_restart = cboost_restart
        }
      }
    })
    return(out)
    },

    .predict = function(task) {
      #browser()
      newdata = task$data(cols = task$feature_names)

      if (self$param_set$values$optimizer == "nesterov") {
        lin_pred = self$model$cboost$predict(newdata)
        if ("cboost_restart" %in% names(self$model)) {
          lin_pred = lin_pred + self$model$cboost_restart$predict(newdata)
        }
        probs = 1 / (1 + exp(-lin_pred))
      } else {
        probs = self$model$cboost$predict(newdata, as_response = TRUE)
      }

      pos = self$model$cboost$response$getPositiveClass()
      neg = setdiff(names(self$model$cboost$response$getClassTable()), pos)
      pmat = matrix(c(probs, 1 - probs), ncol = 2L, nrow = length(probs))
      colnames(pmat) = c(pos, neg)
      if (self$predict_type == "prob") {
        list(prob = pmat)
      }
      if (self$predict_type == "response") {
        list(response = ifelse(probs > self$model$cboost$response$getThreshold(), pos, neg))
      } else {
        list(prob = pmat)
      }
    }
  )
)
mlr_learners$add("classif.compboost", LearnerClassifCompboost)

#lr1 = lrn("classif.compboost", optimizer = "nesterov", use_stopper = TRUE, eps_for_break = 0, patience = 2, oob_fraction = 0.3, predict_type = "prob", mstop = 10L, restart = FALSE)

#design = benchmark_grid(
  #tasks = tsk("sonar"),
  #learners = lr1,
  #resamplings = rsmp("cv", folds = 3)
#)

#bmr = benchmark(design, store_models = TRUE)
#bmr$aggregate(msrs(c("classif.auc", "classif.ce")))



