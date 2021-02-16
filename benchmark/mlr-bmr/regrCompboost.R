LearnerRegrCompboost = R6Class("LearnerRegrCompboost",
  inherit = LearnerRegr,
  public = list(

    #' @description
    #' Create a `LearnerRegrCompboost` object.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamDbl$new(id = "df", default = 5, lower = 1),
          ParamInt$new(id = "mstop", default = 100L, lower = 1L),
          ParamDbl$new(id = "learning_rate", default = 0.05, lower = 0),
          ParamDbl$new(id = "n_knots", default = 20L, lower = 4),
          ParamFct$new(id = "optimizer", default = "cod", levels = c("cod", "nesterov")),
          ParamInt$new(id = "ncores", default = 1L, lower = 1L, upper = parallel::detectCores()),
          ParamDbl$new(id = "momentum", default = 0.0005, lower = 0),
          ParamDbl$new(id = "oob_fraction", default = 0, lower = 0, upper = 0.9),
          ParamLgl$new(id = "use_stopper", default = FALSE),
          ParamInt$new(id = "patience", default = 5, lower = 1),
          ParamDbl$new(id = "eps_for_break", default = 0),
          ParamDbl$new(id = "bin_root", default = 0, lower = 0, upper = 4),
          ParamDbl$new(id = "df_cat", default = 1, lower = 1)
        )
      )
      #ps$add_dep("momentum", "optimizer", CondEqual$new("nesterov"))
      #ps$add_dep("use_stopper", "oob_fraction", Condition$new(oob_fraction > 0))
      #ps$add_dep("patience", "use_stopper", CondEqual$new(TRUE))
      #ps$add_dep("eps_for_break", "use_stopper", CondEqual$new(TRUE))

      super$initialize(
        id = "regr.compboost",
        packages = "compboost",
        feature_types = c("numeric", "factor", "integer", "character"),
        predict_types = "response",
        param_set = ps
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

      if (self$param_set$values$optimizer == "cod") optimizer = compboost::OptimizerCoordinateDescent$new(self$param_set$values$ncores)
      if (self$param_set$values$optimizer == "nesterov") optimizer = compboost::OptimizerAGBM$new(self$param_set$values$momentum, self$param_set$values$ncores)


      if (self$param_set$values$use_stopper) {
        stop_args = list(patience = self$param_set$values$patience, eps_for_break = self$param_set$values$eps_for_break)
      } else {
        stop_args = list()
      }

      # browser()

      out = list()
      seed = sample(seq_len(1e6), 1)

      nuisance = capture.output({
        set.seed(seed)
        cboost = compboost::boostSplines(
          data = task$data(),
          target = task$target_names,
          iterations = self$param_set$values$mstop,
          optimizer = optimizer,
          loss = compboost::LossQuadratic$new(),
          df = self$param_set$values$df,
          learning_rate = self$param_set$values$learning_rate,
          oob_fraction = self$param_set$values$oob_fraction,
          stop_args = stop_args,
          bin_root = self$param_set$values$bin_root,
          df_cat = self$param_set$values$df_cat)

        out$cboost = cboost
        iters = length(cboost$getSelectedBaselearner())

        ### Restart:
        if (self$param_set$values$optimizer == "nesterov") {
          iters_remaining = self$param_set$values$mstop - iters

          if (iters_remaining > 0) {
            set.seed(seed)
            cboost_restart = compboost::boostSplines(
              data = task$data(),
              target = task$target_names,
              iterations = iters_remaining,
              optimizer = compboost::OptimizerCoordinateDescent$new(self$param_set$values$ncores),
              loss = compboost::LossQuadratic$new(cboost$predict(task$data()), TRUE),
              df = self$param_set$values$df,
              learning_rate = self$param_set$values$learning_rate,
              bin_root = self$param_set$values$bin_root,
              df_cat = self$param_set$values$df_cat)

            out$cboost_restart = cboost_restart
          }
        }
      })
      return (out)
    },

    .predict = function(task) {
      #browser()
      newdata = task$data(cols = task$feature_names)

      if (self$param_set$values$optimizer == "nesterov") {
        lin_pred = self$model$cboost$predict(newdata)
        lin_pred = lin_pred + self$model$cboost_restart$predict(newdata)
        #lin_pred = self$model$cboost_restart$predict(newdata)
      } else {
        lin_pred = self$model$cboost$predict(newdata)
      }

      return (list(response = lin_pred))
    }
  )
)

mlr_learners$add("regr.compboost", LearnerRegrCompboost)



#lr1 = lrn("regr.compboost", optimizer = "nesterov", use_stopper = TRUE, eps_for_break = 0, patience = 5, oob_fraction = 0.3, mstop = 5000L)
#lr2 = lrn("regr.compboost", mstop = 5000L)

#lr1$train(tsk("mtcars"))
#lr2$train(tsk("mtcars"))

#p1 = lpo1$predict(tsk("mtcars"))
#p2 = lpo2$predict(tsk("mtcars"))

#p1
#p2
#
#design = benchmark_grid(
  #tasks = tsk("boston_housing"),
  #learners = list(lr1, lr2),
  #resamplings = rsmp("cv", folds = 3)
#)

#bmr = benchmark(design)
#bmr$aggregate(msrs(c("regr.mse", "regr.mae")))

