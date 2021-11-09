
cat("[", as.character(Sys.Date()),   "] Loading new learner\n", sep = "")

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
          ParamFct$new(id = "bin_method", default = "linear", levels = c("linear", "quantile")),
          ParamDbl$new(id = "df_cat", default = 1, lower = 1),
          ParamLgl$new(id = "restart", default = TRUE),
          ParamLgl$new(id = "stop_both", default = FALSE),
          ParamLgl$new(id = "df_autoselect", default = FALSE),
          ParamInt$new(id = "oob_seed", default = sample(seq_len(1e6), 1), lower = 1L)
        )
      )
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

      lg = lgr::get_logger("mlr3")
      #browser()

      pdefaults = self$param_set$default
      pars = self$param_set$values
      for (id in self$param_set$ids()) {
        if (is.null(pars[[id]])) pars[[id]] = pdefaults[[id]]
      }
      self$param_set$values = pars
      if (self$param_set$values$df_autoselect) {
        factor_cols = task$feature_types$id[task$feature_types$type == "factor"]
        if (length(factor_cols) > 0) {
          df_cat_min = min(vapply(
            X = task$data(cols = factor_cols),
            FUN = function(fc) length(unique(fc)),
            FUN.VALUE = integer(1L)
          ))
          df = min(c(df_cat_min, self$param_set$values$n_knots))
          if (df <= 3) df = 5L

          self$param_set$values$df = df
          self$param_set$values$df_cat = df_cat_min
        } else {
          self$param_set$values$df = self$param_set$values$n_knots
        }
      }

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
      seed = sample(seq_len(100000), 1)
      #seed = self$param_set$values$oob_seed
      #browser()

      lg$info("[LGCOMPBOOST] Running compboost with df %f and df_cat %f", self$param_set$values$df, self$param_set$values$df_cat)


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
      })

      out$cboost = cboost
      iters = length(cboost$getSelectedBaselearner())

      ### Restart:
      if ((self$param_set$values$optimizer == "nesterov") && self$param_set$values$restart) {
        iters_remaining = self$param_set$values$mstop - iters

        if (iters_remaining > 0) {
          if (self$param_set$values$stop_both) {
            #browser()
            nuisance = capture.output({
            set.seed(seed)
            cboost_restart = compboost::boostSplines(
              data = task$data(),
              target = task$target_names,
              iterations = iters_remaining,
              optimizer = compboost::OptimizerCoordinateDescent$new(self$param_set$values$ncores),
              loss = compboost::LossBinomial$new(cboost$predict(), TRUE),
              df = self$param_set$values$df,
              stop_args = c(stop_args, list(oob_offset = cboost$response_oob$getPrediction())),
              oob_fraction = self$param_set$values$oob_fraction,
              learning_rate = self$param_set$values$learning_rate,
              bin_root = self$param_set$values$bin_root,
              bin_method = self$param_set$values$bin_method,
              df_cat = self$param_set$values$df_cat)
            })
          } else {
            nuisance = capture.output({
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
            })
          }
          out$cboost_restart = cboost_restart
          lg$info("[LGCOMPBOOST] Completed fitting of restarted compboost model with optimizer %s",
            self$param_set$values$optimizer)
        }
      }
      rintercept = out$cboost$getInbagRisk()[1]
      rintercept_oob = NA
      rcwb  = racwb = rhcwb = NA
      rcwb_oob = racwb_oob = rhcwb_oob = NA
      stop_cwb  = stop_acwb = stop_hcwb = NA

      opt = self$param_set$values$optimizer

      if (opt == "cod") {
        rcwb = tail(out$cboost$getInbagRisk(), 1)
        stop_cwb = length(out$cboost$getSelectedBaselearner())
        if (self$param_set$values$use_stopper) {
          rintercept_oob = out$cboost$getLoggerData()$oob_risk[1]
          rcwb_oob = tail(out$cboost$getLoggerData()$oob_risk, 1)
        }
      }
      if (opt == "nesterov") {
        racwb = tail(out$cboost$getInbagRisk(), 1)
        stop_acwb = length(out$cboost$getSelectedBaselearner())
        if (self$param_set$values$use_stopper) {
          rintercept_oob = out$cboost$getLoggerData()$oob_risk[1]
          racwb_oob = tail(out$cboost$getLoggerData()$oob_risk, 1)
        }
      }
      if ("cboost_restart" %in% names(out)) {
        rhcwb = tail(out$cboost_restart$getInbagRisk(), 1)
        stop_hcwb = length(out$cboost_restart$getSelectedBaselearner())
        if (self$param_set$values$use_stopper) rhcwb_oob = tail(out$cboost_restart$getLoggerData()$oob_risk, 1)
      }

      lg$info("[LGCOMPBOOST] iterations:'stop_cwb',%i,'stop_acwb',%i,'stop_hcwb',%i",
        stop_cwb, stop_acwb, stop_hcwb)
      lg$info("[LGCOMPBOOST] risk_inbag:'risk_intercept',%f,'risk_cwb',%f,'risk_acwb',%f,'risk_hcwb',%f",
        rintercept, rcwb, racwb, rhcwb)
      lg$info(paste0("[LGCOMPBOOST] risk_oob:'risk_intercept_oob',%f,'risk_cwb_oob',%f,'risk_acwb_oob',%f,",
        "'risk_hcwb_oob',%f"), rintercept_oob, rcwb_oob, racwb_oob, rhcwb_oob)

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


#suppressMessages(library(mlr3))
#suppressMessages(library(mlr3tuning))
#suppressMessages(library(mlrintermbo))
#suppressMessages(library(mlr3learners))
#suppressMessages(library(mlr3extralearners))
#suppressMessages(library(mlr3pipelines))
#suppressMessages(library(paradox))
#suppressMessages(library(R6))


#lr1 = lrn("classif.compboost", optimizer = "nesterov", use_stopper = TRUE,
  #eps_for_break = 0, patience = 2, oob_fraction = 0.3, predict_type = "prob",
  #mstop = 5000L, restart = TRUE, stop_both = TRUE, df_autoselect = TRUE,
  #oob_seed = 100)

#lr1$train(tsk("sonar"))

#length(lr1$model$cboost$getSelectedBaselearner())
#length(lr1$model$cboost_restart$getSelectedBaselearner())
#gridExtra::grid.arrange(
  #lr1$model$cboost$plotInbagVsOobRisk() + ggplot2::ylim(0, 1),
  #lr1$model$cboost_restart$plotInbagVsOobRisk() + ggplot2::ylim(0, 1),
  #ncol = 2)

#design = benchmark_grid(
  #tasks = tsk("sonar"),
  #learners = lr1,
  #resamplings = rsmp("cv", folds = 3)
#)

#bmr = benchmark(design, store_models = TRUE)
#bmr$aggregate(msrs(c("classif.auc", "classif.ce")))



