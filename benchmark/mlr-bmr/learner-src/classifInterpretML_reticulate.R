#suppressMessages(library(mlr3))
#suppressMessages(library(mlr3tuning))
#suppressMessages(library(mlrintermbo))
#suppressMessages(library(mlr3learners))
#suppressMessages(library(mlr3extralearners))
#suppressMessages(library(mlr3pipelines))
#suppressMessages(library(paradox))

LearnerClassifInterpretML_reticulate = R6Class("LearnerClassifInterpretML_reticulate",
  inherit = LearnerClassif,
  public = list(

    #' @description
    #' Create a `LearnerClassifInterpretML` object.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamInt$new(id = "max_bins", default = 255L, lower = 100L),
          ParamInt$new(id = "outer_bags", default = 16L, lower = 0L),
          ParamInt$new(id = "inner_bags", default = 0L, lower = 0),
          ParamDbl$new(id = "learning_rate", default = 0.01, lower = 1e-6),
          ParamDbl$new(id = "validation_size", default = 0, lower = 0, upper = 1),
          #ParamInt$new(id = "early_stopping_rounds", default = 1L, lower = 1L, upper = parallel::detectCores()),
          #ParamDbl$new(id = "early_stopping_tolerance", default = 0.0005, lower = 0),
          ParamInt$new(id = "max_rounds", default = 5000L, lower = 200L, upper = 10000L),
          ParamInt$new(id = "n_jobs", default = 1L),
          #ParamInt$new(id = "max_leaves"),
          #ParamInt$new(id = "min_samples_leaf"),
          ParamInt$new(id = "random_state")
        )
      )

      super$initialize(
        id = "classif.interpretML_reticulate",
        packages = "reticulate",
        feature_types = c("numeric", "factor", "integer", "character"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("twoclass")
      )
    }
  ),

  private = list(
    .train = function(task) {

      # Use python virtualenv:
      reticulate::use_python("~/venv/ebm/bin/python", required = TRUE)
      reticulate::use_virtualenv("~/venv/ebm", required = TRUE)
      ebm = reticulate::import("interpret.glassbox")

      #browser()

      pdefaults = self$param_set$default
      pars = self$param_set$values
      for (id in self$param_set$ids()) {
        if (is.null(pars[[id]])) pars[[id]] = pdefaults[[id]]
      }
      self$param_set$values = pars

      X = as.data.frame(task$data(cols = task$feature_names))
      y = abs(as.integer(task$data()[[task$target_names]]) - 2)

      #browser()
      mod = ebm$ExplainableBoostingClassifier(
        max_bins = self$param_set$values$max_bins,
        outer_bags = self$param_set$values$outer_bags,
        inner_bags = self$param_set$values$inner_bags,
        learning_rate = self$param_set$values$learning_rate,
        validation_size = 0.15,
        early_stopping_rounds = self$param_set$values$max_rounds,
        n_jobs = self$param_set$values$n_jobs)
      mod$fit(X,y)
      return(mod)
    },

    .predict = function(task) {
      #browser()
      probs = self$model$predict_proba(X = task$data(cols = task$feature_names))

      pos = task$positive
      neg = task$negative
      #pmat = matrix(c(probs, 1 - probs), ncol = 2L, nrow = length(probs))
      pmat = probs[,c(2, 1)]
      colnames(pmat) = c(pos, neg)
      if (self$predict_type == "prob") {
        list(prob = pmat)
      }
      if (self$predict_type == "response") {
        list(response = ifelse(probs > 0.5, pos, neg))
      } else {
        list(prob = pmat)
      }
    }
  )
)
mlr_learners$add("classif.interpretML_reticulate", LearnerClassifInterpretML_reticulate)



#task = tsk("spam")
#lrn = lrn("classif.interpretML_reticulate", max_rounds = 500L, predict_type = "prob", learning_rate = 0.01,
  #validation_size = 0, random_state = 1337)

#lrn$train(task)
#pred = lrn$predict(task)

#pred
#pred$confusion
#pred$score(msr("classif.auc"))


#X = task$data(cols = task$feature_names)
#y = as.integer(task$data()[[task$target_names]]) - 1

#mod = interpret::ebm_classify(X = X, y = y)
#ebm_predict_proba(mod, X)

