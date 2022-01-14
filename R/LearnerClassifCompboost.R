dfAutoselect = function(task, df, nbases) {
  out = list(df = NA, df_cat = NA)
  factor_cols = task$feature_types$id[task$feature_types$type == "factor"]
  if (length(factor_cols) > 0) {
    df_cat_min = min(vapply(
      X = task$data(cols = factor_cols),
      FUN = function(fc) length(unique(fc)),
      FUN.VALUE = integer(1L)
    ))
    df = min(c(df_cat_min, nbases))
    if (df <= 3) df = 5L

    out$num = df
    out$cat = df_cat_min
  } else {
    out$num = nbases
  }
  return(out)
}

checkCores = function(task, ncores) {
  if (ncores > length(task$feature_names)) {
    warning("Number of cores ", ncores, " exceeds number of features ",
      length(task$feature_names), ". The number of cores are set to ", length(task$feature_names), "!")
    return(length(task$feature_names))
  }
  return(ncores)
}

getAUCLoss = function() {
  aucLoss = function(truth, response) return((1 - mlr::measureAUC(response, truth, negative = -1, positive = 1)) * length(truth))
  aucGrad = function(truth, response) return(rep(0, length(truth)))
  aucInit = function(truth) {
    p = mean(truth == 1)
    return(0.5 * p / (1 - p))
  }
  my_auc_loss = LossCustom$new(aucLoss, aucGrad, aucInit)
  return(my_auc_loss)
}

getAUCLossInit = function(init) {
  aucLoss = function(truth, response) return((1 - mlr::measureAUC(response, truth, negative = -1, positive = 1)) * length(truth))
  aucGrad = function(truth, response) return(rep(0, length(truth)))
  aucInit = function(truth) return(init)

  my_auc_loss = LossCustom$new(aucLoss, aucGrad, aucInit)
  return(my_auc_loss)
}

#' @title Component-wise Boosting
#'
#' @name mlr_learners_classif.compboost
#'
#' @description
#' A [mlr3::LearnerClassif] for a compbonent-wise boosting model implemented in
#' [Compboost] in package \CRANpkg{compboost}.
#'
#' @templateVar id classif.compboost
#' @template learner
#' @template seealso_learner
#' @export
LearnerClassifCompboost = R6::R6Class("LearnerClassifCompboost",
  inherit = mlr3::LearnerClassif,
  public = list(

    #' @field transition (`integer(1)`)\cr
    #' Transition value when the model stops.
    transition = NULL,
    iter = NULL,

    #' @description
    #' Creates a [LearnerClassifCompboost] object.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamInt$new(id = "iterations", default = 100L, lower = 1L),
          ParamDbl$new(id = "learning_rate", default = 0.05, lower = 0),
          ParamDbl$new(id = "df", default = 5, lower = 1),
          ParamDbl$new(id = "df_cat", default = 1, lower = 1),
          ParamDbl$new(id = "n_knots", default = 20L, lower = 4),
          ParamLgl$new(id = "df_autoselect", default = TRUE),
          ParamInt$new(id = "ncores", default = 1L, lower = 1L, upper = parallel::detectCores()),

          ParamDbl$new(id = "oob_fraction", default = 0, lower = 0, upper = 0.9),
          ParamLgl$new(id = "use_stopper", default = FALSE),
          ParamLgl$new(id = "just_log", default = TRUE),
          ParamLgl$new(id = "use_stopper_auc", default = FALSE),
          ParamLgl$new(id = "just_log_auc", default = TRUE),
          ParamInt$new(id = "patience", default = 10, lower = 1),
          ParamDbl$new(id = "eps_for_break", default = 0.00001),

          ParamDbl$new(id = "momentum", default = 0, lower = 0),

          ParamFct$new(id = "bin_method", default = "linear", levels = c("linear", "quantile")),
          ParamDbl$new(id = "bin_root", default = 0, lower = 0, upper = 4),

          ParamLgl$new(id = "show_output", default = FALSE),
          ParamInt$new(id = "oob_seed", default = sample(seq_len(1e6), 1), lower = 1L),
          ParamUty$new(id = "additional_auc_task", default = list())
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
    },
    #' @description
    #' Wrapper for the internal train method.
    #' @param iter New iteration
    setToIteration = function(iter) {
      if (is.null(self$transition)) {
        stop("No trained model!")
      } else {
        self$model$cboost$train(iter)
        self$iter = iter
      }
    }
  ),

  private = list(
    .train = function(task) {

      ## Merge default hyperparameter with the specified ones:
      self$param_set$values = mlr3misc::insert_named(self$param_set$default, self$param_set$values)

      ## Check number of cores:
      ncores = checkCores(task, self$param_set$values$ncores)

      use_validation = FALSE
      ## Split data into train and validation:
      if (self$param_set$values$oob_fraction > 0) {
        oobf = self$param_set$values$oob_fraction

        set.seed(self$param_set$values$oob_seed)
        val_idx   = sample(seq_len(task$nrow), trunc(oobf * task$nrow))
        train_idx = setdiff(seq_len(task$nrow), val_idx)

        ttask = task$clone(deep = TRUE)$filter(train_idx)
        vtask = task$clone(deep = TRUE)$filter(val_idx)

        use_validation = TRUE
      } else {
        ttask = task
      }

      ## Automatically select degrees of freedom:
      if (self$param_set$values$df_autoselect) {
        dfs = dfAutoselect(task, self$param_set$values$df, self$param_set$values$n_knots)

        self$param_set$values$df     = dfs$num
        self$param_set$values$df_cat = dfs$cat
      }

      ## Define optimizer:
      if (self$param_set$values$momentum == 0)
        optimizer = OptimizerCoordinateDescent$new(ncores)
      else
        optimizer = OptimizerAGBM$new(self$param_set$values$momentum, ncores)

      out  = list()

      ## Build compboost model:
      model = Compboost$new(
        data = ttask$data(),
        target = ttask$target_names,
        optimizer = optimizer,
        loss = LossBinomial$new(),
        learning_rate = self$param_set$values$learning_rate)

      for (feat in ttask$feature_names) {
        if (is.numeric(ttask$data()[[feat]])) {
          model$addBaselearner(feat, "spline", BaselearnerPSpline, InMemoryData,
            degree = 3,
            n_knots = self$param_set$values$n_knots,
            df = self$param_set$values$df,
            differences = 2,
            bin_root = self$param_set$values$bin_root,
            bin_method = "linear",
            cache_type = "inverse")
        } else {
          checkmate::assertNumeric(self$param_set$values$df_cat, len = 1L, lower = 1)
          if (length(unique(feat)) > self$param_set$values$df_cat)
            stop("Categorical degree of freedom must be smaller than the number of classes (here <",
              length(unique(feat)), ")")
          model$addBaselearner(feat, "ridge", BaselearnerCategoricalRidge, InMemoryData, df = self$param_set$values$df_cat)
        }
      }
      model$addLogger(LoggerTime, FALSE, "time", 0, "microseconds")

      if (self$param_set$values$use_stopper && use_validation) {
        model$addLogger(logger = LoggerOobRisk,
          use_as_stopper = !self$param_set$values$just_log,
          logger_id      = "oob_risk",
          used_loss      = LossBinomial$new(),
          esp_for_break  = self$param_set$values$eps_for_break,
          patience       = self$param_set$values$patience,
          oob_data       = model$prepareData(vtask$data()),
          oob_response   = model$prepareResponse(vtask$data()[[vtask$target_names]]))
      }
      if (self$param_set$values$use_stopper_auc && use_validation) {
        model$addLogger(logger = LoggerOobRisk,
          use_as_stopper = !self$param_set$values$just_log_auc,
          logger_id      = "val_auc",
          used_loss      = getAUCLoss(),
          esp_for_break  = self$param_set$values$eps_for_break,
          patience       = self$param_set$values$patience,
          oob_data       = model$prepareData(vtask$data()),
          oob_response   = model$prepareResponse(vtask$data()[[vtask$target_names]]))
      }
      if (inherits(self$param_set$values$additional_auc_task, "Task")) {
        ts = self$param_set$values$additional_auc_task
        model$addLogger(logger = LoggerOobRisk,
          use_as_stopper = FALSE,
          logger_id      = "test_auc",
          used_loss      = getAUCLoss(),
          esp_for_break  = self$param_set$values$eps_for_break,
          patience       = self$param_set$values$patience,
          oob_data       = model$prepareData(ts$data()),
          oob_response   = model$prepareResponse(ts$data()[[ttask$target_names]]))
      }

      model$train(self$param_set$values$iterations)

      iters = length(model$getSelectedBaselearner())

      ### Reset iterations if early stopping was used:
      if (iters < self$param_set$values$iterations) {
        if (iters <= (self$param_set$values$patience + 1)) {
          stop("CWB was not able to learn anything! Use a featureless learner or something more powerful!")
        } else {
          iters = iters - (self$param_set$values$patience + 1)
          model$train(iters)
        }
      }
      self$transition = iters
      out$cboost      = model

      return(out)
    },

    .predict = function(task) {
      newdata = task$data(cols = task$feature_names)

      if (is.null(self$model$cboost))
        lin_pred = 0
      else
        lin_pred = self$model$cboost$predict(newdata)

      probs = 1 / (1 + exp(-lin_pred))

      pos = self$model$cboost$response$getPositiveClass()
      neg = setdiff(names(self$model$cboost$response$getClassTable()), pos)
      pmat = matrix(c(probs, 1 - probs), ncol = 2L, nrow = length(probs))
      colnames(pmat) = c(pos, neg)
      if (self$predict_type == "prob") {
        return(list(prob = pmat))
      }
      if (self$predict_type == "response") {
        return(list(response = ifelse(probs > self$model$cboost$response$getThreshold(), pos, neg)))
      } else {
        return(list(prob = pmat))
      }
    }
  )
)

#add_tuning_space(
  #id = "regr.ranger.default",
  #values = vals,
  #tags = c("default", "regression"),
  #learner = "regr.ranger",
  #package = "mlr3learners"
#)

if (FALSE) {
  devtools::document()
  library(mlr3tuningspaces)

# tune learner with default search space
instance = tune(
  method = "random_search",
  task = tsk("sonar"),
  learner = lts(lrn("classif.compboost", predict_type = "prob")),
  resampling = rsmp("holdout"),
  measure = msr("classif.auc"),
  term_evals = 5
)

# best performing hyperparameter configuration
instance$result

}
