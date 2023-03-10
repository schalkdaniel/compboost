tagIndex = function(ps, tag) {
  vapply(ps$tags, function(tgs) tag %in% tgs, logical(1))
}

ochecker = function(x) {
  ccheck = grepl("Rcpp_Optimizer", class(x))
  if (ccheck || is.null(x)) {
    return(TRUE)
  } else {
    return("`optimizer` must be an object of class `Rcpp_Optimizer*`")
  }
}

lchecker = function(x) {
  ccheck = grepl("Rcpp_Optimizer", class(x))
  if (ccheck || is.null(x)) {
    return(TRUE)
  } else {
    return("`loss` must be an object of class `Rcpp_Loss*`")
  }
}

ichecker = function(x) {
  checkmate::checkDataFrame(x, col.names = c("feat1", "feat2"), null.ok = TRUE)
}

#' @title Classification component-wise boosting learner
#'
#' @name mlr_learners_classif.compboost
#'
#' @description
#' A [LearnerClassif] for a component-wise boosting model implemented in [compboost::Compboost]
#' in package \CRANpkg{compboost}.
#'
#' @section Initial parameter values:
#' * Parameter `xval` is initialized to 0 in order to save some computation time.
#'
#' @importFrom mlr3 mlr_learners LearnerClassif
#' @export
LearnerClassifCompboost = R6::R6Class("LearnerClassifCompboost", inherit = LearnerClassif,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = paradox::ps(
        show_output = paradox::p_lgl(default = TRUE),

        baselearner   = paradox::p_fct(c("linear", "spline", "components"), default = "spline"),
        learning_rate = paradox::p_dbl(0, Inf, default = 0.05, tags = "train"),
        iterations    = paradox::p_int(1, Inf, default = 100, tags = "train"),
        df            = paradox::p_dbl(1, Inf, default = 5, tags = "train"),
        df_cat        = paradox::p_dbl(1, Inf, default = 2, tags = "train"),
        bin_root      = paradox::p_int(0, Inf, default = 0, tags = "train"),

        degree      = paradox::p_int(1, Inf, default = 3, tags = "train", depends = baselearner == "spline"),
        n_knots     = paradox::p_int(1, Inf, default = 20, tags = "train", depends = baselearner == "spline"),
        penalty     = paradox::p_dbl(0, Inf, default = 0, tags = "train", depends = baselearner == "spline"),
        differences = paradox::p_int(1, Inf, default = 2, tags = "train", depends = baselearner == "spline"),

        optimizer    = paradox::p_uty(default = NULL, tags = "train", custom_check = ochecker),
        loss         = paradox::p_uty(default = NULL, tags = "train", custom_check = lchecker),
        interactions = paradox::p_uty(default = NULL),

        val_fraction = paradox::p_dbl(0, 1, default = 0.3),
        early_stop   = paradox::p_lgl(default = FALSE, tags = "early_stop"),
        patience     = paradox::p_int(1, Inf, default = 5, depends = early_stop == TRUE, tags = "early_stop"),
        eps_for_break = paradox::p_dbl(-Inf, Inf, default = 0, depends = early_stop == TRUE, tags = "early_stop")
      )
      ps$values$show_output = FALSE
      ps$values$iterations = 100L
      ps$values$baselearner = "spline"
      ps$values$df = 5
      ps$values$df_cat = 2
      ps$values$early_stop = FALSE

      super$initialize(
        id = "classif.compboost",
        packages = "compboost",
        feature_types = c("integer", "numeric", "factor", "ordered"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("twoclass", "missings", "importance", "selected_features"),
        label = "Component-wise boosting",
        man = "mlr3::mlr_learners_classif.compboost"
      )
    },

    #' @description
    #' The importance scores are extracted from the model slot `$calculateFEatureImportance()`.
    #' @return Named `numeric()`.
    importance = function() {
      if (is.null(self$model)) {
        stopf("No model stored")
      }
      # importance is only present if there is at least on split
      return(self$model$calculateFeatureImportance())
    },

    #' @description
    #' Selected features are extracted from the model slot `$getSelectedBaselearner()`.
    #' @return `character()`.
    selected_features = function() {
      if (is.null(self$model)) {
        stopf("No model stored")
      }
      names(table(self$model$getSelectedBaselearner()))
    },

    #' @description
    #' Save the model to a JSON file.
    #' @param file (`character(1)`\cr
    #' Name/path to the file.
    #' @return `character(1)`
    save_json = function (file) {
      checkmate::assertString(file)
      if (is.null(self$model)) {
        stopf("No model stored")
      }
      self$model$saveToJson(file)
    }
  ),

  private = list(
    .train = function(task) {
      pv = self$param_set$get_values(tags = "train")
      if (self$param_set$values$baselearner == "linear") {
        f = compboost::pboostLinear
      } else if (self$param_set$values$baselearner == "spline") {
        f = compboost::boostSplines
      }
      if (self$param_set$early_stop) {
        if (is.null(self$param_set$values$val_fraction) || (self$param_set$values$val_fraction == 0)) {
          stop("`val_fraction > 0` required for early stopping.")
        }
        pv_es = self$param_set$default[tagIndex(self$param_set, "early_stop")]
        mlr3misc::insert_named(pv_es, self$param_set$get_values(tags = "early_stop"))
        pv$stop_args = pv_es
      }
      if (! self$param_set$values$show_output) {
        tmp = capture.output({
          cboost = mlr3misc::invoke(f, data = task$data(), target = task$target_names, .args = pv)
        })
      } else {
        cboost = mlr3misc::invoke(f, data = task$data(), target = task$target_names, .args = pv)
      }
      if (! is.null(self$param_set$values$interactions)) {
        iters = pv$iterations
        pv = insert_named(pv, list(iterations = 0))
        cboost = mlr3misc::invoke(f, data = task$data(), target = task$target_names, .args = pv)

        # Add tensors here:
        #idat = self$param_set$values$interactions
        #for (i in seq_len(nrow(idat)))
        #cboost$addTensor(idat$feat1[i], idat$feat2[i])
        if (! self$param_set$values$show_output) {
          tmp = capture.output(cboost$train(iters))
        } else {
          cboost$train(iters)
        }
      }
      return(cboost)
    },

    .predict = function(task) {
      newdata = task$data(cols = task$feature_names)
      response = prob = NULL

      if ("response" %in% self$predict_type) {
        return(self$model$predict(newdata, as_response = TRUE))
      } else if ("prob" %in% self$predict_type) {
        return(self$model$predict(newdata, as_response = FALSE))
      }

      list(response = response, prob = prob)
    }
  )
)
