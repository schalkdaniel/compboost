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
  ccheck = grepl("Rcpp_Loss", class(x))
  if (ccheck || is.null(x)) {
    return(TRUE)
  } else {
    return("`loss` must be an object of class `Rcpp_Loss*`")
  }
}

ichecker = function(x) {
  if (is.null(x)) return(TRUE)
  if (checkmate::testDataFrame(x, col.names = "named")) {
    cnames = c("feat1", "feat2")
    if (any(! vapply(cnames, checkmate::testChoice, logical(1), choices = names(x)))) {
      return("x must have column names {'feat1', 'feat2'}")
    } else {
      n = nrow(x)
      if (! checkmate::testCharacter(x$feat1)) return(checkmate::checkCharacter(x$feat1))
      if (! checkmate::testCharacter(x$feat2)) return(checkmate::checkCharacter(x$feat2))
      if ("isotrop" %in% names(x)) {
        if (! checkmate::testLogical(x$isotrop)) return(checkmate::checkLogical(x$isotrop))
      }
      return(TRUE)
    }
  } else {
    return(checkmate::checkDataFrame(x, col.names = "named"))
  }
}

#' @title Component-wise boosting learner
#'
#' @name mlr_learners.compboost
#'
#' @description
#' A [Learner] for a component-wise boosting model implemented in [compboost::Compboost]
#' in package \CRANpkg{compboost}.
#'
#' @importFrom mlr3 mlr_learners Learner
LearnerCompboost = R6::R6Class("LearnerCompboost", inherit = Learner,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param ttype (`character(1)`) \cr
    #' The task type of the learner. Must be `regr` or `classif`.
    initialize = function(ttype) {
      checkmate::assertChoice(ttype, c("regr", "classif"))
      ps = paradox::ps(
        show_output = paradox::p_lgl(default = FALSE),

        baselearner   = paradox::p_fct(c("linear", "spline", "components"), default = "spline"),
        learning_rate = paradox::p_dbl(0, Inf, default = 0.05, tags = "train"),
        iterations    = paradox::p_int(1, Inf, default = 100, tags = "train"),
        df            = paradox::p_dbl(1, Inf, default = 5, tags = "train"),
        df_cat        = paradox::p_dbl(1, Inf, default = 2, tags = "train"),
        bin_root      = paradox::p_int(0, Inf, default = 0, tags = "train"),

        degree      = paradox::p_int(1, Inf, default = 3, tags = "train", depends = baselearner %in% c("spline", "components")),
        n_knots     = paradox::p_int(1, Inf, default = 20, tags = "train", depends = baselearner %in% c("spline", "components")),
        differences = paradox::p_int(1, Inf, default = 2, tags = "train", depends = baselearner %in% c("spline", "components")),

        optimizer    = paradox::p_uty(default = NULL, tags = "train", custom_check = ochecker),
        loss         = paradox::p_uty(default = NULL, tags = "train", custom_check = lchecker),

        interactions = paradox::p_uty(default = NULL, custom_check = ichecker),
        just_interactions = paradox::p_lgl(default = FALSE),

        oob_fraction = paradox::p_dbl(0, 1, default = 0.3, tags = "train"),
        early_stop   = paradox::p_lgl(default = FALSE),
        patience     = paradox::p_int(1, Inf, default = 5, depends = early_stop == TRUE, tags = "early_stop"),
        eps_for_break = paradox::p_dbl(-Inf, Inf, default = 0, depends = early_stop == TRUE, tags = "early_stop")
      )
      ps$values$show_output = FALSE
      ps$values$iterations = 100L
      ps$values$baselearner = "spline"
      ps$values$df = 5
      ps$values$df_cat = 2
      ps$values$early_stop = FALSE
      ps$values$just_interactions = FALSE

      ptypes = "response"
      props = c("importance", "selected_features")
      if (ttype == "classif") {
        ptypes = c("prob", ptypes)
        props = c("twoclass", props)
      }

      super$initialize(
        id = sprintf("%s.compboost", ttype),
        task_type = ttype,
        packages = "compboost",
        feature_types = c("integer", "numeric", "factor", "ordered"),
        predict_types = ptypes,
        param_set = ps,
        properties = props,
        label = "Component-wise boosting",
        man = sprintf("mlr3::mlr_learners_%s.compboost", ttype))
        # data_format is set by super class `Learner`.
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
        f = compboost::boostLinear
      } else if (self$param_set$values$baselearner == "spline") {
        f = compboost::boostSplines
      } else if (self$param_set$values$baselearner == "components") {
        f = compboost::boostComponents
      }
      if (self$param_set$values$early_stop) {
        if (is.null(self$param_set$values$oob_fraction) || (self$param_set$values$oob_fraction == 0)) {
          stop("`oob_fraction > 0` required for early stopping.")
        }
        pv_es = self$param_set$default[tagIndex(self$param_set, "early_stop")]
        pv_es = mlr3misc::insert_named(pv_es, self$param_set$get_values(tags = "early_stop"))
        pv$stop_args = pv_es
      }
      pv = pv[intersect(names(pv), formalArgs(f))]
      if (! self$param_set$values$show_output) {
        tmp = capture.output({
          cboost = mlr3misc::invoke(f, data = task$data(), target = task$target_names, .args = pv)
        })
      } else {
        cboost = mlr3misc::invoke(f, data = task$data(), target = task$target_names, .args = pv)
      }
      if (! is.null(self$param_set$values$interactions)) {
        iters = pv$iterations
        pv = mlr3misc::insert_named(pv, list(iterations = 0))
        pvs = pv[intersect(names(pv), c("degree", "n_knots", "differences"))]
        if (self$param_set$values$just_interactions) {
          pv = pv[setdiff(names(pv), c(names(pvs), "iterations", "df", "df_cat"))]
          cboost = mlr3misc::invoke(Compboost$new, data = task$data(), target = task$target_names, .args = pv)
        } else {
          cboost = mlr3misc::invoke(f, data = task$data(), target = task$target_names, .args = pv)
        }

        invisible(lapply(seq_len(nrow(self$param_set$values$interactions)), function(k) {
          i = self$param_set$values$interactions[k, ]
          checkmate::assertChoice(i$feat1, choices = task$feature_names)
          checkmate::assertChoice(i$feat2, choices = task$feature_names)
          if (is.null(i$isotrop)) i$isotrop = FALSE
          if (task$feature_types$type[task$feature_types$id == i$feat1] %in% c("integer", "numeric")) {
            df1 = self$param_set$values$df
          } else {
            df1 = self$param_set$values$df_cat
          }
          if (task$feature_types$type[task$feature_types$id == i$feat2] %in% c("integer", "numeric")) {
            df2 = self$param_set$values$df
          } else {
            df2 = self$param_set$values$df_cat
          }
          mlr3misc::invoke(cboost$addTensor, feature1 = i$feat1, feature2 = i$feat2,
            isotrop = i$isotrop, df1 = df1, df2 = df2, .args = pvs)
        }))

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

      # The score are probabilities in classification and predictions in regression:
      score = as.numeric(self$model$predict(newdata, TRUE))

      # In regression, just return the score:
      if (task$task_type == "regr") {
        return(list(response = score))
      # For classification distinguish between response and prob type:
      } else if (task$task_type == "classif") {
        cnames = names(self$model$response$getClassTable())
        pos = self$model$response$getPositiveClass()
        neg = setdiff(cnames, pos)
        if ("response" %in% self$predict_type) {
          return(list(response = ifelse(score > self$model$response$getThreshold(), pos, neg)))
        } else if ("prob" %in% self$predict_type) {
          pmat = cbind(score, 1 - score)
          colnames(pmat) = c(pos, neg)
          return(list(prob = pmat))
        }
      }
    }
  )
)

#' @title Classification component-wise boosting learner
#'
#' @name mlr_learners_classif.compboost
#'
#' @description
#' A [Learner] for a component-wise boosting model implemented in [compboost::Compboost]
#' in package \CRANpkg{compboost}.
#'
#' @examples
#' l = lrn("classif.compboost", baselearner = "components",
#'   df = 5, iterations = 20)
#' task = tsk("german_credit")
#' l$train(task)
#' l$selected_features()
#' @export
LearnerClassifCompboost = R6::R6Class("LearnerClassifCompboost", inherit = LearnerCompboost,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize("classif")
    }
  )
)

#' @title Regression component-wise boosting learner
#'
#' @name mlr_learners_regr.compboost
#'
#' @description
#' A [Learner] for a component-wise boosting model implemented in [compboost::Compboost]
#' in package \CRANpkg{compboost}.
#'
#' @examples
#' l = lrn("regr.compboost", baselearner = "linear", iterations = 50)
#' task = tsk("mtcars")
#' l$train(task)
#' l$importance()
#' @export
LearnerRegrCompboost = R6::R6Class("LearnerRegrCompboost", inherit = LearnerCompboost,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize("regr")
    }
  )
)

