#' @useDynLib compboost, .registration = TRUE
NULL

#' @import Rcpp
#' @import checkmate
#' @import paradox
#' @import mlr3misc
#' @importFrom R6 R6Class
#' @importFrom mlr3 mlr_learners LearnerClassif LearnerRegr
#' @importFrom stats predict
NULL

.onLoad = function(libname, pkgname) { # nolint
# nocov start
  mlr_learners$add("classif.compboost", LearnerClassifCompboost)

  mlr3tuningspaces:::add_tuning_space(
    id = "classif.compboost.default",
    values = list(
      iterations    = to_tune(10L, 10000L),
      learning_rate = to_tune(0, 0.5),
      df            = to_tune(1, 10L),
      df_cat        = to_tune(1, 10L)
    ),
    tags = c("default", "classification"),
    learner = "classif.compboost",
    package = "mlr3learners"
  )
} # nocov end

leanify_package()
