#' @useDynLib compboost, .registration = TRUE
NULL

#' @import Rcpp
#' @import Matrix
NULL

register_mlr3 = function () {
  x = utils::getFromNamespace("mlr_learners", ns = "mlr3")

  x$add("classif.compboost", LearnerClassifCompboost)
  #x$add("regr.compboost", LearnerRegrCompboost)
}

.onLoad = function(libname, pkgname) { # nolint
  # nocov start
  register_mlr3()

  setHook(packageEvent("mlr3", "onLoad"), function(...) register_mlr3(), action = "append")
} # nocov end

dummy_import = function() {
  # this function is required to silence R CMD check
  Matrix::sparseMatrix
  R6::R6Class
  methods::kronecker
}
