#' @useDynLib compboost, .registration = TRUE
NULL

#' @import Rcpp
#' @import Matrix
NULL

dummy_import = function() {
  # this function is required to silence R CMD check
  Matrix::sparseMatrix
  R6::R6Class
  methods::kronecker
}
