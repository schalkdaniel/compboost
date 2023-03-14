#' @param bin_root (`integer(1)`)\cr
#' The binning root to reduce the data to \eqn{n^{1/\text{binroot}}} data points
#' (default `bin_root = 1`, which means no binning is applied).
#' A value of `bin_root = 2` is suggested for the best approximation
#' error (cf. *Wood et al. (2017) Generalized additive models for gigadata:
#' modeling the UK black smoke network daily data*).
