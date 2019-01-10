assertRcppClass = function (x, x.class, stop.when.error = TRUE)
{
  cls = class(x)
  rcpp.class = TRUE
  if (! grepl("Rcpp", cls)) {
    stop("Object was not exposed by Rcpp.")
  }
  if (! grepl(x.class, cls)) {
    stop("Object does not belong to class ", x.class, ".")
  }
}