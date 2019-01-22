.assertRcppClass = function (x, x.class, stop.when.error = TRUE)
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

.vectorToResponse = function (vec, target)
{
  # Transform factor or character labels to -1 and 1
  if (! is.numeric(vec)) {
    vec = as.factor(vec)

    if (length(levels(vec)) > 2) {
      stop("Multiclass classification is not supported.")
    }
    # self$positive.category = levels(vec)[1]
    # Transform to vector with -1 and 1:
    vec = as.integer(vec) * (1 - as.integer(vec)) + 1
    return (ResponseBinaryClassif$new(target, as.matrix(vec)))
  } else {
    return (ResponseRegr$new(target, as.matrix(vec)))
  }
}


# .vectorToResponse = function (vec, target)
# {
#   # Classification:
#   if (is.character(vec)) {
#     vec = as.factor(vec)
#
#     if (length(levels(vec)) == 2) return (ResponseBinaryClassif$new(target, as.matrix(vec)))
#     if (length(levels(vec)) > 2) stop("Multiclass classification is not supported.")
#   }
#   # Regression:
#   if (is.numeric(vec)) return (ResponseRegr$new(target, as.matrix(vec)))
# }
