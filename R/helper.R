assertRcppClass = function (x, x_class, stop_when.error = TRUE)
{
  cls = class(x)
  
  if (! grepl("Rcpp", cls)) {
    stop("Object was not exposed by Rcpp.")
  }
  if (! grepl(x_class, cls)) {
    stop("Object does not belong to class ", x_class, ".")
  }
}

isRcppClass = function (x, x_class)
{
  cls = class(x)
  is_rcpp_class = TRUE
  
  if (! grepl("Rcpp", cls)) {
    is_rcpp_class = FALSE
  }
  if (! grepl(x_class, cls)) {
    is_rcpp_class = FALSE
  }
  return(is_rcpp_class)
}

vectorToResponse = function (vec, target)
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

rowwiseTensor <- function(A, B)
{
  m <- NROW(A)
  if(m != NROW(B)) stop("Matrix must have the same number of rows.")
  # create an empty matrix for the result
  C <- matrix(nrow = m, ncol = NCOL(A) * NCOL(B))
  for(i in 1:m){
    # but there is a method for the 'conventional' TP:
    C[i,] <- kronecker(A[i,], B[i,])
  }
  return(C)
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

checkModelPlotAvailability = function (cboost_obj, check_ggplot = TRUE)
{
  if (is.null(cboost_obj$model)) { stop("Train the model to get logger data.") }
  if ((! requireNamespace("ggplot2", quietly = TRUE)) && check_ggplot) { stop("Please install ggplot2 to create plots.") }
}
