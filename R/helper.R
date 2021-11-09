assertRcppClass = function(x, x_class, stop_on_error = TRUE) {
  cls = class(x)

  if (! grepl("Rcpp", cls)) {
    stop("Object was not exposed by Rcpp.")
  }
  if (! grepl(x_class, cls)) {
    stop("Object does not belong to class ", x_class, ".")
  }
}

isRcppClass = function(x, x_class) {
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

vectorToResponse = function(vec, target, pos_class = NULL) {
  checkmate::assertCharacter(pos_class, len = 1, null.ok = TRUE)
  # Transform factor or character labels to -1 and 1
  if (! is.numeric(vec)) {
    vec = as.factor(vec)

    if (is.null(pos_class)) pos_class = levels(vec)[1]
    if (length(levels(vec)) > 2) stop("Multiclass classification is not supported.")

    return(ResponseBinaryClassif$new(target, pos_class, as.character(vec)))
  } else {
    return(ResponseRegr$new(target, as.matrix(vec)))
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

checkModelPlotAvailability = function(cboost_obj, check_ggplot = TRUE) {
  if (is.null(cboost_obj$model)) stop("Train the model to get logger data.")
  if ((! requireNamespace("ggplot2", quietly = TRUE)) && check_ggplot) {
    stop("Please install ggplot2 to create plots.")
  }
}
