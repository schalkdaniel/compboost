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

checkModelPlotAvailability = function(cboost_obj, check_ggplot = TRUE) {
  if (is.null(cboost_obj$model)) stop("Train the model to get logger data.")
  if ((! requireNamespace("ggplot2", quietly = TRUE)) && check_ggplot) {
    stop("Please install ggplot2 to create plots.")
  }
}

catchInternalException = function(e, x, fn, df = NULL) {
  checkmate::assertString(fn)
  checkmate::assertNumber(df, null.ok = TRUE)
  msg = attr(e, "condition")$message
  sadd = ""
  saddc = ""
  soerror = sprintf("\nOriginal error:\n%s", msg)
  if (! is.null(df)) {
    sadd = sprintf(" with `df = %s`", df)
    if (! is.numeric(x)) {
      nxu = length(unique(x))
      if (df > nxu) {
        saddc = sprintf("\n%s %s",
          sprintf("I also realized that you are trying to add a categorical feature with more degrees of freedom (%s) than number of classes (%s).", df, nxu),
          sprintf("Please try again with `df <= %s`.", nxu))
      }
    }
  }
  if (grepl("toms748", msg)) {
    msg = sprintf("%s\n%s %s %s%s",
      sprintf("Error during the base learner creation for feature %s%s", fn, sadd),
      "Trying to catch univariate optimization error thrown from `boost::math::tools::toms748_solve` (C++).",
      "This most likely happened because the degrees of freedom are set too big.",
      saddc, soerror)
  }
  if (grepl("chol", msg)) {
    msg = sprintf("%s %s %s %s%s",
      "Failed to calculate the Cholesky decomposition.",
      "A reason may be highly correlated columns in the design matrix of the base learner.",
      "Try to increase the penalty/lower the degrees of freedom or decrease the number of basis functions.",
      "For example by using less knots for splines.",
      soerror)
  }
  return(msg)
}

getBaselearnerFeatureType = function(f) {
  blnum = c("pspline", "polynomial", "centered")
  fmn = f$getModelName()
  return(ifelse(fmn %in% blnum, "numeric", "categorical"))
}
