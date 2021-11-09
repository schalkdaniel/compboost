predict.compboostExtract = function(object, newdata) {
  feats = names(object)
  feats = feats[feats != "offset"]
  out = lapply(feats, function(ft) {
    if (! ft %in% names(newdata)) {
      warning("New data should contain all features used for building the model! Feature ", ft, " is missing.")
      return(NULL)
    } else {
      eff = object[[ft]]$predict(newdata[[ft]])
      pe = eff$linear + eff$nonlinear
      return(list(pe = pe, effects = eff))
    }
  })
  names(out) = feats
  pred = rowSums(do.call(cbind, lapply(out, function(x) x$pe))) + object$offset
  return(list(pred = pred, pe = out, offset = object$offset))
}


