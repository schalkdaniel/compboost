extractArchive = function(bmr) {
  extractCWBIters = function(l) {
    mod = l$model
    lid = names(mod)[grepl("ps_", names(mod))]
    mod0 = mod[[lid]]
    tt = mod0$train_time

    cwb_ids = c("ps_cboost1", "ps_cboost2", "ps_cwb1", "ps_cwb1_bin")
    acwb_ids = c("ps_cboost_nesterov1_norestart", "ps_cboost_nesterov2_norestart", "ps_cwb2", "ps_cwb2_bin")
    hcwb_ids = c("ps_cboost_nesterov1", "ps_cboost_nesterov2", "ps_cwb3", "ps_cwb3_bin")

    icwb  = NA
    iacwb = NA
    ihcwb = NA

    rintercept = NA
    rcwb  = NA
    racwb = NA
    rhcwb = NA

    if (lid %in% cwb_ids) {
      icwb = length(mod0$model$cboost$getSelectedBaselearner())
      rintercept = mod0$model$cboost$getInbagRisk()[1]
      rcwb = tail(mod0$model$cboost$getInbagRisk(), 1)
    }
    if (lid %in% acwb_ids) {
      iacwb = length(mod0$model$cboost$getSelectedBaselearner())
      rintercept = mod0$model$cboost$getInbagRisk()[1]
      racwb = tail(mod0$model$cboost$getInbagRisk(), 1)
    }
    if (lid %in% hcwb_ids) {
      iacwb = length(mod0$model$cboost$getSelectedBaselearner())
      racwb = tail(mod0$model$cboost$getInbagRisk(), 1)
      rintercept = mod0$model$cboost$getInbagRisk()[1]
      if ("cboost_restart" %in% names(mod0$model)) {
        ihcwb = length(mod0$model$cboost_restart$getSelectedBaselearner())
        rhcwb = tail(mod0$model$cboost_restart$getInbagRisk(), 1)
      }
    }
    data.frame(train_time = tt, iters_cwb = icwb, iters_acwb = iacwb, iters_restart = ihcwb,
     risk_intercept = rintercept, risk_cwb = rcwb, risk_acwb = racwb, risk_hcwb = rhcwb, lid = lid)
  }

  rr = bmr$resample_results$resample_result[[1]]

  out = lapply(as.data.table(rr)$learner, function(ll) {
    arx = ll$archive

    b = as.data.table(arx$benchmark_result)$learner
    lid = b[[1]]$graph$ids(TRUE)
    lid = lid[grepl("ps_", lid)]

    ll_arx = list()

    #browser()
    ll_arx[[1]] = arx$data
    ll_arx[[2]] = arx$benchmark_result$score(msrs(c("classif.auc", "time_train")))
    if (grepl("cboost", lid) || grepl("cwb", lid)) ll_arx[[3]] = do.call(rbind, lapply(b, extractCWBIters))

    return(do.call(cbind, ll_arx))
  })
  return(out)
}
