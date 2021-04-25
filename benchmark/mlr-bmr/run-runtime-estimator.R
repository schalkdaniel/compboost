tsks_classif = rbind(
  data.frame(type = "oml", name = "54"),           # Hepatitis
  data.frame(type = "oml", name = "37"),           # Diabetes
  data.frame(type = "oml", name = "4534"),         # Analcat Halloffame
  data.frame(type = "mlr", name = "spam"),         # Spam
  data.frame(type = "oml", name = "7592"),         # Adult
  data.frame(type = "oml", name = "168335"),       # MiniBooNE
  data.frame(type = "script", name = "albert"),    # Albert
  data.frame(type = "oml", name = "168337"),       # Guillermo
  data.frame(type = "oml", name = "359994")        # SF Police Incidents
)

learners = c(
  "classif_lrn_cboost1",            # CWB (without binning)
  "classif_lrn_cboost_bin1",        #     (with binning)
  "classif_lrn_cboost4",            # CWB cosine annealing (without binning)
  "classif_lrn_cboost_bin4",        #                      (with binning)
  "classif_lrn_cboost3",            # ACWB (without binning)
  "classif_lrn_cboost_bin3",        #      (with binning)
  "classif_lrn_cboost2",            # hCWB (without binning)
  "classif_lrn_cboost_bin2",        #      (with binning)
  "classif_lrn_xgboost",            # Boosted trees
  "classif_lrn_gamboost",           # CWB (mboost variant)
  "classif_lrn_ranger",             # Random forest
  "classif_lrn_interpretML"         # Interpret
)

for (i in seq_len(nrow(tsks_classif))) {
  for (j in seq_along(learners)) {
    config_runtime = list(tidx = i, lidx = j)
    save(config_runtime, file = "config-runtime-estimator.Rda")
    system("Rscript runtime-estimator.R")
  }
}
