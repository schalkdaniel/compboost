task_table = c(
  "54" = "Hepatitis",
  "37" = "Diabetes",
  "31" = "German Credit",
  "4534" = "Analcat Halloffame",
  "spam" = "Spam",
  "168337" = "Guillermo",
  "7592" = "Adult",
  "168335" = "MiniBooNE",
  "albert" = "Albert",
  "359994" = "SF Police Incidents")

learner_table = c(
  cboost1 = "CWB (no binning)",
  cboost_bin1 = "CWB (binning)",
  cboost4 = "CWB Cosine Annealing (no binning)",
  cboost_bin4 = "CWB Cosine Annealing (binning)",
  cboost3 = "ACWB (no binning)",
  cboost_bin3 = "ACWB (binning)",
  cboost2 = "hCWB (no binning)",
  cboost_bin2 = "hCWB (binning)",
  ranger = "Random forest",
  xgboost = "Boosted trees",
  gamboost = "CWB (mboost)",
  interpretML = "interpretML")

extractStringBetween = function (str, left, right) {
  tmp = sapply(strsplit(str, left), function (x) x[2])
  sapply(strsplit(tmp, right), function (x) x[1])
}
getTaskFromFile = function (file_name) {
  tsks = extractStringBetween(file_name, "-task", "-classif")
  unname(task_table[sapply(tsks, function (ts) which(ts == names(task_table)))])
}
getLearnerFromFile = function (file_name) {
  lrns = extractStringBetween(file_name, "-classif_lrn_", "[.]Rda")
  lrns_idx = sapply(lrns, function (l) which(l == names(learner_table)))
  unname(learner_table[lrns_idx])
}
extractBMRData = function (file_name) {
  lapply(file_name, function (file) {
    load(file)
    tmp = bmr_res[[3]]
    idx_select = sapply(
      c("classif.auc", "classif.ce", "classif.bbrier", "time_train", "time_predict", "time_both"),
      function (m) which(m == names(tmp)))
    tmp = tmp[, idx_select]
    tmp$task = getTaskFromFile(file)
    tmp$learner = getLearnerFromFile(file)
    return(tmp)
  })
}

files = list.files("res-results", full.names = TRUE)
#getTaskFromFile(files)
#getLearnerFromFile(files)

df_bmr = do.call(rbind, extractBMRData(files))

save(df_bmr, file = "df_bmr.Rda")
load("bmr-aggr/df_bmr.Rda")

df_bmr$learner = factor(df_bmr$learner, labels = learner_table[-12])

if (FALSE) {
  library(ggplot2)
  library(dplyr)

  df_bmr %>%
    group_by(learner, task) %>%
    summarize(med = median(classif.auc[1:3]), sd = sd(classif.auc[1:3]))

    summarize(med = median(classif.auc), sd = sd(classif.auc))

  ggplot(df_bmr, aes(x = learner, y = classif.auc, color = learner, fill = learner)) +
    geom_boxplot(aes(alpha = ifelse(grepl("[(]binning[)]", learner), 0.2, 0.4))) +
    facet_wrap(. ~ task, ncol = 3, scales = "free")
}
