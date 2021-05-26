library(dplyr)
library(tidyr)

task_table = c(
  #"54" = "Hepatitis",
  #"37" = "Diabetes",
  #"31" = "German Credit",
  #"4534" = "Analcat Halloffame",
  "spam" = "Spam",
  #"168337" = "Guillermo",
  "7592" = "Adult",
  "9977" = "namao",
  "168335" = "MiniBooNE",
  "albert" = "Albert",
  "359994" = "SF Police")# Incidents")

learner_table = c(

  cwb = "CWB (nb, no mstop)",
  cwb_bin = "CWB (b, no mstop)",
  acwb = "ACWB (nb, no mstop)",
  acwb_bin = "ACWB (b, no mstop)",
  hcwb = "hCWB (nb, no mstop)",
  hcwb_bin = "hCWB (b, no mstop)",

  cwb_notune = "CWB (nb, no mstop, notune)",
  cwb_notune_bin = "CWB (b, no mstop, notune)",
  acwb_notune = "ACWB (nb, no mstop, notune)",
  acwb_notune_bin = "ACWB (b, no mstop, notune)",
  hcwb_notune = "hCWB (nb, no mstop, notune)",
  hcwb_notune_bin = "hCWB (b, no mstop, notune)",

  cboost1 = "CWB (nb)",
  cboost_bin1 = "CWB (b)",
  cboost4 = "CWB CA (nb)",
  cboost_bin4 = "CWB CA (b)",
  cboost3 = "ACWB (nb)",
  cboost_bin3 = "ACWB (b)",
  cboost2 = "hCWB (nb)",
  cboost_bin2 = "hCWB (b)",
  ranger = "Random forest",
  xgboost = "Boosted trees",
  interpretML = "EBM",
  gamboost = "CWB (mboost)")

extractStringBetween = function(str, left, right) {
  tmp = sapply(strsplit(str, left), function(x) x[2])
  sapply(strsplit(tmp, right), function(x) x[1])
}
getTaskFromFile = function(file_name) {
  tsks = extractStringBetween(file_name, "-task", "-classif")
  unname(task_table[sapply(tsks, function(ts) which(ts == names(task_table)))])
}
getLearnerFromFile = function(file_name) {
  ext = tools::file_ext(file_name)
  lrns = extractStringBetween(file_name, "-classif_lrn_", paste0("[.]", ext))
  lrns_idx = sapply(lrns, function(l) which(l == names(learner_table)))
  unname(learner_table[lrns_idx])
}




base_dir = "~/repos/compboost/benchmark/mlr-bmr/"
cwb_variants = learner_table[1:12]
files = list.files(paste0(base_dir, "log-files"), full.names = TRUE)
files = files[grepl("mlr3log", files)]
files_cv = files[getLearnerFromFile(files) %in% cwb_variants]


#ftest = "~/repos/compboost/benchmark/mlr-bmr/log-files/mlr3log-2021-05-25-taskalbert-classif_lrn_cwb_bin.txt"
#bm_trace = readLines(ftest)

extractCompboostMsg = function(lines) {
  lines = lines[grepl("LGCOMPBOOST", lines)]
  extract = vapply(
    X = strsplit(lines, split = "[[]LGCOMPBOOST[]] "),
    FUN.VALUE = character(1L),
    FUN = function(l) gsub(" ", "", l[2], fixed = TRUE))
}
extractIters = function(lines) {
  lines = extractCompboostMsg(lines)
  lines = lines[grepl("iterations:", lines)]
  stringr::str_remove(lines, "iterations:")
}
extractDFIter = function(file) {
  bm_trace = readLines(file)
  extrct = extractIters(bm_trace)
  tmp = read.csv(text = paste0(extrct, collapse = "\n"), header = FALSE, quote = "\'")
  names(tmp)[c(2, 4, 6)] = c("iters_cwb", "iters_acwb", "iters_restart")
  tmp %>%
    select(-starts_with("V")) %>%
    pivot_longer(cols = starts_with("iters_"), names_to = "method", values_to = "iters") %>%
    select(-method) %>%
    na.omit() %>%
    mutate(task = getTaskFromFile(file), learner = getLearnerFromFile(file))
}

ll_iter = lapply(files_cv, extractDFIter)
save(ll_iter, file = "ll_iter.Rda")


