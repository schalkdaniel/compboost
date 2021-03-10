tsks_classif = rbind(
  data.frame(type = "oml", name = c(359994L, 168337L, 7592L, 168335L, 31L, 37L, 54L, 4534L)),
  data.frame(type = c("script", "mlr"), name = c("albert", "spam")))

learners = c("classif_lrn_cboost1",
  "classif_lrn_cboost2",
  "classif_lrn_cboost3",
  "classif_lrn_cboost4",
  "classif_lrn_cboost_bin1",
  "classif_lrn_cboost_bin2",
  "classif_lrn_cboost_bin3",
  "classif_lrn_cboost_bin4",
  "classif_lrn_xgboost",
  "classif_lrn_gamboost",
  #"classif_lrn_rpart",
  "classif_lrn_ranger")

idx_large_data = c(1, 2, 4, 9)
idx_other = setdiff(seq_len(nrow(tsks_classif)), idx_large_data)

if (FALSE) {
  idx_test = 5
  tsks_classif = tsks_classif[idx_test, ]
}

## Bigger server has much more RAM, therefore, use large data on bigger server.
if (parallel::detectCores() > 30) {
  tsks_classif = tsks_classif[idx_large_data, ]
} else {
  tsks_classif = tsks_classif[idx_other, ]
}

counter = 1L
if (! dir.exists("res-results")) dir.create("res-results")
if (! dir.exists("log-files")) dir.create("log-files")

for (i in seq_len(nrow(tsks_classif))) {
  cat("[", as.character(Sys.time()), "] Task ", tsks_classif$name[i],
    " (", counter, "/", nrow(tsks_classif), ")\n", sep = "")
  for (j in seq_along(learners)) {
    config = list(task = tsks_classif$name[i], type = tsks_classif$type[i], learner = learners[j])
    save(config, file = "config.Rda")

    cat("\t[", as.character(Sys.time()), "] Start benchmark of ", learners[j], " (", j, "/", length(learners), ")\n", sep = "")
    lf = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-task", config$task, "-", learners[j], "-log.txt")
    mf = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-task", config$task, "-", learners[j], "-mem.txt")

    system("chmod +x measure-time.sh")
    #system(paste0("./measure-time.sh ", mf, " & echo $! > pid.txt"))
    system(paste0("./measure-time.sh ", mf))
    system(paste0("Rscript mlr-run.R > ", lf))
    pid = readLines("pid.txt")
    #system(paste0("kill ", pid))
    system("pkill -x measure-time.sh && pkill -x inotifywait")
    cat("\t[", as.character(Sys.time()), "] Finish benchmark of ", lr, "\n", sep = "")

    counter = counter + 1L
  }
}
