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
  "classif_lrn_ranger")#,
  #"classif_lrn_interpretML")


tsks_classif = tsks_classif[tsks_classif$name %in% c("54", "37", "31"), ]

#idx_large_data = c(1, 2, 4, 9)
#idx_other = setdiff(seq_len(nrow(tsks_classif)), idx_large_data)

#if (FALSE) {
  #idx_test = 5
  #tsks_classif = tsks_classif[idx_test, ]
  #learners = learners[9]
#}

# Bigger server has much more RAM, therefore, use large data on bigger server.
#if (parallel::detectCores() > 30) {
  #tsks_classif = tsks_classif[idx_large_data, ]
#} else {
  #tsks_classif = tsks_classif[idx_other, ]
#}

if (! dir.exists("res-results")) dir.create("res-results")
if (! dir.exists("log-files")) dir.create("log-files")

for (i in seq_len(nrow(tsks_classif))) {
  cat("[", as.character(Sys.time()), "] Task ", tsks_classif$name[i],
    " (", i, "/", nrow(tsks_classif), ")\n", sep = "")
  for (j in seq_along(learners)) {
    done = list.files("res-results")
    is_done = any(grepl(learners[j], done) & grepl(tsks_classif$name[i], done))

    cat("\t[", as.character(Sys.time()), "] Start benchmark of ", learners[j],
      " (", j, "/", length(learners), ")\n", sep = "")

    if (! is_done) {
      config = list(date = as.character(Sys.time()), task = tsks_classif$name[i],
        type = tsks_classif$type[i], learner = learners[j])

      save(config, file = "config.Rda")

      lf = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-task", config$task, "-", learners[j], "-log.txt")
      mf = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-task", config$task, "-", learners[j], "-mem.txt")

      system("chmod +x measure-mem.sh")
      system(paste0("./measure-mem.sh ", mf), wait = FALSE)
      system(paste0("Rscript mlr-run.R > ", lf))
      system("pkill -x measure-mem.sh && pkill -x inotifywait")
      cat("\t[", as.character(Sys.time()), "] Finish benchmark of ", learners[j], "\n", sep = "")
    } else {
      cat("\t[", as.character(Sys.time()), "] INTERRUPT: Job already done\n", sep = "")
    }
  }
}
