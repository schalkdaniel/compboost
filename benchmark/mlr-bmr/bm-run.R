tsks_classif = rbind(
  data.frame(type = "oml", name = c(359994L, 168337L, 7592L, 168335L, 31L, 37L, 54L, 4534L)),
  data.frame(type = c("script", "mlr"), name = c("albert", "spam")))

counter = 1L
for (i in seq_along(tsks_classif)) {
  cat("[", as.character(Sys.time()), "] Task ", tsks_classif$name[i], "(", counter, "/", nrow(tsks_classif), ")\n")

  if (! dir.exists("res-results")) dir.create("res-results")
  if (! dir.exists("log-files")) dir.create("log-files")

  config = list(task = tsks_classif$name[i], type = tsks_classif$type[i])
  save(config, file = "config.Rda")

  cat("\t[", as.character(Sys.time()), "] Start benchmark\n")
  log_file = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-", config$task, ".txt")
  system(paste0("Rscript mlr-run.R > ", log_file))
  cat("\t[", as.character(Sys.time()), "] Finish benchmark\n")

  counter = counter + 1L
}
