if (FALSE) {
  install.packages(c("processx", "callr", "mlr3", "mlr3tuning", "mlr3learners", "mlr3pipelines",
    "paradox", "xgboost", "ranger", "mboost", "mlr3oml", "reticulate", "mlrMBO",
    "mlrintermbo", "DiceKriging"))
  remotes::install_github("mlr-org/mlr3extralearners")
  remotes::install_github("schalkdaniel/compboost", ref = "ba044d3a6f6814080eb097acca2e59fd8bad9805")
}

tsks_classif = rbind(
  data.frame(type = "oml", name = "54"),           # Hepatitis
  data.frame(type = "oml", name = "37"),           # Diabetes
  data.frame(type = "oml", name = "4534"),         # Analcat Halloffame
  data.frame(type = "mlr", name = "spam"),         # Spam
  data.frame(type = "oml", name = "7592"),         # Adult
  data.frame(type = "oml", name = "168335"),       # MiniBooNE
  data.frame(type = "script", name = "albert"),    # Albert
  data.frame(type = "oml", name = "168337"),       # Guillermo
  data.frame(type = "oml", name = "359994"),       # SF Police Incidents

  # Additional tasks:
  data.frame(type = "oml", name = "3945"),         # KDDCup09_appetency (231 feats, 50' feats)
  data.frame(type = "oml", name = "9977"),         # namao (119 feats, 34465 rows)
  data.frame(type = "oml", name = "168908"),       # Christine (1637 feats, 5418 rows)
  data.frame(type = "oml", name = "168896")        # gina (970 feats, 3153 rows)

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

# Each sever gets just a few tasks to efficiently distribute
# over several machines (name of the server is saved in '/etc/hostname'
#
# @param on_host logical(1) Indicate if host selector should be applied.
#   If `on_host = FALSE` all tasks are selected.
serverSelector = function(on_host = FALSE) {
  if (on_host) {
    host = readLines("/etc/hostname")
    host_tasks = list(
      "bigger-benchmarks2" = c(1, 2),
      "cacb1" = c(3, 4),
      "cacb2" = 5,
      "cacb3" = 4,
      "cacb4" = 7,
      "cacb5" = 8,
      "cacb6" = 9)
    idx = host_tasks[[host]]
    if (is.null(idx[1])) stop("Server is not one of {", paste(names(host_tasks), collapse = ", "), "}")
    return(idx)
  } else {
    return(seq_len(nrow(tsks_classif)))
  }
}
# Apply selector:
tsks_classif = tsks_classif[serverSelector(TRUE), ]

if (! dir.exists("res-results")) dir.create("res-results")
if (! dir.exists("log-files")) dir.create("log-files")

for (i in seq_len(nrow(tsks_classif))) {
  cat("[", as.character(Sys.time()), "] Task ", as.character(tsks_classif$name[i]),
    " (", i, "/", nrow(tsks_classif), ")\n", sep = "")
  for (j in seq_along(learners)) {
    # Check if job was already executed:
    done = list.files("res-results")
    is_done = any(grepl(learners[j], done) & grepl(tsks_classif$name[i], done))

    cat("\t[", as.character(Sys.time()), "] Start benchmark of ", learners[j],
      " (", j, "/", length(learners), ")\n", sep = "")

    if (! is_done) {
      # Define job. A job is one task + one algorithm:
      config = list(date = as.character(Sys.time()), task = tsks_classif$name[i],
        type = tsks_classif$type[i], learner = learners[j])

      # Save config that it can be loaded by other R sessions:
      save(config, file = "config.Rda")

      # Log file names:
      lf = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-task",
        config$task, "-", learners[j], "-log.txt")
      mf = paste0("log-files/log-", format(Sys.Date(), "%Y-%m-%d"), "-task",
        config$task, "-", learners[j], "-mem.txt")

      # Make memory measurement executeable and start memory measurement:
      system("chmod +x measure-mem.sh")
      system(paste0("./measure-mem.sh ", mf), wait = FALSE)

      # Run benchmark for given configuration in a new R session:
      system(paste0("Rscript mlr-run.R > ", lf))

      # Kill memory measurement:
      system("pkill -x measure-mem.sh && pkill -x inotifywait")
      cat("\t[", as.character(Sys.time()), "] Finish benchmark of ", learners[j], "\n", sep = "")
    } else {
      cat("\t[", as.character(Sys.time()), "] INTERRUPT: Job already done\n", sep = "")
    }
  }
}
