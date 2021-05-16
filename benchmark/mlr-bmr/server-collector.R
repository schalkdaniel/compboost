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

learners = c( "classif_lrn_cboost1",            # CWB (without binning)
  "classif_lrn_cboost_bin1",                    #     (with binning)
  "classif_lrn_cboost4",                        # CWB cosine annealing (without binning)
  "classif_lrn_cboost_bin4",                    #                      (with binning)
  "classif_lrn_cboost3",                        # ACWB (without binning)
  "classif_lrn_cboost_bin3",                    #      (with binning)
  "classif_lrn_cboost2",                        # hCWB (without binning)
  "classif_lrn_cboost_bin2",                    #      (with binning)
  "classif_lrn_xgboost",                        # Boosted trees
  "classif_lrn_gamboost",                       # CWB (mboost variant)
  "classif_lrn_ranger",                         # Random forest
  "classif_lrn_interpretML"                     # Interpret
)

# Each sever gets just a few tasks to efficiently distribute
# over several machines (name of the server is saved in '/etc/hostname'
#
# @param on_host logical(1) Indicate if host selector should be applied.
#   If `on_host = FALSE` all tasks are selected.
serverSelector = function(on_host = FALSE, host = NULL) {
  if (on_host) {
    if (is.null(host)) host = readLines("/etc/hostname")
    host_tasks = list(
      "bigger-benchmarks2" = c(1, 2),
      "cacb1" = c(3, 4),
      "cacb2" = 5,
      "cacb3" = 6,
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
ips = c("bigger-benchmarks2" = "138.246.235.7",
  "cacb1" = "138.246.233.36",
  "cacb2" = "138.246.233.123",
  "cacb3" = "138.246.233.138",
  "cacb4" = "138.246.232.140",
  "cacb5" = "138.246.233.164",
  "cacb6" = "138.246.233.146")
nevals = c("bigger-benchmarks2" = 50,
  "cacb1" = (50 + 5) / 2,
  "cacb2" = 5,
  "cacb3" = 1,
  "cacb4" = 1,
  "cacb5" = 5,
  "cacb6" = 1)

if (! dir.exists("~/repos/compboost/benchmark/mlr-bmr/bmr-aggr")) {
  cat("Create new directory!\n")
  dir.create("~/repos/compboost/benchmark/mlr-bmr/bmr-aggr")
}

just_get_status = FALSE
just_pull = TRUE

callOnAllSever = function(ips, call) {
  for (i in seq_along(ips)) {
    cat("(", i, "/", length(ips), ") ", ips[i], ": \n", sep = "")
    system(call)
  }
}
gitPullOnServer = function(ips, silent = FALSE) {
  for (i in seq_along(ips)) {
    if (! silent) cat("(", i, "/", length(ips), ") ", ips[i], ": \n", sep = "")
    git_pull = paste0("ssh -i ~/.ssh/lrz-key -l debian ", ips[i],
      " 'cd repos/compboost; git pull origin agbm_optim'")
    system(git_pull)
  }
}
gitPushResultsOnServer = function(ips) {
  for (i in seq_along(ips)) {
    cat("(", i, "/", length(ips), ") ", ips[i], ": \n", sep = "")
    pat = readLines("~/repos/compboost/benchmark/mlr-bmr/.gittoken")
    git_push = paste0("ssh -i ~/.ssh/lrz-key -l debian ", ips[i], " \"cd repos/compboost;",
      c("git add benchmark/mlr-bmr/res-results/*",
        paste0("git commit -m 'update latest results from ", ips[i], "'"),
        paste0("git push 'https://schalkdaniel:", pat, "@github.com/schalkdaniel/compboost.git'")),
      "\"")

    gitPullOnServer(ips[i], TRUE)
    for (push in git_push) {
      system(push)
    }
  }
}

#killAndRestartBenchmark = function(ips) {
  #callOnAllServers(ips, "killall R")
  #callOnAllServers(ips, paste0("nohup Rscript ~/repos/compboost/benchmark/mlr-bmr/bm-run.R > bm-log",
    #Sys.Date(), ".txt"))
#}

#gitPullOnServer(ips)
gitPushResultsOnServer(ips)

