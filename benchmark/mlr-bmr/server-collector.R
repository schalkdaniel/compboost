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

ll_done = list()
ll_data = list()
for (i in seq_along(ips)) {
  data_file = paste0("~/repos/compboost/benchmark/mlr-bmr/bmr-aggr/", names(ips)[i], ".Rda")

  git_push = paste0("ssh -i ~/.ssh/lrz-key -l debian ", ips[i], " \"cd repos/compboost;",
    c("git add benchmark/mlr-bmr/res-results/*",
      'git commit -m 'update latest restuls'",
      "git push 'https://schalkdaniel:e06b26c8eb3f14d8aa59878fd7af561a17001784@github.com/schalkdaniel/compboost.git'"),
    "\"")
  git_pull = paste0("ssh -i ~/.ssh/lrz-key -l debian ", ips[i],
    " 'cd repos/compboost; git pull origin agbm_optim'")
  create_data_call = paste0("ssh -i ~/.ssh/lrz-key -l debian ", ips[i],
    " 'Rscript ~/repos/compboost/benchmark/mlr-bmr/summarize-results.R'")
  pull_data_call = paste0("scp -i ~/.ssh/lrz-key debian@", ips[i],
    ":~/repos/compboost/benchmark/mlr-bmr/df_bmr.Rda ", data_file,
    collapse = "")

  if (! just_get_status) {
    cat("(", i, "/", length(ips), "): Pull latest changes ", names(ips)[i], "\n", sep = "")
    system(git_pull)
    for (push in git_push) {
      system(push)
    }
  }
  cat("(", i, "/", length(ips), "): Generate data at server ", names(ips)[i], "\n", sep = "")
  system(create_data_call)
  cat("(", i, "/", length(ips), "): Pull data from ", names(ips)[i], "\n", sep = "")
  system(pull_data_call)

  load(data_file)
  ll_data[[ips[i]]] = df_bmr
  if (is.null(nrow(df_bmr))) {
    dd = data.frame(
      server = names(ips)[i],
      done = 0,
      total = length(serverSelector(TRUE, names(ips)[i])) * length(learners))
  } else {
    dd = data.frame(
      server = names(ips)[i],
      done = nrow(df_bmr) / nevals[i],
      total = length(serverSelector(TRUE, names(ips)[i])) * length(learners))
  }
  dd$percent = paste0(round(dd$done / dd$total, 4) * 100, " %")
  ll_done[[ips[i]]] = dd
}
df_done = do.call(rbind, ll_done)
df_bmr = do.call(rbind, ll_data)
df_done
df_bmr

save(df_bmr, file = "~/repos/compboost/benchmark/mlr-bmr/bmr-aggr/df_bmr.Rda")

if (FALSE) {
  library(dplyr)
  library(ggplot2)
  library(ggsci)
  library(gridExtra)

  df_all = expand.grid(learner = unique(df_bmr$learner), task = unique(df_bmr$task),
    classif.auc = NA, time_per_iter = NA)

  df_bmr = df_all %>% full_join(df_bmr, by = c("learner", "task"))

  df_bmr$classif.auc.x = NULL
  df_bmr$classif.auc = df_bmr$classif.auc.y

  df_bmr$time_per_iter.x = NULL
  df_bmr$time_per_iter = df_bmr$time_per_iter.y

  df_bmr$learner = factor(df_bmr$learner, levels = c("CWB (no binning)",
    "CWB (binning)", "CWB Cosine Annealing (no binning)", "CWB Cosine Annealing (binning)",
    "ACWB (no binning)", "ACWB (binning)", "hCWB (no binning)", "hCWB (binning)",
    "Random forest", "Boosted trees", "CWB (mboost)", "interpretML"))

  df_bmr$task = factor(df_bmr$task, levels = c("Hepatitis",
    "Diabetes",
    #"German Credit",
    "Analcat Halloffame",
    "Spam",
    "Guillermo",
    "Adult",
    "MiniBooNE",
    "Albert",
    "SF Police Incidents"
    ))

  gg1 = ggplot(df_bmr, aes(x = learner, y = classif.auc, color = learner, fill = learner)) +
    geom_boxplot(alpha = 0.2, lwd = 0.3) +
    scale_fill_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
    scale_color_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
    theme(axis.text.x = element_blank(), legend.position = "bottom") +
    labs(fill = "Algorithm", color = "Algorithm") +
    guides(fill = guide_legend(nrow = 4), color = guide_legend(nrow = 4)) +
    xlab("") +
    ylab("AUC") +
    facet_wrap(. ~ task, scales = "free", ncol = 4)

  gg2 = ggplot(df_bmr, aes(x = learner, y = time_per_model, color = learner, fill = learner)) +
    geom_boxplot(alpha = 0.2, lwd = 0.3) +
    scale_fill_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
    scale_color_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
    theme(axis.text.x = element_blank(), legend.position = "bottom") +
    labs(fill = "Algorithm", color = "Algorithm") +
    guides(fill = guide_legend(nrow = 4), color = guide_legend(nrow = 4)) +
    xlab("") +
    ylab("Training time (seconds)") +
    facet_wrap(. ~ task, scales = "free", ncol = 4)

  grid.arrange(gg1, gg2, nrow = 2)
}
