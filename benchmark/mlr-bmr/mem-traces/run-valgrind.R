tsks_classif = rbind(
  data.frame(type = "oml", name = c(359994L, 7592L, 168335L, 168337L)),
  data.frame(type = "script", name = "albert"))

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
  "classif_lrn_ranger",
  "classif_lrn_interpretML")

for (i in seq_len(nrow(tsks_classif))) {
  cat("[", as.character(Sys.time()), "]: Run task ", tsks_classif$name[i], " (", i, "/", nrow(tsks_classif), ")\n", sep = "")
  for (j in seq_along(learners)) {
    config = list(learner = learners[j], task = tsks_classif$name[i], type = tsks_classif$type[i])
    save(config, file = "config.Rda")

    massif_file = paste0("massif.task", tsks_classif$name[i], "-", learners[j])
    massif_print_file = paste0("massif-print.task", tsks_classif$name[i], "-", learners[j], ".txt")
    log_file = paste0("log.task", tsks_classif$name[i], "-", learners[j])

    sys_call = paste0(
      "R -d \"valgrind --tool=massif --stacks=yes",
      " --threshold=0 --detailed-freq=1 --time-unit=ms --verbose --trace-children=yes",
      " --massif-out-file=", massif_file, " --log-file=", log_file,
      "\" -e \"source('train-model.R')\"")

    system(sys_call)
    system(paste0("ms_print ", massif_file, " > ", massif_print_file))
    file.remove(massif_file)
  }
}

extractLineData = function (line) {
  l = stringr::str_replace_all(line, "[,]", "")
  as.integer(unlist(regmatches(l, gregexpr("[[:digit:]]+", l))))
}

getSnapshotData = function (file) {
  lines = readLines(file)
  idx = grep("stacks[(]B[)]", lines)

  out = do.call(rbind, lapply(idx, function (i) extractLineData(lines[i+2])))
  colnames(out) = c("n", "time(ms)", "total(B)", "useful-heap(B)", "extra-heap(B)", "stacks(B)")
  out = as.data.frame(out)

  lrn_tsk = strsplit(x = strsplit(x = file, split = "task")[[1]][[2]], split = "-")[[1]]

  out$task = as.character(lrn_tsk[1])
  out$learner = strsplit(x = lrn_tsk[2], split = ".txt")[[1]][1]

  out$seconds = out$`time(ms)` * 0.001
  out$mb = out$`total(B)` / 1024^2

  idx_cboost = rep(NA, nrow(out))
  if (grepl("cboost", file)) {
    idx_cboost[which.min(idx <= min(grep("compboost", lines))) - 1] = TRUE
  }
  out$idx_cboost = idx_cboost

  return(out)
}

#file = "massif-print.task359994-classif_lrn_cboost1.txt"
#dats = getSnapshotData(file)

files = list.files()
files = files[grepl("massif-print", files)]

df_mem = do.call(rbind, lapply(files, getSnapshotData))

library(dplyr)
tmp = df_mem %>%
  group_by(learner, task) %>%
  mutate(ma = zoo::rollmean(mb, 10, fill = NA))

tmp %>% filter(learner == "classif_lrn_xgboost") %>% select(mb, ma) %>% as.data.frame()

library(ggplot2)
ggplot(tmp, aes(x = seconds / 60, y = mb, color = learner, fill = learner)) +
  geom_line(alpha = 0.2) +
  geom_line(aes(y = ma)) +
  geom_hline(yintercept = df_mem$mb[!is.na(df_mem$idx_cboost)][1]) +
  xlab("Minutes") +
  ylab("Used memory (MB)") +
  facet_wrap(. ~ paste0("Task: ", task), scales = "free") +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1")

