library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)

#file_classif = "res-results/bmr-classif-1-2021-Feb-05.Rda"
#load(file_classif)

showtext::showtext_auto()
extrafont::loadfonts()
font_scale = 3

font = "Lato-Regular"
ft = extrafont::fonttable()
sysfonts::font_add(font, regular = ft[ft$FontName == font, "fontfile"])

font_scale = 6

getLearner = function(x) {
  lrns = sapply(
    X = strsplit(
      x = x,
      split = "_after[.]"),
    FUN = function(x) ifelse(
      grepl(x[2], pattern = "encode[.]"),
      substr(x[2], 8, nchar(x[2])),
      x[2])
  )
  lrns = sapply(strsplit(x = lrns, split = "[.]tuned"), function(x) x[1])
  lrns = sapply(strsplit(x = lrns, split = "ps_"), function(x) x[2])
  return(lrns)
}
getTask = function(x) {
  getOMLTask = function (tn) {
    tsks = sapply(strsplit(split = ": ", x = tn), function(ts) ts[2])
    sapply(strsplit(split = " [(]", x = tsks), function(ts) ts[1])
  }
  tsks = ifelse(x %in% c("adult", "spam", "wine", "advert", "crime", "boston_housing"), x, getOMLTask(x))
  return(tsks)
}
paramExtract = function(param_list, param) {
  sapply(param_list, function(pl) {
    pl = as.data.frame(pl)
    nms = names(pl)
    idx = grep(param, nms)
    if (length(idx) == 0) {
      return(NA)
    } else {
      return(as.numeric(pl[1,idx]))
    }
  })
}


### Prepare classification benchmark:
### =======================================

files = list.files("res-results", full.names = TRUE)
files = files[grepl(x = files, pattern = "bmr-classif")]

files = files[grepl(pattern = "Feb-16", x = files)]

ll_classif = list()
ll_classif_msrs = list()
ll_classif_scrs = list()

for (fn in files) {
  load(fn)

  lrn  = sapply(strsplit(x = sapply(bmr_res[[1]], function(x) names(x)[1]), split = "[.]"), function(x) {
    if (x[1] == "encode") {
      return(x[2])
    } else {
      return(x[1])
    }})
  params = bmr_res[[1]]

  df_params = data.frame(
    learner = lrn,
    learning_rate = paramExtract(params, "learning_rate"),
    momentum = paramExtract(params, "momentum"),
    classif.auc = paramExtract(params, "classif[.]auc"),
    task = unique(bmr_res[[2]]$task_id),
    binning = ifelse(grepl(pattern = "2", lrn), "yes", "no"),
    mstop = paramExtract(params, "mstop"),
    iteration = as.integer(sapply(unique(lrn), function(lr) seq_len(sum(lr == lrn)))))

  df_params$task = getTask(df_params$task)

  ll_classif[[fn]] = df_params

  df_tmp = bmr_res[[2]]
  df_tmp$learner_id = getLearner(df_tmp$learner_id)
  df_tmp$task_id = getTask(df_tmp$task_id)

  ll_classif_msrs[[fn]] = df_tmp[, -5]

  df_scr = bmr_res[[3]]
  df_scr$learner = getLearner(df_scr$learner_id)
  df_scr$task = getTask(df_scr$task_id)

  ll_classif_scrs[[fn]] = df_scr[, -which(names(df_scr) %in% c("learner_id", "task_id"))]
}

df_params = do.call(rbind, ll_classif)
rownames(df_params) = NULL
df_msrs = do.call(rbind, ll_classif_msrs)
rownames(df_msrs) = NULL
df_scrs = do.call(rbind, ll_classif_scrs)
rownames(df_scrs) = NULL
df_scrs$uhash = NULL

a = df_scrs %>%
  mutate(learner = paste0("ps_", learner)) %>%
  select(task, learner, time_train, classif.auc) %>%
  group_by(learner, task) %>%
  summarize(iteration = order(classif.auc), auc = classif.auc, time_train = time_train)

b = df_params %>%
    select(task, learner, mstop, classif.auc) %>%
    group_by(learner, task) %>%
    summarize(iteration = order(classif.auc), auc_p = classif.auc, mstop = mstop)

tmp = a %>% left_join(b, by = c("task", "learner", "iteration"))

df_mstop = df_params %>%
  group_by(learner, task) %>%
  summarize(mstop = median(mstop))

lrn_names = c(cboost1 = "CWB (no binning)", cboost2 = "CWB (binning)", cboost_nesterov1 = "ACWB (no binning)",
  cboost_nesterov2 = "ACWB (binning)", ranger = "Random forest", rpart = "CART", xgboost = "Boosted trees",
  gamboost = "CWB (mboost)")

idx_nm = sapply(df_mstop$learner, function(lr) which(paste0("ps_", names(lrn_names)) == lr))
df_mstop$learner = factor(lrn_names[idx_nm], levels = lrn_names)

idx_nm = sapply(tmp$learner, function(lr) which(paste0("ps_", names(lrn_names)) == lr))
tmp$learner = factor(lrn_names[idx_nm], levels = lrn_names)

ggViz = function (yf) {
  ggplot(data = tmp, aes_string(x = "learner", y = yf, color = "learner", fill = "learner")) +
    geom_boxplot(alpha = 0.2) +
    facet_wrap(. ~ task, ncol = 5L) +
    theme_bw(base_family = font) +
    theme(
      strip.background = element_rect(fill = rgb(47,79,79,maxColorValue = 255), color = "white"),
      strip.text = element_text(color = "white", face = "bold", size = 4 * font_scale),
      axis.text.x = element_blank(),
      axis.text = element_text(size = 8 * font_scale),
      axis.title = element_text(size = 10 * font_scale),
      legend.text = element_text(size = 6 * font_scale),
      legend.title = element_text(size = 8 * font_scale)) +
    xlab("") +
    ylab(yf) +
    labs(color = "Learner", fill = "Learner") +
    scale_color_viridis(discrete = TRUE) +
    scale_fill_viridis(discrete = TRUE)
}

gg = ggViz("auc") + ylab("AUC")
dinA4width = 210 * font_scale
scale_fac = 1 / 2
ggsave(
  plot = gg,
  filename = "fig-bm-table-auc.pdf",
  width = dinA4width * scale_fac,
  height = dinA4width * scale_fac * 1 / 3,
  units = "mm")

gg = ggViz("time_train") + ylab("Training time\n(seconds)")
dinA4width = 210 * font_scale
scale_fac = 1 / 2
ggsave(
  plot = gg,
  filename = "fig-bm-table-time-train.pdf",
  width = dinA4width * scale_fac,
  height = dinA4width * scale_fac * 1 / 3,
  units = "mm")




### Get tables:
### ======================================

latexResults = function (rres, minimize = TRUE) {

  minmax = function (x, minimize) { if (minimize) { return (min(x)) } else { return (max(x)) }}

  max_idx = rres %>%
    select(learner, task, med) %>%
    group_by(task) %>%
    summarize(max_med = minmax(med, minimize))

  lt_lines = rres %>%
    left_join(max_idx, by = "task") %>%
    mutate(cell_text = paste0("$", ifelse(med == max_med, paste0("\\bm{", round(med, 4), "}"), round(med, 4)), "\\pm", round(sd, 4), "$")) %>%
    select(learner, task, cell_text) %>%
    pivot_wider(names_from = learner, values_from = cell_text) %>%
    mutate(task = paste0("\\textbf{", task, "}")) #%>%
  #knitr::kable(format = "latex", escape = FALSE)

  #lt_lines = lt_lines[,c("task", "ps_cboost2", "ps_cboost1", "ps_cboost_nesterov2", "ps_cboost_nesterov1", "ps_xgboost", "ps_rpart", "ps_ranger")]

  lines = paste0(apply(lt_lines, 1, paste0, collapse = " \n\t\t\t& "), "\n\t\t\t\\\\")
  lines = as.character(rbind(lines, "\\hline"))
  lines = paste0("\t\t", lines, " \n")

  lines_header = c("\t\t\\hline\n\t\t\\diagbox{\\textbf{Dataset}}{\\textbf{Learner}} \n\t\t\t& \\makecell{\\textbf{CWB} \\\\ \\textbf{(no binning)}} \n\t\t\t& \\makecell{\\textbf{CWB} \\\\ \\textbf{(binning)}} \n\t\t\t& \\makecell{\\textbf{ACWB} \\\\ \\textbf{(no binning)}} \n\t\t\t& \\makecell{\\textbf{ACWB} \\\\ \\textbf{(binning)}} \n\t\t\t&\\makecell{\\textbf{Boosted} \\\\ \\textbf{trees}} \n\t\t\t& \\textbf{CART} \n\t\t\t& \\makecell{\\textbf{Random} \\\\ \\textbf{forest}} \n\t\t\t& \\makecell{\\textbf{CWB} \\\\ \\textbf{(no binning; mboost)}} \\\\\n\t\t\\hline\\hline\n")
  lines_start = "\t\\begin{tabular}{r|c|c|c|c|c|c|c|c}\n"
  lines_end = "\t\t\\hline\n\t\\end{tabular}"

  cat(c(lines_start, lines_header, lines, lines_end, "\n\n"))
}

rres_classif = tmp %>%
  select(learner, task, auc) %>%
  group_by(learner, task) %>%
  summarize(med = median(auc), min = min(auc), max = max(auc), sd = sd(auc))

latexResults(rres_classif, FALSE)


### Prepare regression  benchmark:
### =======================================

files = list.files("res-results", full.names = TRUE)
files = files[grepl(x = files, pattern = "bmr-regr")]

files = files[grepl(pattern = "Feb-19", x = files)]

ll_classif = list()
ll_classif_msrs = list()
ll_classif_scrs = list()

for (fn in files) {
  load(fn)

  lrn  = sapply(strsplit(x = sapply(bmr_res[[1]], function(x) names(x)[1]), split = "[.]"), function(x) {
    if (x[1] == "encode") {
      return(x[2])
    } else {
      return(x[1])
    }})
  params = bmr_res[[1]]

  df_params = data.frame(
    learner = lrn,
    learning_rate = paramExtract(params, "learning_rate"),
    momentum = paramExtract(params, "momentum"),
    regr.mse = paramExtract(params, "regr[.]mse"),
    task = unique(bmr_res[[2]]$task_id),
    binning = ifelse(grepl(pattern = "2", lrn), "yes", "no"),
    mstop = paramExtract(params, "mstop"),
    iteration = as.integer(sapply(unique(lrn), function(lr) seq_len(sum(lr == lrn)))))

  df_params$task = getTask(df_params$task)

  ll_classif[[fn]] = df_params

  df_tmp = bmr_res[[2]]
  df_tmp$learner_id = getLearner(df_tmp$learner_id)
  df_tmp$task_id = getTask(df_tmp$task_id)

  ll_classif_msrs[[fn]] = df_tmp[, -5]

  df_scr = bmr_res[[3]]
  df_scr$learner = getLearner(df_scr$learner_id)
  df_scr$task = getTask(df_scr$task_id)

  ll_classif_scrs[[fn]] = df_scr[, -which(names(df_scr) %in% c("learner_id", "task_id"))]
}

df_params = do.call(rbind, ll_classif)
rownames(df_params) = NULL
df_msrs = do.call(rbind, ll_classif_msrs)
rownames(df_msrs) = NULL
df_scrs = do.call(rbind, ll_classif_scrs)
rownames(df_scrs) = NULL
df_scrs$uhash = NULL

a = df_scrs %>%
  mutate(learner = paste0("ps_", learner)) %>%
  select(task, learner, time_train, regr.mse) %>%
  group_by(learner, task) %>%
  summarize(iteration = order(regr.mse), mse = regr.mse, time_train = time_train)

b = df_params %>%
    select(task, learner, mstop, regr.mse) %>%
    group_by(learner, task) %>%
    summarize(iteration = order(regr.mse), mse_p = regr.mse, mstop = mstop)

tmp = a %>% left_join(b, by = c("task", "learner", "iteration"))

df_mstop = df_params %>%
  group_by(learner, task) %>%
  summarize(mstop = median(mstop))

lrn_names = c(cboost1 = "CWB (no binning)", cboost2 = "CWB (binning)", cboost_nesterov1 = "ACWB (no binning)",
  cboost_nesterov2 = "ACWB (binning)", ranger = "Random forest", rpart = "CART", xgboost = "Boosted trees",
  gamboost = "CWB (mboost)")

idx_nm = sapply(df_mstop$learner, function(lr) which(paste0("ps_", names(lrn_names)) == lr))
df_mstop$learner = factor(lrn_names[idx_nm], levels = lrn_names)

idx_nm = sapply(tmp$learner, function(lr) which(paste0("ps_", names(lrn_names)) == lr))
tmp$learner = factor(lrn_names[idx_nm], levels = lrn_names)


ggViz("mse") + ylab("MSE")
ggViz("time_train / 100") + ylab("Training time\n(seconds)")



rres_regr = tmp %>%
  select(learner, task, mse) %>%
  group_by(learner, task) %>%
  summarize(med = median(mse), min = min(mse), max = max(mse), sd = sd(mse))

latexResults(rres_regr, TRUE)



### Visualize:
### ======================================


rres_classif %>%
  group_by(task, learner) %>%
  summarize(auc_med = median(classif.auc), auc_min = min(classif.auc), auc_max = max(classif.auc)) %>%
  as.data.frame()

bmrl = as.data.table(bmr)$learner
l = bmrl[[1]]
l$tuning_results


#sysfonts::font_add("Gyre Bonum",
    #regular = "/usr/share/texmf-dist/fonts/opentype/public/tex-gyre/texgyrebonum-regular.otf",
    #bold = "/usr/share/texmf-dist/fonts/opentype/public/tex-gyre/texgyrebonum-bold.otf")
#showtext::showtext_auto()

#font_scale = 3

#gg = rres %>%
  #pivot_longer(names_to = "measure", values_to = "measure_value", cols = starts_with("classif.")) %>%
    #ggplot(aes(x = measure, y = measure_value, fill = learner)) +
      #geom_boxplot() +
      #theme_minimal(base_family = "Gyre Bonum") +
      #theme(
        #strip.background = element_rect(fill = rgb(47,79,79,maxColorValue = 255), color = "white"),
        #strip.text = element_text(color = "white", face = "bold", size = 8 * font_scale),
        #axis.text.x = element_text(angle = 45, hjust = 1),
        #axis.text = element_text(size = 8 * font_scale),
        #axis.title = element_text(size = 10 * font_scale),
        #legend.text = element_text(size = 6 * font_scale),
        #legend.title = element_text(size = 8 * font_scale)
      #) +
      #scale_fill_viridis(discrete = TRUE) +
      #xlab("") +
      #ylab("") +
      #labs(fill = "") +
      #facet_grid(paste0(task, "\nnrow: ", task_nrow) ~ .)

#dinA4width = 210 * font_scale
#ggsave(plot = gg, filename = "rres.pdf", width = dinA4width * 1/3, height = dinA4width, units = "mm")



#as.data.table(bmr$resample_results$resample_result[[1]])$learner[[1]]$tuning_result








#library("paradox")
#library("mlr3")
#library("mlr3tuning")
#library("mlr3learners")

#learner = lrn("classif.rpart")
#tune_ps = ParamSet$new(list(
  #ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  #ParamInt$new("minsplit", lower = 1, upper = 10)
#))
#terminator = trm("evals", n_evals = 10)
#tuner = tnr("random_search")

#at = AutoTuner$new(
  #learner = learner,
  #resampling = rsmp("holdout"),
  #measure = msr("classif.ce"),
  #search_space = tune_ps,
  #terminator = terminator,
  #tuner = tuner,
  #store_tuning_instance = TRUE
#)

#grid = benchmark_grid(
  #task = tsk("pima"),
  #learner = list(at, lrn("classif.rpart")),
  #resampling = rsmp("cv", folds = 3)
#)

#bmr = benchmark(grid, store_models = TRUE)

#bmrl = as.data.table(bmr)$learner
#l = bmrl[[1]]
#l$tuning_result




latexResults = function (rres, minimize = TRUE) {

  minmax = function (x, minimize) { if (minimize) { return (min(x)) } else { return (max(x)) }}

  max_idx = rres %>%
    select(learner, task, med) %>%
    group_by(task) %>%
    summarize(max_med = minmax(med, minimize))

  lt_lines = rres %>%
    left_join(max_idx, by = "task") %>%
    mutate(cell_text = paste0(ifelse(med == max_med, paste0("\\textbf{", round(med, 4), "}"), round(med, 4)), "\\ \\ \\makecell[c]{\\scriptsize{", round(max, 4), "} \\\\ \\scriptsize{", round(min, 4), "}}")) %>%
    select(learner, task, cell_text) %>%
    pivot_wider(names_from = learner, values_from = cell_text) %>%
    mutate(task = paste0("\\textbf{", task, "}")) #%>%
  #knitr::kable(format = "latex", escape = FALSE)

  lines = paste0(apply(lt_lines, 1, paste0, collapse = " \n\t\t\t& "), " \\\\")
  lines = as.character(rbind(lines, "\\hline"))
  lines = paste0("\t\t", lines, " \n")

  lines_header = c("\t\t\\hline\n\t\t\\diagbox{\\textbf{Dataset}}{\\textbf{Learner}} \n\t\t\t& \\makecell{\\textbf{CWB} \\\\ \\textbf{(binning)}} \n\t\t\t& \\makecell{\\textbf{CWB} \\\\ \\textbf{(no binning)}} \n\t\t\t& \\makecell{\\textbf{ACWB} \\\\ \\textbf{(binning)}} \n\t\t\t& \\makecell{\\textbf{ACWB} \\\\ \\textbf{(no binning)}} \n\t\t\t&\\makecell{\\textbf{Boosted} \\\\ \\textbf{trees}} \n\t\t\t& \\textbf{CART} \n\t\t\t& \\makecell{\\textbf{Random} \\\\ \\textbf{forest}} \\\\\n\t\t\\hline\\hline\n")
  lines_start = "\t\\begin{tabular}{r|c|c|c|c|c|c|c}\n"
  lines_end = "\t\t\\hline\n\t\\end{tabular}"

  cat(c(lines_start, lines_header, lines, lines_end, "\n\n"))
}

