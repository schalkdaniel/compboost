## ============================================================ ##
##
##                         SETUP
##
## ============================================================ ##

library(dplyr)
library(tidyr)
library(ggplot2)
library(ggsci)
library(gridExtra)

font = "TeX Gyre Bonum"

sysfonts::font_add(font,
    #regular = paste0(base_dir, "/paper-figures/gyre-bonum/texgyrebonum-regular.ttf"),
    #bold = paste0(base_dir, "/paper-figures/gyre-bonum/texgyrebonum-bold.ttf"))
    regular = "/usr/share/texmf-dist/fonts/opentype/public/tex-gyre/texgyrebonum-regular.otf",
    bold = "/usr/share/texmf-dist/fonts/opentype/public/tex-gyre/texgyrebonum-bold.otf")
#showtext::showtext_auto()
extrafont::font_import(paths = "~/repos/bm-CompAspCboost/paper-figures/gyre-bonum", prompt = FALSE)
extrafont::loadfonts()

theme_set(
  theme_minimal(base_family = font) +
  ggplot2::theme(
    strip.background = element_rect(fill = rgb(47, 79, 79, maxColorValue = 255), color = "white"),
    strip.text = element_text(color = "white", face = "bold", size = 8),
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 11),
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 7),
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5)
  )
)
#my_color = scale_color_viridis(discrete = TRUE)
#my_fill = scale_fill_viridis(discrete = TRUE)

#my_color = scale_color_npg()
#my_fill = scale_fill_npg()

my_color = scale_color_uchicago()
my_fill = scale_fill_uchicago()

#my_color = scale_color_aaas()
#my_fill = scale_fill_aaas()

dinA4width = 162

#extract legend
#https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
g_legend = function(a_gplot) {
  tmp = ggplot_gtable(ggplot_build(a_gplot))
  leg = which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend = tmp$grobs[[leg]]
  return(legend)
}

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

llevels = learner_table


## ============================================================ ##
##
##                         FUNCTIONS
##
## ============================================================ ##

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
extractBMRData = function(file_name) {
  lapply(file_name, function(file) {
    load(file)
    tmp = bmr_res[[3]]
    idx_select = sapply(
      c("classif.auc", "classif.ce", "classif.bbrier", "time_train", "time_predict", "time_both"),#, "n_evals"),
      function(m) which(m == names(tmp)))
    tmp = tmp[, idx_select]
    tmp$task = getTaskFromFile(file)
    tmp$learner = getLearnerFromFile(file)
    return(tmp)
  })
}


## ============================================================ ##
##
##                         LOAD DATA
##
## ============================================================ ##

base_dir = "~/repos/compboost/benchmark/mlr-bmr/"
files = list.files(paste0(base_dir, "res-results"), full.names = TRUE)
idx_files = extractStringBetween(files, "-task", "-classif_") %in% names(task_table)
files = files[idx_files]

df_bmr = do.call(rbind, extractBMRData(files))
df_all = expand.grid(learner = unique(df_bmr$learner), task = unique(df_bmr$task))

df_bmr = df_all %>% full_join(df_bmr, by = c("learner", "task"))

df_bmr$learner = factor(df_bmr$learner, levels = llevels)
df_bmr$task = factor(df_bmr$task, levels = task_table)

equalBreaks = function(n = 4, s = 0.05, ...) {
  function(x) {
    # rescaling
    d = s * diff(range(x)) / (1 + 2 * s)
    seq(min(x) + d, max(x) - d, length = n)
  }
}

## ============================================================ ##
##
##                           FIGURES
##
## ============================================================ ##

### OVERALL PLOTS:
### =================================

df_plt1 = df_bmr %>%
  select(learner, task, classif.auc, time_train) %>%
  filter(classif.auc > 0.5)

df_space = expand.grid(learner = paste0("space", 1:3), task = unique(df_plt1$task), classif.auc = NA, time_train = NA)
df_plt1 = df_plt1 %>% rbind(df_space)
df_plt1$learner = factor(df_plt1$learner, levels = c(llevels[1:6], "space1", llevels[7:12], "space2", llevels[13:20], "space3", llevels[21:24]))

### Overall Plot:
gg_tt = ggplot(df_plt1, aes(x = learner, y = time_train, color = learner, fill = learner)) +#,
  geom_boxplot(alpha = 0.2, lwd = 0.3) +
  scale_fill_manual(values = c(pal_uchicago()(6), pal_aaas()(6), pal_jco()(8), pal_locuszoom()(4))) +
  scale_color_manual(values = c(pal_uchicago()(6), pal_aaas()(6), pal_jco()(8), pal_locuszoom()(4))) +
  #scale_x_discrete(breaks = unname(c(llevels[1:8], "space1", llevels[9:12])), labels = unname(c(llevels[1:8], "", llevels[9:12]))) +
  #scale_y_continuous(breaks = equalBreaks(), trans = "log10") +
  scale_y_continuous(trans = "log10") +
  theme(axis.text.x = element_blank(), legend.position = "bottom", axis.text.y = element_text(size = 7)) +
  labs(fill = "Algorithm", color = "Algorithm") +
  guides(fill = guide_legend(nrow = 4), color = guide_legend(nrow = 4), linetype = FALSE) +
  xlab("") +
  ylab("Time train") +
  facet_wrap(~ task, scales = "free_y", nrow = 2)
gg_tt

gg_auc = ggplot(df_plt1, aes(x = learner, y = classif.auc, color = learner, fill = learner)) +
  geom_boxplot(alpha = 0.2, lwd = 0.3) +
  scale_fill_manual(values = c(pal_uchicago()(6), pal_aaas()(6), pal_jco()(8), pal_locuszoom()(4))) +
  scale_color_manual(values = c(pal_uchicago()(6), pal_aaas()(6), pal_jco()(8), pal_locuszoom()(4))) +
  #scale_x_discrete(breaks = unname(c(llevels[1:8], "space1", llevels[9:12])), labels = unname(c(llevels[1:8], "", llevels[9:12]))) +
  scale_y_continuous(breaks = equalBreaks()) +
  theme(axis.text.x = element_blank(), legend.position = "bottom", axis.text.y = element_text(size = 7)) +
  labs(fill = "Algorithm", color = "Algorithm") +
  guides(fill = guide_legend(nrow = 4), color = guide_legend(nrow = 4), linetype = FALSE) +
  xlab("") +
  ylab("AUC") +
  facet_wrap(~ task, scales = "free_y", nrow = 2)
gg_auc


### hCWB SWITCH:
### =================================

cwb_variants = learner_table[1:12]


files = list.files(paste0(base_dir, "res-results"), full.names = TRUE)
idx_files = extractStringBetween(files, "-task", "-classif_") %in% names(task_table)
files_cv = files[getLearnerFromFile(files) %in% cwb_variants]

fcv = files_cv[1]

extractArchive = function(fcv) {
  load(fcv)
  tmp = bmr_res$archive
  tmp$learner = getLearnerFromFile(fcv)
  tmp$task = getTaskFromFile(fcv)
  tmp$binning = ifelse(grepl("[(]b,", tmp$learner), "yes", "no")
  tmp$learner_id = NULL
  tmp$task_id = NULL
  tmp
}

df_cv = do.call(rbind, lapply(files_cv, extractArchive)) %>%
  pivot_longer(cols = starts_with("iters"), names_to = "method", values_to = "iters") %>%
  filter(! is.na(iters))

df_iters = df_cv %>%
  ggplot(aes(x = iters, color = method, fill = method)) +
    my_color + my_fill +
    geom_density(aes(linetype = binning), alpha = 0.2) +
    facet_wrap(~ task, scales = "free_y", nrow = 2)
df_iters




load(paste0(base_dir, "ll_iter.Rda"))
df_iter = do.call(rbind, ll_iter)

df_iter = df_iter %>%
  mutate(iters_hcwb_restart = iters_restart, iters_hcwb_burnin = ifelse(!is.na(iters_restart), iters_acwb, NA))

df_iter %>% filter(! is.na(iters_hcwb)) %>%
  ggplot() +
    geom_density(aes(x = iters_acwb, color = "ACWB", fill = "ACWB"), alpha = 0.2) +
    geom_density(aes(x = iters_restart, color = "hCWB - ACWB", fill = "hCWB - ACWB"), alpha = 0.2) +
    my_color + my_fill +
    labs(fill = "Iters", color = "Iters") +
    facet_wrap(~ task, scales = "free_y", nrow = 2)

df_iter %>%
  select(-iters_restart) %>%
  select(-starts_with("V")) %>%
  pivot_longer(cols = starts_with("iters_"), names_to = "method", values_to = "iters") %>%
  #filter(grepl("hcwb", method)) %>% na.omit()
  #select(-method) %>%
  na.omit() %>%
  mutate(binning = ifelse(grepl("[(]b,", learner), "yes", "no")) %>%
  mutate(variant = ifelse(grepl("hCWB", learner), "hCWB", ifelse(grepl("ACWB", learner), "ACWB", "CWB"))) %>%
  mutate(tuned = ifelse(grepl("notune", learner), " notune", " tuned")) %>%
  mutate(learner = paste0(variant, tuned)) %>%
  filter(grepl("tuned", learner)) %>%
  ggplot(aes(x = iters, color = method, fill = method)) +
    my_color + my_fill +
    #geom_density(aes(linetype = binning), alpha = 0.2) +
    geom_density(alpha = 0.2) +
    facet_wrap(~ task, scales = "free_y", nrow = 2)



files = list.files("best-runs", full.names = TRUE)

ll = list()
for (fn in files) {
  load(fn)
  ll = c(ll, list(df_best))
}
df_run = do.call(rbind, ll)

tsks = df_run$task
df_run$task = unname(task_table[sapply(tsks, function(ts) which(ts == names(task_table)))])

lrns = substr(as.character(df_run$learner), "13", nchar(as.character(df_run$learner)))
lrns_idx = sapply(lrns, function(l) which(l == names(learner_table)))
df_run$learner = unname(learner_table[lrns_idx])

df_run$learner = factor(df_run$learner, levels = llevels)

df_run$task = factor(df_run$task, levels = c("Hepatitis",
  "Diabetes",
  #"German Credit",
  "Analcat Halloffame",
  "Spam",
  "Guillermo",
  "Adult",
  "MiniBooNE",
  "namao",
  "Albert",
  "SF Police"# Incidents"
  ))

df_plt1 = df_bmr %>% select(learner, task, classif.auc, time_train) %>% filter(task != "Analcat Halloffame") %>% filter(classif.auc > 0.5)
df_space = data.frame(learner = "space1", task = unique(df_plt1$task), classif.auc = NA, time_train = NA)
df_plt1 = df_plt1 %>% rbind(df_space)
df_plt1$learner = factor(df_plt1$learner, levels = c(llevels[1:8], "space1", llevels[9:12]))

### Overall Plot:
gg1 =
gg1 =

  ggplot(df_plt1, aes(x = learner, y = time_train, color = learner, fill = learner)) +#,

  ggplot(df_plt1, aes(x = learner, y = classif.auc, color = learner, fill = learner)) +#,
    #linetype = ifelse(!grepl("[(]binning", learner), "binning", "no binning"))) +
  geom_boxplot(alpha = 0.2, lwd = 0.3) +
  #scale_fill_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
  #scale_color_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
  #scale_x_discrete(breaks = unname(c(llevels[1:8], "space1", llevels[9:12])), labels = unname(c(llevels[1:8], "", llevels[9:12]))) +
  scale_y_continuous(breaks = equalBreaks()) +
  theme(axis.text.x = element_blank(), legend.position = "bottom", axis.text.y = element_text(size = 7)) +
  labs(fill = "Algorithm", color = "Algorithm") +
  guides(fill = guide_legend(nrow = 4), color = guide_legend(nrow = 4), linetype = FALSE) +
  xlab("") +
  ylab("AUC") +
  facet_wrap(~ task, scales = "free_y", nrow = 2)




df_plt1 = df_bmr %>% mutate(classif.auc = time_train) %>% select(learner, task, classif.auc) %>% filter(task != "Analcat Halloffame")
df_space = data.frame(learner = "space1", task = unique(df_plt1$task), classif.auc = NA)
df_plt1 = df_plt1 %>% rbind(df_space)
df_plt1$learner = factor(df_plt1$learner, levels = c(llevels[1:8], "space1", llevels[9:12]))


### Overall Plot:
gg1 = ggplot(df_plt1, aes(x = learner, y = classif.auc, color = learner, fill = learner)) +#,
    #linetype = ifelse(!grepl("[(]binning", learner), "binning", "no binning"))) +
  geom_boxplot(alpha = 0.2, lwd = 0.3) +
  scale_fill_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
  scale_color_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
  scale_x_discrete(breaks = unname(c(llevels[1:8], "space1", llevels[9:12])), labels = unname(c(llevels[1:8], "", llevels[9:12]))) +
  scale_y_continuous(breaks = equalBreaks()) +
  theme(axis.text.x = element_blank(), legend.position = "bottom", axis.text.y = element_text(size = 7)) +
  labs(fill = "Algorithm", color = "Algorithm") +
  guides(fill = guide_legend(nrow = 4), color = guide_legend(nrow = 4), linetype = FALSE) +
  xlab("") +
  ylab("AUC") +
  facet_wrap(~ task, scales = "free_y", nrow = 2)

a = df_plt1 %>% filter(learner %in% c( "hCWB (nb)", "hCWB (b)", "ACWB (nb)", "ACWB (b)") )
sum(a$classif.auc) / 60 / 60 / 7
a = df_plt1 %>% filter(learner %in% c( "hCWB (nb)", "hCWB (b)", "ACWB (nb)", "ACWB (b)"), task != "Hepatitis")
sum(a$classif.auc) / 60 / 60 / 7
df_plt1 %>% filter(learner %in% c( "hCWB (nb)", "hCWB (b)", "ACWB (nb)", "ACWB (b)"), task != "Hepatitis") %>%
  group_by(task) %>%
  summarize(time = sum(classif.auc) / 60^2)


#gg1

df_plt2 = df_run %>% filter(task != "Analcat Halloffame")
df_space = data.frame(learner = "space1", task = unique(df_plt2$task), iteration = 1, time_train = NA)
df_plt2 = df_plt2 %>% rbind(df_space)
df_plt2$learner = factor(df_plt2$learner, levels = c(llevels[1:8], "space1", llevels[9:12]))

gg2 = ggplot(df_plt2, aes(x = learner, y = time_train, color = learner, fill = learner)) + #,
  geom_boxplot(alpha = 0.2, lwd = 0.3, show.legend = FALSE) +
  scale_fill_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
  scale_color_manual(values = c(pal_uchicago()(9), pal_aaas()(3))) +
  theme(axis.text.y = element_text(size = 7), axis.text.x = element_text(size = 7, angle = 60, hjust = 1)) +#, vjust = 0.5, hjust = 1)) +
  scale_x_discrete(breaks = unname(c(llevels[1:8], "space1", llevels[9:12])), labels = unname(c(llevels[1:8], "", llevels[9:12]))) +
  scale_y_continuous(breaks = equalBreaks()) +
  labs(fill = "Algorithm", color = "Algorithm") +
  xlab("") +
  ylab("Training time (seconds)") +
  facet_wrap(~ task, scales = "free_y", nrow = 2)

#gg2

my_legend = g_legend(gg1)
gg1 = gg1 + theme(legend.position = "none")
gg2 = gg2 + theme(legend.position = "none")

gt1 = ggplot_gtable(ggplot_build(gg1))
gt2 = ggplot_gtable(ggplot_build(gg2))

gt1$widths = gt2$width

p3 = arrangeGrob(
  gt1,
  gt2,
  nrow = 2,
  ncol = 1,
  heights = c(4,5))

dev.off()
plot(p3)

ggsave(
  plot = p3,
  filename = "fig-bmr-res.pdf",
  width = dinA4width,
  height = dinA4width * 0.75,
  units = "mm")


### EQ1:
### =============================000

colors = c(pal_uchicago()(9), pal_aaas()(3))[c(1,2,8,11,5,6,7,3,10,9,4,12)]

colors = c(pal_uchicago()(6), pal_aaas()(6), pal_jco()(8), pal_locuszoom()(4))
names(colors) = learner_table
box_width = 0.25
box_fatten = 1.25
hline_size = 0.2
width_ggsep = 0.3

baseline_lrn = "CWB (nb, no mstop)"
additional_lrn = c("CWB (b, no mstop)", "ACWB (nb, no mstop)", "ACWB (b, no mstop)",
  "hCWB (nb, no mstop)", "hCWB (b, no mstop)")

qlower = 0.25
qupper = 0.75
df_tmp = df_bmr %>%
  filter(learner %in% c(baseline_lrn, additional_lrn)) %>%
  group_by(task) %>%
  mutate(
    classif.auc = (classif.auc - classif.auc[learner == baseline_lrn]) / classif.auc[learner == baseline_lrn],
    time_train = time_train[learner == baseline_lrn] / time_train
    ) %>%
  group_by(task, learner) %>%
  summarize(
    med_auc = median(classif.auc, na.rm = TRUE),
    lower_auc = quantile(classif.auc, qlower, na.rm = TRUE),
    upper_auc = quantile(classif.auc, qupper, na.rm = TRUE),
    med_time = median(time_train, na.rm = TRUE),
    lower_time = quantile(time_train, qlower, na.rm = TRUE),
    upper_time = quantile(time_train, qupper, na.rm = TRUE)
  ) %>%
  filter(learner != baseline_lrn)

#equalBreaks = function(n = 4, s = 0.05, ...) {
  #function(x) {
    #d = s * diff(range(x)) / (1 + 2 * s)
    #round(seq(round(min(x) + d, 2), round(max(x) - d, 2), length = n), 2)
  #}
#}


gg1 = df_tmp %>%
  ggplot(aes(color = learner)) +
    geom_point(aes(x = med_time, y = med_auc), size = 1) +
    geom_errorbar(aes(x = med_time, ymin = lower_auc, ymax = upper_auc), alpha = 0.8, size = 0.4) +
    geom_errorbarh(aes(xmin = lower_time, xmax = upper_time, y = med_auc), alpha = 0.8, size = 0.4) +
    my_color +
    my_fill +
    xlab("Speedup") +
    ylab("AUC improvement") +
    labs(color = "", fill = "") +
    #scale_y_continuous(breaks = equalBreaks(3)) +
    scale_x_continuous(trans = "log2", breaks = scales::pretty_breaks(n = 5)) +
    theme(strip.text = element_text(color = "white", face = "bold", size = 8),
      axis.text.x = element_text(size = 6, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 6)) +
    facet_wrap(. ~ task, scales = "free")






gg1 = df_bmr %>%
  filter(learner %in% c(baseline_lrn, additional_lrn)) %>%
  filter(task != "Analcat Halloffame") %>%
  group_by(task) %>%
  summarize(auc = (classif.auc -  classif.auc[learner == baseline_lrn]) /classif.auc[learner == baseline_lrn], learner = learner) %>%
  filter(learner != baseline_lrn) %>%#, auc < 0.2, auc > -0.2) %>%
  ggplot(aes(x = task, y = auc, color = learner, fill = learner)) +
    geom_hline(yintercept = 0, size = hline_size, color = "dark red", linetype = "dashed") +
    geom_boxplot(alpha = 0.2, size = box_width, fatten = box_fatten) +
    geom_vline(xintercept = seq_len(5) + 0.5, size = width_ggsep, alpha = 0.3) +
    #my_color +
    scale_color_manual(values = colors[additional_lrn]) +
    #my_fill +
    scale_fill_manual(values = colors[additional_lrn]) +
    xlab("") +
    ylab(" AUC \nimprovement  ") +
    labs(color = "", fill = "") +
    theme(axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 8, angle = 45, hjust = 0.5, vjust = 0.5), legend.position = "right")

gg2 = df_bmr %>%
  filter(learner %in% c(baseline_lrn, additional_lrn)) %>%
  filter(task != "Analcat Halloffame") %>%
  group_by(task) %>%
  summarize(time = time_train[learner == baseline_lrn] / time_train, learner = learner) %>%
  filter(learner != baseline_lrn) %>%
  ggplot(aes(x = task, y = time, color = learner, fill = learner)) +
    geom_hline(yintercept = 1, color = "dark red", linetype = "dashed", size = hline_size) +
    geom_boxplot(alpha = 0.2, size = box_width, fatten = box_fatten) +
    geom_vline(xintercept = seq_len(5) + 0.5, size = width_ggsep, alpha = 0.3) +
    #my_color +
    scale_color_manual(values = colors[additional_lrn]) +
    #my_fill +
    scale_fill_manual(values = colors[additional_lrn]) +
    xlab("") +
    ylab("\nSpeedup") +
    labs(color = "Algorithm", fill = "Algorithm") +
    theme(axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 8, angle = 45, hjust = 0.5, vjust = 0.5), legend.position = "bottom")

my_legend = g_legend(gg1)
gg1 = gg1 + theme(legend.position = "none")
gg2 = gg2 + theme(legend.position = "none")

gt1 = ggplot_gtable(ggplot_build(gg1))
gt2 = ggplot_gtable(ggplot_build(gg2))
gt2$widths = gt1$width

dev.off()

p3 = arrangeGrob(
  gt1,
  gt2,
  arrangeGrob(
    my_legend,
    ggplot() + theme_void(),
    nrow = 2L
  ),
  nrow = 1,
  ncol = 3,
  widths = c(4,4,2))

plot(p3)

ggsave(
  plot = p3,
  filename = "fig-eq1.pdf",
  width = dinA4width,
  height = dinA4width * 0.27,
  units = "mm")

#system("evince fig-eq1.pdf &")


### EQ2:
### ----------------------------

cwb_files = files[grepl("_cwb[.]", files)]

load(cwb_files[1])


baseline_lrn = "CWB (mboost)"
additional_lrn = c("CWB (nb)", "CWB (b)", "hCWB (b)")
gg1 = df_run %>%
  filter(learner %in% c(baseline_lrn, additional_lrn)) %>%
  filter(task != "Analcat Halloffame") %>%
  group_by(task) %>%
  summarize(time =  time_train[learner == baseline_lrn] / time_train, learner = learner) %>%
  filter(learner != baseline_lrn) %>%
  ggplot(aes(x = task, y = time, color = learner, fill = learner)) +
    geom_hline(yintercept = 1, color = "dark red", linetype = "dashed", size = hline_size) +
    geom_boxplot(alpha = 0.2, fatten = box_fatten, size = box_width) +
    geom_vline(xintercept = seq_len(5) + 0.5, size = width_ggsep, alpha = 0.3) +
    my_color +
    #scale_color_manual(values = colors[additional_lrn]) +
    my_fill +
    #scale_fill_manual(values = colors[additional_lrn]) +
    xlab("") +
    ylab("Speedup  ") +
    labs(color = "", fill = "") +
    theme(axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 8, angle = 45, hjust = 0.5, vjust = 0.5), legend.position = "right")

ggsave(
  plot = gg1,
  filename = "fig-eq2.pdf",
  width = 0.55*dinA4width,
  height = dinA4width * 0.27,
  units = "mm")

#system("evince fig-eq2.pdf &")

### EQ3:
### ==========================

baseline_lrn = "hCWB (b, no mstop)"
additional_lrn = c("Boosted trees", "EBM")
gg1 = df_bmr %>%
  filter(learner %in% c(baseline_lrn, additional_lrn)) %>%
  filter(task != "Analcat Halloffame") %>%
  group_by(task) %>%
  summarize(auc = classif.auc / classif.auc[learner == baseline_lrn] - 1, learner = learner) %>%
  filter(learner != baseline_lrn, auc < 0.2, auc > -0.2) %>%
  ggplot(aes(x = task, y = auc, color = learner, fill = learner)) +
    geom_hline(yintercept = 0, color = "dark red", linetype = "dashed", size = hline_size) +
    geom_boxplot(alpha = 0.2, size = box_width, fatten = box_fatten) +
    geom_vline(xintercept = seq_len(5) + 0.5, size = width_ggsep, alpha = 0.3) +
    #my_color +
    scale_color_manual(values = colors[additional_lrn]) +
    #my_fill +
    scale_fill_manual(values = colors[additional_lrn]) +
    xlab("") +
    ylab(" AUC \nimprovement  ") +
    labs(color = "", fill = "") +
    theme(axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 8, angle = 45, hjust = 0.5, vjust = 0.5), legend.position = "right")

baseline_lrn = "hCWB (b, no mstop)"
gg2 = df_bmr %>%
  filter(learner %in% c(baseline_lrn, additional_lrn)) %>%
  filter(task != "Analcat Halloffame") %>%
  group_by(task) %>%
  summarize(time =  time_train / time_train[learner == baseline_lrn], learner = learner) %>%
  filter(learner != baseline_lrn) %>%
  ggplot(aes(x = task, y = time, color = learner, fill = learner)) +
    geom_hline(yintercept = 1, color = "dark red", linetype = "dashed", size = hline_size) +
    geom_boxplot(alpha = 0.2, size = box_width, fatten = box_fatten) +
    geom_vline(xintercept = seq_len(5) + 0.5, size = width_ggsep, alpha = 0.3) +
    my_color +
    #scale_color_manual(values = colors[additional_lrn]) +
    my_fill +
    #scale_fill_manual(values = colors[additional_lrn]) +
    xlab("") +
    ylab("\nSlowdown") +
    labs(color = "Algorithm", fill = "Algorithm") +
    theme(axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 8, angle = 45, hjust = 0.5, vjust = 0.5), legend.position = "bottom")

my_legend = g_legend(gg1)
gg1 = gg1 + theme(legend.position = "none")
gg2 = gg2 + theme(legend.position = "none")

gt1 = ggplot_gtable(ggplot_build(gg1))
gt2 = ggplot_gtable(ggplot_build(gg2))
gt2$widths = gt1$width

dev.off()

p3 = arrangeGrob(
  gt1,
  gt2,
  arrangeGrob(
    my_legend,
    ggplot() + theme_void(),
    nrow = 2L
  ),
  nrow = 1,
  ncol = 3,
  widths = c(4,4,2))

#plot(p3)

ggsave(
  plot = p3,
  filename = "fig-eq3.pdf",
  width = dinA4width,
  height = dinA4width * 0.27,
  units = "mm")

#system("evince fig-eq3.pdf &")





if (FALSE) {
  library(tidyr)

  lrns = c("CWB (nb)", "CWB (b)", "ACWB (b)", "hCWB (b)", "Boosted trees", "EBM", "CWB (mboost)")
  cellSummary = function(x) {
    if (length(x) > 1) {
      paste0("$", round(mean(x), 3), "\\pm ", round(sd(x), 3), "$")
    } else {
      paste0("$", round(mean(x), 3), "$")
    }
  }
  df_tab = df_bmr %>%
    filter(task != "Analcat Halloffame") %>%
    filter(learner %in% lrns) %>%
    select(task, learner, classif.auc) %>%
    group_by(task, learner) %>%
    summarize(auc = cellSummary(classif.auc)) %>%
    #summarize(auc = median(classif.auc)) %>%
    mutate(measure = "AUC") %>%
    mutate(Learner = learner, learner = NULL) %>%
    select(measure, Learner, task, auc) %>%
    pivot_wider(names_from = "task", values_from = "auc") %>%
    rbind(
      df_run %>%
        filter(task != "Analcat Halloffame") %>%
        filter(learner %in% lrns) %>%
        select(task, learner, time_train) %>%
        group_by(task, learner) %>%
        summarize(runtime = cellSummary(time_train)) %>%
        mutate(measure = "Runtime") %>%
        mutate(Learner = learner, learner = NULL) %>%
        select(measure, Learner, task, runtime) %>%
        pivot_wider(names_from = "task", values_from = "runtime")
    )
  df_tab$measure = c("\\multirow{6}{*}{AUC}", rep("", length(lrns) - 1), "\\multirow{6}{*}{Runtime}", rep("", length(lrns) - 1))
  names(df_tab) = c("", paste0("\\textbf{", names(df_tab)[-1], "}"))
  df_tab %>% knitr::kable(format = "latex", escape = FALSE)

  # How much percent is a model better than vanilla CWB:
  df_bmr %>%
    filter(task != "Analcat Halloffame") %>%
    group_by(task, learner) %>%
    summarize(auc = median(classif.auc), sd = sd(classif.auc))  %>%
    group_by(task) %>%
    summarize(learner = learner, auc_diff = auc / auc[learner == "CWB (nb)"]) %>%
    as.data.frame()

  # Same for runtime:
  df_run %>%
    filter(task != "Analcat Halloffame") %>%
    group_by(task, learner) %>%
    summarize(time_train = median(time_train), sd = sd(time_train)) %>%
    group_by(task) %>%
    summarize(learner = learner, run_diff =   time_train / time_train[learner == "hCWB (b)"]) %>%
    #filter(learner == "CWB (nb)") %>%
    as.data.frame()

  # Latex table with median(auc) +- sd(auc)
  df_bmr %>%
    filter(task != "Analcat Halloffame") %>%
    group_by(task, learner) %>%
    summarize(label = paste0("$", round(median(classif.auc), 3), ifelse(!is.na(sd(time_train)), paste0(" \\pm ", round(sd(classif.auc), 3)), ""), "$")) %>%
    pivot_wider(values_from = label, names_from = task) %>%
    knitr::kable(format = "latex", escape = FALSE)

  # Same for runtime:
  df_run %>%
    filter(task != "Analcat Halloffame") %>%
    group_by(task, learner) %>%
    summarize(label = paste0("$", round(median(time_train), 3), ifelse(!is.na(sd(time_train)), paste0(" \\pm ", round(sd(time_train), 3)), ""), "$")) %>%
    pivot_wider(values_from = label, names_from = task) %>%
    knitr::kable(format = "latex", escape = FALSE)

  # Get average ranks of the learners w.r.t. AUC:
  df_bmr %>%
    group_by(task, learner) %>%
    summarize(mauc = median(classif.auc), sd = sd(classif.auc)) %>%
    group_by(task) %>%
    summarize(learner = learner, auc = mauc, rank = length(learner) + 1 - rank(mauc)) %>%
    group_by(learner) %>%
    summarize(medrank = median(rank), avgrank = mean(rank)) %>%
    as.data.frame()

  # Same for time:
  df_run %>%
    group_by(task, learner) %>%
    summarize(mauc = median(time_train), sd = sd(time_train)) %>%
    group_by(task) %>%
    summarize(learner = learner, auc = mauc, rank = rank(mauc)) %>%
    group_by(learner) %>%
    summarize(medrank = median(rank), avgrank = mean(rank)) %>%
    as.data.frame()

  # Train time total
  sum(df_bmr[["time_both"]], na.rm = TRUE) / 60^2 / 24 / 7

  cwb_lrns_all = learner_table[1:8]
  cwb_lrns = learner_table[c(1,2,8)]

  time_others = (df_bmr %>%
    filter(! learner %in% cwb_lrns_all))[["time_both"]] %>%
    sum(na.rm = TRUE) / 60^2 / 24

  time_cboost = (df_bmr %>%
    filter(learner %in% cwb_lrns))[["time_both"]] %>%
    sum(na.rm = TRUE) / 60^2 / 24

  time_others
  time_cboost

  ts_filter = "Hepatitis"
  (df_bmr %>% filter(learner == "hCWB (b)") %>% filter(task == ts_filter))[["time_both"]] %>% sum(na.rm = TRUE) / 60^2
  (df_bmr %>% filter(learner == "Boosted trees") %>% filter(task == ts_filter))[["time_both"]] %>% sum(na.rm = TRUE) / 60^2


  library(dplyr)

  df_mod_run = df_run %>%
    filter(learner %in% learner_table[c(1,2,5,6,7,8)]) %>%
    filter(task != "Analcat Halloffame") %>%
    mutate(
      binning   = ifelse(grepl("[(]b[)]", learner), "yes", "no"),
      optimizer = ifelse(grepl("ACWB", learner), "nesterov", ifelse(grepl("hCWB", learner), "hybrid", "cod"))) %>%
    group_by(task, iteration) %>%
    mutate(rel_time = time_train[learner == "CWB (nb)"] / time_train) %>%
    filter(learner != "CWB (nb)") %>%
    ungroup() %>%
    select(time_train, task, binning, optimizer, rel_time)

  mod_run = lm(rel_time ~ task*binning*optimizer, data = df_mod_run)

  #mod_run = lm(time_train ~ . + binning*task + optimizer*task + binning*optimizer, data = df_mod_run)
  #mod_run = lm(time_train ~ ., data = df_mod_run)
  summary(mod_run)
  anova(mod_run)

  df_mod_auc = df_bmr %>%
    filter(learner %in% learner_table[1:8]) %>%
    filter(task != "Analcat Halloffame") %>%
    mutate(
      binning   = ifelse(grepl("[(]b[)]", learner), "yes", "no"),
      optimizer = ifelse(grepl("ACWB", learner), "nesterov", ifelse(grepl("hCWB", learner), "hybrid", "cod"))) %>%
    select(classif.auc, task, binning, optimizer)

  mod_auc = lm(classif.auc ~ . + binning*task + optimizer*task + binning*optimizer, data = df_mod_auc)
  summary(mod_auc)
  anova(mod_auc)

  library(mgcv)
  mod_auc = gam(classif.auc ~ task + binning + optimizer + binning*task + optimizer*task + binning*optimizer, data = df_mod_auc,
    family = betar)

  sma = summary(mod_auc)
  knitr::kable(sma$p.table, format = "latex")
  knitr::kable(sma$pTerms.table, format = "latex")
  sma = anova(mod_auc)

}
