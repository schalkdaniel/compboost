if (FALSE) {
  devtools::install("~/repos/compboost")
  install.packages(c("mlr3", "mlr3tuning", "mlr3learners", "mlr3pipelines", "paradox", "xgboost", "ranger", "mboost", "mlr3oml"))
  remotes::install_github("mlr-org/mlr3extralearners")
}

library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3pipelines)
library(paradox)

base_dir = here::here()
bm_dir = paste0(base_dir, "/benchmark/mlr-bmr/")

library(R6)
source(paste0(bm_dir, "classifCompboost.R"))
source(paste0(bm_dir, "regrCompboost.R"))


### Benchmark:
### ==========================================

test = TRUE
if (test) {
  n_evals = 1L
  resampling_inner = rsmp("holdout")
  resampling_outer = rsmp("holdout")
} else {
  n_evals = 100L
  resampling_inner = rsmp("cv", folds = 3)
  resampling_outer = rsmp("cv", folds = 10)
}
measure_classif = msr("classif.auc")
measure_regr = msr("regr.mse")

source(paste0(bm_dir, "tasks.R"))
if (FALSE) {
  tasks_classif = tasks_classif[[1]]
  tasks_regr = tasks_regr[[1]]
}
source(paste0(bm_dir, "param-sets.R"))
source(paste0(bm_dir, "learners.R"))
source(paste0(bm_dir, "design.R"))


## Run benchmark:
## -----------------------

time = proc.time()
bmr_classif  = benchmark(design_classif, store_models = TRUE)
bmr_regr  = benchmark(design_regr, store_models = TRUE)
time = proc.time() - time
#bmr_classif$aggregate()
#bmr_regr$aggregate()

cat("\n\n>> FINISHED BENCHMARK\n\n")

time

#n_rda = sum(grepl(".Rda", list.files()))
#file_name = paste0("bmr", n_rda + 1, ".Rda")
#save(bmr, file = file_name)
#load_name = paste0("bmr", n_rda, ".Rda")
#load(load_name)



### Visualize:
### ======================================

#load(paste0(base_dir, "/benchmark/mlr-bmr/bmr2.Rda"))

#library(ggplot2)
#library(dplyr)
#library(tidyr)
#library(viridis)

#rres = bmr$score(measure = msrs(list("classif.auc", "classif.ce")))
#rres$learner = sapply(strsplit(rres$learner_id, "ps_"), function(x) strsplit(x[2], ".tuned")[[1]])
#rres$task = gsub("([0-9]+).*$", "\\1", rres$task_id)
#rres$task_nrow = sapply(rres$resampling, function (rr) rr$task_nrow)

#rres %>%
  #group_by(task, learner) %>%
  #summarize(auc_med = median(classif.auc), auc_min = min(classif.auc), auc_max = max(classif.auc)) %>%
  #as.data.frame()

#bmrl = as.data.table(bmr)$learner
#l = bmrl[[1]]
#l$tuning_results




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





