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


### Benchmark:
### ==========================================

test = FALSE
if (test) {
  n_evals = 1L
  resampling_inner = rsmp("holdout")
  resampling_outer = rsmp("holdout")
} else {
  n_evals = 100L
  resampling_inner = rsmp("cv", folds = 3)
  resampling_outer = rsmp("cv", folds = 3)
}
#measures = msrs(list("classif.auc", "classif.ce", "classif.mbrier"))
measure = msr("classif.auc")

source(paste0(bm_dir, "tasks.R"))
source(paste0(bm_dir, "param-sets.R"))
source(paste0(bm_dir, "learners.R"))
source(paste0(bm_dir, "design.R"))

if (test) tasks = tasks[[1]]

## Run benchmark:
## -----------------------

time = proc.time()
bmr = benchmark(design)
time = proc.time() - time
# bmr$aggregate()

cat("\n\n>> FINISHED BENCHMARK\n\n")

time

n_rda = sum(grepl(".Rda", list.files()))
file_name = paste0("bmr", n_rda + 1, ".Rda")
save(bmr, file = file_name)
#load("bmr.Rda")


