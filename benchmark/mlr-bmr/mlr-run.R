library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3pipelines)
library(paradox)

base_dir = here::here()
bm_dir = paste0(base_dir, "/bm-scripts/benchmark/")
if (FALSE) devtools::install("~/repos/compboost")

library(R6)
source(paste0(bm_dir, "classifCompboost.R"))


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
bmr$aggregate()


save(bmr, file = "bmr.Rda")
