library(batchtools)
library(mvtnorm)
library(mboost)
library(compboost)

# Setting:
# ---------------------------------------------------

my.setting = list(
  replications = 5L,
  cores = 4,
  overwrite = FALSE,
  packages = c("mvtnorm", "mboost", "compboost"),

  # iters which are tested:
  bm.iters = c(100, 500, 1000, 2000, 5000, 10000, 15000),

  # data dimensions which are tested:
  n = c(1000, 2000, 5000, 10000, 20000, 50000, 100000),
  p = c(10, 50, 100, 500, 1000, 2000, 4000)
)
