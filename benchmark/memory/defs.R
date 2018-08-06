library(batchtools)
library(mvtnorm)
library(mboost)
library(compboost)

# Setting:
# ---------------------------------------------------

my.setting = list(
  replications = 5L,
  cores = 1,
  overwrite = FALSE,
  packages = c("mvtnorm", "mboost", "compboost")
)
