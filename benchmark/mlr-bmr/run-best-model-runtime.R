files = list.files(path = "~/repos/compboost/benchmark/mlr-bmr/res-results", full.names = TRUE)

extractStringBetween = function(str, left, right) {
  tmp = sapply(strsplit(str, left), function(x) x[2])
  sapply(strsplit(tmp, right), function(x) x[1])
}

if (! dir.exists("~/repos/compboost/benchmark/mlr-bmr/best-runs"))
  dir.create("~/repos/compboost/benchmark/mlr-bmr/best-runs")

k = 1L
for (fn in files) {
  is_done = any(grepl(fn, list.files("best-runs")))
  if (is_done) {
    cat(k, "/", length(files), ": Already done\n", sep = "")
  } else {
    config_runtime = list(
      ts = extractStringBetween(fn, "-task", "-classif_"),
      ln = paste0("classif_", extractStringBetween(fn, "classif_", ".Rda")),
      file = fn)

    cat(k, "/", length(files), ": ", fn, "\n", sep = "")

    save(config_runtime, file = "config-best-model.Rda")
    system("Rscript runtime-estimator.R")
  }
  k = k + 1L
}


