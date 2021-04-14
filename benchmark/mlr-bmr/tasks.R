## Tasks:
## ---------------------

## CLASSIFICATION

if (! "config" %in% ls()) stop("No config file given")

suppressMessages(requireNamespace("mlr3oml"))
tasks_classif = list()

if (config$type == "oml") {
  e = try({
    ts = tsk("oml", task_id = as.integer(as.character(config$task)))
    if (as.character(config$task) == "MiniBooNE") {
      dat = ts$data()
      for (i in seq_along(dat)) {
        if (is.numeric(dat[[i]])) {
          idx_na = dat[[i]] == -999
          dat[[i]][idx_na] = NA
        }
      }
      ts = TaskClassif$new(id = ts$id, backend = dat, target = "signal")
    }
    ts
  }, silent = TRUE)
  if (! "try-error" %in% class(e)) {
    if ("twoclass" %in% e$properties) {
      if (! all(is.na(e$data()))) tasks_classif[[as.character(config$task)]] = e
    }
  } else {
    cat(e)
  }
}

if (config$type == "script") {
  source(paste0("load-", as.character(config$task), ".R"))
  tasks_classif[[as.character(config$task)]] = ts_file
}

if (config$type == "mlr") {
  tasks_classif[[as.character(config$task)]] = tsk(as.character(config$task))
}


