## Tasks:
## ---------------------

## CLASSIFICATION

if ("config" %in% ls()) stop("No config file given")

suppressMessages(requireNamespace("mlr3oml"))
tasks_classif = list()

if (config$type == "oml") {
  e = try({
    tsk("oml", task_id = as.integer(config$task))
  }, silent = TRUE)
  if (! "try-error" %in% class(e)) {
    if ("twoclass" %in% e$properties) {
      if (! all(is.na(e$data()))) tasks_classif[[config$task]] = e
    }
  } else {
    cat(e)
  }
}

if (config$type == "script") {
  source(paste0("load-", config$name, ".R"))
  tasks_classif[[config$name]] = ts_file
}

if (config$type == "mlr") {
  tasks_classif[[config$task]] = tsk(config$task)
}


