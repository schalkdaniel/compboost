## Tasks:
## ---------------------

## CLASSIFICATION

requireNamespace("mlr3oml")
task_classif = list()

load("config.Rda")

if (config$type == "oml") {
  e = try({
    tsk("oml", task_id = as.integer(config$task))
  }, silent = TRUE)
  if (class(e) != "try-error") {
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


## REGRESSION
## On hold for now

if (FALSE) {
## UCI machine learning repository:
# Crime:
crime = read.csv("uci-data/communities.data", header = FALSE, na.strings = "?")
for (i in seq_along(crime)) if (is.character(crime[[i]])) crime[[i]] = as.factor(crime[[i]])
target = tail(names(crime), n = 1)
task_crime = TaskRegr$new(id = "crime", backend = crime, target = target)

# Wine:
wine = read.csv("uci-data/winequality-red.csv", sep = ";")
target = "quality"
task_wine = TaskRegr$new(id = "wine", backend = wine, target = target)

# Housing:
task_housing = tsk("boston_housing")

tasks_regr = list()

### Collect:
tasks_regr[["crime"]] = task_crime
tasks_regr[["wine"]] = task_wine
tasks_regr[["housing"]] = task_housing
}


### Summarize:
#if (FALSE) {
  #ll = lapply(tasks_classif, function (t) {
    #pfac = sum(t$feature_types$type %in% c("factor", "character"))
    #id = t$id
    #if (grepl(":", id)) {
      #tid = readr::parse_number(id)
      #idn = strsplit(x = strsplit(x = id, split = ": ")[[1]][2], split = " [(]")[[1]][1]
      #data.frame(name = idn, id = tid, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
    #} else {
      #idn = id
      #data.frame(name = id, id = NA, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
    #}
  #})

  #knitr::kable(do.call(rbind, unname(ll)), format = "latex")

  #ll = lapply(tasks_regr, function (t) {
    #pfac = sum(t$feature_types$type %in% c("factor", "character"))
    #id = t$id
    #if (grepl(":", id)) {
      #tid = readr::parse_number(id)
      #idn = strsplit(x = strsplit(x = id, split = ": ")[[1]][2], split = " [(]")[[1]][1]
      #data.frame(name = idn, id = tid, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
    #} else {
      #idn = id
      #data.frame(name = id, id = NA, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
    #}
  #})

  #knitr::kable(do.call(rbind, unname(ll)), format = "latex")
#}
