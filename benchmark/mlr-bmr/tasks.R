## Tasks:
## ---------------------

requireNamespace("mlr3oml")

### Classification

## UCI machine learning repository:
# Adult:
adult = read.csv("uci-data/adult.data", header = FALSE)
for (i in seq_along(adult)) {
  if (is.character(adult[[i]])) adult[[i]] = as.factor(adult[[i]])
}
target = tail(names(adult), n = 1)
adult[[target]] = as.factor(adult[[target]])
task_adult = TaskClassif$new(id = "adult", backend = adult, target = target)

task_spam = tsk("spam")

# Internet Advertisements
adv = read.csv("uci-data/ad.data", header = FALSE)
for (i in seq_along(adv)) {
  if (is.character(adv[[i]])) adv[[i]] = as.factor(adv[[i]])
}
idx_binary = 4:1558
target = tail(names(adv), n = 1)
for (i in idx_binary) { adv[[paste0("V", i)]] = as.factor(adv[[paste0("V", i)]]) }
adv[[target]] = as.factor(adv[[target]])

task_adv = TaskClassif$new(id = "advert", backend = adv, target = target)




## CC18 tasks:
tsk_ids = c(1590, 1510, 1497, 1475, 1461, 1468, 1485, 1486, 1487, 1489,
 23517, 23381, 4534, 4538, 4134, 6332, 40975, 40978, 40979, 40994, 40996,
  41027, 40982, 40983, 40984, 40966, 40701, 40668, 40670, 40923, 40927,
  40499, 1462, 38, 37, 54, 1480, 12, 6, 11, 31, 28, 29, 32, 151, 46, 44,
  300, 469, 1053, 1050, 1049, 1063, 1478, 1501, 23, 22, 16, 14, 15, 18,
  188, 182, 307, 458, 554, 1068, 1067, 1464, 50, 3, 1494)

tasks_classif = list()
for (tid in tsk_ids) {
  e = try({tsk("oml", task_id = tid)}, silent = TRUE)
  if (class(e) != "try-error") {
    if ("twoclass" %in% e$properties) {
      if (! all(is.na(e$data()))) tasks_classif[[as.character(tid)]] = e
    }
  }
}
# id 3 is corrupt

tasks_classif[["adult"]] = task_adult
tasks_classif[["spam"]] = task_spam
tasks_classif[["advert"]] = task_adv



### Regression

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



### Summarize:

ll = lapply(tasks_classif, function (t) {
  pfac = sum(t$feature_types$type %in% c("factor", "character"))
  id = t$id
  if (grepl(":", id)) {
    tid = readr::parse_number(id)
    idn = strsplit(x = strsplit(x = id, split = ": ")[[1]][2], split = " [(]")[[1]][1]
    data.frame(name = idn, id = tid, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
  } else {
    idn = id
    data.frame(name = id, id = NA, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
  }
})

knitr::kable(do.call(rbind, unname(ll)), format = "latex")


ll = lapply(tasks_regr, function (t) {
  pfac = sum(t$feature_types$type %in% c("factor", "character"))
  id = t$id
  if (grepl(":", id)) {
    tid = readr::parse_number(id)
    idn = strsplit(x = strsplit(x = id, split = ": ")[[1]][2], split = " [(]")[[1]][1]
    data.frame(name = idn, id = tid, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
  } else {
    idn = id
    data.frame(name = id, id = NA, n = t$nrow, pnum = t$ncol - pfac, pfac = pfac)
  }
})

knitr::kable(do.call(rbind, unname(ll)), format = "latex")

