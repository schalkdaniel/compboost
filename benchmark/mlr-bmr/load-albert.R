filename = "~/data/albert.csv"
description = "~/data/albert.json"

albert_info = rjson::fromJSON(file = "~/Downloads/albert.json")
feat_classes = sapply(albert_info$features, function (x) x$type)
feat_classes[feat_classes == "nominal"] = "factor"
albert = read.csv("~/Downloads/albert.csv", na.strings = c("?", "NaN"), colClasses = feat_classes)

ts_file = TaskClassif$new(id = "albert", backend = albert, target = "class")

