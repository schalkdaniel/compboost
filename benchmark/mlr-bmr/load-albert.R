filename        = "~/data/albert.csv"
filedescription = "~/data/albert.json"

albert_info = rjson::fromJSON(file = filedescription)
feat_classes = sapply(albert_info$features, function (x) x$type)
feat_classes[feat_classes == "nominal"] = "factor"
albert = read.csv(filename, na.strings = c("?", "NaN"), colClasses = feat_classes)

ts_file = TaskClassif$new(id = "albert", backend = albert, target = "class")

