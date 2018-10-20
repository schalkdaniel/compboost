library(Rcpp)

sourceCpp("other/module_test.cpp")

test = Test$new()
test$getList()
test$doSomethingWithList()

list_test = list(a = 2, b = 3, c = 4, param_a = 4L, param_b = "hi", boolean = TRUE, vec = 1:10, mat = matrix(1:16, 4, 4), charmat = LETTERS, df = iris)
getListTypes(list_test)

test = Test$new(list_test)
test$doSomethingWithList()





library(inline)

fx <- inline::cxxfunction(signature(), plugin="Rcpp", include=readLines("other/module_test.cpp"))

## assumes fx_unif <- cxxfunction(...) ran
unif_module <- Module("unif_module", getDynLib(fx))
Uniform <- unif_module$Uniform

Uniform@fields

u <- new(Uniform, 0, 10)
u$draw(10L)



library(nycflights13)

data(flights)
# flights = as.data.frame(na.omit(flights))
flights = na.omit(flights)

class(flights)

cboost = Compboost$new(flights, target = "arr_delay", loss = LossQuadratic$new())
cboost$addBaselearner("month", "spline", BaselearnerPSpline)
cboost$getBaselearnerNames()
cboost$train(100)

cboost$plot("month_spline")
cboost$plot("month_spline", iters = c(5, 10, 20, 50, 100))