library(Rcpp)

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

sourceCpp(file = "tutorials/custom_cpp_learner.cpp")

custom.cpp.factory = CustomCppBlearnerFactory$new(X, "my_variable_name", dataFunSetter(),
  trainFunSetter(), predictFunSetter())

custom.cpp.factory$testTrain(y)
custom.cpp.factory$testGetParameter()
custom.cpp.factory$testPredict()
