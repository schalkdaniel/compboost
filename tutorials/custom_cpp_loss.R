library(Rcpp)

sourceCpp(file = "tutorials/custom_cpp_loss.cpp")

true.value = rnorm(100)
prediction = rnorm(100)

custom.cpp.loss = CustomCppLoss$new(lossFunSetter(), gradFunSetter(), constInitFunSetter())


all.equal(
  as.matrix((true.value - prediction)^2 / 2), 
  custom.cpp.loss$testLoss(true.value, prediction)
)

all.equal(
  as.matrix(prediction - true.value), 
  custom.cpp.loss$testGradient(true.value, prediction)
)

all.equal(custom.cpp.loss$testConstantInitializer(true.value), mean(true.value))
