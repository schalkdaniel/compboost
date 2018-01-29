---
title: "The four or maybe five Stages of a Custom Learner"
output: 
  html_document:
    keep_md: true
---




# About This Document

Since we want to be as flexible as possible we provide two custom baselearner
classes (see next). The use of this classes requires two different levels of
programming. 

The easier one is to program everything in `R` and then scoop the functions
to the learner. The same procedure, which is a bit more complex in detail, 
is also provided for `C++` functions.

We now want to go through all these stages and create a custom linear
baselearner. In the end we benchmark everything against each other to get an
idea how slow or fast it is to use custom learner.

But first of all we have to load the package and define some data:

```r
devtools::load_all()
## Loading compboost

n.sim = 10

X = matrix(1:n.sim, ncol = 1)
y = 2 * (1:n.sim)^2 + rnorm(n.sim, 0, 20)
```


# What Stages?

`compboost` gives the opportunity to define own custom baselearner. This can 
be done int wo different ways:

1. Define all the functions within `R`, and give the functions to the 
   `CustomFactory` object. This may be good for prototyping. Nevertheless, it
   slows down the whole system. Here we are working with `SEXP` types.
  
2. Define all the function within `C++`. This sounds very tricky. Actually, it
   isn't. Just take a look at the `custom_cpp_learner.R` tutorial. It is then
   possible to set `C++` functions as members within the `CustomCppFactory`.
   Here we work with `XPtr` types.
   
# Custom `R` Functions

The easierst one is to define your own `R` functions. We use the most 
inefficient way one can think of. Therefore we define everything using
`R`s `lm` function:


```r
instantiateDataFun = function (X) {
  return(X)
}

trainFun = function (y, X) {
  return(lm(y ~ 0 + X))
}

predictFun = function (model, newdata) {
  return(as.matrix(predict(model, as.data.frame(newdata))))
}

extractParameter = function (model) {
  return(as.matrix(coef(model)))
}
```

Now we define the factory:


```r
custom.r.learner1 = CustomBlearner$new(X, "varname", instantiateDataFun, 
  trainFun, predictFun, extractParameter)
```

And basically thats it. Now we can test if our factory works correctly:


```r
custom.r.learner1$getData()
##       [,1]
##  [1,]    1
##  [2,]    2
##  [3,]    3
##  [4,]    4
##  [5,]    5
##  [6,]    6
##  [7,]    7
##  [8,]    8
##  [9,]    9
## [10,]   10
custom.r.learner1$train(y)
custom.r.learner1$getParameter()
##          [,1]
## [1,] 14.58757

trainFun(y, X)
## 
## Call:
## lm(formula = y ~ 0 + X)
## 
## Coefficients:
##     X  
## 14.59
```

Looks nice. Now we can get to the next stage.

# A smarter Custom `R` Learner

Not very hard to see, the first custom learner is super stupid. `R`s `lm` does
so much more than it have to. We now can define smarter functions. To do this 
a bit smarter we define the functions by hand and compute the estimator:


```r
instantiateDataFun = function (X) {
  return(X)
}

trainFun = function (y, X) {
  return(solve(t(X) %*% X) %*% t(X) %*% y)
}

predictFun = function (model, newdata) {
  return(as.matrix(newdata %*% model))
}

extractParameter = function (model) {
  return(as.matrix(model))
}
```

Now we define the factory:


```r
custom.r.learner2 = CustomBlearner$new(X, "varname", instantiateDataFun, 
  trainFun, predictFun, extractParameter)
```

And basically thats it. Now we can test if our factory works correctly:


```r
custom.r.learner2$getData()
##       [,1]
##  [1,]    1
##  [2,]    2
##  [3,]    3
##  [4,]    4
##  [5,]    5
##  [6,]    6
##  [7,]    7
##  [8,]    8
##  [9,]    9
## [10,]   10
custom.r.learner2$train(y)
custom.r.learner2$getParameter()
##          [,1]
## [1,] 14.58757

trainFun(y, X)
##          [,1]
## [1,] 14.58757
```

# A more smarter Custom `R` Learner

Now we use something special. We use `Rcpp` and define custom `C++` learner 
which then are exposed to `R`. This basically exactly the same as the previous
learner. Thats why this is maybe the fift learner. Nevertheless, we have always 
a communication between `C++` and `R`. We also note, that the result of the 
train function is stored in an extra object within the custom baselearner. This 
is reused to do predictions:


```r
instantiateDataFun.code = "
arma::mat instantiateDataFun (arma::mat& X)
{
  return X;
}
"

trainFun.code = "
arma::mat trainFun (arma::vec& y, arma::mat& X)
{
  return arma::solve(X, y);
}
"

predictFun.code = "
arma::mat predictFun (arma::mat& parameter, arma::mat& newdata)
{
  return newdata * parameter;
}
"

extractParameter.code = "
arma::mat extractParameter (arma::mat& parameter)
{
  return parameter;
}
"

Rcpp::cppFunction(instantiateDataFun.code, depends = "RcppArmadillo")
Rcpp::cppFunction(trainFun.code, depends = "RcppArmadillo")
Rcpp::cppFunction(predictFun.code, depends = "RcppArmadillo")
Rcpp::cppFunction(extractParameter.code, depends = "RcppArmadillo")
```

Now we can give this function to the new factory and test if it works:


```r
custom.r.learner3 = CustomBlearner$new(X, "varname", instantiateDataFun, 
  trainFun, predictFun, extractParameter)

custom.r.learner3$getData()
##       [,1]
##  [1,]    1
##  [2,]    2
##  [3,]    3
##  [4,]    4
##  [5,]    5
##  [6,]    6
##  [7,]    7
##  [8,]    8
##  [9,]    9
## [10,]   10
custom.r.learner3$train(y)
custom.r.learner3$getParameter()
##          [,1]
## [1,] 14.58757

trainFun(y, X)
##          [,1]
## [1,] 14.58757
```

Looks alright!

# Own `C++` Function

This done in `custom_cpp_learner.cpp`. We now just copy and paste the code
from `custom_cpp_learner.R`. We note, that we have predefined function setter
which just returns the pointer and give them directly to the underlying `C++`
class:


```r
Rcpp::sourceCpp(file = "custom_cpp_learner.cpp")
## Warning in normalizePath(path.expand(path), winslash, mustWork):
## path[1]="C:/Users/schal/OneDrive/GitHub_repos/compboost/tutorials/../inst/
## include": Das System kann den angegebenen Pfad nicht finden

custom.cpp.learner = CustomCppBlearner$new(X, "varname", dataFunSetter(),
  trainFunSetter(), predictFunSetter())

custom.cpp.learner$getData()
##       [,1]
##  [1,]    1
##  [2,]    2
##  [3,]    3
##  [4,]    4
##  [5,]    5
##  [6,]    6
##  [7,]    7
##  [8,]    8
##  [9,]    9
## [10,]   10
custom.cpp.learner$train(y)
custom.cpp.learner$getParameter()
##          [,1]
## [1,] 14.58757
```

This also seems to work, nice!

# The Implemented One

The last option here is to use the pre implemented one:


```r
linear.learner = PolynomialBlearner$new(X, "varname", 1)

linear.learner$getData()
##       [,1]
##  [1,]    1
##  [2,]    2
##  [3,]    3
##  [4,]    4
##  [5,]    5
##  [6,]    6
##  [7,]    7
##  [8,]    8
##  [9,]    9
## [10,]   10
linear.learner$train(y)
linear.learner$getParameter()
##          [,1]
## [1,] 14.58757
```

This works as expected. 

# Benchmark

Now we want to see where differences are in terms of performance of the 
training:


```r
microbenchmark::microbenchmark(
  "custom_r_1"    = custom.r.learner1$train(y),
  "custom_r_2"    = custom.r.learner2$train(y),
  "custom_r_3"    = custom.r.learner3$train(y),
  "custom_cpp"    = custom.cpp.learner$train(y),
  "pre_installed" = linear.learner$train(y)
)
## Unit: microseconds
##           expr     min       lq       mean    median        uq      max neval  cld
##     custom_r_1 681.534 952.4480 1228.86020 1126.5130 1455.9525 2976.471   100    d
##     custom_r_2  80.803 104.2125  173.23460  149.9005  172.5555 2572.459   100   c 
##     custom_r_3  52.484  76.0835  119.18403  102.3250  129.6995 1429.144   100  bc 
##     custom_cpp  11.706  18.8795   33.98667   28.6965   44.1770   98.549   100 ab  
##  pre_installed  11.706  22.4665   32.22713   28.5080   40.4015   84.201   100 a
```

As expected, the first `R` learner is the slowest. The third `R` one which 
already uses `C++` is a bit faster than the second `R` learner, but not as fast 
as the other two. This two learner are very similar. Therefore we have two 
options to extend `compboost` with own baselearner. Of course this benchmark 
has to be done for the whole training which uses the custom learner. But we 
get an idea how the performance differ between the different type of custom
learner.
