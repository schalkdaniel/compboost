# context("Baselearner works")

# test_that("polynomial baselearner works correctly", {

#   x       = 1:10
#   x.cubic = x^3
#   X = matrix(x, ncol = 1)
#   y = 3 * 1:10 + rnorm(10)
#   y.cubic = 0.5 * x.cubic + rnorm(10)
  
#   data.source     = InMemoryData$new(X, "myvariable")
#   data.target.lin = InMemoryData$new()
#   data.target.cub = InMemoryData$new()
  
#   newdata = InMemoryData$new(as.matrix(runif(10, 1, 10)), "myvariable")
  
#   linear.blearner = PolynomialBlearner$new(data.source, data.target.lin, 1, FALSE)
#   cubic.blearner  = PolynomialBlearner$new(data.source, data.target.cub, 3, FALSE)

#   linear.blearner$train(y)
#   cubic.blearner$train(y.cubic)

#   mod = lm(y ~ 0 + x)
#   mod.cubic = lm(y.cubic ~ 0 + x.cubic)

#   expect_equal(linear.blearner$getData(), X)
#   expect_equal(as.numeric(linear.blearner$getParameter()), unname(coef(mod)))
#   expect_equal(linear.blearner$predict(), as.matrix(unname(predict(mod))))
#   expect_equal(
#     linear.blearner$predictNewdata(newdata),
#     as.matrix(unname(predict(mod, newdata = data.frame(x = newdata$getData()))))
#   )

#   expect_equal(cubic.blearner$getData(), X^3)
#   expect_equal(as.numeric(cubic.blearner$getParameter()), unname(coef(mod.cubic)))
#   expect_equal(cubic.blearner$predict(), as.matrix(unname(predict(mod.cubic))))
#   expect_equal(
#     cubic.blearner$predictNewdata(newdata),
#     as.matrix(unname(predict(mod.cubic, newdata = data.frame(x.cubic = newdata$getData()^3))))
#   )

# })

# test_that("custom baselearner works correctly", {

#   instantiateData = function (X)
#   {
#     return(X);
#   }
#   trainFun = function (y, X) {
#     return(solve(t(X) %*% X) %*% t(X) %*% y)
#   }
#   predictFun = function (model, newdata) {
#     return(newdata %*% model)
#   }
#   extractParameter = function (model) {
#     return(model)
#   }

#   x = 1:10
#   X = matrix(x, ncol = 1)
#   y = 3 * 1:10 + rnorm(10)
  
#   data.source = InMemoryData$new(X, "myvariable")
#   data.target = InMemoryData$new()
  
#   newdata = InMemoryData$new(as.matrix(runif(10, 1, 10)), "myvariable")

#   custom.blearner = CustomBlearner$new(data.source, data.target, instantiateData, 
#     trainFun, predictFun, extractParameter)

#   custom.blearner$train(y)
#   mod = lm(y ~ 0 + x)

#   expect_equal(custom.blearner$getData(), X)
#   expect_equal(as.numeric(custom.blearner$getParameter()), unname(coef(mod)))
#   expect_equal(custom.blearner$predict(), as.matrix(unname(predict(mod))))
#   expect_equal(
#     custom.blearner$predictNewdata(newdata),
#     as.matrix(unname(predict(mod, newdata = data.frame(x = newdata$getData()))))
#   )
# })

# test_that("CustomCpp baselearner works", {

#   Rcpp::sourceCpp(code = '
#     // [[Rcpp::depends(RcppArmadillo)]]
#     #include <RcppArmadillo.h>

#     typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
#     typedef arma::mat (*trainFunPtr) (const arma::vec& y, const arma::mat& X);
#     typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);


#     // instantiateDataFun:
#     // -------------------

#     arma::mat instantiateDataFun (const arma::mat& X)
#     {
#     return X;
#     }

#     // trainFun:
#     // -------------------

#     arma::mat trainFun (const arma::vec& y, const arma::mat& X)
#     {
#     return arma::solve(X, y);
#     }

#     // predictFun:
#     // -------------------

#     arma::mat predictFun (const arma::mat& newdata, const arma::mat& parameter)
#     {
#     return newdata * parameter;
#     }


#     // Setter function:
#     // ------------------

#     // [[Rcpp::export]]
#     Rcpp::XPtr<instantiateDataFunPtr> dataFunSetter ()
#     {
#     return Rcpp::XPtr<instantiateDataFunPtr> (new instantiateDataFunPtr (&instantiateDataFun));
#     }

#     // [[Rcpp::export]]
#     Rcpp::XPtr<trainFunPtr> trainFunSetter ()
#     {
#     return Rcpp::XPtr<trainFunPtr> (new trainFunPtr (&trainFun));
#     }

#     // [[Rcpp::export]]
#     Rcpp::XPtr<predictFunPtr> predictFunSetter ()
#     {
#     return Rcpp::XPtr<predictFunPtr> (new predictFunPtr (&predictFun));
#     }'
#   )

#   x = 1:10
#   X = matrix(x, ncol = 1)
#   y = 3 * 1:10 + rnorm(10)
  
#   data.source = InMemoryData$new(X, "myvariable")
#   data.target = InMemoryData$new()
  
#   newdata = InMemoryData$new(as.matrix(runif(10, 1, 10)), "myvariable")
  
#   # newdata = runif(10, 1, 10)

#   custom.cpp.blearner = CustomCppBlearner$new(data.source, data.target, 
#     dataFunSetter(), trainFunSetter(), predictFunSetter())

#   custom.cpp.blearner$train(y)

#   mod = lm(y ~ 0 + x)

#   expect_equal(custom.cpp.blearner$getData(), X)
#   expect_equal(as.numeric(custom.cpp.blearner$getParameter()), unname(coef(mod)))
#   expect_equal(custom.cpp.blearner$predict(), as.matrix(unname(predict(mod))))
#   expect_equal(
#     custom.cpp.blearner$predictNewdata(newdata),
#     as.matrix(unname(predict(mod, newdata = data.frame(x = newdata$getData()))))
#   )
# })
