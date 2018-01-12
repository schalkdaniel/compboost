// Example for a linear baselearner:
// ---------------------------------


#include <RcppArmadillo.h>

typedef arma::mat (*instantiateDataFunPtr) (arma::mat& X);
typedef arma::mat (*trainFunPtr) (arma::vec& y, arma::mat& X);
typedef arma::mat (*predictFunPtr) (arma::mat& newdata, arma::mat& parameter);


// instantiateDataFun:
// -------------------

arma::mat instantiateDataFun (arma::mat& X)
{
  return X;
}

// trainFun:
// -------------------

arma::mat trainFun (arma::vec& y, arma::mat& X)
{
  return arma::solve(X, y);
}

// predictFun:
// -------------------

arma::mat predictFun (arma::mat& newdata, arma::mat& parameter)
{
  return newdata * parameter;
}


// Setter function:
// ------------------

// Now here we wrap the function to an XPtr. This one stores the pointer
// to the function and can be used as parameter for the CustomCppFactory.

// Note that we don't have to export the upper functions since we are just
// interested in the pointer of the functions.

// [[Rcpp::export]]
Rcpp::XPtr<instantiateDataFunPtr> dataFunSetter ()
{
  return Rcpp::XPtr<instantiateDataFunPtr> (new instantiateDataFunPtr (&instantiateDataFun));
}

// [[Rcpp::export]]
Rcpp::XPtr<trainFunPtr> trainFunSetter ()
{
  return Rcpp::XPtr<trainFunPtr> (new trainFunPtr (&trainFun));
}

// [[Rcpp::export]]
Rcpp::XPtr<predictFunPtr> predictFunSetter ()
{
  return Rcpp::XPtr<predictFunPtr> (new predictFunPtr (&predictFun));
}
