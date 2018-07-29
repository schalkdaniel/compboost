// Example for quadratic loss:
// ---------------------------------


#include <RcppArmadillo.h>

typedef arma::vec (*lossFunPtr) (const arma::vec& true_value, const arma::vec& prediction);
typedef arma::vec (*gradFunPtr) (const arma::vec& true_value, const arma::vec& prediction);
typedef double (*constInitFunPtr) (const arma::vec& true_value);


// Loss function:
// -------------------

arma::vec lossFun (const arma::vec& true_value, const arma::vec& prediction)
{
  return arma::pow(true_value - prediction, 2) / 2;
}

// trainFun:
// -------------------

arma::vec gradFun (const arma::vec& true_value, const arma::vec& prediction)
{
  return prediction - true_value;
}

// predictFun:
// -------------------

double constInitFun (const arma::vec& true_value)
{
  return arma::mean(true_value);
}


// Setter function:
// ------------------

// Now here we wrap the function to an XPtr. This one stores the pointer
// to the function and can be used as parameter for the BaselearnerCustomCppFactory.

// Note that we don't have to export the upper functions since we are just
// interested in the pointer of the functions.

// [[Rcpp::export]]
Rcpp::XPtr<lossFunPtr> lossFunSetter ()
{
  return Rcpp::XPtr<lossFunPtr> (new lossFunPtr (&lossFun));
}

// [[Rcpp::export]]
Rcpp::XPtr<gradFunPtr> gradFunSetter ()
{
  return Rcpp::XPtr<gradFunPtr> (new gradFunPtr (&gradFun));
}

// [[Rcpp::export]]
Rcpp::XPtr<constInitFunPtr> constInitFunSetter ()
{
  return Rcpp::XPtr<constInitFunPtr> (new constInitFunPtr (&constInitFun));
}
