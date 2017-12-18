#include "compboost.h"

#include <RcppArmadillo.h>
#include <vector>

using namespace Rcpp;

//' @title timesTwo test
//'
//' @description
//'   Multiplied by two!
//' @param x [\code{numeric}] \cr
//'   Vector which should be multiplied by 2.
//' @return [\code{numeric}] \cr
//'   New vector.
//' @export
// [[Rcpp::export]]
double timesTwo(std::vector<double> x) {

  double sum = 0;

  for (unsigned int i = 0; i < x.size(); i++) {
    sum = sum + x[i];
  }
  return sum;
}


//' @title timesTwo test
//'
//' @description
//'   Multiplied by two!
//' @param x [\code{numeric}] \cr
//'   Vector which should be multiplied by 2.
//' @return [\code{numeric}] \cr
//'   New vector.
//' @export
// [[Rcpp::export]]
double timesTwoPtr(std::vector<double> &x) {

  double sum = 0;

  for (unsigned int i = 0; i < x.size(); i++) {
    sum = sum + x[i];
  }
  return sum;
}

//' @title RcppArmadillo test
//'
//' @description
//'   This function is used to play around with Armadillo.
//' @param x [\code{numeric}] \cr
//'   Numeric vector for the armadillo to play around with.
//' @return [\code{numeric}] \cr
//'   The vector after the torture.
//' @export
// [[Rcpp::export]]
arma::vec playArma (arma::vec &x) {

  return x * 0.5;
}