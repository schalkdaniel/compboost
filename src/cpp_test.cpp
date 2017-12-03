#include "compboost.h"

#include <Rcpp.h>
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
//' @useDynLib compboost
//' @importFrom Rcpp evalCpp
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
//' @useDynLib compboost
//' @importFrom Rcpp evalCpp
//' @export
// [[Rcpp::export]]
double timesTwoPtr(std::vector<double> &x) {

  double sum = 0;

  for (unsigned int i = 0; i < x.size(); i++) {
    sum = sum + x[i];
  }
  return sum;
}