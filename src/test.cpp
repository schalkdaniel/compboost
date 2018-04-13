#include <Rcpp.h>

//' My cpp mean function
//' 
//' @param x 
//'   Input vector to calculate the mean from
//' @return 
//'   A double containing the mean
// [[Rcpp::export]]
double myMean (Rcpp::NumericVector x) {
  
  std::vector<double> v = Rcpp::as<std::vector<double> >(x);
  double sum = std::accumulate(v.begin(), v.end(), 0);
  
  return sum / v.size();
}


