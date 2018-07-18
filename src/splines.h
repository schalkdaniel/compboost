#ifndef SPLINE_CPP_
#define SPLINE_CPP_

#include <RcppArmadillo.h>

arma::mat penaltyMat (const unsigned int&, const unsigned int&);
unsigned int findSpan (const double&, const arma::vec&);
arma::vec createKnots (const arma::vec&, const unsigned int&,const unsigned int&);
arma::mat createBasis (const arma::vec&, const unsigned int&, const arma::vec&);
arma::sp_mat createSparseBasis (const arma::vec&, const unsigned int&, const arma::vec&);

# endif // SPLINE_CPP_