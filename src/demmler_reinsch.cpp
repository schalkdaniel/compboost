// ========================================================================== //
//                                 ___.                          __           //
//        ____  ____   _____ ______\_ |__   ____   ____  _______/  |_         //
//      _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\        //
//      \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |          //
//       \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|          //
//           \/            \/|__|       \/                  \/                //
//                                                                            //
// ========================================================================== //
//
// Compboost is free software: you can redistribute it and/or modify
// it under the terms of the MIT License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// MIT License for more details. You should have received a copy of
// the MIT License along with compboost.
//
// =========================================================================== #

#include "demmler_reinsch.h"


namespace demrei
{


// Define function to calculate the degrees of freedom depending on the penalty term. The root of this function
// corresponds to the desired penalty term:
double calculateDegreesOfFreedom (const double& x, const arma::vec& singular_values, const double& degrees_of_freedom)
{
  return 2 * arma::accu(1 / (1 + x * singular_values)) - arma::accu(1 / arma::pow(1 + x * singular_values, 2)) - degrees_of_freedom;
}

// Use TOMS Algorithm 748: it uses a mixture of cubic, quadratic and linear (secant) interpolation to locate the root of f(x)
// Included from the boost library: https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/roots_noderiv/TOMS748.html
double findLambdaWithToms748 (const arma::vec& singular_values, const double& degrees_of_freedom, const double& lower_bound = 0.,
  const double& upper_bound = 1e15) 
{  
  boost::uintmax_t max_iter = 500;
  boost::math::tools::eps_tolerance<double> tol(30);
 
  // Conduct the root finding:
  std::pair<double, double> r = boost::math::tools::toms748_solve(std::bind(calculateDegreesOfFreedom, std::placeholders::_1, singular_values, degrees_of_freedom), lower_bound, upper_bound, tol, max_iter);

  return (r.first + r.second) / 2;
}

double demmlerReinsch (const arma::mat& XtX, const arma::mat& penalty_mat, const double& degrees_of_freedom)
{
  const double eps = 1e-9;

  arma::mat cholesky = arma::chol(XtX + penalty_mat * eps);
  arma::mat cholesky_inv = arma::inv(cholesky);

  arma::mat Ld  = cholesky_inv.t() * penalty_mat * cholesky_inv;

  arma::vec singular_values = svd(Ld);

  return findLambdaWithToms748(singular_values, degrees_of_freedom);
}



/// - Legacy
// compute the smoothing parameter
double getLambda (const arma::mat X, const arma::mat W, const arma::mat D, const arma::mat y, int df)
{
  
  // Calculate Everything needed
  arma::mat W2 = arma::sqrt(W);
  arma::mat R = arma::chol(X.t() * W2 * W2 * X);
  R = arma::inv(R);
  arma::mat L = R.t() * D * R;
  arma::vec s = arma::svd(L);

  auto lamfun = [](double x, arma::vec s, int df)
  {
    double out = 0;
    for(int i = 0; i<s.n_cols; i++)
    {
    out = out + 1/(1+x*s(i));
    }
    out = out - df;
    return out;
  };
    
  // FIXME 
  // now we only need to find the largest root of lamfun
  double root = 0;
  
  return root;
}

// compute the degrees of freedom
double getDF (const arma::mat X, const arma::mat W, const arma::mat D, const arma::mat y, double lambda)
{
  arma::mat W2 = arma::sqrt(W);
  arma::mat R = arma::chol(X.t() * W2 * W2 * X);
  R = arma::inv(R);
  arma::mat L = R.t() * D * R;
  arma::vec s = arma::svd(L);

  double df = 0;

  for(int i = 0; i<s.n_cols; i++)
  {
    df = df + 1/(1+lambda*s(i));
  }

  return df;
}

}





