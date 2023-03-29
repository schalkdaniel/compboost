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
// it under the terms of the LGPL-3 License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// LGPL-3 License for more details. You should have received a copy of
// the license along with compboost.
//
// =========================================================================== #

#include "demmler_reinsch.h"

namespace dro {

// Define function to calculate the degrees of freedom depending on the penalty term. The root of this function
// corresponds to the desired penalty term:
double calculateDegreesOfFreedom (const double x, const arma::vec& singular_values, const double degrees_of_freedom)
{
  return 2 * arma::accu(1 / (1 + x * singular_values)) - arma::accu(1 / arma::pow(1 + x * singular_values, 2)) - degrees_of_freedom;
}

double calculateDegreesOfFreedomRidge (const double x, const arma::vec& table, const double degrees_of_freedom)
{
  return arma::accu((table % (table + 2 * x)) / arma::pow(table + x, 2)) - degrees_of_freedom;
}


// Use TOMS Algorithm 748: it uses a mixture of cubic, quadratic and linear (secant) interpolation to locate the root of f(x)
// Included from the boost library: https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/roots_noderiv/TOMS748.html
double findLambdaWithToms748 (const arma::vec& singular_values, const double degrees_of_freedom, const double lower_bound,
  const double upper_bound)
{
  boost::uintmax_t max_iter = 500;
  boost::math::tools::eps_tolerance<double> tol(30);

  // Conduct the root finding:
  std::pair<double, double> r = boost::math::tools::toms748_solve(std::bind(calculateDegreesOfFreedom, std::placeholders::_1, singular_values, degrees_of_freedom), lower_bound, upper_bound, tol, max_iter);

  return (r.first + r.second) / 2;
}

double demmlerReinschRidge (const arma::vec& table, const double degrees_of_freedom, const double lower_bound,
  const double upper_bound)
{
  boost::uintmax_t max_iter = 500;
  boost::math::tools::eps_tolerance<double> tol(30);

  // Conduct the root finding:
  std::pair<double, double> r = boost::math::tools::toms748_solve(std::bind(calculateDegreesOfFreedomRidge, std::placeholders::_1, table, degrees_of_freedom), lower_bound, upper_bound, tol, max_iter);

  return (r.first + r.second) / 2;

}


double demmlerReinsch (const arma::mat& XtX, const arma::mat& penalty_mat, const double degrees_of_freedom,
  const double eps)
{
  arma::mat cholesky;
  try {
    cholesky = arma::chol(XtX + penalty_mat * eps);
  } catch (const std::exception& e) {
    std::string msg = "From demmlerReinsch: Trying cholesky decomposition of XtX." + std::string(e.what());
    throw std::runtime_error(msg);
  }

  arma::mat cholesky_inv;
  try {
    cholesky_inv = arma::inv(cholesky);
  } catch (const std::exception& e) {
    std::string msg = "From demmlerReinsch: Trying to invert XtX." + std::string(e.what());
    throw std::runtime_error(msg);
  }

  arma::mat Ld  = cholesky_inv.t() * penalty_mat * cholesky_inv;

  arma::vec singular_values = svd(Ld);

  return findLambdaWithToms748(singular_values, degrees_of_freedom);
}

} // namespace dro
