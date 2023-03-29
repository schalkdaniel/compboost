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
// ========================================================================== //

#include "lossoptim.h"

namespace lossoptim
{

double calculateRiskForConstant (const double constant, const arma::mat& truth, const std::shared_ptr<const loss::Loss>& sh_ptr_loss)
{
  arma::mat temp(truth.n_rows, truth.n_cols);
  temp.fill(constant);

  return sh_ptr_loss->calculateEmpiricalRisk(truth, temp);
}

double calculateWeightedRiskForConstant (const double constant, const arma::mat& truth, const arma::mat& weights,
  const std::shared_ptr<const loss::Loss>& sh_ptr_loss)
{
  arma::mat temp(truth.n_rows, truth.n_cols);
  temp.fill(constant);

  return sh_ptr_loss->calculateWeightedEmpiricalRisk(truth, temp, weights);
}

double findOptimalLossConstant (const arma::mat& truth, const std::shared_ptr<const loss::Loss>& sh_ptr_loss,
  const double lower_bound, const double upper_bound)
{
  boost::uintmax_t max_iter = 500;
  int bits = std::numeric_limits<double>::digits;

  // Conduct the root finding:
  std::pair<double, double> r = boost::math::tools::brent_find_minima(
    std::bind(calculateRiskForConstant, std::placeholders::_1, truth, sh_ptr_loss),
    lower_bound, upper_bound, bits, max_iter);

  return r.first;
}

double findOptimalWeightedLossConstant (const arma::mat& truth, const arma::mat& weights, const std::shared_ptr<const loss::Loss>& sh_ptr_loss,
  const double lower_bound, const double upper_bound)
{
  boost::uintmax_t max_iter = 500;
  int bits = std::numeric_limits<double>::digits;

  // Conduct the root finding:
  std::pair<double, double> r = boost::math::tools::brent_find_minima(
    std::bind(calculateWeightedRiskForConstant, std::placeholders::_1, truth,
    weights, sh_ptr_loss), lower_bound, upper_bound, bits, max_iter);

  return r.first;
}

} // namespace lossoptim
