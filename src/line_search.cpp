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

#include "line_search.h"

namespace linesearch {

/**
 * \brief Calculate risk for a given step size
 *
 * This function calculates risk obtained by a given step size (and all the other components).
 * Hence, it defines the objective function we want to minimize to get the optimal step size.
 *
 * \param step_size `double`
 *
 * \param sh_ptr_loss `std::shared_ptr<loss::Loss>`
 *
 * \param target `arma::vec`
 *
 * \param model_prediction `arma::vec`
 *
 * \param baselearner_prediction `arma::vec`
 *
 * \returns `double` Risk evaluated at the given step size
 */
double calculateRisk (const double step_size, const std::shared_ptr<loss::Loss> sh_ptr_loss, const arma::vec& target, const arma::vec& model_prediction,
  const arma::vec& baselearner_prediction)
{
  return arma::accu(sh_ptr_loss->definedLoss(target, model_prediction + step_size * baselearner_prediction)) / model_prediction.size();
}

/**
 * \brief Conduct line search
 *
 * This function calculates the step sized used in boosting to shrink the parameter.
 * It uses the Brent methods from boost to find the minimum. Included from the boost library:
 * https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/brent_minima.html
 *
 * \param sh_ptr_loss `std::shared_ptr<loss::Loss>`
 *
 * \param target `arma::vec`
 *
 * \param model_prediction `arma::vec`
 *
 * \param baselearner_prediction `arma::vec`
 *
 * \returns `double` Optimal step size.
 */
double findOptimalStepSize (const std::shared_ptr<loss::Loss> sh_ptr_loss, const arma::vec& target, const arma::vec& model_prediction,
  const arma::vec& baselearner_prediction, const double lower_bound, const double upper_bound)
{
  boost::uintmax_t max_iter = 500;
  // boost::math::tools::eps_tolerance<double> tol(30);
  int bits = std::numeric_limits<double>::digits;

  // Conduct the root finding:
  std::pair<double, double> r = boost::math::tools::brent_find_minima(std::bind(calculateRisk, std::placeholders::_1, sh_ptr_loss, target, model_prediction, baselearner_prediction), lower_bound, upper_bound, bits, max_iter);

  // return (r.first + r.second) / 2;
  return r.first;
}

} // namespace linesearch
