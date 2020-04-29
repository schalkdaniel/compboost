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
// ========================================================================== //

#ifndef LOSSOPTIM_H_
#define LOSSOPTIM_H_

#include "loss.h"

#include <RcppArmadillo.h>
#include <memory>

#include <boost/math/tools/minima.hpp>

namespace lossoptim
{

double calculateRiskForConstant (const double&, const arma::mat&, const std::shared_ptr<const loss::Loss>);
double calculateWeightedRiskForConstant (const double&, const arma::mat&, const arma::mat&, const std::shared_ptr<const loss::Loss>);

double findOptimalLossConstant (const arma::mat&, const std::shared_ptr<const loss::Loss>,
  const double& = -std::numeric_limits<double>::infinity(), const double& = std::numeric_limits<double>::infinity());
double findOptimalWeightedLossConstant (const arma::mat&, const arma::mat&, const std::shared_ptr<const loss::Loss>,
  const double& = -std::numeric_limits<double>::infinity(), const double& = std::numeric_limits<double>::infinity());

} // namespace lossoptim

# endif // LOSSOPTIM_H_
