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

#ifndef LINE_SEARCH_H_
#define LINE_SEARCH_H_

#include <RcppArmadillo.h>
#include <boost/cstdint.hpp>
#include <boost/math/tools/minima.hpp>

#include "loss.h"

namespace linesearch {

double calculateRisk       (const double, const std::shared_ptr<loss::Loss>, const arma::vec&, const arma::vec&, const arma::vec&);
double findOptimalStepSize (const std::shared_ptr<loss::Loss>, const arma::vec&, const arma::vec&, const arma::vec&, const double = 0., const double = 100);

} // namespace linesearch

#endif // LINE_SEARCH_H_
