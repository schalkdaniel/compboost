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
#ifndef TENSORS_H_
#define TENSORS_H_

#include "RcppArmadillo.h"

namespace tensors
{
arma::mat rowWiseKronecker (const arma::mat&, const arma::mat&);
arma::mat penaltySumKronecker (const arma::mat&, const arma::mat&);
arma::vec trapezWeights (const arma::vec&);
std::map<std::string, arma::mat>  centerDesignMatrix (const arma::mat&, const arma::mat&, const arma::mat&);
} // namespace tensors

# endif // TENSORS_H_
