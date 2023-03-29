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
#ifndef TENSORS_H_
#define TENSORS_H_

#include "RcppArmadillo.h"

namespace tensors
{
arma::mat rowWiseKronecker (const arma::mat&, const arma::mat&);
arma::sp_mat rowWiseKroneckerSparse (const arma::sp_mat&, const arma::sp_mat&);
arma::mat penaltySumKronecker (const arma::mat&, const arma::mat&);
arma::mat centerDesignMatrix (const arma::mat&, const arma::mat&);
arma::mat centerDesignMatrix (const arma::mat&, const arma::mat&, const arma::uvec&);
} // namespace tensors

# endif // TENSORS_H_
