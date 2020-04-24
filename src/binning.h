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

#ifndef BINNING_H_
#define BINNING_H_

#include <iostream>
#include <RcppArmadillo.h>
#include <cmath>

namespace binning {

// Calculate binned vector and index vector:
arma::vec  binVectorCustom      (const arma::vec&, const unsigned int);
arma::vec  binVector            (const arma::vec&);
arma::uvec calculateIndexVector (const arma::vec&, const arma::vec&);

// Matrix multiplication on binned vectors:
arma::mat binnedMatMult                (const arma::mat&, const arma::uvec&, const arma::vec&);
arma::mat binnedMatMultResponse        (const arma::mat&, const arma::vec&, const arma::uvec&, const arma::vec&);
arma::mat binnedSparseMatMult          (const arma::sp_mat&, const arma::uvec&, const arma::vec&);
arma::mat binnedSparseMatMultResponse  (const arma::sp_mat&, const arma::vec&, const arma::uvec&, const arma::vec&);
arma::mat binnedSparsePrediction       (const arma::sp_mat&, const arma::mat&, const arma::uvec&);

} // namespace binning

# endif // BINNING_H_
