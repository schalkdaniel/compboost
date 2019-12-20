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

#ifndef COORDINATE_DESCENT_H_
#define COORDINATE_DESCENT_H_

#include <RcppArmadillo.h>

namespace coodesc
{

arma::sp_mat subsetSparseCols (const arma::sp_mat&, const unsigned int&);
arma::colvec subsetRows (const arma::colvec&, const unsigned int&);
arma::colvec coordinateDescent (const arma::mat&, const arma::sp_mat&, const double&, const double&, const unsigned int&);

} // namespace coodesc

# endif // COORDINATE_DESCENT_H_
