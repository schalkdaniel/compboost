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

#ifndef SPLINE_H_
#define SPLINE_H_

#include <RcppArmadillo.h>

namespace splines {

arma::mat penaltyMat (const unsigned int&, const unsigned int&);
unsigned int findSpan (const double&, const arma::vec&);
arma::vec createKnots (const arma::vec&, const unsigned int&,const unsigned int&);
arma::mat createSplineBasis (const arma::vec&, const unsigned int&, const arma::vec&);
arma::sp_mat createSparseSplineBasis (const arma::vec&, const unsigned int&, const arma::vec&);
arma::mat filterKnotRange (const arma::mat&, const double&, const double&, const std::string&);

} // namespace splines

# endif // SPLINE_H_