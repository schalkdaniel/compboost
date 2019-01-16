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

#ifndef HELPER_H_
#define HELPER_H_

#include <RcppArmadillo.h>
#include <sstream>
#include <string>

namespace helper
{

bool stringInNames (std::string, std::vector<std::string>);
Rcpp::List argHandler (Rcpp::List, Rcpp::List, bool);
double calculateSumOfSquaredError (const arma::mat&, const arma::mat&);
arma::mat sigmoid (const arma::mat&);
arma::mat transformToBinaryResponse (const arma::mat&, const double&, const double&, const double&);
void checkForBinaryClassif (const arma::mat&, const int&, const int&);
void checkMatrixDim (const arma::mat&, const arma::mat&);

} // namespace helper

# endif // HELPER_H_