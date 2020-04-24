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
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include "data.h"

namespace helper
{

bool stringInNames (std::string, std::vector<std::string>);
void assertChoice (const std::string, const std::vector<std::string>&);
Rcpp::List argHandler (Rcpp::List, Rcpp::List, bool);
double calculateSumOfSquaredError (const arma::mat&, const arma::mat&);
arma::mat sigmoid (const arma::mat&);
std::map<std::string, unsigned int> tableResponse (const std::vector<std::string>&);
arma::vec stringVecToBinaryVec(const std::vector<std::string>&, const std::string&);
arma::mat transformToBinaryResponse (const arma::mat&, const double&, const double&, const double&);
void checkForBinaryClassif (const std::vector<std::string>&);
void checkMatrixDim (const arma::mat&, const arma::mat&);
bool checkTracePrinter (const unsigned int&, const unsigned int&);
double matrixQuantile (const arma::mat&, const double&);
arma::SpMat<unsigned int> binaryMat (const arma::Row<unsigned int>&);
arma::uvec binaryToIndex (const arma::mat&);
arma::uvec binaryToIndex (const arma::sp_mat&);
arma::mat predictBinaryIndex (const arma::uvec&, const double);
void getMatStatus (const arma::mat&, const std::string);
arma::mat solveCholesky (const arma::mat&, const arma::mat&);
arma::mat cboostSolver (const std::pair<std::string, arma::mat>&, const arma::mat&);
} // namespace helper

# endif // HELPER_H_
