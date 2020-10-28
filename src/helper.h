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

extern bool _DEBUG_PRINT;

namespace helper
{

void debugPrint (const std::string);

bool        stringInNames              (std::string, std::vector<std::string>);
void        assertChoice               (const std::string, const std::vector<std::string>&);
Rcpp::List  argHandler                 (Rcpp::List, Rcpp::List, bool);
double      calculateSumOfSquaredError (const arma::mat&, const arma::mat&);
arma::mat   sigmoid                    (const arma::mat&);

std::map<std::string, unsigned int> tableResponse (const std::vector<std::string>&);

arma::vec  stringVecToBinaryVec      (const std::vector<std::string>&, const std::string&);
arma::mat  transformToBinaryResponse (const arma::mat&, const double&, const double&, const double&);
void       checkForBinaryClassif     (const std::vector<std::string>&);
void       checkMatrixDim            (const arma::mat&, const arma::mat&);
bool       checkTracePrinter         (const unsigned int&, const unsigned int&);
double     matrixQuantile            (const arma::mat&, const double&);

std::string getMatStatus    (const arma::mat&);
void        printMatStatus  (const arma::mat&, const std::string);
arma::mat   solveCholesky   (const arma::mat&, const arma::mat&);
arma::mat   cboostSolver    (const std::pair<std::string, arma::mat>&, const arma::mat&);

// template<typename SH_PTR>
// inline unsigned int countSharedPointer (const SH_PTR&);
template<typename SH_PTR>
inline unsigned int countSharedPointer (const SH_PTR& sh_ptr)
{
  return sh_ptr.use_count();
}

// inline arma::mat predictBinaryIndex (const arma::uvec&, const unsigned int, const double);
inline arma::mat predictBinaryIndex (const arma::uvec& idx, const unsigned int nobs, const double value)
{
  arma::mat out(nobs, 1, arma::fill::zeros);
  for (unsigned int i = 0; i < idx.size(); i++) {
    out(idx(i)) = value;
  }
  return out;
}


} // namespace helper

# endif // HELPER_H_
