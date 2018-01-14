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
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with Compboost. If not, see <http://www.gnu.org/licenses/>.
//
// This file contains:
// -------------------
//
//   Implementation of the Loss class.
//
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
// ========================================================================== //

#include "loss.h"

namespace loss
{

// Parent class:
// -----------------------

Loss::~Loss () {
  std::cout << "Call Loss Destructor" << std::endl;
}

// -------------------------------------------------------------------------- //
// Child classes:
// -------------------------------------------------------------------------- //

// Quadratic loss:
// -----------------------

arma::vec Quadratic::DefinedLoss (arma::vec &true_value, arma::vec &prediction)
{
  // for debugging:
  // std::cout << "Calculate loss of child class Quadratic!" << std::endl;
  return arma::pow(true_value - prediction, 2) / 2;
}

arma::vec Quadratic::DefinedGradient (arma::vec &true_value, arma::vec &prediction)
{
  // for debugging:
  // std::cout << "Calculate gradient of child class Quadratic!" << std::endl;
  return prediction - true_value;
}

double Quadratic::ConstantInitializer (arma::vec &true_value)
{
  return arma::mean(true_value);
}


// Absolute loss:
// -----------------------


arma::vec Absolute::DefinedLoss (arma::vec &true_value, arma::vec &prediction)
{
  // for debugging:
  // std::cout << "Calculate loss of child class Absolute!" << std::endl;
  return arma::abs(true_value - prediction);
}

arma::vec Absolute::DefinedGradient (arma::vec &true_value, arma::vec &prediction)
{
  // for debugging:
  // std::cout << "Calculate gradient of child class Absolute!" << std::endl;
  return arma::sign(prediction - true_value);
}

double Absolute::ConstantInitializer (arma::vec &true_value)
{
  return arma::median(true_value);
}


// Custom loss:
// -----------------------

// This one is a special one. It allows to use a custom loss predefined in R.
// The convenience here comes from the 'Rcpp::Function' class and the use of
// a special constructor which defines the three needed functions!

// Note that there is one conversion step. There is no predefined conversion
// from 'Rcpp::Function' (which acts as SEXP) to 'double'. But it is possible
// to go the step above 'Rcpp::NumericVector'. Therefore the custom functions
// returns a 'Rcpp::NumericVector' which then is able to be converted to
// 'double' by just selecting one element.


CustomLoss::CustomLoss (Rcpp::Function lossFun, Rcpp::Function gradientFun, Rcpp::Function initFun) 
  : lossFun( lossFun ), 
    gradientFun( gradientFun ), 
    initFun( initFun )
{
  std::cout << "Be careful! You are using a custom loss out of R!"
            << "This will slow down everything!"
            << std::endl;
}

arma::vec CustomLoss::DefinedLoss (arma::vec &true_value, arma::vec &prediction)
{
  // for debugging:
  // std::cout << "Calculate loss for a custom loss!" << std::endl;
  Rcpp::NumericVector out = lossFun(true_value, prediction);
  return out;
}

arma::vec CustomLoss::DefinedGradient (arma::vec &true_value, arma::vec &prediction)
{
  // for debugging:
  // std::cout << "Calculate gradient for a custom loss!" << std::endl;
  Rcpp::NumericVector out = gradientFun(true_value, prediction);
  return out;
}

// Conversion step from 'SEXP' to double via 'Rcpp::NumericVector' which 
// knows how to convert a 'SEXP':
double CustomLoss::ConstantInitializer (arma::vec &true_value)
{
  // for debugging:
  // std::cout << "Initialize custom loss!" << std::endl;
  
  Rcpp::NumericVector out = initFun(true_value);
  
  return out[0];
}


} // namespace loss
