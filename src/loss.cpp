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
  // Rcpp::Rcout << "Call Loss Destructor" << std::endl;
}

// -------------------------------------------------------------------------- //
// Child classes:
// -------------------------------------------------------------------------- //

// QuadraticLoss loss:
// -----------------------

/**
 * \brief Definition of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the loss function
 */

arma::vec QuadraticLoss::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate loss of child class Quadratic!" << std::endl;
  return arma::pow(true_value - prediction, 2) / 2;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the gradient
 */

arma::vec QuadraticLoss::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate gradient of child class Quadratic!" << std::endl;
  return prediction - true_value;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * 
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */

double QuadraticLoss::constantInitializer (const arma::vec& true_value) const
{
  return arma::mean(true_value);
}


// Absolute loss:
// -----------------------

/**
 * \brief Definition of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the loss function
 */

arma::vec AbsoluteLoss::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate loss of child class Absolute!" << std::endl;
  return arma::abs(true_value - prediction);
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the gradient
 */

arma::vec AbsoluteLoss::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate gradient of child class Absolute!" << std::endl;
  return arma::sign(prediction - true_value);
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * 
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */

double AbsoluteLoss::constantInitializer (const arma::vec& true_value) const
{
  return arma::median(true_value);
}


// Absolute loss:
// -----------------------

/**
* \brief Definition of the loss function (see description of the class)
* 
* \param true_value `arma::vec` True value of the response
* \param prediction `arma::vec` Prediction of the true value
* 
* \returns `arma::vec` vector of elementwise application of the loss function
*/

arma::vec BernoulliLoss::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
{
  return arma::log(1 + arma::exp(- true_value % prediction));
}

/**
* \brief Definition of the gradient of the loss function (see description of the class)
* 
* \param true_value `arma::vec` True value of the response
* \param prediction `arma::vec` Prediction of the true value
* 
* \returns `arma::vec` vector of elementwise application of the gradient
*/

arma::vec BernoulliLoss::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
{
  return - true_value / (1 + arma::exp(true_value % prediction));
}

/**
* \brief Definition of the constant risk initialization (see description of the class)
* 
* \param true_value `arma::vec` True value of the response
* 
* \returns `double` constant which minimizes the empirical risk for the given true value
*/

double BernoulliLoss::constantInitializer (const arma::vec& true_value) const
{
  double p = arma::accu(true_value + 1) / (2 * true_value.size());
  return 0.5 * std::log(p / (1 - p));
}



// Custom loss:
// -----------------------

/**
 * \brief Default constructor of custom loss class
 * 
 * \param lossFun `Rcpp::Function` `R` function to calculate the loss
 * \param gradientFun `Rcpp::Function` `R` function to calculate the gradient 
 *   of the loss function
 * \param initFun `Rcpp::Function` `R` function to initialize a constant (here
 *   it is not neccessary to initialize in a loss/risk optimal manner)
 * 
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
CustomLoss::CustomLoss (Rcpp::Function lossFun, Rcpp::Function gradientFun, Rcpp::Function initFun) 
  : lossFun( lossFun ), 
    gradientFun( gradientFun ), 
    initFun( initFun )
{
  // Rcpp::Rcout << "Be careful! You are using a custom loss out of R!"
  //           << "This will slow down everything!"
  //           << std::endl;
}

/**
 * \brief Definition of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the loss function
 */

arma::vec CustomLoss::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate loss for a custom loss!" << std::endl;
  Rcpp::NumericVector out = lossFun(true_value, prediction);
  return out;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the gradient
 */

arma::vec CustomLoss::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate gradient for a custom loss!" << std::endl;
  Rcpp::NumericVector out = gradientFun(true_value, prediction);
  return out;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * 
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */

double CustomLoss::constantInitializer (const arma::vec& true_value) const
{
  // for debugging:
  // Rcpp::Rcout << "Initialize custom loss!" << std::endl;
  
  Rcpp::NumericVector out = initFun(true_value);
  
  return out[0];
}


} // namespace loss
