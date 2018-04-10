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
 * \brief Default constructor of `QuadraticLoss`
 * 
 */

QuadraticLoss::QuadraticLoss () { }

/**
 * \brief Constructor to initialize custom offset of `QuadraticLoss`
 * 
 * \param custom_offset0 `double` Offset which is used instead of the pre 
 *   defined initialization
 * 
 */

QuadraticLoss::QuadraticLoss (const double& custom_offset0)
{ 
  custom_offset = custom_offset0;
  use_custom_offset = true;
}


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
  if (use_custom_offset) { return custom_offset; }
  return arma::mean(true_value);
}


// Absolute loss:
// -----------------------

/**
 * \brief Default constructor of `AbsoluteLoss`
 * 
 */

AbsoluteLoss::AbsoluteLoss () { }

/**
 * \brief Constructor to initialize custom offset of `AbsoluteLoss`
 * 
 * \param custom_offset0 `double` Offset which is used instead of the pre 
 *   defined initialization
 * 
 */

AbsoluteLoss::AbsoluteLoss (const double& custom_offset0)
{ 
  custom_offset = custom_offset0;
  use_custom_offset = true;
}

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
  if (use_custom_offset) { return custom_offset; }
  return arma::median(true_value);
}


// Binomial loss:
// -----------------------

/**
 * \brief Default constructor of `BinomialLoss`
 * 
 */

BinomialLoss::BinomialLoss () { }

/**
* \brief Constructor to initialize custom offset of `AbsoluteLoss`
* 
* \param custom_offset0 `double` Offset which is used instead of the pre 
*   defined initialization
* 
*/

BinomialLoss::BinomialLoss (const double& custom_offset0)
{ 
  if (custom_offset0 > 1 || custom_offset0 < -1) {
    
    Rcpp::warning("BinomialLoss allows just values between -1 and 1 as offset. Continuing with default offset.");
      
  } else {
    
    custom_offset = custom_offset0;
    use_custom_offset = true;
    
  }
}

/**
* \brief Definition of the loss function (see description of the class)
* 
* \param true_value `arma::vec` True value of the response
* \param prediction `arma::vec` Prediction of the true value
* 
* \returns `arma::vec` vector of elementwise application of the loss function
*/

arma::vec BinomialLoss::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
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

arma::vec BinomialLoss::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
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

double BinomialLoss::constantInitializer (const arma::vec& true_value) const
{
  if (use_custom_offset) { return custom_offset; }
  
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


// Custom cpp loss:
// -----------------------

/**
* \brief Default constructor of custom cpp loss class
* 
* \param lossFun `Rcpp::Function` `R` function to calculate the loss
* \param gradientFun `Rcpp::Function` `R` function to calculate the gradient 
*   of the loss function
* \param initFun `Rcpp::Function` `R` function to initialize a constant (here
*   it is not neccessary to initialize in a loss/risk optimal manner)
* 
* \returns `double` constant which minimizes the empirical risk for the given true value
*/

CustomCppLoss::CustomCppLoss (SEXP lossFun0, SEXP gradFun0, SEXP constInitFun0)
{
  // Set functions:
  Rcpp::XPtr<lossFunPtr> myTempLoss (lossFun0);
  lossFun = *myTempLoss;
  
  Rcpp::XPtr<gradFunPtr> myTempGrad (gradFun0);
  gradFun = *myTempGrad;
  
  Rcpp::XPtr<constInitFunPtr> myTempConstInit (constInitFun0);
  constInitFun = *myTempConstInit;
}

/**
 * \brief Definition of the loss function (see description of the class)
 * 
 * \param true_value `arma::vec` True value of the response
 * \param prediction `arma::vec` Prediction of the true value
 * 
 * \returns `arma::vec` vector of elementwise application of the loss function
 */

arma::vec CustomCppLoss::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
{
  return lossFun(true_value, prediction);
}

/**
* \brief Definition of the gradient of the loss function (see description of the class)
* 
* \param true_value `arma::vec` True value of the response
* \param prediction `arma::vec` Prediction of the true value
* 
* \returns `arma::vec` vector of elementwise application of the gradient
*/

arma::vec CustomCppLoss::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
{
  return gradFun(true_value, prediction);
}

/**
* \brief Definition of the constant risk initialization (see description of the class)
* 
* \param true_value `arma::vec` True value of the response
* 
* \returns `double` constant which minimizes the empirical risk for the given true value
*/

double CustomCppLoss::constantInitializer (const arma::vec& true_value) const
{
  return constInitFun(true_value);
}



} // namespace loss
