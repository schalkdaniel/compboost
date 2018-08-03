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
// Written by:
// -----------
//
//   Daniel Schalk
//   Department of Statistics
//   Ludwig-Maximilians-University Munich
//   Ludwigstrasse 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
//   Contact
//   e: contact@danielschalk.com
//   w: danielschalk.com
//
// =========================================================================== #

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

// LossQuadratic loss:
// -----------------------

/**
 * \brief Default constructor of `LossQuadratic`
 * 
 */

LossQuadratic::LossQuadratic () { }

/**
 * \brief Constructor to initialize custom offset of `LossQuadratic`
 * 
 * \param custom_offset0 `double` Offset which is used instead of the pre 
 *   defined initialization
 * 
 */

LossQuadratic::LossQuadratic (const double& custom_offset0)
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

arma::vec LossQuadratic::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
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

arma::vec LossQuadratic::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
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

double LossQuadratic::constantInitializer (const arma::vec& true_value) const
{
  if (use_custom_offset) { return custom_offset; }
  return arma::mean(true_value);
}

/**
 * \brief Definition of the response function
 * 
 * \param score `arma::vec` The score trained during the fitting process
 * 
 * \returns `arma::vec` The transforemd score.
 */
arma::vec LossQuadratic::responseTransformation (const arma::vec& score) const 
{
  return score;
}


// Absolute loss:
// -----------------------

/**
 * \brief Default constructor of `LossAbsolute`
 * 
 */

LossAbsolute::LossAbsolute () { }

/**
 * \brief Constructor to initialize custom offset of `LossAbsolute`
 * 
 * \param custom_offset0 `double` Offset which is used instead of the pre 
 *   defined initialization
 * 
 */

LossAbsolute::LossAbsolute (const double& custom_offset0)
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

arma::vec LossAbsolute::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
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

arma::vec LossAbsolute::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
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

double LossAbsolute::constantInitializer (const arma::vec& true_value) const
{
  if (use_custom_offset) { return custom_offset; }
  return arma::median(true_value);
}

/**
 * \brief Definition of the response function
 * 
 * \param score `arma::vec` The score trained during the fitting process
 * 
 * \returns `arma::vec` The transforemd score.
 */
arma::vec LossAbsolute::responseTransformation (const arma::vec& score) const 
{
  return score;
}


// Binomial loss:
// -----------------------

/**
 * \brief Default constructor of `LossBinomial`
 * 
 */

LossBinomial::LossBinomial () { }

/**
* \brief Constructor to initialize custom offset of `LossAbsolute`
* 
* \param custom_offset0 `double` Offset which is used instead of the pre 
*   defined initialization
* 
*/

LossBinomial::LossBinomial (const double& custom_offset0)
{ 
  if (custom_offset0 > 1 || custom_offset0 < -1) {
    
    Rcpp::warning("LossBinomial allows just values between -1 and 1 as offset. Continuing with default offset.");
      
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

arma::vec LossBinomial::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
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

arma::vec LossBinomial::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
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

double LossBinomial::constantInitializer (const arma::vec& true_value) const
{
  arma::vec unique_values = arma::unique(true_value);
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (unique_values.size() != 2) {
      Rcpp::stop("Binomial loss does not support multiclass classification.");
    }
    if (! arma::all((true_value == -1) || (true_value == 1))) {
      Rcpp::stop("Labels must be coded as -1 and 1.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) { 
    ::Rf_error( "c++ exception (unknown reason)" ); 
  }

  if (use_custom_offset) { return custom_offset; }
  
  double p = arma::accu(true_value + 1) / (2 * true_value.size());
  return 0.5 * std::log(p / (1 - p));
}

/**
 * \brief Definition of the response function
 * 
 * \param score `arma::vec` The score trained during the fitting process
 * 
 * \returns `arma::vec` The transforemd score.
 */
arma::vec LossBinomial::responseTransformation (const arma::vec& score) const 
{
  return 1 / (1 + arma::exp(-score));
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

LossCustom::LossCustom (Rcpp::Function lossFun, Rcpp::Function gradientFun, Rcpp::Function initFun) 
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

arma::vec LossCustom::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
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

arma::vec LossCustom::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
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

double LossCustom::constantInitializer (const arma::vec& true_value) const
{
  // for debugging:
  // Rcpp::Rcout << "Initialize custom loss!" << std::endl;
  
  Rcpp::NumericVector out = initFun(true_value);
  
  return out[0];
}

/**
 * \brief Definition of the response function
 * 
 * \param score `arma::vec` The score trained during the fitting process
 * 
 * \returns `arma::vec` The transforemd score.
 */
arma::vec LossCustom::responseTransformation (const arma::vec& score) const 
{
  return score;
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

LossCustomCpp::LossCustomCpp (SEXP lossFun0, SEXP gradFun0, SEXP constInitFun0)
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

arma::vec LossCustomCpp::definedLoss (const arma::vec& true_value, const arma::vec& prediction) const
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

arma::vec LossCustomCpp::definedGradient (const arma::vec& true_value, const arma::vec& prediction) const
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

double LossCustomCpp::constantInitializer (const arma::vec& true_value) const
{
  return constInitFun(true_value);
}

/**
 * \brief Definition of the response function
 * 
 * \param score `arma::vec` The score trained during the fitting process
 * 
 * \returns `arma::vec` The transforemd score.
 */
arma::vec LossCustomCpp::responseTransformation (const arma::vec& score) const 
{
  return score;
}



} // namespace loss
