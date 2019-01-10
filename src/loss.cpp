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

#include "loss.h"

namespace loss
{

// Parent class:
// -----------------------

std::string Loss::getTaskId () const
{
  return task_id;
}

double Loss::calculateEmpiricalRisk (const arma::mat& true_value, const arma::mat& prediction) const
{
  return arma::accu(definedLoss(true_value, prediction)) / true_value.size();
}

arma::mat Loss::calculatePseudoResiduals (const arma::mat& true_value, const arma::mat& prediction) const
{
  return -definedGradient(true_value, prediction);
}

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
LossQuadratic::LossQuadratic ()
{
  task_id = "regression";
}

/**
 * \brief Constructor to initialize custom offset of `LossQuadratic`
 *
 * \param custom_offset0 `double` Offset which is used instead of the pre
 *   defined initialization
 *
 */
LossQuadratic::LossQuadratic (const double& custom_offset0)
{
  task_id = "regression";
  custom_offset = custom_offset0;
  use_custom_offset = true;
}


/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the loss function
 */
arma::mat LossQuadratic::definedLoss (const arma::mat& true_value, const arma::mat& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate loss of child class Quadratic!" << std::endl;
  return arma::pow(true_value - prediction, 2) / 2;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the gradient
 */
arma::mat LossQuadratic::definedGradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate gradient of child class Quadratic!" << std::endl;
  return prediction - true_value;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossQuadratic::constantInitializer (const arma::mat& true_value) const
{
  if (use_custom_offset) { return custom_offset; }
  arma::mat out(1, 1);
  out.fill(arma::accu(true_value) / true_value.size());
  return out;
}


// Absolute loss:
// -----------------------

/**
 * \brief Default constructor of `LossAbsolute`
 *
 */
LossAbsolute::LossAbsolute ()
{
  task_id = "regression"; // set parent
}

/**
 * \brief Constructor to initialize custom offset of `LossAbsolute`
 *
 * \param custom_offset0 `double` Offset which is used instead of the pre
 *   defined initialization
 *
 */
LossAbsolute::LossAbsolute (const double& custom_offset0)
{
  task_id = "regression"; // set parent
  custom_offset = custom_offset0;
  use_custom_offset = true;
}

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the loss function
 */
arma::mat LossAbsolute::definedLoss (const arma::mat& true_value, const arma::mat& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate loss of child class Absolute!" << std::endl;
  return arma::abs(true_value - prediction);
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the gradient
 */
arma::mat LossAbsolute::definedGradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate gradient of child class Absolute!" << std::endl;
  return arma::sign(prediction - true_value);
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossAbsolute::constantInitializer (const arma::mat& true_value) const
{
  if (use_custom_offset) { return custom_offset; }
  arma::vec temp = true_value;
  arma::mat out(1, 1);
  out.fill(arma::median(temp));
  return out;
}

// Binomial loss:
// -----------------------

/**
 * \brief Default constructor of `LossBinomial`
 *
 */
LossBinomial::LossBinomial ()
{
  task_id = "binary_classif"; // set parent
}

/**
* \brief Constructor to initialize custom offset of `LossAbsolute`
*
* \param custom_offset0 `double` Offset which is used instead of the pre
*   defined initialization
*
*/
LossBinomial::LossBinomial (const double& custom_offset0)
{
  task_id = "binary_classif"; // set parent
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
* \param true_value `arma::mat` True value of the response
* \param prediction `arma::mat` Prediction of the true value
*
* \returns `arma::mat` vector of elementwise application of the loss function
*/
arma::mat LossBinomial::definedLoss (const arma::mat& true_value, const arma::mat& prediction) const
{
  return arma::log(1 + arma::exp(-2 * true_value % prediction));
}

/**
* \brief Definition of the gradient of the loss function (see description of the class)
*
* \param true_value `arma::mat` True value of the response
* \param prediction `arma::mat` Prediction of the true value
*
* \returns `arma::mat` vector of elementwise application of the gradient
*/
arma::mat LossBinomial::definedGradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  return -2 * true_value / (1 + arma::exp(true_value % prediction));
}

/**
* \brief Definition of the constant risk initialization (see description of the class)
*
* \param true_value `arma::mat` True value of the response
*
* \returns `double` constant which minimizes the empirical risk for the given true value
*/
arma::mat LossBinomial::constantInitializer (const arma::mat& true_value) const
{
  arma::vec unique_values = arma::unique(true_value);
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (unique_values.size() != 2) {
      Rcpp::stop("Binomial loss does not support multiclass classification.");
    }
    if (! arma::all((unique_values == -1) || (unique_values == 1))) {
      Rcpp::stop("Labels must be coded as -1 and 1.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }

  if (use_custom_offset) { return custom_offset; }

  double p = arma::accu(true_value + 1) / (2 * true_value.size());
  arma::mat out(1, 1);
  out.fill(0.5 * std::log(p / (1 - p)));
  return out;
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
  task_id = "custom";
  // Rcpp::Rcout << "Be careful! You are using a custom loss out of R!"
  //           << "This will slow down everything!"
  //           << std::endl;
}

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the loss function
 */
arma::mat LossCustom::definedLoss (const arma::mat& true_value, const arma::mat& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate loss for a custom loss!" << std::endl;
  Rcpp::NumericVector out = lossFun(true_value, prediction);
  return out;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the gradient
 */
arma::mat LossCustom::definedGradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  // for debugging:
  // Rcpp::Rcout << "Calculate gradient for a custom loss!" << std::endl;
  Rcpp::NumericVector out = gradientFun(true_value, prediction);
  return out;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossCustom::constantInitializer (const arma::mat& true_value) const
{
  // for debugging:
  // Rcpp::Rcout << "Initialize custom loss!" << std::endl;

  Rcpp::NumericVector out = initFun(true_value);
  arma::mat out_single(1, 1);
  out_single.fill(out[0]);
  return out_single;
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
  task_id = "custom"; // set parent
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
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of elementwise application of the loss function
 */
arma::mat LossCustomCpp::definedLoss (const arma::mat& true_value, const arma::mat& prediction) const
{
  return lossFun(true_value, prediction);
}

/**
* \brief Definition of the gradient of the loss function (see description of the class)
*
* \param true_value `arma::mat` True value of the response
* \param prediction `arma::mat` Prediction of the true value
*
* \returns `arma::mat` vector of elementwise application of the gradient
*/
arma::mat LossCustomCpp::definedGradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  return gradFun(true_value, prediction);
}

/**
* \brief Definition of the constant risk initialization (see description of the class)
*
* \param true_value `arma::mat` True value of the response
*
* \returns `double` constant which minimizes the empirical risk for the given true value
*/
arma::mat LossCustomCpp::constantInitializer (const arma::mat& true_value) const
{
  arma::mat out(1, 1);
  out.fill(constInitFun(true_value));
  return out;
}

} // namespace loss
