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
// ========================================================================== //

/**
 *  @file    loss.h
 *  @author  Daniel Schalk (github: schalkdaniel)
 *
 *  @brief Loss class definition
 *
 *  @section DESCRIPTION
 *
 *  This file contains the different loss implementations.
 *
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <cmath>
#include <memory>

#include "helper.h"

namespace loss
{

// Parent class:
// -----------------------

/**
 * \class Loss
 *
 * \brief Abstract loss class
 *
 * This class defines the minimal requirements of every loss class. Note that
 * the custom offset uses two members. The initial idea of assigning `NAN` to
 * the `custom_offset` fails.
 *
 */
class Loss
{
public:
  /// Get the task id
  std::string getTaskId () const;
  std::string getLossType () const;
  std::string loss_type;

  /// Specific loss function
  virtual arma::mat definedLoss (const arma::mat&, const arma::mat&) const = 0;
  arma::mat weightedLoss (const arma::mat&, const arma::mat&, const arma::mat&) const;

  /// Gradient of loss functions for pseudo residuals
  virtual arma::mat definedGradient (const arma::mat&, const arma::mat&) const = 0;
  arma::mat weightedGradient (const arma::mat&, const arma::mat&, const arma::mat&) const;

  /// Constant initialization of the empirical risk
  virtual arma::mat constantInitializer (const arma::mat&) const = 0;
  virtual arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const = 0;

  double calculateEmpiricalRisk (const arma::mat&, const arma::mat&) const;
  double calculateWeightedEmpiricalRisk (const arma::mat&, const arma::mat&, const arma::mat&) const;

  arma::mat calculatePseudoResiduals (const arma::mat&, const arma::mat&) const;
  arma::mat calculateWeightedPseudoResiduals (const arma::mat&, const arma::mat&, const arma::mat&) const;

  virtual ~Loss ();

protected:
  /// The id of the task, e.g. regression or binary classification
  std::string task_id;

  /// Custom offset:
  arma::mat custom_offset;

  /// Tag if a custom offset is used
  bool use_custom_offset = false;
};

// -------------------------------------------------------------------------- //
// Loss implementations as child classes:
// -------------------------------------------------------------------------- //

// LossQuadratic loss:
// -----------------------

/**
 * \class LossQuadratic
 *
 * \brief Quadratic loss for regression tasks.
 *
 * This loss can be used for regression with \f$y \in \mathbb{R}\f$.
 *
 * **Loss Function:**
 * \f[
 *   L(y, f(x)) = \frac{1}{2}\left( y - f(x) \right)^2
 * \f]
 * **Gradient:**
 * \f[
 *   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = f(x) - y
 * \f]
 * **Initialization:**
 * \f[
 *   \hat{f}^{[0]}(x) = \underset{c\in\mathbb{R}}{\mathrm{arg~min}}\ \frac{1}{n}\sum\limits_{i=1}^n
 *   L\left(y^{(i)}, c\right) = \bar{y}
 * \f]
 *
 */
class LossQuadratic : public Loss
{
public:

  /// Default Constructor
  LossQuadratic ();

  /// Constructor to initialize custom offset
  LossQuadratic (const double&);

  /// Specific loss function
  arma::mat definedLoss (const arma::mat&, const arma::mat&) const;

  /// Gradient of loss functions for pseudo residuals
  arma::mat definedGradient (const arma::mat&, const arma::mat&) const;

  /// Constant initialization of the empirical risk
  arma::mat constantInitializer (const arma::mat&) const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

// LossAbsolute loss:
// -----------------------

/**
 * \class LossAbsolute
 *
 * \brief Absolute loss for regression tasks.
 *
 * **Loss Function:**
 * \f[
 *   L(y, f(x)) = \left| y - f(x) \right|
 * \f]
 * **Gradient:**
 * \f[
 *   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = \mathrm{sign}\left( f(x) - y \right)
 * \f]
 * **Initialization:**
 * \f[
 *   \hat{f}^{[0]}(x) = \underset{c\in\mathbb{R}}{\mathrm{arg~min}}\ \frac{1}{n}\sum\limits_{i=1}^n
 *   L\left(y^{(i)}, c\right) = \mathrm{median}(y)
 * \f]
 *
 */
class LossAbsolute : public Loss
{
public:

  /// Default Constructor
  LossAbsolute ();

  /// Constructor to initialize custom offset
  LossAbsolute (const double&);

  /// Specific loss function
  arma::mat definedLoss (const arma::mat&, const arma::mat&) const;

  /// Gradient of loss functions for pseudo residuals
  arma::mat definedGradient (const arma::mat&, const arma::mat&) const;

  /// Constant initialization of the empirical risk
  arma::mat constantInitializer (const arma::mat&) const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

// Binomial loss:
// -----------------------

/**
 * \class LossBinomial
 *
 * \brief 0-1 Loss for binary classification derifed of the binomial distribution
 *
 * This loss can be used for binary classification. The coding we have chosen
 * here acts on
 * \f[
 *   y \in \{-1, 1\}.
 * \f]
 *
 * **Loss Function:**
 * \f[
 *   L(y, f(x)) = \log\left\{1 + \exp\left(-2yf(x)\right)\right\}
 * \f]
 * **Gradient:**
 * \f[
 *   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = - \frac{2y}{1 + \exp\left(2yf\right)}
 * \f]
 * **Initialization:**
 * \f[
 *   \hat{f}^{[0]}(x) = \frac{1}{2}\log\left(\frac{p}{1 - p}\right)
 * \f]
 * with
 * \f[
 *   p = \frac{1}{n}\sum\limits_{i=1}^n\mathbb{1}_{\{y_i > 0\}}
 * \f]
 *
 */

class LossBinomial : public Loss
{
public:


  /// Default Constructor
  LossBinomial ();

  /// Constructor to initialize custom offset
  LossBinomial (const double&);

  /// Specific loss function
  arma::mat definedLoss (const arma::mat&, const arma::mat&) const;

  /// Gradient of loss functions for pseudo residuals
  arma::mat definedGradient (const arma::mat&, const arma::mat&) const;

  /// Constant initialization of the empirical risk
  arma::mat constantInitializer (const arma::mat&) const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

// Custom loss:
// -----------------------

/**
 * \class LossCustom
 *
 * \brief With this loss it is possible to define custom functions out of `R`
 *
 * This one is a special one. It allows to use a custom loss predefined in R.
 * The convenience here comes from the 'Rcpp::Function' class and the use of
 * a special constructor which defines the three needed functions.
 *
 * **Note** that there is one conversion step. There is no predefined conversion
 * from `Rcpp::Function` (which acts as SEXP) to a `arma::mat`. But it is
 * possible by using `Rcpp::NumericVector`. Therefore the custom functions
 * returns a `Rcpp::NumericVector` which then is able to be converted to a
 * `arma::mat`.
 *
 * **Also Note:** This class doesn't have a constructor to initialize a
 * custom offset. Because this is not necessary here since the user can
 * define a custom offset within the `initFun` function.
 *
 */
class LossCustom : public Loss
{
private:

  /// `R` loss function
  Rcpp::Function lossFun;

  /// `R` gradient of loss function
  Rcpp::Function gradientFun;

  /// `R` constant initializer of empirical risk
  Rcpp::Function initFun;

public:

  /// Default constructor
  LossCustom (Rcpp::Function, Rcpp::Function, Rcpp::Function);

  /// Specific loss function
  arma::mat definedLoss (const arma::mat&, const arma::mat&) const;

  /// Gradient of loss functions for pseudo residuals
  arma::mat definedGradient (const arma::mat&, const arma::mat&) const;

  /// Constant initialization of the empirical risk
  arma::mat constantInitializer (const arma::mat&) const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

// Custom loss:
// -----------------------

/**
* \class LossCustomCpp
*
* \brief With this loss it is possible to define custom functions in `C++`
*
* This one is a special one. It allows to use a custom loss programmed in `C++`.
* The key is to use external pointer to set the corresponding functions. The
* big advantage of this is to provide a (not too complicated) method to define
* custom `C++` losses without recompiling compboost.
*
* **Note:** This class doesn't have a constructor to initialize a
* custom offset. Because this is not necessary here since the user can
* define a custom offset within the `initFun` function.
*
*/

typedef arma::mat (*lossFunPtr) (const arma::mat& true_value, const arma::mat& prediction);
typedef arma::mat (*gradFunPtr) (const arma::mat& true_value, const arma::mat& prediction);
typedef double (*constInitFunPtr) (const arma::mat& true_value);

class LossCustomCpp : public Loss
{
private:

  /// Pointer to `C++` function to define the loss
  lossFunPtr lossFun;

  /// Pointer to `C++` function to define the gradient of the loss function
  gradFunPtr gradFun;

  /// Pointer to `C++` function to initialize the model
  constInitFunPtr constInitFun;

public:

  /// Default constructor to set pointer (`Rcpp`s `XPtr` class) out of
  /// external pointer wrapped by SEXP
  LossCustomCpp (SEXP, SEXP, SEXP);

  /// Specific loss function
  arma::mat definedLoss (const arma::mat&, const arma::mat&) const;

  /// Gradient of loss functions for pseudo residuals
  arma::mat definedGradient (const arma::mat&, const arma::mat&) const;

  /// Constant initialization of the empirical risk
  arma::mat constantInitializer (const arma::mat&) const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

} // namespace loss

#endif // LOSS_H_
