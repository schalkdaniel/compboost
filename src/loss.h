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

/** 
 *  @file    loss.h
 *  @author  Daniel Schalk (github: schalkdaniel)
 *  
 *  @brief Loss class definition
 *
 *  @section DESCRIPTION
 *  
 *  This file contains the different loss implementations. The structure is:
 *   
 *  ``` 
 *  class SpecificLoss: public Loss
 *  {
 *    arma::vec definedLoss      { IMPLEMENTATION };
 *    arma::vec definedGradient  { IMPLEMENTATION };
 *    double constantInitializer { IMPLEMENTATION };
 *  }
 *  ```
 *
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <RcppArmadillo.h>

#include <iostream>

namespace loss
{

// Parent class:
// -----------------------

/**
 * \class Loss
 * 
 * \brief Abstract loss class
 * 
 * This class defines the minimal requirements of every loss class.
 * 
 */
class Loss
{
public:
  
  /// Specific loss function
  virtual arma::vec definedLoss (const arma::vec&, const arma::vec&) const = 0;
  
  /// Gradient of loss functions for pseudo residuals
  virtual arma::vec definedGradient (const arma::vec&, const arma::vec&) const = 0;
  
  /// Constant initialization of the empirical risk
  virtual double constantInitializer (const arma::vec&) const = 0;
  
  virtual ~Loss ();
  
protected:
  
  /// Custom offset:
  double custom_offset = NULL;
  
  /// Weights:
  arma::vec weights;
};

// -------------------------------------------------------------------------- //
// Loss implementations as child classes:
// -------------------------------------------------------------------------- //

// QuadraticLoss loss:
// -----------------------

/**
 * \class QuadraticLoss
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
class QuadraticLoss : public Loss
{
public:
  
  /// Default Constructor
  QuadraticLoss ();
  
  /// Constructor to initialize custom offset
  QuadraticLoss (const double&);
  
  /// Specific loss function
  arma::vec definedLoss (const arma::vec&, const arma::vec&) const;
  
  /// Gradient of loss functions for pseudo residuals
  arma::vec definedGradient (const arma::vec&, const arma::vec&) const;
  
  /// Constant initialization of the empirical risk
  double constantInitializer (const arma::vec&) const;
};

// AbsoluteLoss loss:
// -----------------------

/**
 * \class AbsoluteLoss
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
class AbsoluteLoss : public Loss
{
public:
  
  /// Default Constructor
  AbsoluteLoss ();
  
  /// Constructor to initialize custom offset
  AbsoluteLoss (const double&);
  
  /// Specific loss function
  arma::vec definedLoss (const arma::vec&, const arma::vec&) const;
  
  /// Gradient of loss functions for pseudo residuals
  arma::vec definedGradient (const arma::vec&, const arma::vec&) const;
  
  /// Constant initialization of the empirical risk
  double constantInitializer (const arma::vec&) const;
};

// Bernoulli loss:
// -----------------------

/**
 * \class BernoulliLoss
 * 
 * \brief 0-1 Loss for binary classification derifed of the bernoulli distribution
 * 
 * This loss can be used for binary classification. The coding we have chosen
 * here acts on 
 * \f[
 *   y \in \{-1, 1\}.
 * \f]
 * 
 * **Loss Function:**
 * \f[
 *   L(y, f(x)) = \log\left\{1 + \exp\left(-yf(x)\right)\right\}
 * \f]
 * **Gradient:**
 * \f[
 *   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = - \frac{y}{1 + \exp\left(yf\right)}
 * \f]
 * **Initialization:**
 * \f[
 *   \hat{f}^{[0]}(x) = \frac{1}{2}\log\left\{\frac{p}{1 - p}\right\}
 * \f]
 * with
 * \f[
 *   p = \frac{1}{n}\sum\limits_{i=1}^n\mathbb{1}_{\{y_i > 0\}}
 * \f]
 * 
 */

class BernoulliLoss : public Loss
{
public:
  
  /// Default Constructor
  BernoulliLoss ();
  
  /// Constructor to initialize custom offset
  BernoulliLoss (const double&);
  
  /// Specific loss function
  arma::vec definedLoss (const arma::vec&, const arma::vec&) const;
  
  /// Gradient of loss functions for pseudo residuals
  arma::vec definedGradient (const arma::vec&, const arma::vec&) const;
  
  /// Constant initialization of the empirical risk
  double constantInitializer (const arma::vec&) const;
};

// Custom loss:
// -----------------------

/**
 * \class CustomLoss
 * 
 * \brief With this loss it is possible to define custom functions out of `R`
 * 
 * This one is a special one. It allows to use a custom loss predefined in R.
 * The convenience here comes from the 'Rcpp::Function' class and the use of
 * a special constructor which defines the three needed functions.
 *
 * **Note** that there is one conversion step. There is no predefined conversion
 * from `Rcpp::Function` (which acts as SEXP) to a `arma::vec`. But it is 
 * possible by using `Rcpp::NumericVector`. Therefore the custom functions 
 * returns a `Rcpp::NumericVector` which then is able to be converted to a
 * `arma::vec`.
 * 
 * **Also Note:** This class doesn't have a constructor to initialize a
 * custom offset. Because this is not necessary here since the user can
 * define a custom offset within the `initFun` function.
 * 
 */
class CustomLoss : public Loss
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
  CustomLoss (Rcpp::Function, Rcpp::Function, Rcpp::Function);

  /// Specific loss function
  arma::vec definedLoss (const arma::vec&, const arma::vec&) const;
  
  /// Gradient of loss functions for pseudo residuals
  arma::vec definedGradient (const arma::vec&, const arma::vec&) const;
  
  /// Constant initialization of the empirical risk
  double constantInitializer (const arma::vec&) const;
};

} // namespace loss

#endif // LOSS_H_
