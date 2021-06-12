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
// #include "lossoptim.h"


namespace loss
{

// Loss specific helper:
// ------------------------------------

arma::mat doubleToMat (const double);

// Parent class:
// ------------------------------------

/**
 * \class Loss
 *
 * \brief Abstract loss class
 *
 * This class defines the minimal functionality.
 *
 */
class Loss : public std::enable_shared_from_this<Loss>
{
protected:
  const std::string _task_id;
  const arma::mat   _custom_offset;
  const bool        _use_custom_offset = false;

  Loss (const std::string);
  Loss (const std::string, const arma::mat&);

public:
  // Virtual functions
  virtual arma::mat loss     (const arma::mat&, const arma::mat&) const = 0;
  virtual arma::mat gradient (const arma::mat&, const arma::mat&) const = 0;

  virtual arma::mat constantInitializer         (const arma::mat&)                   const = 0;
  virtual arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const = 0;

  // Setter/Getter
  std::string getTaskId () const;

  // Other member functions
  arma::mat weightedLoss     (const arma::mat&, const arma::mat&, const arma::mat&) const;
  arma::mat weightedGradient (const arma::mat&, const arma::mat&, const arma::mat&) const;

  double calculateEmpiricalRisk         (const arma::mat&, const arma::mat&)                   const;
  double calculateWeightedEmpiricalRisk (const arma::mat&, const arma::mat&, const arma::mat&) const;

  arma::mat calculatePseudoResiduals         (const arma::mat&, const arma::mat&)                   const;
  arma::mat calculateWeightedPseudoResiduals (const arma::mat&, const arma::mat&, const arma::mat&) const;

  // Destructor
  virtual ~Loss ();
};

} // namespace loss


// Forward declaring lossoptim:
namespace lossoptim
{
  double findOptimalLossConstant (const arma::mat&, const std::shared_ptr<const loss::Loss>&,
    const double, const double);
  double findOptimalWeightedLossConstant (const arma::mat&, const arma::mat&,
    const std::shared_ptr<const loss::Loss>&, const double, const double);
}


namespace loss
{

// -------------------------------------------------------------------------- //
// Loss implementations
// -------------------------------------------------------------------------- //

// LossQuadratic:
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
  LossQuadratic ();
  LossQuadratic (const double);
  LossQuadratic (const arma::mat&);

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

// LossAbsolute:
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
  LossAbsolute ();
  LossAbsolute (const double);

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};


// LossQuantile
// -----------------------

class LossQuantile : public Loss
{
private:
  const double _quantile;

public:
  LossQuantile (const double);
  LossQuantile (const double, const double);

  double getQuantile () const;

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};


// LossHuber
// -----------------------


class LossHuber : public Loss
{
private:
  const double _delta;

public:
  LossHuber (const double);
  LossHuber (const double, const double);

  double getDelta () const;

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
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
  LossBinomial ();
  LossBinomial (const double);
  LossBinomial (const arma::mat&);

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

// Custom loss:
// -----------------------

/**
 * \class LossCustom
 *
 * \brief Loss to define custom functions out of `R`
 *
 * This special loss allows to use a custom loss predefined in R.
 * The use of the 'Rcpp::Function' class and a special constructor
 * allows it to define the functions required for the loss.
 *
 * **Note:** This class does not have a constructor to initialize a
 * custom offset. This is not necessary since the user can
 * define the custom offset within the `initFun` function.
 *
 */
class LossCustom : public Loss
{
private:
  const Rcpp::Function _lossFun;
  const Rcpp::Function _gradientFun;
  const Rcpp::Function _initFun;

public:
  LossCustom (const Rcpp::Function&, const Rcpp::Function&, const Rcpp::Function&);

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};


// Custom loss:
// -----------------------

/**
* \class LossCustomCpp
*
* \brief With this loss it is possible to set custom `C++` functions
*
* This loss allows to use a custom loss programmed in `C++`. The key is to
* use external pointers to set the corresponding functions. The advantage of
* this procedure is to provide a  method to define custom `C++` losses without
* recompiling `compboost`.
*
* **Note:** This class does not have a constructor to initialize a
* custom offset. This is not necessary  since the user can
* define a custom offset within the `initFun` function.
*
*/
typedef arma::mat (*lossFunPtr)      (const arma::mat& true_value, const arma::mat& prediction);
typedef arma::mat (*gradFunPtr)      (const arma::mat& true_value, const arma::mat& prediction);
typedef double    (*constInitFunPtr) (const arma::mat& true_value);

class LossCustomCpp : public Loss
{
private:
  lossFunPtr      _lossFun;
  gradFunPtr      _gradFun;
  constInitFunPtr _constInitFun;

public:
  LossCustomCpp (const SEXP&, const SEXP&, const SEXP&);

  arma::mat loss     (const arma::mat&, const arma::mat&) const;
  arma::mat gradient (const arma::mat&, const arma::mat&) const;

  arma::mat constantInitializer         (const arma::mat&)                   const;
  arma::mat weightedConstantInitializer (const arma::mat&, const arma::mat&) const;
};

} // namespace loss

#endif // LOSS_H_
