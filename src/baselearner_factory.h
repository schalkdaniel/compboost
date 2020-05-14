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

/**
 *  @file    baselearner_factory.h
 *  @author  Daniel Schalk (github: schalkdaniel)
 *
 *  @brief Definition of baselearner factory classes
 *
 *  @section DESCRIPTION
 *
 *  This file defines the baselearner factory classes. Every baselearner should
 *  have a corresponding factory class. This factory class just exists to
 *  crate baselearner objects.
 *
 *  the factories are also there to instantiate the data at the moment the
 *  factory is instantiated. This is done by taking a data source and transform
 *  it baselearner dependent into a data target object. This data object is then
 *  used the whole time.
 *
 */

#ifndef BASELEARNERFACTORY_H_
#define BASELEARNERFACTORY_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <string>
#include <memory>

#include "baselearner.h"
#include "data.h"
#include "splines.h"
#include "binning.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

class BaselearnerFactory
{
protected:
  const std::string                  _blearner_type;
  const std::shared_ptr<data::Data>  _sh_ptr_data_source;

public:
  BaselearnerFactory (const std::string);
  BaselearnerFactory (const std::string, const std::shared_ptr<data::Data>);

  // Virtual methods
  virtual arma::mat  instantiateData          (const arma::mat&) const = 0;
  virtual arma::mat  getData                  ()                 const = 0;
  virtual arma::mat  calculateLinearPredictor (const arma::mat&) const = 0;
  virtual arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const = 0;

  virtual std::string getDataIdentifier   () const;
  virtual std::shared_ptr<blearner::Baselearner>  createBaselearner () = 0;

  // Getter/Setter
  std::string getBaselearnerType  () const;

  // Destructor:
  virtual ~BaselearnerFactory ();
};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomialFactory:
// -----------------------------

class BaselearnerPolynomialFactory : public BaselearnerFactory
{
private:
  const unsigned int           _degree;
  const bool                   _intercept;
  std::shared_ptr<data::Data>  _sh_ptr_data_target;

public:
  BaselearnerPolynomialFactory (const std::string, std::shared_ptr<data::Data>,
    const unsigned int, const bool);

  arma::mat  instantiateData          (const arma::mat&) const;
  arma::mat  getData                  ()                 const;
  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
};


// BaselearnerPSplineFactory:
// -----------------------------

class BaselearnerPSplineFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::PSplineData> _sh_ptr_psdata;

public:
  BaselearnerPSplineFactory (const std::string, std::shared_ptr<data::Data>, const unsigned int,
    const unsigned int, const double, const unsigned int, const bool, const unsigned int, const std::string);

  arma::mat  instantiateData          (const arma::mat&) const;
  arma::mat  getData                  ()                 const;
  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
};


// BaselearnerCategoricalRidgeFactory:
// ------------------------------------------------

class BaselearnerCategoricalRidgeFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::CategoricalData> _sh_ptr_cdata;

public:
  BaselearnerCategoricalRidgeFactory (const std::string, std::shared_ptr<data::CategoricalData>&);
  BaselearnerCategoricalRidgeFactory (const std::string, std::shared_ptr<data::CategoricalData>&, const double);

  arma::mat  instantiateData          (const arma::mat&) const;
  arma::mat  getData                  ()                 const;
  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  std::string getDataIdentifier () const;

};


// BaselearnerCategoricalBinaryFactory:
// -----------------------------------------

class BaselearnerCategoricalBinaryFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::CategoricalBinaryData> _sh_ptr_bcdata;

public:
  BaselearnerCategoricalBinaryFactory (const std::string, std::shared_ptr<data::Data>);

  arma::mat  instantiateData          (const arma::mat&) const;
  arma::mat  getData                  ()                 const;
  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
};


// BaselearnerCustomFactory:
// -----------------------------

class BaselearnerCustomFactory : public BaselearnerFactory
{
private:
  const std::shared_ptr<data::Data> _sh_ptr_data_target;

  const Rcpp::Function _instantiateDataFun;
  const Rcpp::Function _trainFun;
  const Rcpp::Function _predictFun;
  const Rcpp::Function _extractParameter;

public:

  BaselearnerCustomFactory (const std::string, const std::shared_ptr<data::Data>,
    const Rcpp::Function, const Rcpp::Function, const Rcpp::Function, const Rcpp::Function);

  arma::mat  instantiateData          (const arma::mat&) const;
  arma::mat  getData                  ()                 const;
  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
};

// BaselearnerCustomCppFactory:
// -----------------------------

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);

class BaselearnerCustomCppFactory : public BaselearnerFactory
{
private:
  const std::shared_ptr<data::Data> _sh_ptr_data_target;

  const SEXP _instantiateDataFun;
  const SEXP _trainFun;
  const SEXP _predictFun;

public:
  BaselearnerCustomCppFactory (const std::string, const std::shared_ptr<data::Data>,
    const SEXP, const SEXP, const SEXP);

  arma::mat  instantiateData          (const arma::mat&) const;
  arma::mat  getData                  ()                 const;
  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const std::shared_ptr<data::Data>&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_
