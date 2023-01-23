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
#include "demmler_reinsch.h"
#include "helper.h"
#include "init.h"
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

namespace blearnerfactory {

typedef std::shared_ptr<data::Data> sdata;
typedef std::shared_ptr<data::BinnedData> sbindata;
typedef std::map<std::string, sdata> mdata;

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

class BaselearnerFactory
{
protected:
  const std::string  _blearner_type;
  const sdata        _sh_ptr_data_source;

public:
  BaselearnerFactory (const std::string);
  BaselearnerFactory (const std::string, const sdata&);
  BaselearnerFactory (const json&, const mdata&);

  // Virtual methods
  virtual bool       usesSparse           ()                const = 0;
  virtual sdata      instantiateData      (const mdata&)    const = 0;

  virtual sdata      getInstantiatedData ()                 const = 0;
  virtual arma::mat  getData             ()                 const = 0;
  virtual arma::vec  getDF               ()                 const = 0;
  virtual arma::vec  getPenalty          ()                 const = 0;
  virtual arma::mat  getPenaltyMat       ()                 const = 0;

  virtual arma::mat  calculateLinearPredictor (const arma::mat&) const = 0;
  virtual arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const = 0;

  virtual std::string getDataIdentifier  () const;
  virtual std::shared_ptr<blearner::Baselearner>  createBaselearner () = 0;

  virtual json toJson            ()           const = 0;
  virtual json extractDataToJson (const bool) const = 0;

  // Getter/Setter
  sdata       getDataSource       () const;
  std::string getBaselearnerType  () const;

  json dataSourceToJson ()                  const;
  json baseToJson       (const std::string) const;

  // Destructor:
  virtual ~BaselearnerFactory ();
};

std::shared_ptr<BaselearnerFactory> jsonToBaselearnerFactory (const json&, const mdata&, const mdata&);

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomialFactory:
// -----------------------------

class BaselearnerPolynomialFactory : public BaselearnerFactory
{
private:
  sbindata _sh_ptr_bindata;
  const std::shared_ptr<init::PolynomialAttributes> _attributes = std::make_shared<init::PolynomialAttributes>();

public:
  BaselearnerPolynomialFactory (const std::string, std::shared_ptr<data::Data>,
    const unsigned int, const bool, const unsigned int, const double = 0, const double = 0);
  BaselearnerPolynomialFactory (const json&, const mdata&, const mdata&);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  json toJson () const;
  json extractDataToJson (const bool) const;
};


// BaselearnerPSplineFactory:
// -----------------------------

class BaselearnerPSplineFactory : public BaselearnerFactory
{
private:
  sbindata _sh_ptr_bindata;
  std::shared_ptr<init::PSplineAttributes> _attributes = std::make_shared<init::PSplineAttributes>();

public:
  BaselearnerPSplineFactory (const std::string, const std::shared_ptr<data::Data>&, const unsigned int,
    const unsigned int, const double, const double, const unsigned int, const bool, const unsigned int,
    const std::string);
  BaselearnerPSplineFactory (const json&, const mdata&, const mdata&);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  json toJson () const;
  json extractDataToJson (const bool) const;
};


// BaselearnerTensorFactory:
// ------------------------------------------------


class BaselearnerTensorFactory : public BaselearnerFactory
{
private:
  // the data is stored in a psdata object:
  std::shared_ptr<data::Data>             _sh_ptr_data;
  std::shared_ptr<init::TensorAttributes> _attributes = std::make_shared<init::TensorAttributes>();

  std::shared_ptr<blearnerfactory::BaselearnerFactory> _blearner1;
  std::shared_ptr<blearnerfactory::BaselearnerFactory> _blearner2;
  const bool _isotrop;

public:
  BaselearnerTensorFactory (const std::string&, std::shared_ptr<blearnerfactory::BaselearnerFactory>,
    std::shared_ptr<blearnerfactory::BaselearnerFactory>, const bool = false);
  BaselearnerTensorFactory (const json&, const mdata&, const mdata&);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner> createBaselearner ();

  json toJson () const;
  json extractDataToJson (const bool) const;
};


// BaselearnerCenteredFactory:
// ------------------------------------------------


class BaselearnerCenteredFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<init::CenteredAttributes> _attributes = std::make_shared<init::CenteredAttributes>();
  std::shared_ptr<data::BinnedData> _sh_ptr_bindata;

  std::shared_ptr<blearnerfactory::BaselearnerFactory> _blearner1;
  std::shared_ptr<blearnerfactory::BaselearnerFactory> _blearner2;

public:
  BaselearnerCenteredFactory (const std::string&, std::shared_ptr<blearnerfactory::BaselearnerFactory>,
    std::shared_ptr<blearnerfactory::BaselearnerFactory>);
  BaselearnerCenteredFactory (const json&, const mdata&, const mdata&);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner> createBaselearner ();
  arma::mat getRotation () const;
  json toJson () const;
  json extractDataToJson (const bool) const;
};



// BaselearnerCategoricalRidgeFactory:
// ------------------------------------------------

class BaselearnerCategoricalRidgeFactory : public BaselearnerFactory
{
private:
  sdata _sh_ptr_data;
  std::shared_ptr<init::RidgeAttributes> _attributes = std::make_shared<init::RidgeAttributes>();

public:
  BaselearnerCategoricalRidgeFactory (const std::string, std::shared_ptr<data::CategoricalDataRaw>&, const double = 0, const double = 0);
  BaselearnerCategoricalRidgeFactory (const json&, const mdata&, const mdata&);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  std::string getDataIdentifier () const;

  std::map<std::string, unsigned int> getDictionary () const;

  json toJson () const;
  json extractDataToJson (const bool) const;
};


// BaselearnerCategoricalBinaryFactory:
// -----------------------------------------

class BaselearnerCategoricalBinaryFactory : public BaselearnerFactory
{
private:
  sdata _sh_ptr_data;
  std::shared_ptr<init::BinaryAttributes> _attributes = std::make_shared<init::BinaryAttributes>();

public:
  BaselearnerCategoricalBinaryFactory (const std::string, const std::string, const std::shared_ptr<data::CategoricalDataRaw>&);
  BaselearnerCategoricalBinaryFactory (const json&, const mdata&, const mdata&);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  std::string getDataIdentifier () const;
  json toJson () const;
  json extractDataToJson (const bool) const;
};


// BaselearnerCustomFactory:
// -----------------------------

class BaselearnerCustomFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::Data> _sh_ptr_data;
  //std::shared_ptr<init::CustomAttributes> _attributes;

  const Rcpp::Function _instantiateDataFun;
  const Rcpp::Function _trainFun;
  const Rcpp::Function _predictFun;
  const Rcpp::Function _extractParameter;

public:

  BaselearnerCustomFactory (const std::string, const std::shared_ptr<data::Data>,
    const Rcpp::Function, const Rcpp::Function, const Rcpp::Function, const Rcpp::Function);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec  getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  json toJson () const;
  json extractDataToJson (const bool) const;
};

// BaselearnerCustomCppFactory:
// -----------------------------

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
typedef arma::mat (*trainFunPtr) (const arma::mat& y, const arma::mat& X);
typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);

class BaselearnerCustomCppFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::Data> _sh_ptr_data;
  std::shared_ptr<init::CustomCppAttributes> _attributes = std::make_shared<init::CustomCppAttributes>();

  //const SEXP _instantiateDataFun;
  //const SEXP _trainFun;
  //const SEXP _predictFun;

public:
  BaselearnerCustomCppFactory (const std::string, const std::shared_ptr<data::Data>, SEXP, SEXP, SEXP);

  bool       usesSparse           ()                 const;
  sdata      instantiateData      (const mdata&)     const;

  sdata      getInstantiatedData  ()                 const;
  arma::mat  getData              ()                 const;
  arma::vec  getDF                ()                 const;
  arma::vec     getPenalty           ()                 const;
  arma::mat  getPenaltyMat        ()                 const;

  arma::mat  calculateLinearPredictor (const arma::mat&) const;
  arma::mat  calculateLinearPredictor (const arma::mat&, const mdata&) const;

  std::shared_ptr<blearner::Baselearner>  createBaselearner ();
  json toJson () const;
  json extractDataToJson (const bool) const;
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_
