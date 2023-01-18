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

#ifndef BASELEARNER_H_
#define BASELEARNER_H_

#include <RcppArmadillo.h>
#include <memory>
#include <string>

#include "data.h"
#include "response.h"
#include "splines.h"
#include "binning.h"
#include "init.h"
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

namespace blearner {

typedef std::shared_ptr<data::Data> sdata;
typedef std::shared_ptr<data::BinnedData> sbindata;
typedef std::map<std::string, sdata> mdata;

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

class Baselearner
{
protected:
  arma::mat          _parameter;
  const std::string  _blearner_type;

public:
  Baselearner (const std::string);
  Baselearner (const json&);

  // Virtual methods
  virtual void         train             (const arma::mat&)       = 0;
  virtual arma::mat    predict           ()                 const = 0;
  virtual arma::mat    predict           (const sdata&)     const = 0;
  virtual std::string  getDataIdentifier ()                 const = 0;
  virtual json         toJson            ()                 const = 0;

  // Getter/Setter
  arma::mat    getParameter        () const;
  std::string  getBaselearnerType  () const;

  json baseToJson (const std::string) const;

  // Destructor
  virtual ~Baselearner ();
};

std::shared_ptr<Baselearner> jsonToBaselearner (const json&, const mdata&);

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

class BaselearnerPolynomial : public Baselearner
{
private:
  const sbindata                                    _sh_ptr_bindata;
  const std::shared_ptr<init::PolynomialAttributes> _attributes;

public:
  BaselearnerPolynomial (const std::string, const sbindata&);
  BaselearnerPolynomial (const std::string, const sbindata&, const std::shared_ptr<init::PolynomialAttributes>&);
  BaselearnerPolynomial (const json&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerPolynomial ();
};


// BaselearnerPSpline:
// -----------------------

/**
 * \class BaselearnerPSpline
 *
 * \brief P-Spline base-learner
 *
 * This class implements the P-Spline base-learner. It is implemented based on de Boors
 * algorithm (from the Nurbs Book) to create the spline basis.
 */
class BaselearnerPSpline : public Baselearner
{
private:
  const sbindata _sh_ptr_bindata;

public:
  BaselearnerPSpline (const std::string, const sbindata&);//, const std::shared_ptr<init::PSplineAttributes>&);
  BaselearnerPSpline (const json&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerPSpline ();
};

// BaselearnerTensor:
// ------------------------------------

class BaselearnerTensor : public Baselearner
{
private:
  const sdata _sh_ptr_data;

public:
  BaselearnerTensor (const std::string, const sdata&);
  BaselearnerTensor (const json&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerTensor ();
};

// BaselearnerCentered:
// ------------------------------------

class BaselearnerCentered : public Baselearner
{
private:
  const sdata _sh_ptr_data;

public:
  BaselearnerCentered (const std::string, const sdata&);
  BaselearnerCentered (const json&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerCentered ();
};

// BaselearnerCategoricalRidge:
// ------------------------------------

class BaselearnerCategoricalRidge : public Baselearner
{
private:
  const sdata _sh_ptr_data;

public:
  BaselearnerCategoricalRidge (const std::string, const sdata&);
  BaselearnerCategoricalRidge (const json&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerCategoricalRidge ();

};


// BaselearnerCategoricalBinary:
// ------------------------------------

class BaselearnerCategoricalBinary : public Baselearner
{
private:
  const sdata _sh_ptr_data;

public:
  BaselearnerCategoricalBinary (const std::string, const sdata&);
  BaselearnerCategoricalBinary (const json&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerCategoricalBinary ();
};


// BaselearnerCustom:
// -----------------------

class BaselearnerCustom : public Baselearner
{
private:
  const sdata _sh_ptr_data;
  SEXP _model;

  const Rcpp::Function _instantiateDataFun;
  const Rcpp::Function _trainFun;
  const Rcpp::Function _predictFun;
  const Rcpp::Function _extractParameter;

public:
  BaselearnerCustom (const std::string, const sdata&, Rcpp::Function,
    Rcpp::Function, Rcpp::Function, Rcpp::Function);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerCustom ();
};


// BaselearnerCustomCpp:
// -----------------------

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
typedef arma::mat (*trainFunPtr) (const arma::mat& y, const arma::mat& X);
typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);

class BaselearnerCustomCpp : public Baselearner
{
private:
  const sdata                                      _sh_ptr_data;
  const std::shared_ptr<init::CustomCppAttributes> _attributes;

public:
  BaselearnerCustomCpp (const std::string, const sdata&, const std::shared_ptr<init::CustomCppAttributes>&);

  void         train             (const arma::mat&);
  arma::mat    predict           ()                  const;
  arma::mat    predict           (const sdata&)      const;
  std::string  getDataIdentifier ()                  const;
  json         toJson            ()                  const;

  ~BaselearnerCustomCpp ();
};

} // namespace blearner

#endif // BASELEARNER_H_
