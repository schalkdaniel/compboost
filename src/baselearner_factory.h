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
public:

  BaselearnerFactory (const std::string, const std::shared_ptr<data::Data>);

  // Create new baselearner with id:
  virtual std::shared_ptr<blearner::Baselearner> createBaselearner (const std::string&) = 0;

  // Getter for data, data identifier and the baselearner type:
  // arma::mat getData () const;
  std::string getDataIdentifier () const;
  std::string getBaselearnerType () const;

  virtual arma::mat instantiateData (const arma::mat&) const = 0;
  virtual arma::mat getData() const = 0;

  // Destructor:
  virtual ~BaselearnerFactory ();

protected:

  // Minimal functionality every baselearner should have:
  const std::string blearner_type;
  const std::shared_ptr<data::Data> data_source;

};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomialFactory:
// -----------------------------

class BaselearnerPolynomialFactory : public BaselearnerFactory
{
private:

  const unsigned int degree;
  bool intercept;
  std::shared_ptr<data::Data> data_target;

public:

  BaselearnerPolynomialFactory (const std::string&, std::shared_ptr<data::Data>, std::shared_ptr<data::Data>, const unsigned int&,
    const bool&);

  std::shared_ptr<blearner::Baselearner> createBaselearner (const std::string&);

  /// Get data used for modeling
  arma::mat getData() const;

  arma::mat instantiateData (const arma::mat&) const;
};


// BaselearnerCategoricalBinaryFactory:
// -----------------------------------------

/**
 * \class BaselearnerCategoricalBinaryFactory
 *
 * \brief Factory to create `BaselearnerCategoricalBinaryFactory` objects
 *
 */
class BaselearnerCategoricalBinaryFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::CategoricalBinaryData> data_target_cat;

public:

  /// Default constructor of class `BaselearnerCategoricalBinaryFactory`
  BaselearnerCategoricalBinaryFactory (const std::string&, std::shared_ptr<data::Data>);

  /// Create new `BaselearnerCategoricalBinaryFactory` object
  std::shared_ptr<blearner::Baselearner> createBaselearner (const std::string&);

  /// Get data used for modelling
  arma::mat getData() const;

  /// Instantiate the design matrix
  arma::mat instantiateData (const arma::mat&) const;
};



// BaselearnerPSplineFactory:
// -----------------------------

/**
 * \class BaselearnerPSplineFactory
 *
 * \brief Factory to create `PSplineBlearner` objects
 *
 */
class BaselearnerPSplineFactory : public BaselearnerFactory
{
private:
  // std::shared_ptr<data::Data> data_target;
  std::shared_ptr<data::PSplineData> sh_ptr_psdata;

  /// Degree of splines
  // const unsigned int degree;

  /// Number of inner knots
  // const unsigned int n_knots;

  /// Regularization parameter
  // const double penalty;

  /// Order of differences used for penalty matrix
  // const unsigned int differences;

  /// Flag if sparse matrices should be used:
  // const bool use_sparse_matrices;

  // Member used for binning:
  // const bool use_binning;

  // Order of binning:
  // const unsigned int bin_root;

public:

  /// Default constructor of class `PSplineBleanerFactory`
  BaselearnerPSplineFactory (const std::string, std::shared_ptr<data::Data>, const unsigned int,
    const unsigned int, const double, const unsigned int, const bool, const unsigned int, const std::string);

  /// Create new `BaselearnerPSpline` object
  std::shared_ptr<blearner::Baselearner> createBaselearner (const std::string&);

  /// Get data used for modelling
  arma::mat getData() const;

  /// Instantiate the design matrix
  arma::mat instantiateData (const arma::mat&) const;
};

// BaselearnerCustomFactory:
// -----------------------------

// This class stores the R functions:

class BaselearnerCustomFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::Data> data_target;

  Rcpp::Function instantiateDataFun;
  Rcpp::Function trainFun;
  Rcpp::Function predictFun;
  Rcpp::Function extractParameter;

public:

  BaselearnerCustomFactory (const std::string&, std::shared_ptr<data::Data>, std::shared_ptr<data::Data>,
    Rcpp::Function, Rcpp::Function, Rcpp::Function, Rcpp::Function);

  std::shared_ptr<blearner::Baselearner> createBaselearner (const std::string&);

  /// Get data used for modelling
  arma::mat getData() const;

  arma::mat instantiateData (const arma::mat&) const;

};

// BaselearnerCustomCppFactory:
// -----------------------------

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);

class BaselearnerCustomCppFactory : public BaselearnerFactory
{
private:
  std::shared_ptr<data::Data> data_target;

  // Cpp functions for a custom baselearner:
  SEXP instantiateDataFun;
  SEXP trainFun;
  SEXP predictFun;

public:

  BaselearnerCustomCppFactory (const std::string&, std::shared_ptr<data::Data>, std::shared_ptr<data::Data>,
    SEXP, SEXP, SEXP);

  std::shared_ptr<blearner::Baselearner> createBaselearner (const std::string&);

  /// Get data used for modelling
  arma::mat getData() const;

  arma::mat instantiateData (const arma::mat&) const;

};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_
