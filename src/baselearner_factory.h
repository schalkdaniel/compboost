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

#include "baselearner.h"
#include "data.h"
#include "splines.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

class BaselearnerFactory
{
public:
  
  // Create new baselearner with id:
  virtual blearner::Baselearner* createBaselearner (const std::string&) = 0;
  
  // Getter for data, data identifier and the baselearner type:
  arma::mat getData () const;
  std::string getDataIdentifier () const;
  std::string getBaselearnerType () const;
  
  virtual arma::mat instantiateData (const arma::mat&) = 0;
  
  void initializeDataObjects (data::Data*, data::Data*);
  
  // Destructor:
  virtual ~BaselearnerFactory ();
  
protected:
  
  // Minimal functionality every baselearner should have:
  std::string blearner_type;
  data::Data* data_source;
  data::Data* data_target;
  
};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// PolynomialBlearnerFactory:
// -----------------------------

class PolynomialBlearnerFactory : public BaselearnerFactory
{
private:
  
  const unsigned int degree;
  
public:
  
  PolynomialBlearnerFactory (const std::string&, data::Data*, data::Data*, const unsigned int&);
  
  blearner::Baselearner* createBaselearner (const std::string&);
  
  arma::mat instantiateData (const arma::mat&);
};

// PSplineBlearnerFactory:
// -----------------------------

/**
 * \class PSplineBlearnerFactory
 * 
 * \brief Factory to creaet `PSplineBlearner` objects
 * 
 */

class PSplineBlearnerFactory : public BaselearnerFactory
{
private:
  
  /// Degree of splines
  const unsigned int degree;
  
  /// Number of inner knots
  const unsigned int n_knots;
  
  /// Regularization parameter
  const double penalty;
  
  /// Order of differences used for penalty matrix
  const unsigned int differences;
  
public:

  /// Default constructor of class `PSplineBleanrerFactory`
  PSplineBlearnerFactory (const std::string&, data::Data*, data::Data*, 
    const unsigned int&, const unsigned int&, const double&, 
    const unsigned int&);
  
  /// Create new `PSplineBlearner` object
  blearner::Baselearner* createBaselearner (const std::string&);
  
  /// Instantiate the design matrix
  arma::mat instantiateData (const arma::mat&);
};

// CustomBlearnerFactory:
// -----------------------------

// This class stores the R functions:

class CustomBlearnerFactory : public BaselearnerFactory
{
private:
  
  Rcpp::Function instantiateDataFun;
  Rcpp::Function trainFun;
  Rcpp::Function predictFun;
  Rcpp::Function extractParameter;
  
public:
  
  CustomBlearnerFactory (const std::string&, data::Data*, data::Data*,
    Rcpp::Function, Rcpp::Function, Rcpp::Function, Rcpp::Function);
  
  blearner::Baselearner* createBaselearner (const std::string&);
  
  arma::mat instantiateData (const arma::mat&);
  
};

// CustomCppBlearnerFactory:
// -----------------------------

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);

class CustomCppBlearnerFactory : public BaselearnerFactory
{
private:
  
  // Cpp functions for a custom baselearner:
  SEXP instantiateDataFun;
  SEXP trainFun;
  SEXP predictFun;
  
public:
  
  CustomCppBlearnerFactory (const std::string&, data::Data*, data::Data*, 
    SEXP, SEXP, SEXP);
  
  blearner::Baselearner* createBaselearner (const std::string&);
  
  arma::mat instantiateData (const arma::mat&);
  
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_
