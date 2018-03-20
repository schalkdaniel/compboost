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
// This file contains:
// -------------------
//
//   "BaselearnerFactory" class. 
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

#ifndef BASELEARNERFACTORY_H_
#define BASELEARNERFACTORY_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <string>

#include "baselearner.h"
#include "data.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

class BaselearnerFactory
{
public:
  
  // Create new baselearner with id:
  virtual blearner::Baselearner* CreateBaselearner (const std::string&) = 0;
  
  // Getter for data, data identifier and the baselearner type:
  arma::mat GetData () const;
  std::string GetDataIdentifier () const;
  std::string GetBaselearnerType () const;
  
  virtual arma::mat InstantiateData (const arma::mat&) = 0;
  
  void InitializeDataObjects (data::Data*, data::Data*);
  
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

// PolynomialBlearner:
// -----------------------

class PolynomialBlearnerFactory : public BaselearnerFactory
{
private:
  
  const unsigned int degree;
  
public:
  
  PolynomialBlearnerFactory (const std::string&, data::Data*, data::Data*, const unsigned int&);
  
  blearner::Baselearner* CreateBaselearner (const std::string&);
  
  arma::mat InstantiateData (const arma::mat&);
};

// CustomBlearner:
// -----------------------

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
  
  blearner::Baselearner* CreateBaselearner (const std::string&);
  
  arma::mat InstantiateData (const arma::mat&);
  
};

// CustomCppBlearner:
// -----------------------

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
  
  blearner::Baselearner* CreateBaselearner (const std::string&);
  
  arma::mat InstantiateData (const arma::mat&);
  
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_
