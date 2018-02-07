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
//   "BaselearnerFactory" class. This file implements the factorys. The reason 
//   behind that pattern is that every baselearner has its own data. This data 
//   is stored within the factory. Every baselearner which is created in the 
//   factory points to that data (if possible). This is more memory friendly 
//   than copying the whole time. 
//
//   The abstract BaselearnerFactory parent class has just the following
//   virtual function:
//
//     - blearner::Baselearner* CreateBaselearner (std::string&) = 0;
//
//   This function together with the constructor of the child classes creates
//   the new baselearner.
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

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

// The abstract class must have some basic functionality which is defined
// within itself. This basic functionality is setted by InitializeFactory
// within the constructor of the child classes. It is also possible to add
// child specific elements (as done below). 

// The most crucial part is, that the CreateBaseLearner function is for
// every child class the same. Otherwise it isn't that easy to dynamically
// instantiate the baselearners within the main algorithm.

class BaselearnerFactory
{

  public:
    
    // This function has to be called from every child class:
    void InitializeFactory (std::string, std::string);
    
    // Create new baselearner with id:
    virtual blearner::Baselearner* CreateBaselearner (std::string&) = 0;
    
    // // Check if data is already instantiated. This is important for the first
    // // time the factory creates a new object. Then the data is setted by the
    // // first object. The following objects then doesn't need to instantiate the
    // // data again:
    // bool IsDataInstantiated ();
    
    // Getter for data, data identifier and the baselearner type:
    arma::mat GetData ();
    std::string GetDataIdentifier ();
    
    std::string GetBaselearnerType ();
    
    virtual arma::mat InstantiateData (arma::mat&) = 0;
    
    // Destructor:
    virtual ~BaselearnerFactory ();
    
  protected:
    
    // Minimal functionality every baselearner should have:
    std::string blearner_type;
    arma::mat data;
    std::string data_identifier;
    
    // bool is_data_instantiated = false;
  
};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// PolynomialBlearner:
// -----------------------

// Should be explained by itself:

class PolynomialBlearnerFactory : public BaselearnerFactory
{
  private:
    
    unsigned int degree;
    
  public:
    
    PolynomialBlearnerFactory (std::string, arma::mat, std::string, unsigned int);
    
    blearner::Baselearner* CreateBaselearner (std::string&);
    
    arma::mat InstantiateData (arma::mat&);
};

// CustomBlearner:
// -----------------------

// This class stores the R functions which are mentioned and explained a bit
// in "baselearner.cpp". The idea is that the factory sets this functions // [[Rcpp::export]]
// again and again:

// Issue: https://github.com/schalkdaniel/compboost/issues/53

class CustomBlearnerFactory : public BaselearnerFactory
{
  private:
    
    Rcpp::Function instantiateDataFun;
    Rcpp::Function trainFun;
    Rcpp::Function predictFun;
    Rcpp::Function extractParameter;
    
  public:
    
    CustomBlearnerFactory (std::string, arma::mat, std::string, Rcpp::Function, 
      Rcpp::Function, Rcpp::Function, Rcpp::Function);
    
    blearner::Baselearner* CreateBaselearner (std::string&);
    
    arma::mat InstantiateData (arma::mat&);
    
};

// CustomCppBlearner:
// -----------------------

typedef arma::mat (*instantiateDataFunPtr) (arma::mat& X);
class CustomCppBlearnerFactory : public BaselearnerFactory
{
private:
  
  // Cpp functions for a custom baselearner:
  SEXP instantiateDataFun;
  SEXP trainFun;
  SEXP predictFun;
  
public:
  
  CustomCppBlearnerFactory (std::string, arma::mat, std::string, SEXP, SEXP, 
    SEXP);
  
  blearner::Baselearner* CreateBaselearner (std::string&);
  
  arma::mat InstantiateData (arma::mat&);
  
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_
