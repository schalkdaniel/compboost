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
//   "BaselearnerFactory" class. This file implements the factorys as well as  
//   the classes which are made in those factorys. The reason behind that pattern
//   is that every baselearner have its own data. This data is stored within
//   the factory. Every baselearner which is created in the factory points to
//   that data (if possible).
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
// =========================================================================== #

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


class BaselearnerFactory
{

  public:
    
    void InitializeFactory (std::string, arma::mat, std::string);
    
    virtual blearner::Baselearner *CreateBaselearner (std::string &) = 0;
    
    bool IsDataInstantiated ();
    
    std::string GetBaselearnerType ();
    
  protected:
    
    std::string blearner_type;
    arma::mat data;
    std::string data_identifier;
    bool is_data_instantiated = false;
  
};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// Polynomial:
// -----------------------

class PolynomialFactory : public BaselearnerFactory
{
  private:
    
    unsigned int degree;
    
  public:
    
    PolynomialFactory (std::string, arma::mat, std::string, unsigned int);
    
    blearner::Baselearner *CreateBaselearner (std::string &);
};

// Custom:
// -----------------------

class CustomFactory : public BaselearnerFactory
{
  private:
    
    Rcpp::Function instantiateDataFun;
    Rcpp::Function trainFun;
    Rcpp::Function predictFun;
    Rcpp::Function extractParameter;
    
  public:
    
    CustomFactory (std::string, arma::mat, std::string, Rcpp::Function, 
      Rcpp::Function, Rcpp::Function, Rcpp::Function);
    
    blearner::Baselearner *CreateBaselearner (std::string &);
    
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_