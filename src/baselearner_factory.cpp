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
//   Implementation of the "BaselearnerFactory".
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München

//   https://www.compstat.statistik.uni-muenchen.de
//
// =========================================================================== #

#include "baselearner_factory.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

arma::mat BaselearnerFactory::GetData () const
{
  return data->getData();
}

std::string BaselearnerFactory::GetDataIdentifier () const
{
  return data->getDataIdentifier();
}

std::string BaselearnerFactory::GetBaselearnerType() const
{
  return blearner_type;
}

BaselearnerFactory::~BaselearnerFactory () {};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// PolynomialBlearner:
// -----------------------

PolynomialBlearnerFactory::PolynomialBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data0, const unsigned int& degree)
  : degree ( degree )
{
  blearner_type = blearner_type0;
  data = data0;
  data->setData(InstantiateData(data->getData()));
  blearner_type = blearner_type + " with degree " + std::to_string(degree);
}

blearner::Baselearner* PolynomialBlearnerFactory::CreateBaselearner (const std::string& identifier)
{
  blearner::Baselearner* blearner_obj;
  
  // Create new polynomial baselearner. This one will be returned by the 
  // factory:
  blearner_obj = new blearner::PolynomialBlearner(data, identifier, degree);
  blearner_obj->SetBaselearnerType(blearner_type);
  
  // // Check if the data is already set. If not, run 'InstantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->InstantiateData();
  //   
  //   is_data_instantiated = true;
  //   
  //   // update baselearner type:
  //   blearner_type = blearner_type + " with degree " + std::to_string(degree);
  // }
  return blearner_obj;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat PolynomialBlearnerFactory::InstantiateData (const arma::mat& newdata)
{
  return arma::pow(newdata, degree);
}

// CustomBlearner:
// -----------------------

CustomBlearnerFactory::CustomBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data0, Rcpp::Function instantiateDataFun, Rcpp::Function trainFun, 
  Rcpp::Function predictFun, Rcpp::Function extractParameter)
  : instantiateDataFun ( instantiateDataFun ),
    trainFun ( trainFun ),
    predictFun ( predictFun ),
    extractParameter ( extractParameter )
{
  blearner_type = blearner_type0;
  data = data0;
  data->setData(InstantiateData(data->getData()));
}

blearner::Baselearner *CustomBlearnerFactory::CreateBaselearner (const std::string &identifier)
{
  blearner::Baselearner *blearner_obj;
  
  blearner_obj = new blearner::CustomBlearner(data, identifier, instantiateDataFun, 
    trainFun, predictFun, extractParameter);
  
  // // Check if the data is already set. If not, run 'InstantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->InstantiateData();
  //   
  //   is_data_instantiated = true;
  // }
  return blearner_obj;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat CustomBlearnerFactory::InstantiateData (const arma::mat& newdata)
{
  Rcpp::NumericMatrix out = instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}

// CustomCppBlearner:
// -----------------------

CustomCppBlearnerFactory::CustomCppBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data0, SEXP instantiateDataFun, SEXP trainFun, SEXP predictFun)
  : instantiateDataFun ( instantiateDataFun ),
    trainFun ( trainFun ),
    predictFun ( predictFun )
{
  blearner_type = blearner_type0;
  data = data0;
  data->setData(InstantiateData(data->getData()));
}

blearner::Baselearner* CustomCppBlearnerFactory::CreateBaselearner (const std::string& identifier)
{
  blearner::Baselearner* blearner_obj;
  
  blearner_obj = new blearner::CustomCppBlearner(data, identifier, instantiateDataFun, 
    trainFun, predictFun);
  
  // // Check if the data is already set. If not, run 'InstantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->InstantiateData();
  //   
  //   is_data_instantiated = true;
  // }
  return blearner_obj;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat CustomCppBlearnerFactory::InstantiateData (const arma::mat& newdata)
{
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun);
  instantiateDataFunPtr instantiateDataFun0 = *myTempInstantiation;
  
  return instantiateDataFun0(newdata);
}

} // namespace blearnerfactory
