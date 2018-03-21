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
//   Implementations for "Baselearner" class.
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

#include "baselearner.h"

namespace blearner {

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

// Copy (or initialize) the members in new copied class:
void Baselearner::CopyMembers (const arma::mat& parameter0, 
  const std::string& blearner_identifier0, data::Data* data0)
{
  parameter = parameter0;
  blearner_identifier = blearner_identifier0;
  data_ptr = data0;
}

// Set the data pointer:
void Baselearner::SetData (data::Data* data)
{
  data_ptr = data;
}

// Get the data on which data pointer points:
arma::mat Baselearner::GetData () const
{
  return data_ptr->getData();
}

// Get the data identifier:
std::string Baselearner::GetDataIdentifier () const
{
  return data_ptr->getDataIdentifier();
}

// Get the parameter obtained by training:
arma::mat Baselearner::GetParameter () const
{
  return parameter;
}

// Predict function. This one calls the virtual function with the data pointer:
// arma::mat Baselearner::predict ()
// {
//   return predict(*data_ptr);
// }

// Function to set the identifier (should be unique over all baselearner):
void Baselearner::SetIdentifier (const std::string& id0)
{
  blearner_identifier = id0;
}

// Get the identifier:
std::string Baselearner::GetIdentifier () const
{
  return blearner_identifier;
}

// Function to set the baselearner type:
void Baselearner::SetBaselearnerType (const std::string& blearner_type0)
{
  blearner_type = blearner_type0;
}

// Get the baselearner type:
std::string Baselearner::GetBaselearnerType () const
{
  return blearner_type;
}

// Destructor:
Baselearner::~Baselearner ()
{
  // Rcpp::Rcout << "Call Baselearner Destructor" << std::endl;
  
  // delete blearner_type;
  // delete data_ptr;
  // delete data_identifier_ptr;
}

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// PolynomialBlearner:
// -----------------------

PolynomialBlearner::PolynomialBlearner (data::Data* data, const std::string& identifier, 
  const unsigned int& degree) 
  : degree ( degree )
{
  // Called from parent class 'Baselearner':
  Baselearner::SetData(data);
  Baselearner::SetIdentifier(identifier);
  // Baselearner::SetDataIdentifier(data_identifier);
}

// Copy member:
Baselearner* PolynomialBlearner::Clone ()
{
  Baselearner* newbl = new PolynomialBlearner(*this);
  newbl->CopyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

// // Transform data:
// arma::mat PolynomialBlearner::InstantiateData ()
// {
//   
//   return arma::pow(*data_ptr, degree);
// }
// 
// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat PolynomialBlearner::InstantiateData (const arma::mat& newdata)
{
  
  return arma::pow(newdata, degree);
}

// Train the learner:
void PolynomialBlearner::train (const arma::vec& response)
{
  parameter = arma::solve(data_ptr->getData(), response);
}

// Predict the learner:
arma::mat PolynomialBlearner::predict ()
{
  return data_ptr->getData() * parameter;
}
arma::mat PolynomialBlearner::predict (data::Data* newdata)
{
  return InstantiateData(newdata->getData()) * parameter;
}

// Destructor:
PolynomialBlearner::~PolynomialBlearner () {}

// CustomBlearner Baselearner:
// -----------------------

CustomBlearner::CustomBlearner (data::Data* data, const std::string& identifier, 
  Rcpp::Function instantiateDataFun, Rcpp::Function trainFun, Rcpp::Function predictFun, 
  Rcpp::Function extractParameter) 
  : instantiateDataFun ( instantiateDataFun ), 
    trainFun ( trainFun ),
    predictFun ( predictFun ),
    extractParameter ( extractParameter )
{
  // Called from parent class 'Baselearner':
  Baselearner::SetData (data);
  Baselearner::SetIdentifier (identifier);
  // Baselearner::SetDataIdentifier (data_identifier);
}

// Copy member:
Baselearner* CustomBlearner::Clone ()
{
  Baselearner* newbl = new CustomBlearner (*this);
  newbl->CopyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat CustomBlearner::InstantiateData (const arma::mat& newdata)
{
  Rcpp::NumericMatrix out = instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}

// Train by using the R function 'trainFun'.

// NOTE: It is highly recommended to specify an explicit extractParameter
//       function! Otherwise, it is not possible to estimate the parameter
//       during the whole process:
void CustomBlearner::train (const arma::vec& response)
{
  model     = trainFun(response, data_ptr->getData());
  parameter = Rcpp::as<arma::mat>(extractParameter(model));
}

// Predict by using the R function 'predictFun':
arma::mat CustomBlearner::predict ()
{
  Rcpp::NumericMatrix out = predictFun(model, data_ptr->getData());
  return Rcpp::as<arma::mat>(out);
}
arma::mat CustomBlearner::predict (data::Data* newdata)
{
  Rcpp::NumericMatrix out = predictFun(model, InstantiateData(newdata->getData()));
  return Rcpp::as<arma::mat>(out);
}

// Destructor:
CustomBlearner::~CustomBlearner () {}


// CustomCppBlearner Baselearner:
// -----------------------

CustomCppBlearner::CustomCppBlearner (data::Data* data, const std::string& identifier, 
  SEXP instantiateDataFun0, SEXP trainFun0, SEXP predictFun0)
{
  // Called from parent class 'Baselearner':
  Baselearner::SetData (data);
  Baselearner::SetIdentifier (identifier);
  // Baselearner::SetDataIdentifier (data_identifier);
  
  // Set functions:
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun0);
  instantiateDataFun = *myTempInstantiation;
  
  Rcpp::XPtr<trainFunPtr> myTempTrain (trainFun0);
  trainFun = *myTempTrain;
  
  Rcpp::XPtr<predictFunPtr> myTempPredict (predictFun0);
  predictFun = *myTempPredict;
}

// Copy member:
Baselearner* CustomCppBlearner::Clone ()
{
  Baselearner* newbl = new CustomCppBlearner (*this);
  newbl->CopyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat CustomCppBlearner::InstantiateData (const arma::mat& newdata)
{
  return instantiateDataFun(newdata);
}



// Train by using the external pointer to the function 'trainFun'.

// NOTE: It is highly recommended to specify an explicit extractParameter
//       function! Otherwise, it is not possible to estimate the parameter
//       during the whole process:
void CustomCppBlearner::train (const arma::vec& response)
{
  parameter = trainFun(response, data_ptr->getData());
}

// Predict by using the external pointer to the function 'predictFun':

arma::mat CustomCppBlearner::predict ()
{
  return predictFun (data_ptr->getData(), parameter);
}
arma::mat CustomCppBlearner::predict (data::Data* newdata)
{
  arma::mat temp_mat = InstantiateData(newdata->getData());
  return predictFun (temp_mat, parameter);
}

// Destructor:
CustomCppBlearner::~CustomCppBlearner () {}


} // namespace blearner
