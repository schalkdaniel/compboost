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
// Written by:
// -----------
//
//   Daniel Schalk
//   Department of Statistics
//   Ludwig-Maximilians-University Munich
//   Ludwigstrasse 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
//   Contact
//   e: contact@danielschalk.com
//   w: danielschalk.com
//
// =========================================================================== #

#include "baselearner.h"

namespace blearner {

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

// Copy (or initialize) the members in new copied class:
void Baselearner::copyMembers (const arma::mat& parameter0, 
  const std::string& blearner_identifier0, data::Data* data0)
{
  parameter = parameter0;
  blearner_identifier = blearner_identifier0;
  data_ptr = data0;
}

// Set the data pointer:
void Baselearner::setData (data::Data* data)
{
  data_ptr = data;
}

// // Get the data on which data pointer points:
// arma::mat Baselearner::getData () const
// {
//   return data_ptr->getData();
// }

// Get the data identifier:
std::string Baselearner::getDataIdentifier () const
{
  return data_ptr->getDataIdentifier();
}

// Get the parameter obtained by training:
arma::mat Baselearner::getParameter () const
{
  return parameter;
}

// Predict function. This one calls the virtual function with the data pointer:
// arma::mat Baselearner::predict ()
// {
//   return predict(*data_ptr);
// }

// Function to set the identifier (should be unique over all baselearner):
void Baselearner::setIdentifier (const std::string& id0)
{
  blearner_identifier = id0;
}

// Get the identifier:
std::string Baselearner::getIdentifier () const
{
  return blearner_identifier;
}

// Function to set the baselearner type:
void Baselearner::setBaselearnerType (const std::string& blearner_type0)
{
  blearner_type = blearner_type0;
}

// Get the baselearner type:
std::string Baselearner::getBaselearnerType () const
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

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomial::BaselearnerPolynomial (data::Data* data, const std::string& identifier, 
  const unsigned int& degree, const bool& intercept) 
  : degree ( degree ),
    intercept ( intercept )
{
  // Called from parent class 'Baselearner':
  Baselearner::setData(data);
  Baselearner::setIdentifier(identifier);
}

// Copy member:
Baselearner* BaselearnerPolynomial::clone ()
{
  Baselearner* newbl = new BaselearnerPolynomial(*this);
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

// // Transform data:
// arma::mat BaselearnerPolynomial::instantiateData ()
// {
//   
//   return arma::pow(*data_ptr, degree);
// }
// 
// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerPolynomial::instantiateData (const arma::mat& newdata)
{
  arma::mat temp = arma::pow(newdata, degree);
  if (intercept) {
    arma::mat temp_intercept(temp.n_rows, 1, arma::fill::ones);
    temp = join_rows(temp_intercept, temp);
  }
  return temp;
}

// Train the learner:
void BaselearnerPolynomial::train (const arma::vec& response)
{
  if (data_ptr->getData().n_cols == 1) {
    double y_mean = 0;
    if (intercept) {
      y_mean = arma::as_scalar(arma::mean(response));
    } 

    double slope = arma::as_scalar(arma::sum((data_ptr->getData() - data_ptr->XtX_inv(0,0)) % (response - y_mean)) / arma::as_scalar(data_ptr->XtX_inv(0,1)));
    double intercept = y_mean - slope * data_ptr->XtX_inv(0,0);
      
    if (intercept) {
      arma::mat out(2,1);
        
      out(0,0) = intercept;
      out(1,0) = slope;

      parameter = out;
    } else {
      parameter = slope;
    }
  } else {
    // parameter = arma::solve(data_ptr->getData(), response);
    parameter = data_ptr->XtX_inv * data_ptr->getData().t() * response;
  }
}

// Predict the learner:
arma::mat BaselearnerPolynomial::predict ()
{
  if (data_ptr->getData().n_cols == 1) {
    if (intercept) {
      return parameter(0) + data_ptr->getData() * parameter(1);
    } else {
      return data_ptr->getData() * parameter;
    }
  } else {
    return data_ptr->getData() * parameter;
  }
}
arma::mat BaselearnerPolynomial::predict (data::Data* newdata)
{
  return instantiateData(newdata->getData()) * parameter;
}

// Destructor:
BaselearnerPolynomial::~BaselearnerPolynomial () {}

// PSplienBlearner:
// ----------------------

/**
 * \brief Constructor of `BaselearnerPSpline` class
 * 
 * This constructor sets the members such as n_knots etc. The more computational
 * complex data are stored within the data object which should be initialized
 * first (e.g. in the factory or otherwise). 
 * 
 * One note about the used knots. The number of inner knots is specified
 * by `n_knots`. These inner knots are then wrapped by the minimal and maximal
 * value of the given data. For instance we have a feature 
 * \f[
 *   x = (1, 2, \dots, 2.5, 6)
 * \f]
 * and we want to have 3 knots, then the inner knots with boundaries are:
 * \f[
 *   U = (1.00, 2.25, 3.50, 4.75, 6.00)
 * \f]
 * To get a full base these knots are wrapped by `degree` (\f$p\f$) numbers 
 * on either side. If we choose `degree = 2` then we have 
 * \f$n_\mathrm{knots} + 2(p + 1) = 3 + 2(2 + 1) 9\f$ final knots:
 * \f[
 *   U = (-1.50, -0.25, 1.00, 2.25, 3.50, 4.75, 6.00, 7.25, 8.50)
 * \f]
 * Finally we get a \f$9 - (p + 1)\f$ splines for which we can calculate the
 * base.  
 * 
 * \param data `data::Data*` Target data used for training etc.
 * \param identifier `std::string` Identifier for one specific baselearner
 * \param degree `unsigned int` Polynomial degree of the splines
 * \param n_knots `unsigned int` Number of inner knots used 
 * \param penalty `double` Regularization parameter `penalty = 0` yields
 *   b splines while a bigger penalty forces the splines into a global
 *   polynomial form.
 * \param differences `unsigned int` Number of differences used for the 
 *   penalty matrix.
 */

BaselearnerPSpline::BaselearnerPSpline (data::Data* data, const std::string& identifier,
  const unsigned int& degree, const unsigned int& n_knots, const double& penalty, 
  const unsigned int& differences, const bool& use_sparse_matrices)
  : degree ( degree ),
    n_knots ( n_knots ),
    penalty ( penalty ),
    differences ( differences ),
    use_sparse_matrices ( use_sparse_matrices )
{ 
  // Called from parent class 'Baselearner':
  Baselearner::setData(data);
  Baselearner::setIdentifier(identifier);
}

/**
 * \brief Clean copy of baselearner
 * 
 * \returns `Baselearner*` An exact copy of the actual baselearner.
 */
Baselearner* BaselearnerPSpline::clone ()
{
  Baselearner* newbl = new BaselearnerPSpline (*this);
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

/**
 * \brief Instantiate data matrix (design matrix)
 * 
 * This function is ment to create the design matrix which is then stored
 * within the data object. This should be done just once and then reused all
 * the time.
 * 
 * Note that this function sets the `data_mat` object of the data object!
 * 
 * \param newdata `arma::mat` Input data which is transformed to the design matrix
 * 
 * \returns `arma::mat` of transformed data
 */
arma::mat BaselearnerPSpline::instantiateData (const arma::mat& newdata)
{
  // Data object has to be created prior! That means that data_ptr must have
  // initialized knots, and penalty matrix!
  return createSplineBasis (newdata, degree, data_ptr->knots);
}

/**
 * \brief Training of a baselearner
 * 
 * This function sets the `parameter` member of the parent class `Baselearner`.
 * 
 * \param response `arma::vec` Response variable of the training.
 */
void BaselearnerPSpline::train (const arma::vec& response)
{
  if (use_sparse_matrices) {
    parameter = data_ptr->XtX_inv * (data_ptr->sparse_data_mat * response);
  } else {
    parameter = data_ptr->XtX_inv * data_ptr->data_mat.t() * response;
  }
}

/**
 * \brief Predict on training data
 * 
 * \returns `arma::mat` of predicted values
 */
arma::mat BaselearnerPSpline::predict ()
{
  // arma::mat out;
  if (use_sparse_matrices) {
    // Trick to speed up things. Try to avoid transposing the sparse matrix. The
    // original one (data_ptr->sparse_data_mat * parameter) is about 4 or 5 times
    // slower than that one:
    return (parameter.t() * data_ptr->sparse_data_mat).t();
  } else {
    return data_ptr->data_mat * parameter;
  }
  // return out;
}

/**
 * \brief Predict on newdata
 * 
 * \param newdata `data::Data*` new source data object
 * 
 * \returns `arma::mat` of predicted values
 */
arma::mat BaselearnerPSpline::predict (data::Data* newdata)
{
  return instantiateData(newdata->getData()) * parameter;
}


/// Destructor
BaselearnerPSpline::~BaselearnerPSpline () {}


// BaselearnerCustom:
// -----------------------

BaselearnerCustom::BaselearnerCustom (data::Data* data, const std::string& identifier, 
  Rcpp::Function instantiateDataFun, Rcpp::Function trainFun, Rcpp::Function predictFun, 
  Rcpp::Function extractParameter) 
  : instantiateDataFun ( instantiateDataFun ), 
    trainFun ( trainFun ),
    predictFun ( predictFun ),
    extractParameter ( extractParameter )
{
  // Called from parent class 'Baselearner':
  Baselearner::setData (data);
  Baselearner::setIdentifier (identifier);
}

// Copy member:
Baselearner* BaselearnerCustom::clone ()
{
  Baselearner* newbl = new BaselearnerCustom (*this);
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCustom::instantiateData (const arma::mat& newdata)
{
  Rcpp::NumericMatrix out = instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}

// Train by using the R function 'trainFun'.

// NOTE: It is highly recommended to specify an explicit extractParameter
//       function! Otherwise, it is not possible to estimate the parameter
//       during the whole process:
void BaselearnerCustom::train (const arma::vec& response)
{
  model     = trainFun(response, data_ptr->getData());
  parameter = Rcpp::as<arma::mat>(extractParameter(model));
}

// Predict by using the R function 'predictFun':
arma::mat BaselearnerCustom::predict ()
{
  Rcpp::NumericMatrix out = predictFun(model, data_ptr->getData());
  return Rcpp::as<arma::mat>(out);
}
arma::mat BaselearnerCustom::predict (data::Data* newdata)
{
  Rcpp::NumericMatrix out = predictFun(model, instantiateData(newdata->getData()));
  return Rcpp::as<arma::mat>(out);
}

// Destructor:
BaselearnerCustom::~BaselearnerCustom () {}


// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCpp::BaselearnerCustomCpp (data::Data* data, const std::string& identifier, 
  SEXP instantiateDataFun0, SEXP trainFun0, SEXP predictFun0)
{
  // Called from parent class 'Baselearner':
  Baselearner::setData (data);
  Baselearner::setIdentifier (identifier);
  
  // Set functions:
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun0);
  instantiateDataFun = *myTempInstantiation;
  
  Rcpp::XPtr<trainFunPtr> myTempTrain (trainFun0);
  trainFun = *myTempTrain;
  
  Rcpp::XPtr<predictFunPtr> myTempPredict (predictFun0);
  predictFun = *myTempPredict;
}

// Copy member:
Baselearner* BaselearnerCustomCpp::clone ()
{
  Baselearner* newbl = new BaselearnerCustomCpp (*this);
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->data_ptr);
  
  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCustomCpp::instantiateData (const arma::mat& newdata)
{
  return instantiateDataFun(newdata);
}



// Train by using the external pointer to the function 'trainFun'.

// NOTE: It is highly recommended to specify an explicit extractParameter
//       function! Otherwise, it is not possible to estimate the parameter
//       during the whole process:
void BaselearnerCustomCpp::train (const arma::vec& response)
{
  parameter = trainFun(response, data_ptr->getData());
}

// Predict by using the external pointer to the function 'predictFun':

arma::mat BaselearnerCustomCpp::predict ()
{
  return predictFun (data_ptr->getData(), parameter);
}
arma::mat BaselearnerCustomCpp::predict (data::Data* newdata)
{
  arma::mat temp_mat = instantiateData(newdata->getData());
  return predictFun (temp_mat, parameter);
}

// Destructor:
BaselearnerCustomCpp::~BaselearnerCustomCpp () {}


} // namespace blearner
