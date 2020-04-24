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

#include "baselearner.h"

namespace blearner {

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

// Copy (or initialize) the members in new copied class:
void Baselearner::copyMembers (const arma::mat& parameter0,
  const std::string& blearner_identifier0, std::shared_ptr<data::Data> sh_ptr_data0)
{
  parameter = parameter0;
  blearner_identifier = blearner_identifier0;
  sh_ptr_data = sh_ptr_data0;
}

void Baselearner::setData (std::shared_ptr<data::Data> data) { sh_ptr_data = data; }
arma::mat Baselearner::getParameter () const { return parameter; }
void Baselearner::setIdentifier (const std::string& id) { blearner_identifier = id; }
std::string Baselearner::getIdentifier () const { return blearner_identifier; }
void Baselearner::setBaselearnerType (const std::string& blearner_type0) { blearner_type = blearner_type0; }
std::string Baselearner::getBaselearnerType () const { return blearner_type; }

Baselearner::~Baselearner () { }

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomial::BaselearnerPolynomial (std::shared_ptr<data::Data> data, const std::string& identifier,
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
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->sh_ptr_data);

  return newbl;
}

// // Transform data:
// arma::mat BaselearnerPolynomial::instantiateData ()
// {
//
//   return arma::pow(*sh_ptr_data, degree);
// }
//
// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerPolynomial::instantiateData (const arma::mat& newdata) const
{
  arma::mat temp = arma::pow(newdata, degree);
  if (intercept) {
    arma::mat temp_intercept(temp.n_rows, 1, arma::fill::ones);
    temp = join_rows(temp_intercept, temp);
  }
  return temp;
}

// Train the learner:
void BaselearnerPolynomial::train (const arma::mat& response)
{
  if (sh_ptr_data->getData().n_cols == 1) {
    double y_mean = 0;
    if (intercept) {
      y_mean = arma::as_scalar(arma::accu(response) / response.size());
    }

    double slope = arma::as_scalar(arma::sum((sh_ptr_data->getData() - sh_ptr_data->XtX_inv(0,0)) % (response - y_mean)) / arma::as_scalar(sh_ptr_data->XtX_inv(0,1)));
    double intercept = y_mean - slope * sh_ptr_data->XtX_inv(0,0);

    if (intercept) {
      arma::mat out(2,1);

      out(0,0) = intercept;
      out(1,0) = slope;

      parameter = out;
    } else {
      parameter = slope;
    }
  } else {
    // parameter = arma::solve(sh_ptr_data->getData(), response);
    parameter = sh_ptr_data->XtX_inv * sh_ptr_data->getData().t() * response;
  }
}

// Predict the learner:
arma::mat BaselearnerPolynomial::predict () const
{
  if (sh_ptr_data->getData().n_cols == 1) {
    if (intercept) {
      return parameter(0) + sh_ptr_data->getData() * parameter(1);
    } else {
      return sh_ptr_data->getData() * parameter;
    }
  } else {
    return sh_ptr_data->getData() * parameter;
  }
}
arma::mat BaselearnerPolynomial::predict (std::shared_ptr<data::Data> newdata) const
{
  return instantiateData(newdata->getData()) * parameter;
}
std::string BaselearnerPolynomial::getDataIdentifier () const
{
  return sh_ptr_data->getDataIdentifier();
}

// Destructor:
BaselearnerPolynomial::~BaselearnerPolynomial () {}

// BaselearnerPSpline:
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
 * \param data `std::shared_ptr<data::Data>` Target data used for training etc.
 * \param identifier `std::string` Identifier for one specific baselearner
 * \param degree `unsigned int` Polynomial degree of the splines
 * \param n_knots `unsigned int` Number of inner knots used
 * \param penalty `double` Regularization parameter `penalty = 0` yields
 *   b splines while a bigger penalty forces the splines into a global
 *   polynomial form.
 * \param differences `unsigned int` Number of differences used for the
 *   penalty matrix.
 */

BaselearnerPSpline::BaselearnerPSpline (std::shared_ptr<data::PSplineData> sh_ptr_psdata, const std::string& identifier)
  : sh_ptr_psdata ( sh_ptr_psdata )
{
  // Called from parent class 'Baselearner':
  // Baselearner::setData(data);
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
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->sh_ptr_data);

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
arma::mat BaselearnerPSpline::instantiateData (const arma::mat& newdata) const
{

  // arma::vec knots = sh_ptr_data->knots;

  // check if the new data matrix contains value which are out of range:
  //double range_min = knots[degree];                   // minimal value from original data
  //double range_max = knots[n_knots + degree + 1];     // maximal value from original data

  //arma::mat temp = splines::filterKnotRange(newdata, range_min, range_max, sh_ptr_data->getDataIdentifier());
  arma::mat temp = sh_ptr_psdata->filterKnotRange(newdata);
  // Data object has to be created prior! That means that sh_ptr_data must have
  // initialized knots, and penalty matrix!
  return splines::createSplineBasis (temp, sh_ptr_psdata->degree, sh_ptr_psdata->getKnots());
}

/**
 * \brief Training of a baselearner
 *
 * This function sets the `parameter` member of the parent class `Baselearner`.
 *
 * \param response `arma::vec` Response variable of the training.
 */
void BaselearnerPSpline::train (const arma::mat& response) {

  arma::mat temp;

  if (sh_ptr_psdata->usesBinning()) {
    arma::vec temp_weight(1, arma::fill::ones);
    temp = binning::binnedSparseMatMultResponse(sh_ptr_psdata->sparse_data_mat, response, sh_ptr_psdata->bin_idx, temp_weight);
  } else {
    temp = sh_ptr_psdata->sparse_data_mat * response;
  }
  parameter = helper::cboostSolver(sh_ptr_psdata->getCachedMat(), temp);
}

/**
 * \brief Predict on training data
 *
 * \returns `arma::mat` of predicted values
 */
arma::mat BaselearnerPSpline::predict () const
{
  // arma::mat out;
  if (sh_ptr_psdata->usesSparseMatrix()) {
    if (sh_ptr_psdata->usesBinning()) {
      return binning::binnedSparsePrediction(sh_ptr_psdata->sparse_data_mat, parameter, sh_ptr_psdata->bin_idx);
    } else {
      // Trick to speed up things. Try to avoid transposing the sparse matrix. The
      // original one (sh_ptr_data->sparse_data_mat * parameter) is about 4 or 5 times
      // slower than that one:
      return (parameter.t() * sh_ptr_psdata->sparse_data_mat).t();
    }
  } else {
    return sh_ptr_psdata->data_mat * parameter;
  }
  // return out;
}

/**
 * \brief Predict on newdata
 *
 * \param newdata `std::shared_ptr<data::Data>` new source data object
 *
 * \returns `arma::mat` of predicted values
 */
arma::mat BaselearnerPSpline::predict (std::shared_ptr<data::Data> newdata) const
{
  return instantiateData(newdata->getData()) * parameter;
}

std::string BaselearnerPSpline::getDataIdentifier () const
{
  return sh_ptr_psdata->getDataIdentifier();
}

/// Destructor
BaselearnerPSpline::~BaselearnerPSpline () {}



// BaselearnerCategoricalBinary:
// -------------------------------

BaselearnerCategoricalBinary::BaselearnerCategoricalBinary (std::shared_ptr<data::CategoricalBinaryData> data, const std::string& identifier)
{
  // Called from parent class 'Baselearner':
  // Baselearner::setData(data);
  Baselearner::setIdentifier(identifier);

  // Additionally, cast the data object to the categorical one to reuse it later:
  sh_ptr_bcdata = data;
}

// Copy member:
Baselearner* BaselearnerCategoricalBinary::clone ()
{
  Baselearner* newbl = new BaselearnerCategoricalBinary(*this);
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->sh_ptr_data);

  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCategoricalBinary::instantiateData (const arma::mat& newdata) const
{
  return newdata;
}

// Train the learner:
void BaselearnerCategoricalBinary::train (const arma::mat& response)
{
  // Calculate sum manually due to the idx format:
  double sum_response = 0;
  for (unsigned int i = 0; i < sh_ptr_bcdata->idx.size() - 1; i++) {
    sum_response += response(sh_ptr_bcdata->idx(i), 0);
  }
  arma::mat param_temp(1,1);
  param_temp(0,0) = sh_ptr_bcdata->xtx_inv_scalar * sum_response;
  parameter = param_temp;
  // parameter(0,0) = sh_ptr_data_cat->xtx_inv_scalar * sum_response;
}

// Predict the learner:
arma::mat BaselearnerCategoricalBinary::predict () const
{
  // arma::mat temp(sh_ptr_data_cat->n_obs, 1, arma::fill::zeros);
  // double param_double = arma::as_scalar(parameter);

  // for (unsigned int i = 0; i < sh_ptr_data_cat->idx.size(); i++) {
  //   temp(i,0) = param_double;
  // }
  // return temp;
  arma::mat out = helper::predictBinaryIndex(sh_ptr_bcdata->idx, arma::as_scalar(parameter));
  return out;
}
arma::mat BaselearnerCategoricalBinary::predict (std::shared_ptr<data::Data> newdata) const
{
  return newdata->getData() * parameter;
}

std::string BaselearnerCategoricalBinary::getDataIdentifier () const
{
  return sh_ptr_bcdata->getDataIdentifier();
}


// Destructor:
BaselearnerCategoricalBinary::~BaselearnerCategoricalBinary () {}




// BaselearnerCustom:
// -----------------------

BaselearnerCustom::BaselearnerCustom (std::shared_ptr<data::Data> data, const std::string& identifier,
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
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->sh_ptr_data);

  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCustom::instantiateData (const arma::mat& newdata) const
{
  Rcpp::NumericMatrix out = instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}

// Train by using the R function 'trainFun'.

// NOTE: It is highly recommended to specify an explicit extractParameter
//       function! Otherwise, it is not possible to estimate the parameter
//       during the whole process:
void BaselearnerCustom::train (const arma::mat& response)
{
  model     = trainFun(response, sh_ptr_data->getData());
  parameter = Rcpp::as<arma::mat>(extractParameter(model));
}

// Predict by using the R function 'predictFun':
arma::mat BaselearnerCustom::predict () const
{
  Rcpp::NumericMatrix out = predictFun(model, sh_ptr_data->getData());
  return Rcpp::as<arma::mat>(out);
}

arma::mat BaselearnerCustom::predict (std::shared_ptr<data::Data> newdata) const
{
  Rcpp::NumericMatrix out = predictFun(model, instantiateData(newdata->getData()));
  return Rcpp::as<arma::mat>(out);
}

std::string BaselearnerCustom::getDataIdentifier () const
{
  return sh_ptr_data->getDataIdentifier();
}



// Destructor:
BaselearnerCustom::~BaselearnerCustom () {}


// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCpp::BaselearnerCustomCpp (std::shared_ptr<data::Data> data, const std::string& identifier,
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
  newbl->copyMembers(this->parameter, this->blearner_identifier, this->sh_ptr_data);

  return newbl;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCustomCpp::instantiateData (const arma::mat& newdata) const
{
  return instantiateDataFun(newdata);
}



// Train by using the external pointer to the function 'trainFun'.

// NOTE: It is highly recommended to specify an explicit extractParameter
//       function! Otherwise, it is not possible to estimate the parameter
//       during the whole process:
void BaselearnerCustomCpp::train (const arma::mat& response)
{
  parameter = trainFun(response, sh_ptr_data->getData());
}

// Predict by using the external pointer to the function 'predictFun':
arma::mat BaselearnerCustomCpp::predict () const
{
  return predictFun (sh_ptr_data->getData(), parameter);
}

arma::mat BaselearnerCustomCpp::predict (std::shared_ptr<data::Data> newdata) const
{
  arma::mat temp_mat = instantiateData(newdata->getData());
  return predictFun (temp_mat, parameter);
}
std::string BaselearnerCustomCpp::getDataIdentifier () const
{
  return sh_ptr_data->getDataIdentifier();
}
// Destructor:
BaselearnerCustomCpp::~BaselearnerCustomCpp () {}


} // namespace blearner
