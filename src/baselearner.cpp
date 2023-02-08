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

std::shared_ptr<Baselearner> jsonToBaselearner (const json& j, const mdata& mdat)
{
  std::shared_ptr<Baselearner> bl;

  if (j["Class"] == "BaselearnerPolynomial") {
    bl = std::make_shared<BaselearnerPolynomial>(j, mdat);
  }
  if (j["Class"] == "BaselearnerPSpline") {
    bl = std::make_shared<BaselearnerPSpline>(j, mdat);
  }
  if (j["Class"] == "BaselearnerTensor") {
    bl = std::make_shared<BaselearnerTensor>(j, mdat);
  }
  if (j["Class"] == "BaselearnerCentered") {
    bl = std::make_shared<BaselearnerCentered>(j, mdat);
  }
  if (j["Class"] == "BaselearnerCategoricalRidge") {
    bl = std::make_shared<BaselearnerCategoricalRidge>(j, mdat);
  }
  if (j["Class"] == "BaselearnerCategoricalBinary") {
    bl = std::make_shared<BaselearnerCategoricalBinary>(j, mdat);
  }
  if (bl == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return bl;
}


// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

Baselearner::Baselearner (const std::string blearner_type)
  : _blearner_type ( blearner_type )
{ }

Baselearner::Baselearner (const json& j)
  : _parameter     ( saver::jsonToArmaMat(j["_parameter"]) ),
    _blearner_type ( j["_blearner_type"] )
{ }

arma::mat Baselearner::getParameter () const
{
  return _parameter;
}

std::string Baselearner::getBaselearnerType () const
{
  return _blearner_type;
}

json Baselearner::baseToJson (const std::string cln) const
{
  json j = {
    { "Class",          cln },
    { "_parameter",     saver::armaMatToJson(_parameter) },
    { "_blearner_type", _blearner_type }
  };
  return j;
}

Baselearner::~Baselearner () { }

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomial::BaselearnerPolynomial (const std::string blearner_type, const std::shared_ptr<data::BinnedData>& data)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_bindata          ( data ),
    _attributes              ( std::make_shared<init::PolynomialAttributes>(0, false))
{ }

BaselearnerPolynomial::BaselearnerPolynomial (const std::string blearner_type, const std::shared_ptr<data::BinnedData>& data,
  const std::shared_ptr<init::PolynomialAttributes>& attributes)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_bindata          ( data ),
    _attributes              ( attributes)
{ }

BaselearnerPolynomial::BaselearnerPolynomial (const json& j, const mdata& mdat)
  : Baselearner::Baselearner ( j ),
    _sh_ptr_bindata          ( std::static_pointer_cast<data::BinnedData>(data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdat)) ),
    _attributes              ( std::make_shared<init::PolynomialAttributes>(j["_attributes"]) )
{ }

void BaselearnerPolynomial::train (const arma::mat& response)
{
  arma::vec temp_weight(1, arma::fill::ones);
  if (_attributes->degree == 1) {
    double y_mean = 0;
    if (_attributes->use_intercept) {
      y_mean = arma::as_scalar(arma::accu(response) / response.size());
    }

    arma::mat xtx_inv = _sh_ptr_bindata->getCacheMat();

    unsigned int dcol;
    if (_attributes->use_intercept) {
      dcol = 1;
    } else {
      dcol = 0;
    }
    arma::mat xmx = _sh_ptr_bindata->getDenseData().col(dcol) - xtx_inv(0,0);
    arma::mat ymy = response - y_mean;
    arma::mat xmxdymy;
    if (_sh_ptr_bindata->usesBinning()) {
      xmxdymy = binning::binnedMatMultResponse(xmx, ymy, _sh_ptr_bindata->getBinningIndex(), temp_weight);
    } else {
      xmxdymy = arma::accu(xmx % ymy);
    }

    double slope = arma::as_scalar(xmxdymy / arma::as_scalar(xtx_inv(0,1)));
    double intercept = y_mean - slope * xtx_inv(0,0);

    if (_attributes->use_intercept) {
      arma::mat out(2,1);

      out(0,0) = intercept;
      out(1,0) = slope;

      _parameter = out;
    } else {
      _parameter = slope;
    }
  } else {
    arma::mat temp;
    if (_sh_ptr_bindata->usesBinning()) {
      temp = binning::binnedMatMultResponse(_sh_ptr_bindata->getDenseData(), response, _sh_ptr_bindata->getBinningIndex(), temp_weight).t();
    } else {
      temp = (response.t() * _sh_ptr_bindata->getDenseData()).t();
    }
    _parameter = helper::cboostSolver(_sh_ptr_bindata->getCache(), temp);
  }
}

arma::mat BaselearnerPolynomial::predict () const
{
  if (_sh_ptr_bindata->usesBinning()) {
    return binning::binnedPrediction(_sh_ptr_bindata->getDenseData(), _parameter, _sh_ptr_bindata->getBinningIndex());
  } else {
    return _sh_ptr_bindata->getDenseData() * _parameter;
  }
}

arma::mat BaselearnerPolynomial::predict (const std::shared_ptr<data::Data>& newdata) const
{
  return newdata->getDenseData() * _parameter;
}

std::string BaselearnerPolynomial::getDataIdentifier () const
{
  return _sh_ptr_bindata->getDataIdentifier();
}

json BaselearnerPolynomial::toJson () const
{
  json j = Baselearner::baseToJson("BaselearnerPolynomial");
  j["id_data_init"] = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();

  return j;
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
 * About the used knots: The number of inner knots is specified
 * by `n_knots`. These inner knots are then wrapped by the minimal and maximal
 * value of the given data. For instance, a feature
 * \f[
 *   x = (1, 2, \dots, 2.5, 6)
 * \f]
 * should be used with 3 knots, then the inner knots with boundaries are:
 * \f[
 *   U = (1.00, 2.25, 3.50, 4.75, 6.00)
 * \f]
 * These knots are wrapped by `degree` (\f$p\f$) on either side to get the
 * full base. If we choose `degree = 2` then we get
 * \f$n_\mathrm{knots} + 2(p + 1) = 3 + 2(2 + 1) 9\f$ final knots:
 * \f[
 *   U = (-1.50, -0.25, 1.00, 2.25, 3.50, 4.75, 6.00, 7.25, 8.50)
 * \f]
 * Finally we get \f$9 - (p + 1)\f$ parameter for the splines.
 *
 * \param blearner_type `std::string` The base-learner type, usually set by the factory.
 * \param sh_ptr_psdata `std::shared_ptr<data::PSplineData>` Data container.
 */
BaselearnerPSpline::BaselearnerPSpline (const std::string blearner_type, const std::shared_ptr<data::BinnedData>& sh_ptr_bindata)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_bindata          ( sh_ptr_bindata )
{ }

BaselearnerPSpline::BaselearnerPSpline (const json& j, const mdata& mdat)
  : Baselearner::Baselearner ( j ),
    _sh_ptr_bindata          ( std::static_pointer_cast<data::BinnedData>(data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdat)) )
{ }

void BaselearnerPSpline::train (const arma::mat& response)
{
  arma::mat temp;

  if (_sh_ptr_bindata->usesBinning()) {
    arma::vec temp_weight(1, arma::fill::ones);
    temp = binning::binnedSparseMatMultResponse(_sh_ptr_bindata->getSparseData(), response, _sh_ptr_bindata->getBinningIndex(), temp_weight);
  } else {
    temp = _sh_ptr_bindata->getSparseData() * response;
  }
  _parameter = helper::cboostSolver(_sh_ptr_bindata->getCache(), temp);
}

arma::mat BaselearnerPSpline::predict () const
{
  // Here we have a different handling than in predict(data) because of the possibility to use binning.
  // It does not make sense to also include binning into the prediction of new points! Binning is just
  // a method to fasten the fitting process.
  if (_sh_ptr_bindata->usesBinning()) {
    return binning::binnedSparsePrediction(_sh_ptr_bindata->getSparseData(), _parameter, _sh_ptr_bindata->getBinningIndex());
  } else {
    // Trick to speed up things. Try to avoid transposing the sparse matrix. The
    // original one (sh_ptr_data->sparse_data_mat * parameter) is about 4 or 5 times
    // slower than that one:
    return (_parameter.t() * _sh_ptr_bindata->getSparseData()).t();
  }
}

arma::mat BaselearnerPSpline::predict (const std::shared_ptr<data::Data>& newdata) const
{
  return (_parameter.t() * newdata->getSparseData()).t();
}

std::string BaselearnerPSpline::getDataIdentifier () const
{
  return _sh_ptr_bindata->getDataIdentifier();
}

json BaselearnerPSpline::toJson () const
{
  json j = Baselearner::baseToJson("BaselearnerPSpline");
  j["id_data_init"] = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;

  return j;
}

/// Destructor
BaselearnerPSpline::~BaselearnerPSpline () {}


// BaselearnerTensor:
// ------------------------------------

BaselearnerTensor::BaselearnerTensor (const std::string blearner_type, const std::shared_ptr<data::Data>& sh_ptr_data)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_data             ( sh_ptr_data )
{ }

BaselearnerTensor::BaselearnerTensor (const json& j, const mdata& mdat)
  : Baselearner::Baselearner ( j ),
    _sh_ptr_data             ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdat) )
{ }

void BaselearnerTensor::train (const arma::mat& response)
{
  arma::mat temp;
  if (_sh_ptr_data->usesSparseMatrix()) {
    temp = _sh_ptr_data->getSparseData() * response;
  } else {
    temp = (response.t() * _sh_ptr_data->getDenseData()).t();
  }
  _parameter = helper::cboostSolver(_sh_ptr_data->getCache(), temp);
}

arma::mat BaselearnerTensor::predict () const
{
  return predict(_sh_ptr_data);
  //if (_sh_ptr_data->usesSparseMatrix()) {
    //return (_parameter.t() * _sh_ptr_data->getSparseData()).t();  }
  //} else {
    //return _sh_ptr_data->getDenseData() * _parameter;
  //}
}

arma::mat BaselearnerTensor::predict (const std::shared_ptr<data::Data>& newdata) const
{
  if (newdata->usesSparseMatrix()) {
    return (_parameter.t() * newdata->getSparseData()).t();
  } else {
    return newdata->getDenseData() * _parameter;
  }
}

std::string BaselearnerTensor::getDataIdentifier () const
{
  return _sh_ptr_data->getDataIdentifier();
}

json BaselearnerTensor::toJson () const
{
  json j = Baselearner::baseToJson("BaselearnerTensor");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;

  return j;
}

/// Destructor
BaselearnerTensor::~BaselearnerTensor () {}

// BaselearnerCentered:
// ------------------------------------

BaselearnerCentered::BaselearnerCentered (const std::string blearner_type, const std::shared_ptr<data::Data>& sh_ptr_data)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_data             ( sh_ptr_data )
{ }

BaselearnerCentered::BaselearnerCentered (const json& j, const mdata& mdat)
  : Baselearner::Baselearner ( j ),
    _sh_ptr_data             ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdat) )
{ }

void BaselearnerCentered::train (const arma::mat& response)
{
  arma::mat temp = _sh_ptr_data->getDenseData().t() * response;
  _parameter = helper::cboostSolver(_sh_ptr_data->getCache(), temp);

  //arma::mat temp;
  //if (_sh_ptr_data->usesBinning()) {
    //temp = binning::binnedMatMultResponse(_sh_ptr_bindata->getDenseData(), response, _sh_ptr_bindata->getBinningIndex(), temp_weight).t();
  //} else {
    //temp = (response.t() * _sh_ptr_bindata->getDenseData()).t();
  //}
  //_parameter = helper::cboostSolver(_sh_ptr_bindata->getCache(), temp);
}

arma::mat BaselearnerCentered::predict () const
{
  return predict(_sh_ptr_data);
}

arma::mat BaselearnerCentered::predict (const std::shared_ptr<data::Data>& newdata) const
{
  return newdata->getDenseData() * _parameter;
}

std::string BaselearnerCentered::getDataIdentifier () const
{
  return _sh_ptr_data->getDataIdentifier();
}

json BaselearnerCentered::toJson () const
{
  json j = Baselearner::baseToJson("BaselearnerCentered");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;

  return j;
}

/// Destructor
BaselearnerCentered::~BaselearnerCentered () {}

// BaselearnerCategoricalRidge:
// ---------------------------------------

BaselearnerCategoricalRidge::BaselearnerCategoricalRidge (const std::string blearner_type,
  const std::shared_ptr<data::Data>& data)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_data             ( data )
{ }

BaselearnerCategoricalRidge::BaselearnerCategoricalRidge (const json& j, const mdata& mdat)
  : Baselearner::Baselearner ( j ),
    _sh_ptr_data             ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdat) )
{ }

void BaselearnerCategoricalRidge::train (const arma::mat& response)
{
  _parameter = _sh_ptr_data->getCache().second % (_sh_ptr_data->getSparseData() * response);
}

arma::mat BaselearnerCategoricalRidge::predict () const
{
  return (_parameter.t() * _sh_ptr_data->getSparseData()).t();
}

arma::mat BaselearnerCategoricalRidge::predict (const std::shared_ptr<data::Data>& newdata) const
{
  return (_parameter.t() * newdata->getSparseData()).t();
}

//arma::mat BaselearnerCategoricalRidge::instantiateData (const arma::mat& newdata) const
//{
  //throw std::logic_error("Categorical base-learner do not instantiate data!");
  //return arma::mat(1, 1, arma::fill::zeros);
//}

std::string BaselearnerCategoricalRidge::getDataIdentifier () const
{
  return _sh_ptr_data->getDataIdentifier();
}

json BaselearnerCategoricalRidge::toJson () const
{
  json j = Baselearner::baseToJson("BaselearnerCategoricalRidge");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
  //j["data"] = _sh_ptr_data->toJson();

  return j;
}


BaselearnerCategoricalRidge::~BaselearnerCategoricalRidge () {}

// BaselearnerCategoricalBinary:
// -------------------------------

BaselearnerCategoricalBinary::BaselearnerCategoricalBinary (const std::string blearner_type,
  const std::shared_ptr<data::Data>& data)
  : Baselearner::Baselearner ( std::string(blearner_type) ),
    _sh_ptr_data             ( data )
{ }

BaselearnerCategoricalBinary::BaselearnerCategoricalBinary (const json& j, const mdata& mdat)
  : Baselearner::Baselearner ( j ),
    _sh_ptr_data             ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdat) )
{ }


void BaselearnerCategoricalBinary::train (const arma::mat& response)
{
  _parameter = _sh_ptr_data->getCache().second * (_sh_ptr_data->getSparseData() * response);

  // Calculate sum manually due to the idx format:
  //double sum_response = 0;
  //arma::uvec idx = _sh_ptr_bcdata->getIndex();
  //for (unsigned int i = 0; i < idx.size(); i++) {
    //sum_response += response(idx(i), 0);
  //}
  //arma::mat param_temp(1,1);
  //param_temp(0,0) = _sh_ptr_bcdata->getXtxScalar() * sum_response;
  //_parameter = param_temp;
}

arma::mat BaselearnerCategoricalBinary::predict () const
{
  return (_parameter.t() * _sh_ptr_data->getSparseData()).t();
}

arma::mat BaselearnerCategoricalBinary::predict (const std::shared_ptr<data::Data>& newdata) const
{
  return _parameter * newdata->getSparseData();
  //std::shared_ptr<data::CategoricalDataRaw> sh_ptr_cdata_raw = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);

  //std::vector<std::string> data_raw = sh_ptr_cdata_raw->getRawData();
  //unsigned int nobs = data_raw.size();
  //arma::mat out(nobs, 1, arma::fill::zeros);


  //for (unsigned int i = 0; i < nobs; i++) {
    //if (data_raw.at(i) == _sh_ptr_bcdata->getCategory())
      //out(i) = arma::as_scalar(_parameter);
  //}
  //return out;
}

//arma::mat BaselearnerCategoricalBinary::instantiateData (const arma::mat& newdata) const
//{
  //throw std::logic_error("Categorical base-learner do not instantiate data!");
  //return arma::mat(1, 1, arma::fill::zeros);
//}

std::string BaselearnerCategoricalBinary::getDataIdentifier () const
{
  return _sh_ptr_data->getDataIdentifier();
}

json BaselearnerCategoricalBinary::toJson () const
{
  json j = Baselearner::baseToJson("BaselearnerCategoricalBinary");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier();
  //j["data"] = _sh_ptr_data->toJson();

  return j;
}

/// Destructor:
BaselearnerCategoricalBinary::~BaselearnerCategoricalBinary () {}


// BaselearnerCustom:
// -----------------------

BaselearnerCustom::BaselearnerCustom (const std::string blearner_type, const std::shared_ptr<data::Data>& data,
  //const std::shared_ptr<init::CustomAttributes>& attributes)
  Rcpp::Function instantiateDataFun, Rcpp::Function trainFun, Rcpp::Function predictFun,
  Rcpp::Function extractParameter)
  : Baselearner        ( blearner_type ),
    _sh_ptr_data       ( data ),
    _instantiateDataFun ( instantiateDataFun ),
    _trainFun           ( trainFun ),
    _predictFun         ( predictFun ),
    _extractParameter   ( extractParameter )
{ }

void BaselearnerCustom::train (const arma::mat& response)
{
  _model     = _trainFun(response, _sh_ptr_data->getData());
  _parameter = Rcpp::as<arma::mat>(_extractParameter(_model));
}

arma::mat BaselearnerCustom::predict () const
{
  Rcpp::NumericMatrix out = _predictFun(_model, _sh_ptr_data->getData());
  return Rcpp::as<arma::mat>(out);
}

arma::mat BaselearnerCustom::predict (const std::shared_ptr<data::Data>& newdata) const
{
  Rcpp::NumericMatrix out = _predictFun(_model, newdata->getDenseData());
  return Rcpp::as<arma::mat>(out);
}

//arma::mat BaselearnerCustom::instantiateData (const arma::mat& newdata) const
//{
  //Rcpp::NumericMatrix out = _instantiateDataFun(newdata);
  //return Rcpp::as<arma::mat>(out);
//}

std::string BaselearnerCustom::getDataIdentifier () const
{
  return _sh_ptr_data->getDataIdentifier();
}

json BaselearnerCustom::toJson () const
{
  throw std::logic_error("Cannot transform a custom base learner to JSON.");
  json j = Baselearner::baseToJson("BaselearnerCustom");

  return j;
}

/// Destructor:
BaselearnerCustom::~BaselearnerCustom () {}


// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCpp::BaselearnerCustomCpp (const std::string blearner_type, const std::shared_ptr<data::Data>& data,
  //SEXP instantiateDataFun0, SEXP trainFun0, SEXP predictFun0)
  const std::shared_ptr<init::CustomCppAttributes>& attributes)
  : Baselearner ( blearner_type ),
    _attributes ( attributes )
{
  //Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun0);
  //_instantiateDataFun = *myTempInstantiation;

  //Rcpp::XPtr<trainFunPtr> myTempTrain (trainFun0);
  //_trainFun = *myTempTrain;

  //Rcpp::XPtr<predictFunPtr> myTempPredict (predictFun0);
  //_predictFun = *myTempPredict;
}

void BaselearnerCustomCpp::train (const arma::mat& response)
{
  _parameter = _attributes->trainFun(response, _sh_ptr_data->getData());
}

arma::mat BaselearnerCustomCpp::predict () const
{
  return _attributes->predictFun (_sh_ptr_data->getData(), _parameter);
}

arma::mat BaselearnerCustomCpp::predict (const std::shared_ptr<data::Data>& newdata) const
{
  arma::mat temp_mat = newdata->getData();
  return _attributes->predictFun (temp_mat, _parameter);
}

//sdata BaselearnerCustomCpp::instantiateData (const mdata& data_map) const
//{
  //return _instantiateDataFun(newdata);
//}

std::string BaselearnerCustomCpp::getDataIdentifier () const
{
  return _sh_ptr_data->getDataIdentifier();
}

json BaselearnerCustomCpp::toJson () const
{
  throw std::logic_error("Cannot transform a custom base learner to JSON.");
  json j = Baselearner::baseToJson("BaselearnerCustomCpp");

  return j;
}


/// Destructor:
BaselearnerCustomCpp::~BaselearnerCustomCpp () {}

} // namespace blearner
