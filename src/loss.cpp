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

#include "loss.h"

namespace loss
{

arma::mat doubleToMat (const double x)
{
  arma::mat temp(1,1);
  temp(0,0) = x;
  return temp;
}

// Abstract Loss class:
// -----------------------

Loss::Loss (const std::string task_id)
  : _task_id ( task_id )
{ }

Loss::Loss (const std::string task_id, const arma::mat& custom_offset)
  : _task_id           ( task_id ),
    _custom_offset     ( custom_offset ),
    _use_custom_offset ( true )
{ }

Loss::Loss (const json& j)
  : _task_id           ( j["_task_id"] ),
    _custom_offset     ( saver::jsonToArmaMat( j["_custom_offset"]) ),
    _use_custom_offset ( j["_use_custom_offset"] )
{ }

std::string Loss::getTaskId () const { return _task_id; }

arma::mat Loss::weightedLoss (const arma::mat& true_value, const arma::mat& prediction,
  const arma::mat& weights) const
{
  return weights % loss(true_value, prediction);
}

arma::mat Loss::weightedGradient (const arma::mat& true_value, const arma::mat& prediction,
  const arma::mat& weights) const
{
  return weights % gradient(true_value, prediction);
}

double Loss::calculateEmpiricalRisk (const arma::mat& true_value, const arma::mat& prediction) const
{
  return arma::accu(loss(true_value, prediction)) / true_value.size();
}

double Loss::calculateWeightedEmpiricalRisk (const arma::mat& true_value, const arma::mat& prediction,
  const arma::mat& weights) const
{
  return arma::accu(weightedLoss(true_value, prediction, weights)) / true_value.size();
}

arma::mat Loss::calculatePseudoResiduals (const arma::mat& true_value, const arma::mat& prediction) const
{
  return -gradient(true_value, prediction);
}

arma::mat Loss::calculateWeightedPseudoResiduals (const arma::mat& true_value, const arma::mat& prediction,
  const arma::mat& weights) const
{
  return -weightedGradient(true_value, prediction, weights);
}

json Loss::baseToJson (const std::string cln) const
{
  json j = {
    {"Class",              cln},
    {"_task_id",           _task_id},
    {"_custom_offset",     saver::armaMatToJson(_custom_offset)},
    {"_use_custom_offset", _use_custom_offset}
  };

  return j;
}

// Destructor
Loss::~Loss () { }


// -------------------------------------------------------------------------- //
// Loss implementations:
// -------------------------------------------------------------------------- //

// LossQuadratic:
// -----------------------

LossQuadratic::LossQuadratic ()
  : Loss::Loss ( std::string("regression") ) // Explicit casting to avoid declaration ambiguity
{ }

LossQuadratic::LossQuadratic (const double custom_offset)
  : Loss::Loss ( "regression", doubleToMat(custom_offset) )
{ }

LossQuadratic::LossQuadratic (const arma::mat& custom_offset)
  : Loss::Loss ("regression", custom_offset)
{ }

LossQuadratic::LossQuadratic (const json& j)
  : Loss::Loss (j)
{ }

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the loss function
 */
arma::mat LossQuadratic::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  return arma::pow(true_value - prediction, 2) / 2;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the gradient
 */
arma::mat LossQuadratic::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  return prediction - true_value;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `arma::mat` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossQuadratic::constantInitializer (const arma::mat& true_value) const
{
  if (_use_custom_offset) { return _custom_offset; }

  arma::mat out(1, 1);
  out.fill(arma::accu(true_value) / true_value.size());

  return out;
}
arma::mat LossQuadratic::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  if (_use_custom_offset) { return _custom_offset; }

  arma::mat out(1, 1);
  out.fill(arma::accu(weights % true_value) / true_value.size());

  return out;
}

json LossQuadratic::toJson () const { return baseToJson("LossQuadratic"); }

// LossAbsolute:
// ------------------------------------------

LossAbsolute::LossAbsolute ()
  : Loss::Loss ( std::string("regression") ) // Explicit casting to avoid declaration ambiguity
{ }

LossAbsolute::LossAbsolute (const double custom_offset)
  : Loss::Loss ( "regression", doubleToMat(custom_offset) )
{ }

LossAbsolute::LossAbsolute (const json& j)
  : Loss::Loss (j)
{ }

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the loss function
 */
arma::mat LossAbsolute::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  return arma::abs(true_value - prediction);
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the gradient
 */
arma::mat LossAbsolute::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  return - arma::sign(true_value - prediction);
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossAbsolute::constantInitializer (const arma::mat& true_value) const
{
  if (_use_custom_offset) { return _custom_offset; }

  arma::vec temp = true_value;
  arma::mat out(1, 1);
  out.fill(arma::median(temp));

  return out;
}
arma::mat LossAbsolute::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  return constantInitializer(true_value);
}

json LossAbsolute::toJson () const { return baseToJson("LossQuadratic"); }

// LossQuantile
// -----------------------

LossQuantile::LossQuantile (const double quantile)
  : Loss::Loss ( std::string("regression") ), // Explicit casting to avoid declaration ambiguity
    _quantile  ( quantile )
{
  if ((_quantile > 1) || (_quantile < 0)) {
    Rcpp::stop("Quantile must be in [0,1]");
  }
}

LossQuantile::LossQuantile (const double custom_offset, const double quantile)
  : Loss::Loss ( "regression", doubleToMat(custom_offset) ),
    _quantile  ( quantile )
{
  if ((_quantile > 1) || (_quantile < 0)) {
    Rcpp::stop("Quantile must be in [0,1]");
  }
}

LossQuantile::LossQuantile (const json& j)
  : Loss::Loss (j),
    _quantile  ( j["_quantile"] )
{ }

double LossQuantile::getQuantile () const { return _quantile; }

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the loss function
 */
arma::mat LossQuantile::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  arma::mat residual_mat  = true_value - prediction;
  arma::mat quant_weights = residual_mat;

  quant_weights.transform( [this](double elem) { return (elem < 0) ? double(2 * (1 - _quantile)) : double(2 * _quantile); } );

  return arma::abs(residual_mat) % quant_weights;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element_wise application of the gradient
 */
arma::mat LossQuantile::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  arma::mat residual_mat  = true_value - prediction;
  arma::mat quant_weights = residual_mat;

  quant_weights.transform( [this](double elem) { return (elem < 0) ? double(2 * (1 - _quantile)) : double(2 * _quantile); } );

  return - arma::sign(residual_mat) % quant_weights;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossQuantile::constantInitializer (const arma::mat& true_value) const
{
  if (_use_custom_offset) { return _custom_offset; }

  arma::vec temp = true_value;
  arma::mat out(1, 1);
  out.fill(helper::matrixQuantile(true_value, _quantile));

  return out;
}

arma::mat LossQuantile::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  Rcpp::warning("Quantile loss does not have a weighted offset implementation. Using unweighted initializer.");
  return LossQuantile::constantInitializer(true_value);
}

json LossQuantile::toJson () const
{
  json j = baseToJson("LossQuantile");
  j["_quantile"] = _quantile;

  return j;
}


// LossHuber:
// -----------------------

LossHuber::LossHuber (const double delta)
  : Loss   ( std::string("regression") ), // Explicit casting to avoid declaration ambiguity
    _delta ( delta )
{
  if ((_delta < 0)) {
    Rcpp::stop("Delta must be greater than 0");
  }
}

LossHuber::LossHuber (const double custom_offset, const double delta)
  : Loss   ( "regression", doubleToMat(custom_offset) ),
    _delta ( delta )
{
  if ((_delta < 0)) {
    Rcpp::stop("Delta must be greater than 0");
  }
}

LossHuber::LossHuber (const json& j)
  : Loss::Loss (j),
    _delta     ( j["_delta"] )
{ }

double LossHuber::getDelta () const { return _delta; }

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the loss function
 */
arma::mat LossHuber::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  arma::mat loss_mat = true_value - prediction;
  loss_mat.transform( [this](double elem) {
    return (std::abs(elem) < _delta) ? double(0.5 * std::pow(elem, 2)) : double(_delta * std::abs(elem) - 0.5 * std::pow(_delta, 2));
  } );

  return loss_mat;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the gradient
 */
arma::mat LossHuber::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  arma::mat grad_mat = true_value - prediction;
  grad_mat.transform( [this](double elem) {
    return (std::abs(elem) < _delta) ? double(-elem) : double(-_delta * elem / std::abs(elem));
  } );

  return grad_mat;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossHuber::constantInitializer (const arma::mat& true_value) const
{
  if (_use_custom_offset) { return _custom_offset; }

  double out = lossoptim::findOptimalLossConstant(true_value, shared_from_this(), true_value.min(), true_value.max());
  arma::mat out_mat(1,1);
  out_mat.fill(out);

  return out_mat;
}

arma::mat LossHuber::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  if (_use_custom_offset) { return _custom_offset; }

  double out = lossoptim::findOptimalWeightedLossConstant(true_value, weights, shared_from_this(), true_value.min(), true_value.max());
  arma::mat out_mat(1,1);
  out_mat.fill(out);

  return out_mat;
}

json LossHuber::toJson () const
{
  json j = baseToJson("LossHuber");
  j["_delta"] = _delta;

  return j;
}


// LossBinomial:
// -----------------------

LossBinomial::LossBinomial ()
  : Loss ( std::string("binary_classif") ) // Explicit casting to avoid declaration ambiguity
{ }

LossBinomial::LossBinomial (const double custom_offset)
  : Loss ( "binary_classif", doubleToMat(custom_offset) )
{
  if (custom_offset > 1 || custom_offset < -1) {
    Rcpp::stop("LossBinomial allows just values between -1 and 1 as offset. Continuing with default offset.");
  }
}
LossBinomial::LossBinomial (const arma::mat& custom_offset)
  : Loss ("binary_classif", custom_offset)
{ }

LossBinomial::LossBinomial (const json& j)
  : Loss::Loss (j)
{ }

/**
* \brief Definition of the loss function (see description of the class)
*
* \param true_value `arma::mat` True value of the response
* \param prediction `arma::mat` Prediction of the true value
*
* \returns `arma::mat` vector of element-wise application of the loss function
*/
arma::mat LossBinomial::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  return arma::log(1 + arma::exp(-true_value % prediction));
}

/**
* \brief Definition of the gradient of the loss function (see description of the class)
*
* \param true_value `arma::mat` True value of the response
* \param prediction `arma::mat` Prediction of the true value
*
* \returns `arma::mat` vector of element-wise application of the gradient
*/
arma::mat LossBinomial::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  return -true_value / (1 + arma::exp(true_value % prediction));
}

/**
* \brief Definition of the constant risk initialization (see description of the class)
*
* \param true_value `arma::mat` True value of the response
*
* \returns `double` constant which minimizes the empirical risk for the given true value
*/
arma::mat LossBinomial::constantInitializer (const arma::mat& true_value) const
{
  if (_use_custom_offset) { return _custom_offset; }

  double p = arma::accu(true_value + 1) / (2 * true_value.size());
  arma::mat out(1, 1);
  out.fill(std::log(p / (1 - p)));

  return out;
}
arma::mat LossBinomial::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  Rcpp::warning("Binomial loss does not have a weighted offset implementation. Using unweighted initializer.");
  return LossBinomial::constantInitializer(true_value);
}

json LossBinomial::toJson () const { return baseToJson("LossBinomial"); }

// LossCustom:
// -----------------------

/**
 * \brief Default constructor of custom loss class
 *
 * \param lossFun `Rcpp::Function` `R` function to calculate the loss
 * \param gradientFun `Rcpp::Function` `R` function to calculate the gradient
 *   of the loss function
 * \param initFun `Rcpp::Function` `R` function to initialize a constant (here
 *   it is not neccessary to initialize in a loss/risk optimal manner)
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
LossCustom::LossCustom (const Rcpp::Function& lossFun, const Rcpp::Function& gradientFun,
  const Rcpp::Function& initFun)
  : Loss::Loss   ( std::string("custom") ), // Explicit casting to avoid declaration ambiguity
    _lossFun     ( lossFun ),
    _gradientFun ( gradientFun ),
    _initFun     ( initFun )
{ }

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the loss function
 */
arma::mat LossCustom::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  Rcpp::NumericVector out = _lossFun(true_value, prediction);
  return out;
}

/**
 * \brief Definition of the gradient of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the gradient
 */
arma::mat LossCustom::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  Rcpp::NumericVector out = _gradientFun(true_value, prediction);
  return out;
}

/**
 * \brief Definition of the constant risk initialization (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 *
 * \returns `double` constant which minimizes the empirical risk for the given true value
 */
arma::mat LossCustom::constantInitializer (const arma::mat& true_value) const
{
  Rcpp::NumericVector out = _initFun(true_value);
  arma::mat out_single = out;
  //arma::mat out_single(1, 1);
  //out_single.fill(out[0]);
  return out_single;
}

arma::mat LossCustom::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  Rcpp::warning("Custom loss does not have a weighted offset implementation. Using unweighted initializer.");
  return LossCustom::constantInitializer(true_value);
}

json LossCustom::toJson () const
{
  throw std::logic_error("It is not implemented yet to save custom loss classes as JSON.");
  return baseToJson("LossCustom");
}

// LossCustomCpp:
// -----------------------

LossCustomCpp::LossCustomCpp (const SEXP& lossFun, const SEXP& gradFun, const SEXP& constInitFun)
  : Loss::Loss ( std::string("custom") ) // Explicit casting to avoid declaration ambiguity
{
  Rcpp::XPtr<lossFunPtr> myTempLoss (lossFun);
  _lossFun = *myTempLoss;

  Rcpp::XPtr<gradFunPtr> myTempGrad (gradFun);
  _gradFun = *myTempGrad;

  Rcpp::XPtr<constInitFunPtr> myTempConstInit (constInitFun);
  _constInitFun = *myTempConstInit;
}

/**
 * \brief Definition of the loss function (see description of the class)
 *
 * \param true_value `arma::mat` True value of the response
 * \param prediction `arma::mat` Prediction of the true value
 *
 * \returns `arma::mat` vector of element-wise application of the loss function
 */
arma::mat LossCustomCpp::loss (const arma::mat& true_value, const arma::mat& prediction) const
{
  return _lossFun(true_value, prediction);
}

/**
* \brief Definition of the gradient of the loss function (see description of the class)
*
* \param true_value `arma::mat` True value of the response
* \param prediction `arma::mat` Prediction of the true value
*
* \returns `arma::mat` vector of element-wise application of the gradient
*/
arma::mat LossCustomCpp::gradient (const arma::mat& true_value, const arma::mat& prediction) const
{
  return _gradFun(true_value, prediction);
}

/**
* \brief Definition of the constant risk initialization (see description of the class)
*
* \param true_value `arma::mat` True value of the response
*
* \returns `double` constant which minimizes the empirical risk for the given true value
*/
arma::mat LossCustomCpp::constantInitializer (const arma::mat& true_value) const
{
  arma::mat out(1, 1);
  out.fill(_constInitFun(true_value));

  return out;
}
arma::mat LossCustomCpp::weightedConstantInitializer (const arma::mat& true_value, const arma::mat& weights) const
{
  Rcpp::warning("Custom cpp loss does not have a weighted offset implementation. Using unweighted initializer.");
  return constantInitializer(true_value);
}

json LossCustomCpp::toJson () const
{
  throw std::logic_error("It is not implemented yet to save custom loss classes as JSON.");
  return baseToJson("LossCustomCpp");
}

} // namespace loss
