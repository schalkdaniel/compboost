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

#include "baselearner_factory.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type, std::shared_ptr<data::Data> data_source)
  : _blearner_type      ( blearner_type ),
    _sh_ptr_data_source ( data_source )
{ }

std::string BaselearnerFactory::getDataIdentifier () const { return _sh_ptr_data_source->getDataIdentifier(); }
std::string BaselearnerFactory::getBaselearnerType() const { return _blearner_type; }

/// Destructor
BaselearnerFactory::~BaselearnerFactory () {}

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomialFactory::BaselearnerPolynomialFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, std::shared_ptr<data::Data> data_target0, const unsigned int degree,
  const bool intercept)
  : BaselearnerFactory ( blearner_type, data_source ),
    _degree ( degree ),
    _intercept ( intercept )
{
  _sh_ptr_data_target = data_target0;
  _sh_ptr_data_target->setDataIdentifier(data_source->getDataIdentifier());

  // Prepare computation of intercept and slope of an ordinary linear regression:
  if (data_source->getData().n_cols == 1) {
    // Store centered x values for faster computation:
    _sh_ptr_data_target->setData(arma::pow(data_source->getData(), _degree));

    unsigned int p = 1;
    if (_intercept) p = 2;
    arma::mat temp_mat(1, p, arma::fill::zeros);

    if (_intercept) {
      temp_mat(0,0) = arma::as_scalar(arma::mean(_sh_ptr_data_target->getData()));
    }
    const double slope = arma::as_scalar(arma::sum(arma::pow(_sh_ptr_data_target->getData() - temp_mat(0,0), 2)));
    if (_intercept) {
      temp_mat(0,1) = slope;
    } else {
      temp_mat(0,0) = slope;
    }
    _sh_ptr_data_target->XtX_inv = temp_mat;
  } else {
    _sh_ptr_data_target->setData(instantiateData(data_source->getData()));
    _sh_ptr_data_target->XtX_inv = arma::inv(_sh_ptr_data_target->getData().t() * _sh_ptr_data_target->getData());
  }
}

arma::mat BaselearnerPolynomialFactory::instantiateData (const arma::mat& newdata) const
{
  arma::mat temp = arma::pow(newdata, _degree);
  if (_intercept) {
    arma::mat temp_intercept(temp.n_rows, 1, arma::fill::ones);
    temp = join_rows(temp_intercept, temp);
  }
  return temp;
}

arma::mat BaselearnerPolynomialFactory::getData () const
{
  // In the case of p = 1 we have to treat the getData() function differently
  // due to the saved and already transformed data without intercept. This
  // is annoying but improves performance of the fitting process.
  if (_sh_ptr_data_target->getData().n_cols == 1) {
    if (_intercept) {
      return instantiateData(arma::pow(_sh_ptr_data_target->getData(), 1/_degree));
    } else {
      return _sh_ptr_data_target->getData();
    }
  } else {
    return _sh_ptr_data_target->getData();
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerPolynomialFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPolynomial>(_blearner_type, _sh_ptr_data_target, _degree, _intercept);
}


// BaselearnerPSpline:
// -----------------------

/**
 * \brief Default constructor of class `PSplineBleanrerFactory`
 *
 * The P-Spline constructor has some important tasks which are:
 *   - Set the knots
 *   - Initialize the spline base (knots must be setted prior)
 *   - Compute and store penalty matrix
 *
 * \param blearner_type `std::string` Name of the baselearner type (setted by
 *   the Rcpp Wrapper classes in `compboost_modules.cpp`)
 * \param data_source `std::shared_ptr<data::Data>` Source of the data
 * \param degree `unsigned int` Polynomial degree of the splines
 * \param n_knots `unsigned int` Number of inner knots
 * \param penalty `double` Regularization parameter `penalty = 0` gives
 *   b splines while a bigger penalty forces the splines into a global
 *   polynomial form
 * \param differences `unsigned int` Number of differences used for the
 *   penalty matrix
 * \param use_sparse_matrices `bool` Use sparse matrices for data storage
 * \param use_binning `bool` Use binning to improve runtime performance and reduce memory load
 */
BaselearnerPSplineFactory::BaselearnerPSplineFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, const unsigned int degree, const unsigned int n_knots,
  const double penalty, const unsigned int differences, const bool use_sparse_matrices, const unsigned int bin_root,
  const std::string cache_type)
  : BaselearnerFactory ( blearner_type, data_source )
{
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (data_source->getData().n_cols > 1) {
      Rcpp::stop("Given data should just have one column.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }
  const arma::mat knots       = splines::createKnots(data_source->getData(), n_knots, degree);
  const arma::mat penalty_mat = splines::penaltyMat(n_knots + (degree + 1), differences);

  if (bin_root == 0) { // don't use binning
    _sh_ptr_psdata = std::make_shared<data::PSplineData>(data_source->getDataIdentifier(), degree, knots, penalty_mat);
  } else {             // use binning
    _sh_ptr_psdata =    std::make_shared<data::PSplineData>(data_source->getDataIdentifier(), degree, knots, penalty_mat, bin_root);
    arma::colvec bins = binning::binVectorCustom(data_source->getData(), bin_root);

    _sh_ptr_psdata->setIndexVector(data_source->getData(), bins);
    data_source->setData(bins);
  }

  arma::mat     temp_xtx;
  arma::sp_mat  temp      = splines::createSparseSplineBasis (data_source->getData(), degree, knots).t();

  _sh_ptr_psdata->setSparseData(temp);

  if (_sh_ptr_psdata->usesBinning()) {
    arma::vec temp_weight(1, arma::fill::ones);
    temp_xtx = binning::binnedSparseMatMult(_sh_ptr_psdata->sparse_data_mat, _sh_ptr_psdata->bin_idx, temp_weight);
  } else {
    temp_xtx = _sh_ptr_psdata->sparse_data_mat * _sh_ptr_psdata->sparse_data_mat.t();
  }
  _sh_ptr_psdata->setCache(cache_type, temp_xtx + penalty * penalty_mat);
}

arma::mat BaselearnerPSplineFactory::instantiateData (const arma::mat& newdata) const
{
  arma::mat temp = _sh_ptr_psdata->filterKnotRange(newdata);
  return splines::createSplineBasis (temp, _sh_ptr_psdata->degree, _sh_ptr_psdata->getKnots());
}

arma::mat BaselearnerPSplineFactory::getData () const
{
  return _sh_ptr_psdata->getData();
}

std::shared_ptr<blearner::Baselearner> BaselearnerPSplineFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPSpline>(_blearner_type, _sh_ptr_psdata);
}


// BaselearnerCategoricalBinary:
// ----------------------------------

BaselearnerCategoricalBinaryFactory::BaselearnerCategoricalBinaryFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source)
  : BaselearnerFactory ( blearner_type, data_source )
{
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (data_source->getData().n_cols > 1) {
      Rcpp::stop("Given data should just have one column.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }
  // In the binary case, the matrix XtX_inv is just the inverse of the length of non-zero elements.
  // This is automatically set in the constructor:
  arma::uvec idx;
  if (data_source->usesSparseMatrix()) {
    idx = helper::binaryToIndex(data_source->sparse_data_mat);
  } else {
    idx = helper::binaryToIndex(data_source->getData());
  }
  _sh_ptr_bcdata = std::make_shared<data::CategoricalBinaryData> (idx);
  _sh_ptr_bcdata->setDataIdentifier(data_source->getDataIdentifier());
}

std::shared_ptr<blearner::Baselearner> BaselearnerCategoricalBinaryFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCategoricalBinary>(_blearner_type, _sh_ptr_bcdata);
}

arma::mat BaselearnerCategoricalBinaryFactory::getData () const
{
  return helper::predictBinaryIndex(std::static_pointer_cast<data::CategoricalBinaryData>(_sh_ptr_bcdata)->idx, 1);
}

arma::mat BaselearnerCategoricalBinaryFactory::instantiateData (const arma::mat& newdata) const
{
  return newdata;
}


// BaselearnerCustom:
// -----------------------

BaselearnerCustomFactory::BaselearnerCustomFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, std::shared_ptr<data::Data> data_target, Rcpp::Function instantiateDataFun,
  Rcpp::Function trainFun, Rcpp::Function predictFun, Rcpp::Function extractParameter)
  : BaselearnerFactory   ( blearner_type, data_source ),
    _sh_ptr_data_target  ( data_target ),
    _instantiateDataFun  ( instantiateDataFun ),
    _trainFun            ( trainFun ),
    _predictFun          ( predictFun ),
    _extractParameter    ( extractParameter )
{
  _sh_ptr_data_target->setDataIdentifier(data_source->getDataIdentifier());
  _sh_ptr_data_target->setData(instantiateData(data_source->getData()));
}

std::shared_ptr<blearner::Baselearner> BaselearnerCustomFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCustom>(_blearner_type, _sh_ptr_data_target,
    _instantiateDataFun, _trainFun, _predictFun, _extractParameter);
}

arma::mat BaselearnerCustomFactory::getData () const
{
  return _sh_ptr_data_target->getData();
}

arma::mat BaselearnerCustomFactory::instantiateData (const arma::mat& newdata) const
{
  Rcpp::NumericMatrix out = _instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}


// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCppFactory::BaselearnerCustomCppFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, std::shared_ptr<data::Data> data_target, SEXP instantiateDataFun,
  SEXP trainFun, SEXP predictFun)
  : BaselearnerFactory   ( blearner_type, data_source ),
    _sh_ptr_data_target  ( data_target ),
    _instantiateDataFun  ( instantiateDataFun ),
    _trainFun            ( trainFun ),
    _predictFun          ( predictFun )
{
  _sh_ptr_data_target->setDataIdentifier(data_source->getDataIdentifier());
  _sh_ptr_data_target->setData(instantiateData(data_source->getData()));
}

std::shared_ptr<blearner::Baselearner> BaselearnerCustomCppFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCustomCpp>(_blearner_type, _sh_ptr_data_target,
    _instantiateDataFun, _trainFun, _predictFun);
}

arma::mat BaselearnerCustomCppFactory::getData () const
{
  return _sh_ptr_data_target->getData();
}

arma::mat BaselearnerCustomCppFactory::instantiateData (const arma::mat& newdata) const
{
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (_instantiateDataFun);
  instantiateDataFunPtr instantiateDataFun0 = *myTempInstantiation;

  return instantiateDataFun0(newdata);
}

} // namespace blearnerfactory
