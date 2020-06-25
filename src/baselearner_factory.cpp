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

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type) : _blearner_type ( blearner_type ) {}

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type, std::shared_ptr<data::Data> data_source)
  : _blearner_type      ( blearner_type ),
    _sh_ptr_data_source ( data_source )
{ }

std::string BaselearnerFactory::getDataIdentifier () const
{
  if (_sh_ptr_data_source.use_count() == 0) {
    throw std::logic_error("Data source is not initialized");
  }
  return _sh_ptr_data_source->getDataIdentifier();
}
std::string BaselearnerFactory::getBaselearnerType() const { return _blearner_type; }

/// Destructor
BaselearnerFactory::~BaselearnerFactory () {}

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomialFactory::BaselearnerPolynomialFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, const unsigned int degree, const bool intercept)
  : BaselearnerFactory ( blearner_type, data_source ),
    _degree            ( degree ),
    _intercept         ( intercept )
{
  arma::mat   temp_data_mat;
  arma::mat   temp_xtx;
  std::string cache_type;

  if (data_source->getData().n_cols == 1) {
    temp_data_mat = arma::pow(data_source->getDenseData(), _degree);
    arma::mat temp_mat(1, 2, arma::fill::zeros);

    if (_intercept) {
      temp_mat(0,0) = arma::as_scalar(arma::mean(temp_data_mat));
    }
    temp_mat(0,1) = arma::as_scalar(arma::sum(arma::pow(temp_data_mat - temp_mat(0,0), 2)));
    temp_xtx      = temp_mat;
    cache_type    = "identity";
  } else {
    temp_data_mat = instantiateData(data_source->getDenseData());
    temp_xtx      = temp_data_mat.t() * temp_data_mat;
    cache_type    = "inverse";
  }
  _sh_ptr_data_target = std::make_shared<data::InMemoryData>(data_source->getDataIdentifier(), temp_data_mat);
  _sh_ptr_data_target->setCache(cache_type, temp_xtx);
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
      return instantiateData(arma::pow(_sh_ptr_data_target->getDenseData(), 1/(double)_degree));
    } else {
      return _sh_ptr_data_target->getDenseData();
    }
  } else {
    return _sh_ptr_data_target->getDenseData();
  }
}

arma::mat BaselearnerPolynomialFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return BaselearnerPolynomialFactory::getData() * param;
}

arma::mat BaselearnerPolynomialFactory::calculateLinearPredictor (const arma::mat& param, const std::shared_ptr<data::Data>& newdata) const
{
  arma::mat temp = newdata->getDenseData();
  return this->instantiateData(temp) * param;
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
  const double penalty, const double df, const unsigned int differences, const bool use_sparse_matrices, const unsigned int bin_root,
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
    ::Rf_error( "c++ exception (unknown reason) in constructor of BaselearnerPSplineFactory" );
  }
  const arma::mat knots       = splines::createKnots(data_source->getDenseData(), n_knots, degree);
  const arma::mat penalty_mat = splines::penaltyMat(n_knots + (degree + 1), differences);

  if (bin_root == 0) { // don't use binning
    _sh_ptr_psdata = std::make_shared<data::PSplineData>(data_source->getDataIdentifier(), degree, knots, penalty_mat);
  } else {             // use binning
    arma::colvec bins = binning::binVectorCustom(data_source->getData(), bin_root);
    _sh_ptr_psdata    = std::make_shared<data::PSplineData>(data_source->getDataIdentifier(), degree, knots, penalty_mat, bin_root, data_source->getDenseData(), bins);
    data_source->setDenseData(bins);
  }
  arma::mat     temp_xtx;
  arma::sp_mat  temp      = splines::createSparseSplineBasis (data_source->getDenseData(), degree, knots).t();

  _sh_ptr_psdata->setSparseData(temp);

  if (_sh_ptr_psdata->usesBinning()) {
    arma::vec temp_weight(1, arma::fill::ones);
    temp_xtx = binning::binnedSparseMatMult(_sh_ptr_psdata->getSparseData(), _sh_ptr_psdata->getBinningIndex(), temp_weight);
  } else {
    temp_xtx = _sh_ptr_psdata->getSparseData() * _sh_ptr_psdata->getSparseData().t();
  }
  double used_penalty;
  if (df > 0) {
    used_penalty = dro::demmlerReinsch(temp_xtx, penalty_mat, df);
  } else {
    used_penalty = penalty;
  }
  _sh_ptr_psdata->setCache(cache_type, temp_xtx + used_penalty * penalty_mat);
}

arma::mat BaselearnerPSplineFactory::instantiateData (const arma::mat& newdata) const
{
  arma::mat temp = _sh_ptr_psdata->filterKnotRange(newdata);
  return splines::createSplineBasis (temp, _sh_ptr_psdata->getDegree(), _sh_ptr_psdata->getKnots());
}

arma::mat BaselearnerPSplineFactory::getData () const
{
  return _sh_ptr_psdata->getDenseData();
}

arma::mat BaselearnerPSplineFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return (param.t() * _sh_ptr_psdata->getSparseData()).t();
}

arma::mat BaselearnerPSplineFactory::calculateLinearPredictor (const arma::mat& param, const std::shared_ptr<data::Data>& newdata) const
{
  arma::mat temp = newdata->getDenseData();
  return this->instantiateData(temp) * param;
}


std::shared_ptr<blearner::Baselearner> BaselearnerPSplineFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPSpline>(_blearner_type, _sh_ptr_psdata);
}


// BaselearnerCategoricalRidgeFactory:
// -------------------------------------------

BaselearnerCategoricalRidgeFactory::BaselearnerCategoricalRidgeFactory (const std::string blearner_type,
  std::shared_ptr<data::CategoricalData>& cdata_source)
  : BaselearnerFactory ( blearner_type ),
    _sh_ptr_cdata      ( cdata_source )
{
  // TODO: Throw exception if data object is not categorical!
  _sh_ptr_cdata->initRidgeData();
}

BaselearnerCategoricalRidgeFactory::BaselearnerCategoricalRidgeFactory (const std::string blearner_type,
  std::shared_ptr<data::CategoricalData>& cdata_source, const double df)
  : BaselearnerFactory ( blearner_type ),
    _sh_ptr_cdata      ( cdata_source )
{
  // TODO: Throw exception if data object is not categorical!
  _sh_ptr_cdata->initRidgeData(df);
}

arma::mat BaselearnerCategoricalRidgeFactory::instantiateData (const arma::mat& newdata) const
{
  throw std::logic_error("Categorical base-learner do not instantiate data!");
  return arma::mat(1, 1, arma::fill::zeros);
}

arma::mat BaselearnerCategoricalRidgeFactory::getData () const
{
  return _sh_ptr_cdata->getData();
}

arma::mat BaselearnerCategoricalRidgeFactory::calculateLinearPredictor (const arma::mat& param) const
{
  arma::urowvec classes = _sh_ptr_cdata->getClasses();
  arma::mat out(classes.n_rows, param.n_cols, arma::fill::zeros);
  for (unsigned int i = 0; i < classes.n_rows; i++) {
    out.row(i) = param.row(classes(i));
  }
  return out;
}

arma::mat BaselearnerCategoricalRidgeFactory::calculateLinearPredictor (const arma::mat& param, const std::shared_ptr<data::Data>& newdata) const
{
  std::vector<std::string> classes = std::static_pointer_cast<data::CategoricalDataRaw>(newdata)->getRawData();
  return _sh_ptr_cdata->dictionaryInsert(classes, param);
}

std::shared_ptr<blearner::Baselearner> BaselearnerCategoricalRidgeFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCategoricalRidge>(_blearner_type, _sh_ptr_cdata);
}

std::string BaselearnerCategoricalRidgeFactory::getDataIdentifier () const { return _sh_ptr_cdata->getDataIdentifier(); }

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
    idx = helper::binaryToIndex(data_source->getSparseData());
  } else {
    idx = helper::binaryToIndex(data_source->getData());
  }
  _sh_ptr_bcdata = std::make_shared<data::CategoricalBinaryData> (data_source->getDataIdentifier(), idx);
}

std::shared_ptr<blearner::Baselearner> BaselearnerCategoricalBinaryFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCategoricalBinary>(_blearner_type, _sh_ptr_bcdata);
}

arma::mat BaselearnerCategoricalBinaryFactory::getData () const
{
  return helper::predictBinaryIndex(_sh_ptr_bcdata->getIndex(), 1);
}

arma::mat BaselearnerCategoricalBinaryFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return helper::predictBinaryIndex(_sh_ptr_bcdata->getIndex(), arma::as_scalar(param));
}

arma::mat BaselearnerCategoricalBinaryFactory::calculateLinearPredictor (const arma::mat& param, const std::shared_ptr<data::Data>& newdata) const
{
  // FIX FIX FIX, when having the appropriate structure for categorical data fix this to
  // set elements of the raw data == class to parameter:
  return param;
}


arma::mat BaselearnerCategoricalBinaryFactory::instantiateData (const arma::mat& newdata) const
{
  return newdata;
}


// BaselearnerCustom:
// -----------------------

BaselearnerCustomFactory::BaselearnerCustomFactory (const std::string blearner_type,
  const std::shared_ptr<data::Data> data_source, const Rcpp::Function instantiateDataFun,
  const Rcpp::Function trainFun, const Rcpp::Function predictFun, const Rcpp::Function extractParameter)
  : BaselearnerFactory   ( blearner_type, data_source ),
    _sh_ptr_data_target  ( std::make_shared<data::InMemoryData>(data_source->getDataIdentifier()) ),
    _instantiateDataFun  ( instantiateDataFun ),
    _trainFun            ( trainFun ),
    _predictFun          ( predictFun ),
    _extractParameter    ( extractParameter )
{
  _sh_ptr_data_target->setDenseData(instantiateData(data_source->getData()));
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

arma::mat BaselearnerCustomFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_data_target->getData() * param;
}

arma::mat BaselearnerCustomFactory::calculateLinearPredictor (const arma::mat& param, const std::shared_ptr<data::Data>& newdata) const
{
  arma::mat temp = newdata->getDenseData();
  return this->instantiateData(temp) * param;
}

arma::mat BaselearnerCustomFactory::instantiateData (const arma::mat& newdata) const
{
  Rcpp::NumericMatrix out = _instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}


// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCppFactory::BaselearnerCustomCppFactory (const std::string blearner_type,
  const std::shared_ptr<data::Data> data_source, const SEXP instantiateDataFun, const SEXP trainFun,
  const SEXP predictFun)
  : BaselearnerFactory   ( blearner_type, data_source ),
    _sh_ptr_data_target  ( std::make_shared<data::InMemoryData>(data_source->getDataIdentifier()) ),
    _instantiateDataFun  ( instantiateDataFun ),
    _trainFun            ( trainFun ),
    _predictFun          ( predictFun )
{
  _sh_ptr_data_target->setDenseData(instantiateData(data_source->getData()));
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

arma::mat BaselearnerCustomCppFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_data_target->getData() * param;
}

arma::mat BaselearnerCustomCppFactory::calculateLinearPredictor (const arma::mat& param, const std::shared_ptr<data::Data>& newdata) const
{
  arma::mat temp = newdata->getDenseData();
  return this->instantiateData(temp) * param;
}


arma::mat BaselearnerCustomCppFactory::instantiateData (const arma::mat& newdata) const
{
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (_instantiateDataFun);
  instantiateDataFunPtr instantiateDataFun0 = *myTempInstantiation;

  return instantiateDataFun0(newdata);
}

} // namespace blearnerfactory
