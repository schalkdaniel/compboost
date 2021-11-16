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

std::shared_ptr<data::Data> extractDataFromMap (const std::shared_ptr<data::Data>& sh_ptr_data,
  const std::map<std::string, std::shared_ptr<data::Data>>& data_map)
{
  std::string data_id = sh_ptr_data->getDataIdentifier();
  auto it_data = data_map.find(data_id);
  if (it_data == data_map.end()) {
    throw "Cannot find data " + data_id + " in data map. Using 0 as linear predictor.";
  }
  return it_data->second;
}

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type) : _blearner_type ( blearner_type ) {}

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type, const std::shared_ptr<data::Data>& data_source)
  : _blearner_type      ( blearner_type ),
    _sh_ptr_data_source ( data_source )
{ }


std::string BaselearnerFactory::getDataIdentifier () const
{
  if (_sh_ptr_data_source.use_count() == 0) {
    throw std::logic_error("Data source is not initialized or 'getDataIdentifier()' is not implemented");
  }
  return _sh_ptr_data_source->getDataIdentifier();
}
std::string BaselearnerFactory::getBaselearnerType() const { return _blearner_type; }
//
/// Destructor
BaselearnerFactory::~BaselearnerFactory () {}

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomialFactory::BaselearnerPolynomialFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, const unsigned int degree, const bool intercept,
  const unsigned int bin_root)
  : BaselearnerFactory ( blearner_type, data_source )
{
  _attributes->degree        = degree;
  _attributes->use_intercept = intercept;
  _attributes->bin_root      = bin_root;

  _sh_ptr_bindata = init::initPolynomialData(data_source, _attributes);

  arma::mat temp_xtx;
  if (_attributes->degree == 1) {
    arma::mat mraw = data_source->getDenseData();
    arma::mat temp_mat(1, 2, arma::fill::zeros);

    if (_attributes->use_intercept) {
      temp_mat(0,0) = arma::as_scalar(arma::mean(mraw));
    }
    temp_mat(0,1) = arma::as_scalar(arma::sum(arma::pow(mraw - temp_mat(0,0), 2)));
    temp_xtx      = temp_mat;
    _sh_ptr_bindata->setCache("identity", temp_xtx);
  } else {

    if (_sh_ptr_bindata->usesBinning()) {
      arma::vec temp_weight(1, arma::fill::ones);
      temp_xtx = binning::binnedMatMult(_sh_ptr_bindata->getDenseData(), _sh_ptr_bindata->getBinningIndex(), temp_weight);
    } else {
      temp_xtx = _sh_ptr_bindata->getDenseData().t() * _sh_ptr_bindata->getDenseData();
    }
    _sh_ptr_bindata->setCache("inverse", temp_xtx);
  }
  _attributes->bin_root = 0;

  //arma::mat   temp_data_mat;
  //arma::mat   temp_xtx;
  //std::string cache_type;

  //if (data_source->getData().n_cols == 1) {
    //temp_data_mat = arma::pow(data_source->getDenseData(), _degree);
    //arma::mat temp_mat(1, 2, arma::fill::zeros);

    //if (_intercept) {
      //temp_mat(0,0) = arma::as_scalar(arma::mean(temp_data_mat));
    //}
    //temp_mat(0,1) = arma::as_scalar(arma::sum(arma::pow(temp_data_mat - temp_mat(0,0), 2)));
    //temp_xtx      = temp_mat;
    //cache_type    = "identity";
  //} else {
    //temp_data_mat = instantiateData(data_source->getDenseData());
    //temp_xtx      = temp_data_mat.t() * temp_data_mat;
    //cache_type    = "inverse";
  //}
  //_sh_ptr_data_target = std::make_shared<data::InMemoryData>(data_source->getDataIdentifier(), temp_data_mat);
  //_sh_ptr_data_target->setCache(cache_type, temp_xtx);
}

//arma::mat BaselearnerPolynomialFactory::instantiateData (const arma::mat& newdata) const
sdata BaselearnerPolynomialFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
  return init::initPolynomialData(newdata, _attributes);


  //arma::mat temp = arma::pow(newdata, _degree);
  //if (_intercept) {
    //arma::mat temp_intercept(temp.n_rows, 1, arma::fill::ones);
    //temp = join_rows(temp_intercept, temp);
  //}
  //return temp;
}

bool BaselearnerPolynomialFactory::usesSparse () const { return false; }
sdata BaselearnerPolynomialFactory::getInstantiatedData () const { return _sh_ptr_bindata; }

arma::mat BaselearnerPolynomialFactory::getData () const
{
  return _sh_ptr_bindata->getDenseData();
}

arma::mat BaselearnerPolynomialFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_bindata->getDenseData() * param;

//  // Here we have a different handling than in predict(data) because of the possibility to use binning.
//  // It does not make sense to also include binning into the prediction of new points! Binning is just
//  // a method to fasten the fitting process.
//  if (_sh_ptr_bindata->usesBinning()) {
//    return binning::binnedSparsePrediction(_sh_ptr_bindata->getSparseData(), param, _sh_ptr_bindata->getBinningIndex());
//  } else {
//    // Trick to speed up things. Try to avoid transposing the sparse matrix. The
//    // original one (sh_ptr_data->sparse_data_mat * parameter) is about 4 or 5 times
//    // slower than that one:
//    return (param.t() * _sh_ptr_bindata->getSparseData()).t();
//  }
}

arma::mat BaselearnerPolynomialFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerPolynomialFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  // For newdata, we just extract the sparse data because no binning is used!
  try {
    auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
    return init::initPolynomialData(newdata, _attributes)->getDenseData() * param;

  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerPolynomialFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPolynomial>(_blearner_type, _sh_ptr_bindata, _attributes);//, _degree, _intercept);
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
  const std::shared_ptr<data::Data>& data_source, const unsigned int degree, const unsigned int n_knots,
  const double penalty, const double df, const unsigned int differences, const bool use_sparse_matrices,
  const unsigned int bin_root, const std::string cache_type)
  : BaselearnerFactory ( blearner_type, data_source )
{
  _attributes->degree      = degree;
  _attributes->n_knots     = n_knots;
  _attributes->penalty     = penalty;
  _attributes->df          = df;
  _attributes->differences = differences;
  _attributes->bin_root    = bin_root;
  _attributes->knots       = splines::createKnots(data_source->getDenseData(), n_knots, degree);

  _sh_ptr_bindata = init::initPSplineData(data_source, _attributes);

  const arma::mat penalty_mat = splines::penaltyMat(_attributes->n_knots + (_attributes->degree + 1), _attributes->differences);

  arma::mat temp_xtx;
  if (_sh_ptr_bindata->usesBinning()) {
    arma::vec temp_weight(1, arma::fill::ones);
    temp_xtx = binning::binnedSparseMatMult(_sh_ptr_bindata->getSparseData(), _sh_ptr_bindata->getBinningIndex(), temp_weight);
  } else {
    temp_xtx = _sh_ptr_bindata->getSparseData() * _sh_ptr_bindata->getSparseData().t();
  }
  if (df > 0) {
    try {
      _attributes->penalty = dro::demmlerReinsch(temp_xtx, penalty_mat, df);
    } catch (const std::exception& e) {
      std::string msg = "From constructor of BaselearnerPSplineFactory with data '" + _sh_ptr_bindata->getDataIdentifier() +
        "': Try to run demmlerDemmlerReinsch" + std::string(e.what());
      throw msg;
    }
  }

  _sh_ptr_bindata->setPenaltyMat(_attributes->penalty * penalty_mat);
  _sh_ptr_bindata->setCache(cache_type, temp_xtx + _attributes->penalty * penalty_mat);

  // Set bin_root to zero for later creation of data for predictions. We don't want to
  // use binning there.
  _attributes->bin_root = 0;
}

bool BaselearnerPSplineFactory::usesSparse () const { return true; }
sdata BaselearnerPSplineFactory::getInstantiatedData () const { return _sh_ptr_bindata; }

sdata BaselearnerPSplineFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
  auto attr_temp = _attributes;
  attr_temp->bin_root = 0;

  return init::initPSplineData(newdata, attr_temp);
  //arma::mat temp = _sh_ptr_bindata->filterKnotRange(newdata);
  //return splines::createSplineBasis (temp, _sh_ptr_psdata->getDegree(), _sh_ptr_psdata->getKnots());
}

arma::mat BaselearnerPSplineFactory::getData () const
{
  return arma::mat(_sh_ptr_bindata->getSparseData());
}

arma::mat BaselearnerPSplineFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return (param.t() * _sh_ptr_bindata->getSparseData()).t();
}

arma::mat BaselearnerPSplineFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerPSplineFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  try {
    auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
    return (param.t() * init::initPSplineData(newdata, _attributes)->getSparseData()).t();
  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerPSplineFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPSpline>(_blearner_type, std::static_pointer_cast<data::BinnedData>(_sh_ptr_bindata));
}


// BaselearnerTensorFactory:
// ------------------------------------------------

BaselearnerTensorFactory::BaselearnerTensorFactory (const std::string& blearner_type,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner1,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner2)
  : BaselearnerFactory (blearner_type, std::make_shared<data::InMemoryData>(blearner1->getDataIdentifier() + "_" + blearner2->getDataIdentifier())),
    _blearner1 ( blearner1 ),
    _blearner2 ( blearner2 )
{

  // Get data from both learners
  arma::mat bl1_pen = _blearner1->getInstantiatedData()->getPenaltyMat();
  arma::mat bl2_pen = _blearner2->getInstantiatedData()->getPenaltyMat();

  // TODO! Include lambda * pen:
  arma::mat penalty_mat = tensors::penaltySumKronecker(bl1_pen, bl2_pen);

  // PASS instantiated data of factories to initTensorData
  _sh_ptr_data = init::initTensorData(_blearner1->getInstantiatedData(), _blearner2->getInstantiatedData());

  arma::mat temp_xtx;
  if (_sh_ptr_data->usesSparseMatrix()) {
   temp_xtx = _sh_ptr_data->getSparseData() * _sh_ptr_data->getSparseData().t() + penalty_mat;
  } else {
   temp_xtx = _sh_ptr_data->getDenseData().t() * _sh_ptr_data->getDenseData() + penalty_mat;
  }
  _sh_ptr_data->setCache("cholesky", temp_xtx);
}

bool BaselearnerTensorFactory::usesSparse () const
{
  return _blearner1->usesSparse() | _blearner2->usesSparse();
}
sdata BaselearnerTensorFactory::getInstantiatedData () const { return _sh_ptr_data; }

sdata BaselearnerTensorFactory::instantiateData (const mdata& data_map) const
{
  //auto data1 = extractDataFromMap(_blearner1->getInstantiatedData(), data_map);
  //auto data2 = extractDataFromMap(_blearner2->getInstantiatedData(), data_map);
  //
  // Instantiate data ... again ... FIX
  sdata newdata1 = _blearner1->instantiateData(data_map);
  sdata newdata2 = _blearner2->instantiateData(data_map);

  return init::initTensorData(newdata1, newdata2);
}

arma::mat BaselearnerTensorFactory::getData () const
{
  arma::mat out;
  if (_sh_ptr_data->usesSparseMatrix()) {
    out = _sh_ptr_data->getSparseData();
  } else  {
    out = _sh_ptr_data->getDenseData();
  }
  return out;
}

arma::mat BaselearnerTensorFactory::calculateLinearPredictor (const arma::mat& param) const
{
  if (_sh_ptr_data->usesSparseMatrix()) {
    return (param.t() * _sh_ptr_data->getSparseData()).t();
  } else  {
    return _sh_ptr_data->getDenseData() * param;
  }
}

arma::mat BaselearnerTensorFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  try {
    auto newdata = instantiateData(data_map);
    if (newdata->usesSparseMatrix()) {
      return (param.t() * newdata->getSparseData()).t();
    } else  {
      return newdata->getDenseData() * param;
    }
  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerTensorFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerTensor>(_blearner_type, _sh_ptr_data);
}



// BaselearnerCenterFactory:
// ------------------------------------------------

BaselearnerCenteredFactory::BaselearnerCenteredFactory (const std::string& blearner_type,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner1,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner2)
  : BaselearnerFactory (blearner_type, std::make_shared<data::InMemoryData>(blearner1->getDataIdentifier())),
    _blearner1 ( blearner1 ),
    _blearner2 ( blearner2 )
{
  auto bldat1 = _blearner1->getInstantiatedData();
  auto bldat2 = _blearner2->getInstantiatedData();

  if (bldat1->usesBinning() != bldat2->usesBinning()) {
    std::string msg = "Can just use centering binning or non-binning applied to both base learners.";
    throw msg;
  }

  arma::mat temp1;
  if (bldat1->usesSparseMatrix()) {
    temp1 = bldat1->getSparseData().t();
  } else {
    temp1 = bldat1->getDenseData();
  }
  arma::mat temp2;
  if (bldat2->usesSparseMatrix()) {
    temp2 = bldat2->getSparseData().t();
  } else {
    temp2 = bldat2->getDenseData();
  }
  // usesBinning has to be a function of the parent data class!
  bool uses_binning = bldat1->usesBinning();
  if (uses_binning) {
    _attributes->rotation = tensors::centerDesignMatrix(temp1, temp2, bldat1->getBinningIndex());
  } else {
    _attributes->rotation = tensors::centerDesignMatrix(temp1, temp2);
  }
  _sh_ptr_bindata = init::initCenteredData(bldat1, _attributes);

  if (uses_binning) {
    _sh_ptr_bindata->setIndexVector(bldat1->getBinningIndex());
  }

  arma::mat pen = _attributes->rotation.t() * bldat1->getPenaltyMat() * _attributes->rotation;
  _sh_ptr_bindata->setPenaltyMat(pen);

  auto mcache = bldat1->getCache();
  arma::mat temp_xtx;
  if (mcache.first == "cholesky") {
    temp_xtx = _attributes->rotation.t() * mcache.second;
    temp_xtx = temp_xtx * temp_xtx.t();
  }
  if (mcache.first == "inverse") {
    temp_xtx = _attributes->rotation.t() * arma::inv(mcache.second) * _attributes->rotation;
  }
  if ((mcache.first != "cholesky") && (mcache.first != "inverse")) {
    throw "Can just handle cholesky or inverse cache types.";
  }

  //arma::mat temp_xtx;
  //if (uses_binning) {
    //arma::vec wtmp(1, arma::fill::ones);
    //temp_xtx = _attributes->rotation.t() * *
    //temp_xtx = binning::binnedMatMult(_sh_ptr_bindata->getDenseData(), _sh_ptr_bindata->getBinningIndex(), wtmp);
  //} else {
    //temp_xtx = _sh_ptr_bindata->getDenseData().t() * _sh_ptr_bindata->getDenseData() + pen;
  //}
  _sh_ptr_bindata->setCache(mcache.first, temp_xtx);
}

bool BaselearnerCenteredFactory::usesSparse () const { return false; }
sdata BaselearnerCenteredFactory::getInstantiatedData () const { return _sh_ptr_bindata; }

sdata BaselearnerCenteredFactory::instantiateData (const mdata& data_map) const
{
  sdata newdata = _blearner1->instantiateData(data_map);
  return init::initCenteredData(newdata, _attributes);
}

arma::mat BaselearnerCenteredFactory::getData () const
{
  return _sh_ptr_bindata->getDenseData();
}

arma::mat BaselearnerCenteredFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_bindata->getDenseData() * param;
}

arma::mat BaselearnerCenteredFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  try {
    auto newdata = instantiateData(data_map);
    return newdata->getDenseData() * param;
  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerCenteredFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPolynomial>(_blearner_type, _sh_ptr_bindata);
}

arma::mat BaselearnerCenteredFactory::getRotation() const { return _attributes->rotation; }


// BaselearnerCategoricalRidgeFactory:
// -------------------------------------------

//BaselearnerCategoricalRidgeFactory::BaselearnerCategoricalRidgeFactory (const std::string blearner_type,
  //std::shared_ptr<data::CategoricalData>& cdata_source)
  //: BaselearnerFactory ( blearner_type ),
    //_sh_ptr_cdata      ( cdata_source )
//{
  //_sh_ptr_cdata->initRidgeData();
//}

BaselearnerCategoricalRidgeFactory::BaselearnerCategoricalRidgeFactory (const std::string blearner_type,
  std::shared_ptr<data::CategoricalDataRaw>& cdata_source, const double df)
  : BaselearnerFactory ( blearner_type, cdata_source )
{
  auto          chr_classes = cdata_source->getRawData();
  std::string   chr_class;
  unsigned int  int_class;
  for (unsigned int i = 0; i < chr_classes.size(); i++) {
    chr_class = chr_classes.at(i);
    auto it = _attributes->dictionary.find(chr_class);
    // Add class into dictionary if not already there:
    if (it == _attributes->dictionary.end()) {
      int_class = _attributes->dictionary.size();
      _attributes->dictionary.insert(std::pair<std::string, unsigned int>(chr_class, int_class));
    }
  }
  _sh_ptr_data = init::initRidgeData(cdata_source, _attributes);

  // Calculate and set penalty
  unsigned int nrows = chr_classes.size();


  arma::mat penalty_mat_dummy = arma::diagmat(arma::vec(_attributes->dictionary.size(), arma::fill::ones));
  arma::mat xtx(_sh_ptr_data->getSparseData() * _sh_ptr_data->getSparseData().t());

  double penalty = 0;
  if (df > 0) {
    // penalty = nrows / df - 1;
    penalty = dro::demmlerReinsch(xtx, penalty_mat_dummy, df);
  }
  _sh_ptr_data->setPenaltyMat(penalty * penalty_mat_dummy);
  arma::vec temp_XtX_inv = 1 / (arma::diagvec(xtx) + penalty);
  _sh_ptr_data->setCache("identity", temp_XtX_inv);
}
bool BaselearnerCategoricalRidgeFactory::usesSparse () const { return true; }
sdata BaselearnerCategoricalRidgeFactory::getInstantiatedData () const { return _sh_ptr_data; }

sdata BaselearnerCategoricalRidgeFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = extractDataFromMap(this->_sh_ptr_data, data_map);
  auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
  return init::initRidgeData(cnewdata, _attributes);

  //throw std::logic_error("Categorical base-learner do not instantiate data!");
  //return arma::mat(1, 1, arma::fill::zeros);
}

arma::mat BaselearnerCategoricalRidgeFactory::getData () const
{
  return arma::mat(_sh_ptr_data->getSparseData());
}

arma::mat BaselearnerCategoricalRidgeFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return (param.t() * _sh_ptr_data->getSparseData()).t();
}

arma::mat BaselearnerCategoricalRidgeFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerCategoricalRidgeFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  try {
    helper::debugPrint("| > Extract data object from map");
    auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
    helper::debugPrint("| > Cast to raw categorical data object");
    auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
    helper::debugPrint("| > Initialize new data:");
    auto init_cnewdata = init::initRidgeData(cnewdata, _attributes);
    helper::debugPrint("| > Calling for sparse data and calculate linear predictor");
    arma::mat temp = (param.t() * init_cnewdata->getSparseData()).t();
    return temp;
  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerCategoricalRidgeFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCategoricalRidge>(_blearner_type, _sh_ptr_data);
}

std::string BaselearnerCategoricalRidgeFactory::getDataIdentifier () const { return _sh_ptr_data->getDataIdentifier(); }

std::map<std::string, unsigned int> BaselearnerCategoricalRidgeFactory::getDictionary () const { return _attributes->dictionary; }

// BaselearnerCategoricalBinary:
// ----------------------------------

BaselearnerCategoricalBinaryFactory::BaselearnerCategoricalBinaryFactory (const std::string blearner_type, const std::string cls,
  const std::shared_ptr<data::CategoricalDataRaw>& cdata_source)
  : BaselearnerFactory ( blearner_type )
    //_class             ( cls ),
    //_sh_ptr_cdata      ( cdata_source ),
    //_sh_ptr_bcdata     ( std::make_shared<data::CategoricalBinaryData>(cdata_source->getDataIdentifier(), cls, cdata_source) )
{
  _attributes->cls = cls;
  _sh_ptr_data = init::initBinaryData(cdata_source, _attributes);
  arma::mat xtx_inv(1,1);
  xtx_inv(0,0) = 1 / (double)(_sh_ptr_data->getSparseData().n_nonzero);
  _sh_ptr_data->setCache("identity", xtx_inv);
}

bool BaselearnerCategoricalBinaryFactory::usesSparse () const { return true; }
sdata BaselearnerCategoricalBinaryFactory::getInstantiatedData () const { return _sh_ptr_data; }

std::shared_ptr<blearner::Baselearner> BaselearnerCategoricalBinaryFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCategoricalBinary>(_blearner_type, _sh_ptr_data);
}

arma::mat BaselearnerCategoricalBinaryFactory::getData () const
{
  return arma::mat(_sh_ptr_data->getSparseData());
}

arma::mat BaselearnerCategoricalBinaryFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return (param.t() * _sh_ptr_data->getSparseData()).t();
}

arma::mat BaselearnerCategoricalBinaryFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerCategoricalBinaryFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  try {
    auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
    auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
    return (param.t() * init::initBinaryData(cnewdata, _attributes)->getSparseData()).t();

    //unsigned int nobs = classes.size();
    //arma::mat out(nobs, 1, arma::fill::zeros);

    //for (unsigned int i = 0; i < nobs; i++) {
      //if (classes.at(i) == _sh_ptr_bcdata->getCategory())
        //out(i) = arma::as_scalar(param);
    //}
    //return out;
  } catch (const char* msg) {
    throw msg;
  }
}

sdata BaselearnerCategoricalBinaryFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = extractDataFromMap(this->_sh_ptr_data, data_map);
  auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
  return init::initBinaryData(cnewdata, _attributes);

  //throw std::logic_error("Categorical base-learner do not instantiate data!");
  //return arma::mat(1, 1, arma::fill::zeros);
}

std::string BaselearnerCategoricalBinaryFactory::getDataIdentifier () const { return _sh_ptr_data->getDataIdentifier(); }


// BaselearnerCustom:
// -----------------------

BaselearnerCustomFactory::BaselearnerCustomFactory (const std::string blearner_type,
  const std::shared_ptr<data::Data> data_source, const Rcpp::Function instantiateDataFun,
  const Rcpp::Function trainFun, const Rcpp::Function predictFun, const Rcpp::Function extractParameter)
  : BaselearnerFactory   ( blearner_type, data_source ),
    //_sh_ptr_data_target  ( std::make_shared<data::InMemoryData>(data_source->getDataIdentifier()) ),
    _instantiateDataFun  ( instantiateDataFun ),
    _trainFun            ( trainFun ),
    _predictFun          ( predictFun ),
    _extractParameter    ( extractParameter )
{
  _sh_ptr_data = init::initCustomData(data_source, _instantiateDataFun);
}

bool BaselearnerCustomFactory::usesSparse () const { return false; }
sdata BaselearnerCustomFactory::getInstantiatedData () const { return _sh_ptr_data; }

std::shared_ptr<blearner::Baselearner> BaselearnerCustomFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCustom>(_blearner_type, _sh_ptr_data, _instantiateDataFun,
    _trainFun, _predictFun, _extractParameter);
}

arma::mat BaselearnerCustomFactory::getData () const
{
  return _sh_ptr_data->getDenseData();
}

arma::mat BaselearnerCustomFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_data->getDenseData() * param;
}

arma::mat BaselearnerCustomFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  try {
    auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
    return init::initCustomData(newdata, _instantiateDataFun)->getDenseData() * param;
  } catch (const char* msg) {
    throw msg;
  }
}

sdata BaselearnerCustomFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = extractDataFromMap(this->_sh_ptr_data, data_map);
  return init::initCustomData(newdata, _instantiateDataFun);

  //Rcpp::NumericMatrix out = _instantiateDataFun(newdata);
  //return Rcpp::as<arma::mat>(out);
}


// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCppFactory::BaselearnerCustomCppFactory (const std::string blearner_type,
  const std::shared_ptr<data::Data> data_source, SEXP instantiateDataFun0, SEXP trainFun0,
  SEXP predictFun0)
  : BaselearnerFactory   ( blearner_type, data_source )
    //_sh_ptr_data_target  ( std::make_shared<data::InMemoryData>(data_source->getDataIdentifier()) ),
    //_instantiateDataFun  ( instantiateDataFun ),
    //_trainFun            ( trainFun ),
    //_predictFun          ( predictFun )
{
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun0);
  //instantiateDataFun = *myTempInstantiation;

  Rcpp::XPtr<trainFunPtr> myTempTrain (trainFun0);
  //trainFun = *myTempTrain;

  Rcpp::XPtr<predictFunPtr> myTempPredict (predictFun0);
  //predictFun = *myTempPredict;

  _attributes->instantiateDataFun = *myTempInstantiation;
  _attributes->trainFun = *myTempTrain;
  _attributes->predictFun = *myTempPredict;

  _sh_ptr_data = init::initCustomCppData(data_source, _attributes);
  //_sh_ptr_data_target->setDenseData(instantiateData(data_source->getData()));
}

bool BaselearnerCustomCppFactory::usesSparse () const { return false; }
sdata BaselearnerCustomCppFactory::getInstantiatedData () const { return _sh_ptr_data; }

std::shared_ptr<blearner::Baselearner> BaselearnerCustomCppFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCustomCpp>(_blearner_type, _sh_ptr_data, _attributes);
}

arma::mat BaselearnerCustomCppFactory::getData () const
{
  return _sh_ptr_data->getDenseData();
}

arma::mat BaselearnerCustomCppFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_data->getDenseData() * param;
}

arma::mat BaselearnerCustomCppFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  try {
    auto newdata = extractDataFromMap(this->_sh_ptr_data, data_map);
    return init::initCustomCppData(newdata, _attributes)->getDenseData() * param;
  } catch (const char* msg) {
    throw msg;
  }
}


sdata BaselearnerCustomCppFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = extractDataFromMap(this->_sh_ptr_data_source, data_map);
  return init::initCustomCppData(newdata, _attributes);

  //return instantiateDataFun0(newdata);
}

} // namespace blearnerfactory
