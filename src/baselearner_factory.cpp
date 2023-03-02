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


std::shared_ptr<BaselearnerFactory> jsonToBaselearnerFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
{
  std::shared_ptr<BaselearnerFactory> blf;

  if (j["Class"] == "BaselearnerPolynomialFactory") {
    blf = std::make_shared<BaselearnerPolynomialFactory>(j, mdsource, mdinit);
  }
  if (j["Class"] == "BaselearnerPSplineFactory") {
    blf = std::make_shared<BaselearnerPSplineFactory>(j, mdsource, mdinit);
  }
  if (j["Class"] == "BaselearnerTensorFactory") {
    blf = std::make_shared<BaselearnerTensorFactory>(j, mdsource, mdinit);
  }
  if (j["Class"] == "BaselearnerCenteredFactory") {
    blf = std::make_shared<BaselearnerCenteredFactory>(j, mdsource, mdinit);
  }
  if (j["Class"] == "BaselearnerCategoricalRidgeFactory") {
    blf = std::make_shared<BaselearnerCategoricalRidgeFactory>(j, mdsource, mdinit);
  }
  if (j["Class"] == "BaselearnerCategoricalBinaryFactory") {
    blf = std::make_shared<BaselearnerCategoricalBinaryFactory>(j, mdsource, mdinit);
  }
  if (blf == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return blf;

}


// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type)
  : _blearner_type ( blearner_type )
{ }

BaselearnerFactory::BaselearnerFactory (const std::string blearner_type, const std::shared_ptr<data::Data>& data_source)
  : _blearner_type      ( blearner_type ),
    _sh_ptr_data_source ( data_source )
{ }

BaselearnerFactory::BaselearnerFactory (const json& j, const mdata& mdat)
  : _blearner_type      ( j["_blearner_type"].get<std::string>() ),
    _sh_ptr_data_source ( data::extractDataFromMap(j["id_data_source"].get<std::string>(), mdat) )
{ }

std::vector<std::string> BaselearnerFactory::getDataIdentifier () const
{
  if (_sh_ptr_data_source.use_count() == 0) {
    throw std::logic_error("Data source is not initialized or 'getDataIdentifier()' is not implemented");
  }
  std::vector<std::string> out;
  out.push_back(_sh_ptr_data_source->getDataIdentifier());
  return out;
}

sdata BaselearnerFactory::getDataSource () const
{
  return _sh_ptr_data_source;
}

std::string BaselearnerFactory::getBaselearnerType() const
{
  return _blearner_type;
}

json BaselearnerFactory::baseToJson (const std::string cln) const
{
  json j = {
    {"Class", cln},

    {"_blearner_type", _blearner_type},
    {"id_data_source", _sh_ptr_data_source->getDataIdentifier()}
  };

  return j;
}

json BaselearnerFactory::dataSourceToJson () const
{
  json j;
  j[_sh_ptr_data_source->getDataIdentifier()] = _sh_ptr_data_source->toJson();

  return j;
}

std::vector<sdata> BaselearnerFactory::getVecDataSource () const
{
  std::vector<sdata> out;
  out.push_back(_sh_ptr_data_source);
  return out;
}

/// Destructor
BaselearnerFactory::~BaselearnerFactory () {}

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomialFactory::BaselearnerPolynomialFactory (const std::string blearner_type,
  std::shared_ptr<data::Data> data_source, const unsigned int degree, const bool intercept,
  const unsigned int bin_root, const double df, const double penalty)
  : BaselearnerFactory::BaselearnerFactory ( blearner_type, data_source )
{
  _attributes->df            = df;
  _attributes->penalty       = penalty;
  _attributes->degree        = degree;
  _attributes->use_intercept = intercept;
  _attributes->bin_root      = bin_root;

  _sh_ptr_bindata = init::initPolynomialData(data_source, _attributes);
  _attributes->penalty_mat = arma::diagmat( arma::vec(_sh_ptr_bindata->getNCols(), arma::fill::ones) );

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
    if (df > 0) {
      try {
        _attributes->penalty = dro::demmlerReinsch(temp_xtx, _attributes->penalty_mat, df);
      } catch (const std::exception& e) {
        std::string msg = "From constructor of BaselearnerPolynomialFactory with data '" + _sh_ptr_bindata->getDataIdentifier() +
          "': Try to run demmlerDemmlerReinsch" + std::string(e.what());
        throw msg;
      }
    }
    _sh_ptr_bindata->setCache("cholesky", temp_xtx + _attributes->penalty * _attributes->penalty_mat);
  }
  _attributes->bin_root = 0;
}

BaselearnerPolynomialFactory::BaselearnerPolynomialFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
  : BaselearnerFactory::BaselearnerFactory ( j, mdsource ),
    _sh_ptr_bindata ( std::static_pointer_cast<data::BinnedData>(data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdinit)) ),
    _attributes     ( std::make_shared<init::PolynomialAttributes>(j["_attributes"]) )
{ }

sdata BaselearnerPolynomialFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
  return init::initPolynomialData(newdata, _attributes);
}

bool BaselearnerPolynomialFactory::usesSparse () const
{
  return false;
}

sdata BaselearnerPolynomialFactory::getInstantiatedData () const
{
  return _sh_ptr_bindata;
}

arma::mat BaselearnerPolynomialFactory::getData () const
{
  return _sh_ptr_bindata->getDenseData();
}

arma::vec BaselearnerPolynomialFactory::getDF () const
{
  return arma::vec(1, arma::fill::value(_attributes->df));
}

arma::vec BaselearnerPolynomialFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::value(_attributes->penalty));
}

arma::mat BaselearnerPolynomialFactory::getPenaltyMat () const
{
  return _attributes->penalty_mat;
}

std::string BaselearnerPolynomialFactory::getBaseModelName () const
{
  return std::string("polynomial");
}

std::string BaselearnerPolynomialFactory::getFactoryId () const
{
  return _sh_ptr_bindata->getDataIdentifier() + "_" + _blearner_type;
}

arma::mat BaselearnerPolynomialFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_bindata->getDenseData() * param;
}

arma::mat BaselearnerPolynomialFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerPolynomialFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  // For newdata, we just extract the sparse data because no binning is used!
  try {
    auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
    return init::initPolynomialData(newdata, _attributes)->getDenseData() * param;

  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerPolynomialFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPolynomial>(_blearner_type, _sh_ptr_bindata, _attributes);
}

json BaselearnerPolynomialFactory::toJson () const
{
  json j = BaselearnerFactory::baseToJson("BaselearnerPolynomialFactory");
  j["id_data_init"] = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();

  return j;
}

json BaselearnerPolynomialFactory::extractDataToJson (const bool save_source) const
{
  json j;
  std::string id_dat;

  if (save_source) {
    j = BaselearnerFactory::dataSourceToJson();
  } else {
    id_dat = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
    j[id_dat] = _sh_ptr_bindata->toJson();
  }
  return j;
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
  : BaselearnerFactory::BaselearnerFactory ( blearner_type, data_source )
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
  _attributes->penalty_mat = penalty_mat;;

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
  _sh_ptr_bindata->setCache(cache_type, temp_xtx + _attributes->penalty * _attributes->penalty_mat);

  // Set bin_root to zero for later creation of data for predictions. We don't want to
  // use binning there.
  _attributes->bin_root = 0;
}

BaselearnerPSplineFactory::BaselearnerPSplineFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
  : BaselearnerFactory::BaselearnerFactory ( j, mdsource ),
    _sh_ptr_bindata ( std::static_pointer_cast<data::BinnedData>(data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdinit)) ),
    _attributes     ( std::make_shared<init::PSplineAttributes>(j["_attributes"]) )
{ }

bool BaselearnerPSplineFactory::usesSparse () const
{
  return true;
}

sdata BaselearnerPSplineFactory::getInstantiatedData () const
{
  return _sh_ptr_bindata;
}

sdata BaselearnerPSplineFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
  auto attr_temp = _attributes;
  attr_temp->bin_root = 0;

  return init::initPSplineData(newdata, attr_temp);
}

arma::mat BaselearnerPSplineFactory::getData () const
{
  return arma::mat(_sh_ptr_bindata->getSparseData());
}

arma::vec BaselearnerPSplineFactory::getDF () const
{
  return arma::vec(1, arma::fill::value(_attributes->df));
}

arma::vec BaselearnerPSplineFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::value(_attributes->penalty));
}

arma::mat BaselearnerPSplineFactory::getPenaltyMat () const
{
  return _attributes->penalty_mat;
}

std::string BaselearnerPSplineFactory::getBaseModelName () const
{
  return std::string("pspline");
}

std::string BaselearnerPSplineFactory::getFactoryId () const
{
  return _sh_ptr_bindata->getDataIdentifier() + "_" + _blearner_type;
}

arma::mat BaselearnerPSplineFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return (param.t() * _sh_ptr_bindata->getSparseData()).t();
}

arma::mat BaselearnerPSplineFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerPSplineFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  try {
    auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
    return (param.t() * init::initPSplineData(newdata, _attributes)->getSparseData()).t();
  } catch (const char* msg) {
    throw msg;
  }
}

std::shared_ptr<blearner::Baselearner> BaselearnerPSplineFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerPSpline>(_blearner_type, std::static_pointer_cast<data::BinnedData>(_sh_ptr_bindata));
}

json BaselearnerPSplineFactory::toJson () const
{
  json j = BaselearnerFactory::baseToJson("BaselearnerPSplineFactory");
  j["id_data_init"] = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();

  return j;
}

json BaselearnerPSplineFactory::extractDataToJson (const bool save_source) const
{
  json j;
  std::string id_dat;

  if (save_source) {
    j = BaselearnerFactory::dataSourceToJson();
  } else {
    id_dat = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
    j[id_dat] = _sh_ptr_bindata->toJson();
  }
  return j;
}


// BaselearnerTensorFactory:
// ------------------------------------------------

BaselearnerTensorFactory::BaselearnerTensorFactory (const std::string& blearner_type,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner1,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner2, const bool isotrop)
  : BaselearnerFactory::BaselearnerFactory (blearner_type, std::make_shared<data::InMemoryData>(
        blearner1->getDataSource()->getDataIdentifier() + "_" +
        blearner2->getDataSource()->getDataIdentifier())),
    _blearner1 ( blearner1 ),
    _blearner2 ( blearner2 ),
    _isotrop  ( isotrop )
{
  // Get data from both learners
  arma::mat bl1_penmat = _blearner1->getPenaltyMat();
  arma::mat bl2_penmat = _blearner2->getPenaltyMat();

  // PASS instantiated data of factories to initTensorData
  _sh_ptr_data = init::initTensorData(_blearner1->getInstantiatedData(), _blearner2->getInstantiatedData());

  arma::mat temp_xtx;
  if (_sh_ptr_data->usesSparseMatrix()) {
   temp_xtx = _sh_ptr_data->getSparseData() * _sh_ptr_data->getSparseData().t();
  } else {
   temp_xtx = _sh_ptr_data->getDenseData().t() * _sh_ptr_data->getDenseData();
  }
  // Calculate penalty matrix:
  double df = arma::as_scalar(_blearner1->getDF() * _blearner2->getDF());
  arma::mat penalty_mat;
  if (_isotrop) {
    penalty_mat = tensors::penaltySumKronecker(bl1_penmat, bl2_penmat);

    try {
      _attributes->penalty = dro::demmlerReinsch(temp_xtx, penalty_mat, df);
    } catch (const std::exception& e) {
      std::string msg = "From constructor of BaselearnerTensorFactory with data '" + _sh_ptr_data->getDataIdentifier() +
        "': Try to run demmlerDemmlerReinsch" + std::string(e.what());
      throw msg;
    }

    penalty_mat = penalty_mat * _attributes->penalty;
  } else {
    penalty_mat = tensors::penaltySumKronecker(bl1_penmat * arma::as_scalar(_blearner1->getPenalty()), bl2_penmat * arma::as_scalar(_blearner2->getPenalty()));
    _attributes->penalty = arma::as_scalar(_blearner1->getPenalty() * _blearner2->getPenalty());
  }
  _sh_ptr_data->setCache("cholesky", temp_xtx + penalty_mat);
}

BaselearnerTensorFactory::BaselearnerTensorFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
  : BaselearnerFactory::BaselearnerFactory ( j, mdsource ),
    _sh_ptr_data ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdinit) ),
    _attributes  ( std::make_shared<init::TensorAttributes>(j["_attributes"])),
    _blearner1   ( jsonToBaselearnerFactory(j["_blearner1"], mdsource, mdinit) ),
    _blearner2   ( jsonToBaselearnerFactory(j["_blearner2"], mdsource, mdinit) ),
    _isotrop     ( j["_isotrop"].get<bool>() )
{ }

bool BaselearnerTensorFactory::usesSparse () const
{
  return _blearner1->usesSparse() | _blearner2->usesSparse();
}

sdata BaselearnerTensorFactory::getInstantiatedData () const
{
  return _sh_ptr_data;
}

sdata BaselearnerTensorFactory::instantiateData (const mdata& data_map) const
{
  //auto data1 = data::extractDataFromMap(_blearner1->getInstantiatedData(), data_map);
  //auto data2 = data::extractDataFromMap(_blearner2->getInstantiatedData(), data_map);
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

arma::vec BaselearnerTensorFactory::getDF () const
{
  arma::vec df;
  if (_isotrop) {
    df = arma::vec(1, arma::fill::value(arma::as_scalar(_blearner1->getDF()) * arma::as_scalar(_blearner2->getDF())));
  } else {
    df = {
      arma::as_scalar(_blearner1->getDF()),
      arma::as_scalar(_blearner2->getDF()) };
  }
  return df;
}

arma::vec BaselearnerTensorFactory::getPenalty () const
{
  arma::vec pen;
  if (_isotrop) {
      pen = arma::vec(1, arma::fill::value(_attributes->penalty));
  } else {
    pen = {
      arma::as_scalar(_blearner1->getPenalty()),
      arma::as_scalar(_blearner2->getPenalty()) };
  }
  return pen;
}

arma::mat BaselearnerTensorFactory::getPenaltyMat () const
{
  arma::mat bl1_penmat = _blearner1->getPenaltyMat();
  arma::mat bl2_penmat = _blearner2->getPenaltyMat();

  double bl1_pen = arma::as_scalar(_blearner1->getPenalty());
  double bl2_pen = arma::as_scalar(_blearner2->getPenalty());

  arma::mat penalty_mat;
  if (_isotrop) {
    penalty_mat = tensors::penaltySumKronecker(bl1_penmat, bl2_penmat);
  } else {
    penalty_mat = tensors::penaltySumKronecker(bl1_pen * bl1_penmat, bl2_pen * bl2_penmat);
  }
  return penalty_mat;
}

std::string BaselearnerTensorFactory::getBaseModelName() const
{
  return std::string("tensor");
}

std::string BaselearnerTensorFactory::getFactoryId () const
{
  return _sh_ptr_data->getDataIdentifier() + "_" + _blearner_type;
}

std::shared_ptr<blearnerfactory::BaselearnerFactory> BaselearnerTensorFactory::getBl1 () const
{
  return _blearner1;
}

std::shared_ptr<blearnerfactory::BaselearnerFactory> BaselearnerTensorFactory::getBl2 () const
{
  return _blearner2;
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

std::vector<std::string> BaselearnerTensorFactory::getDataIdentifier () const
{
  std::vector<std::string> bld1 = _blearner1->getDataIdentifier();
  std::vector<std::string> bld2 = _blearner2->getDataIdentifier();
  for (unsigned int i = 0; i < bld2.size(); i++) {
    bld1.push_back(bld2[i]);
  }
  return bld1;
}

std::vector<sdata> BaselearnerTensorFactory::getVecDataSource () const
{
  auto dvec1 = _blearner1->getVecDataSource();
  auto dvec2 = _blearner2->getVecDataSource();

  for (auto& it : dvec2) {
    dvec1.push_back(it);
  }
  return dvec1;
}

std::shared_ptr<blearner::Baselearner> BaselearnerTensorFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerTensor>(_blearner_type, _sh_ptr_data);
}

json BaselearnerTensorFactory::toJson () const
{
  json j = BaselearnerFactory::baseToJson("BaselearnerTensorFactory");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();
  j["_blearner1"] = _blearner1->toJson();
  j["_blearner2"] = _blearner2->toJson();
  j["_isotrop"]  = _isotrop;

  return j;
}

json BaselearnerTensorFactory::extractDataToJson (const bool save_source) const
{
  json j;
  json jsub1;
  json jsub2;
  std::string id_dat;

  if (save_source) {
    // Save source data of all factories:
    j = BaselearnerFactory::dataSourceToJson();  // X
    jsub1 = _blearner1->extractDataToJson(true); // Y
    jsub2 = _blearner2->extractDataToJson(true); // Z
  } else {
    // Save init data of the factories, e.g. the design matrix Z = X x Y.
    id_dat = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
    j[id_dat] = _sh_ptr_data->toJson(); // X

    // Also save the init data design matrices X and Z of the underlying factories:
    jsub1 = _blearner1->extractDataToJson(false); // X
    jsub2 = _blearner2->extractDataToJson(false); // Z

  }
  // Unroll to bring X, Y, and Z to the same level:
  for (auto& it : jsub1.items()) {
    j[it.key()] = it.value();
  }
  for (auto& it : jsub2.items()) {
    j[it.key()] = it.value();
  }
  return j;
}


// BaselearnerCenterFactory:
// ------------------------------------------------

BaselearnerCenteredFactory::BaselearnerCenteredFactory (const std::string& blearner_type,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner1,
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner2)
  : BaselearnerFactory::BaselearnerFactory (blearner_type, std::make_shared<data::InMemoryData>(
        blearner1->getDataSource()->getDataIdentifier() + "_" +
        blearner2->getDataSource()->getDataIdentifier())),
    _blearner1 ( blearner1 ),
    _blearner2 ( blearner2 )
{
  auto bldat1 = _blearner1->getInstantiatedData();
  auto bldat2 = _blearner2->getInstantiatedData();

  if (bldat1->usesBinning() != bldat2->usesBinning()) {
    std::string msg = "Binning is just possible if applied to both base learners.";
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

  arma::mat pen = _attributes->rotation.t() * _blearner1->getPenaltyMat() * _attributes->rotation;

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
  _sh_ptr_bindata->setCache(mcache.first, temp_xtx);
}

BaselearnerCenteredFactory::BaselearnerCenteredFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
  : BaselearnerFactory::BaselearnerFactory ( j, mdsource ),
    _sh_ptr_bindata ( std::static_pointer_cast<data::BinnedData>(data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdinit)) ),
    _blearner1      ( jsonToBaselearnerFactory(j["_blearner1"], mdsource, mdinit) ),
    _blearner2      ( jsonToBaselearnerFactory(j["_blearner2"], mdsource, mdinit) ),
    _attributes     ( std::make_shared<init::CenteredAttributes>(j["_attributes"]) )
{ }


bool BaselearnerCenteredFactory::usesSparse () const
{
  return false;
}

sdata BaselearnerCenteredFactory::getInstantiatedData () const
{
  return _sh_ptr_bindata;
}

sdata BaselearnerCenteredFactory::instantiateData (const mdata& data_map) const
{
  sdata newdata = _blearner1->instantiateData(data_map);
  return init::initCenteredData(newdata, _attributes);
}

arma::mat BaselearnerCenteredFactory::getData () const
{
  return _sh_ptr_bindata->getDenseData();
}

arma::vec BaselearnerCenteredFactory::getDF () const
{
  double df1 = arma::as_scalar(_blearner1->getDF());
  double df2 = arma::as_scalar(_blearner2->getDF());
  return arma::vec(1, arma::fill::value(df1 - df2));
}

arma::vec BaselearnerCenteredFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::value(arma::as_scalar(_blearner1->getPenalty())));
}

arma::mat BaselearnerCenteredFactory::getPenaltyMat () const
{
  return _attributes->rotation.t() * _blearner1->getPenaltyMat() * _attributes->rotation;
}

std::string BaselearnerCenteredFactory::getBaseModelName() const
{
  return std::string("centered");
}

std::string BaselearnerCenteredFactory::getFactoryId () const
{
  return _sh_ptr_bindata->getDataIdentifier() + "_" + _blearner_type;
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
  return std::make_shared<blearner::BaselearnerCentered>(_blearner_type, _sh_ptr_bindata);
}

arma::mat BaselearnerCenteredFactory::getRotation() const
{
  return _attributes->rotation;
}

std::vector<std::string> BaselearnerCenteredFactory::getDataIdentifier () const
{
  std::vector<std::string> bld1 = _blearner1->getDataIdentifier();
  std::vector<std::string> bld2 = _blearner2->getDataIdentifier();
  for (unsigned int i = 0; i < bld2.size(); i++) {
    bld1.push_back(bld2[i]);
  }
  return bld1;
}

std::vector<sdata> BaselearnerCenteredFactory::getVecDataSource () const
{
  auto dvec1 = _blearner1->getVecDataSource();
  auto dvec2 = _blearner2->getVecDataSource();

  for (auto& it : dvec2) {
    dvec1.push_back(it);
  }
  return dvec1;
}


json BaselearnerCenteredFactory::toJson () const
{
  json j = BaselearnerFactory::baseToJson("BaselearnerCenteredFactory");
  j["id_data_init"] = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();
  j["_blearner1"] = _blearner1->toJson();
  j["_blearner2"] = _blearner2->toJson();

  return j;
}

json BaselearnerCenteredFactory::extractDataToJson (const bool save_source) const
{
  json j;
  json jsub1;
  json jsub2;
  std::string id_dat;

  if (save_source) {
    // Save source data of all factories:
    j = BaselearnerFactory::dataSourceToJson();  // X
    jsub1 = _blearner1->extractDataToJson(true); // Y
    jsub2 = _blearner2->extractDataToJson(true); // Z
  } else {
    // Save init data of the factories, e.g. the design matrix Z = X / Y.
    id_dat = _sh_ptr_bindata->getDataIdentifier() + "." + _blearner_type;
    j[id_dat] = _sh_ptr_bindata->toJson(); // X

    // Also save the init data design matrices X and Z of the underlying factories:
    jsub1 = _blearner1->extractDataToJson(false); // X
    jsub2 = _blearner2->extractDataToJson(false); // Z

  }
  // Unroll to bring X, Y, and Z to the same level:
  for (auto& it : jsub1.items()) {
    j[it.key()] = it.value();
  }
  for (auto& it : jsub2.items()) {
    j[it.key()] = it.value();
  }
  return j;
}


// BaselearnerCategoricalRidgeFactory:
// -------------------------------------------


BaselearnerCategoricalRidgeFactory::BaselearnerCategoricalRidgeFactory (const std::string blearner_type,
  std::shared_ptr<data::CategoricalDataRaw>& cdata_source, const double df, const double penalty)
  : BaselearnerFactory::BaselearnerFactory ( blearner_type, cdata_source )
{
  _attributes->df      = df;
  _attributes->penalty = penalty;

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


  _attributes->penalty_mat = arma::diagmat(arma::vec(_attributes->dictionary.size(), arma::fill::ones));
  arma::vec xtx_diag(arma::diagvec((_sh_ptr_data->getSparseData() * _sh_ptr_data->getSparseData().t())));

  if (df > 0) {
    _attributes->penalty = dro::demmlerReinschRidge(xtx_diag, df);
  }
  arma::vec temp_XtX_inv = 1 / (xtx_diag + _attributes->penalty);
  _sh_ptr_data->setCache("identity", temp_XtX_inv);
}

BaselearnerCategoricalRidgeFactory::BaselearnerCategoricalRidgeFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
  : BaselearnerFactory::BaselearnerFactory ( j, mdsource ),
    _sh_ptr_data ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdinit) ),
    _attributes  ( std::make_shared<init::RidgeAttributes>(j["_attributes"]) )
{ }

bool BaselearnerCategoricalRidgeFactory::usesSparse () const
{
  return true;
}

sdata BaselearnerCategoricalRidgeFactory::getInstantiatedData () const
{
  return _sh_ptr_data;
}

sdata BaselearnerCategoricalRidgeFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = data::extractDataFromMap(this->_sh_ptr_data, data_map);
  auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
  return init::initRidgeData(cnewdata, _attributes);
}

arma::mat BaselearnerCategoricalRidgeFactory::getData () const
{
  return arma::mat(_sh_ptr_data->getSparseData());
}

arma::vec BaselearnerCategoricalRidgeFactory::getDF () const
{
  return arma::vec(1, arma::fill::value(_attributes->df));
}

arma::vec BaselearnerCategoricalRidgeFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::value(_attributes->penalty));
}

arma::mat BaselearnerCategoricalRidgeFactory::getPenaltyMat () const
{
  return _attributes->penalty_mat;
}

std::string BaselearnerCategoricalRidgeFactory::getBaseModelName () const
{
  return std::string("cridge");
}

std::string BaselearnerCategoricalRidgeFactory::getFactoryId () const
{
  return _sh_ptr_data->getDataIdentifier() + "_" + _blearner_type;
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
    auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
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

std::vector<std::string> BaselearnerCategoricalRidgeFactory::getDataIdentifier () const
{
  std::vector<std::string> out;
  out.push_back(_sh_ptr_data->getDataIdentifier());
  return out;
}

std::map<std::string, unsigned int> BaselearnerCategoricalRidgeFactory::getDictionary () const
{
  return _attributes->dictionary;
}

json BaselearnerCategoricalRidgeFactory::toJson () const
{
  json j = BaselearnerFactory::baseToJson("BaselearnerCategoricalRidgeFactory");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();

  return j;
}

json BaselearnerCategoricalRidgeFactory::extractDataToJson (const bool save_source) const
{
  json j;
  std::string id_dat;

  if (save_source) {
    j = BaselearnerFactory::dataSourceToJson();
  } else {
    id_dat = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
    j[id_dat] = _sh_ptr_data->toJson();
  }
  return j;
}


// BaselearnerCategoricalBinary:
// ----------------------------------

BaselearnerCategoricalBinaryFactory::BaselearnerCategoricalBinaryFactory (const std::string blearner_type, const std::string cls,
  const std::shared_ptr<data::CategoricalDataRaw>& cdata_source)
  : BaselearnerFactory ( blearner_type, cdata_source )
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

BaselearnerCategoricalBinaryFactory::BaselearnerCategoricalBinaryFactory (const json& j, const mdata& mdsource, const mdata& mdinit)
  : BaselearnerFactory::BaselearnerFactory ( j, mdsource ),
    _sh_ptr_data ( data::extractDataFromMap(j["id_data_init"].get<std::string>(), mdinit) ),
    _attributes  ( std::make_shared<init::BinaryAttributes>(j["_attributes"]) )
{ }


bool BaselearnerCategoricalBinaryFactory::usesSparse () const
{
  return true;
}

sdata BaselearnerCategoricalBinaryFactory::getInstantiatedData () const
{
  return _sh_ptr_data;
}

std::shared_ptr<blearner::Baselearner> BaselearnerCategoricalBinaryFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCategoricalBinary>(_blearner_type, _sh_ptr_data);
}

arma::mat BaselearnerCategoricalBinaryFactory::getData () const
{
  return arma::mat(_sh_ptr_data->getSparseData());
}

arma::vec BaselearnerCategoricalBinaryFactory::getDF () const
{
  return arma::vec(1, arma::fill::ones);
}

arma::vec BaselearnerCategoricalBinaryFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::zeros);
}

arma::mat BaselearnerCategoricalBinaryFactory::getPenaltyMat () const
{
  return arma::mat(1, 1, arma::fill::ones);
}

std::string BaselearnerCategoricalBinaryFactory::getBaseModelName () const
{
  return std::string("cbinary");
}

std::string BaselearnerCategoricalBinaryFactory::getFactoryId () const
{
  return _sh_ptr_data->getDataIdentifier() + "_" + _blearner_type;
}

arma::mat BaselearnerCategoricalBinaryFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return (param.t() * _sh_ptr_data->getSparseData()).t();
}

arma::mat BaselearnerCategoricalBinaryFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  helper::debugPrint("From 'BaselearnerCategoricalBinaryFactory::calculateLinearPredictor' for feature " + this->_sh_ptr_data_source->getDataIdentifier());
  try {

    auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
    auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
    return (param.t() * init::initBinaryData(cnewdata, _attributes)->getSparseData()).t();

  } catch (const char* msg) {
    throw msg;
  }
}

sdata BaselearnerCategoricalBinaryFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
  auto cnewdata = std::static_pointer_cast<data::CategoricalDataRaw>(newdata);
  return init::initBinaryData(cnewdata, _attributes);

  //throw std::logic_error("Categorical base-learner do not instantiate data!");
  //return arma::mat(1, 1, arma::fill::zeros);
}

std::vector<std::string> BaselearnerCategoricalBinaryFactory::getDataIdentifier () const
{
  std::vector<std::string> out;
  out.push_back(_sh_ptr_data_source->getDataIdentifier());
  return out;
}

json BaselearnerCategoricalBinaryFactory::toJson () const
{
  json j = BaselearnerFactory::baseToJson("BaselearnerCategoricalBinaryFactory");
  j["id_data_init"] = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
  j["_attributes"] = _attributes->toJson();

  return j;
}

json BaselearnerCategoricalBinaryFactory::extractDataToJson (const bool save_source) const
{
  json j;
  std::string id_dat;

  if (save_source) {
    j = BaselearnerFactory::dataSourceToJson();
  } else {
    id_dat = _sh_ptr_data->getDataIdentifier() + "." + _blearner_type;
    j[id_dat] = _sh_ptr_data->toJson();
  }
  return j;
}


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
  _is_initialized = true;
}

bool BaselearnerCustomFactory::usesSparse () const
{
  return false;
}

sdata BaselearnerCustomFactory::getInstantiatedData () const
{
  return _sh_ptr_data;
}

std::shared_ptr<blearner::Baselearner> BaselearnerCustomFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCustom>(_blearner_type, _sh_ptr_data, _instantiateDataFun,
    _trainFun, _predictFun, _extractParameter);
}

arma::mat BaselearnerCustomFactory::getData () const
{
  return _sh_ptr_data->getDenseData();
}

arma::vec BaselearnerCustomFactory::getDF () const
{
  return arma::vec(1, arma::fill::value(_sh_ptr_data->getNCols()));
}

arma::vec BaselearnerCustomFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::zeros);
}

arma::mat BaselearnerCustomFactory::getPenaltyMat () const
{
  return arma::diagmat( arma::vec(_sh_ptr_data->getNCols(), arma::fill::ones) );
}

std::string BaselearnerCustomFactory::getBaseModelName () const
{
  return std::string("custom");
}

std::string BaselearnerCustomFactory::getFactoryId () const
{
  return _sh_ptr_data->getDataIdentifier() + "_" + _blearner_type;
}

arma::mat BaselearnerCustomFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_data->getDenseData() * param;
}

arma::mat BaselearnerCustomFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  try {
    auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
    return init::initCustomData(newdata, _instantiateDataFun)->getDenseData() * param;
  } catch (const char* msg) {
    throw msg;
  }
}

sdata BaselearnerCustomFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = data::extractDataFromMap(this->_sh_ptr_data, data_map);
  return init::initCustomData(newdata, _instantiateDataFun);
}

json BaselearnerCustomFactory::toJson () const
{
  throw std::logic_error("Cannot save a custom factory to JSON.");
  json j = BaselearnerFactory::baseToJson("BaselearnerCustomFactory");
  return j;
}

json BaselearnerCustomFactory::extractDataToJson (const bool save_source) const
{
  throw std::logic_error("Cannot save a custom factory to JSON.");
  json j = BaselearnerFactory::dataSourceToJson();
  return j;
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

  _is_initialized = true;
  //_sh_ptr_data_target->setDenseData(instantiateData(data_source->getData()));
}

bool BaselearnerCustomCppFactory::usesSparse () const
{
  return false;
}

sdata BaselearnerCustomCppFactory::getInstantiatedData () const
{
  return _sh_ptr_data;
}

std::shared_ptr<blearner::Baselearner> BaselearnerCustomCppFactory::createBaselearner ()
{
  return std::make_shared<blearner::BaselearnerCustomCpp>(_blearner_type, _sh_ptr_data, _attributes);
}

arma::mat BaselearnerCustomCppFactory::getData () const
{
  return _sh_ptr_data->getDenseData();
}

arma::vec BaselearnerCustomCppFactory::getDF () const
{
  return arma::vec(1, arma::fill::value(_sh_ptr_data->getNCols()));
}

arma::vec BaselearnerCustomCppFactory::getPenalty () const
{
  return arma::vec(1, arma::fill::zeros);
}

arma::mat BaselearnerCustomCppFactory::getPenaltyMat () const
{
  return arma::diagmat( arma::vec(_sh_ptr_data->getNCols(), arma::fill::ones) );
}

std::string BaselearnerCustomCppFactory::getBaseModelName () const
{
  return std::string("customcpp");
}

std::string BaselearnerCustomCppFactory::getFactoryId () const
{
  return _sh_ptr_data->getDataIdentifier() + "_" + _blearner_type;
}

arma::mat BaselearnerCustomCppFactory::calculateLinearPredictor (const arma::mat& param) const
{
  return _sh_ptr_data->getDenseData() * param;
}

arma::mat BaselearnerCustomCppFactory::calculateLinearPredictor (const arma::mat& param, const mdata& data_map) const
{
  try {
    auto newdata = data::extractDataFromMap(this->_sh_ptr_data, data_map);
    return init::initCustomCppData(newdata, _attributes)->getDenseData() * param;
  } catch (const char* msg) {
    throw msg;
  }
}

sdata BaselearnerCustomCppFactory::instantiateData (const mdata& data_map) const
{
  auto newdata = data::extractDataFromMap(this->_sh_ptr_data_source, data_map);
  return init::initCustomCppData(newdata, _attributes);
}

json BaselearnerCustomCppFactory::toJson () const
{
  throw std::logic_error("Cannot save a custom factory to JSON.");
  json j = BaselearnerFactory::baseToJson("BaselearnerCustomCppFactory");
  return j;
}

json BaselearnerCustomCppFactory::extractDataToJson (const bool save_source) const
{
  throw std::logic_error("Cannot save a custom factory to JSON.");
  json j = BaselearnerFactory::dataSourceToJson();
  return j;
}


} // namespace blearnerfactory
