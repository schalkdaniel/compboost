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
// it under the terms of the LGPL-3 License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// LGPL-3 License for more details. You should have received a copy of
// the license along with compboost.
//
// =========================================================================== #

#include "data.h"

namespace data
{

sdata jsonToData (const json& j)
{
  std::shared_ptr<Data> d;

  if (j["Class"] == "InMemoryData") {
    d = std::make_shared<InMemoryData>(j);
  }
  if (j["Class"] == "BinnedData") {
    d = std::make_shared<BinnedData>(j);
  }
  if (j["Class"] == "CategoricalDataRaw") {
    d = std::make_shared<CategoricalDataRaw>(j);
  }
  if (d == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return d;
}

mdata jsonToDataMap (const json& j)
{
  std::map<std::string, std::shared_ptr<Data>> mdat;
  for (auto& it : j.items()) {
    mdat[it.key()] = jsonToData(it.value());
  }
  return mdat;
}

json dataMapToJson (const mdata& mdat, const bool rm_data)
{
  json j;
  std::string id_dat;
  sdata sh_ptr_data;
  for (auto& it : mdat) {
    sh_ptr_data = it.second;
    id_dat = sh_ptr_data->getDataIdentifier();
    j[id_dat] = sh_ptr_data->toJson(rm_data);
  }
  return j;
}


sdata extractDataFromMap (const std::string did, const mdata& mdat)
{
  auto it_data = mdat.find(did);
  if (it_data == mdat.end()) {
    std::string msg = "Cannot find data '" + did + "' in data map.";
    throw std::logic_error(msg);
  }
  sdata dout = it_data->second;
  return dout;
}


sdata extractDataFromMap (const sdata& sh_ptr_data, const mdata& mdat)
{
  std::string data_id = sh_ptr_data->getDataIdentifier();
  return extractDataFromMap(data_id, mdat);
}



Data::Data (const std::string data_identifier, const std::string type)
  : _type            ( type ),
    _data_identifier ( data_identifier )
{ }

Data::Data (const std::string data_identifier, const std::string type,
  const std::vector<double>& minmax)
  : _type            ( type ),
    _data_identifier ( data_identifier ),
    _minmax          ( minmax )
{ }

Data::Data (const std::string data_identifier, const std::string type, const arma::mat& data_mat)
  : _type            ( type ),
    _data_identifier ( data_identifier ),
    _data_mat        ( data_mat ),
    _minmax          ( std::vector<double>{data_mat.min(), data_mat.max()} )
{ }

Data::Data (const std::string data_identifier, const std::string type, const arma::mat& data_mat,
  const std::vector<double>& minmax)
  : _type            ( type ),
    _data_identifier ( data_identifier ),
    _data_mat        ( data_mat ),
    _minmax          ( minmax )
{ }

Data::Data (const std::string data_identifier, const std::string type, const arma::sp_mat& sparse_data_mat)
  : _type ( type ),
    _data_identifier ( data_identifier ),
    _use_sparse      ( true ),
    _sparse_data_mat ( sparse_data_mat ),
    _minmax          ( std::vector<double>{sparse_data_mat.min(), sparse_data_mat.max()} )
{ }

Data::Data (const std::string data_identifier, const std::string type, const arma::sp_mat& sparse_data_mat,
  const std::vector<double>& minmax)
  : _type ( type ),
    _data_identifier ( data_identifier ),
    _use_sparse      ( true ),
    _sparse_data_mat ( sparse_data_mat ),
    _minmax          ( minmax )
{ }

Data::Data (const json& j)
  : _type            ( j["_type"].get<std::string>() ),
    _data_identifier ( j["_data_identifier"].get<std::string>() ),
    _mat_cache       (std::make_pair(
      j["_mat_cache"]["type"].get<std::string>(),
      saver::jsonToArmaMat(j["_mat_cache"]["mat"])
    )),
    _use_sparse      ( j["_use_sparse"].get<bool>() ),
    _use_binning     ( j["_use_binning"].get<bool>() ),
    _data_mat        ( saver::jsonToArmaMat(j["_data_mat"]) ),
    _bin_idx         ( saver::jsonToArmaUvec(j["_bin_idx"]) ),
    _sparse_data_mat ( saver::jsonToArmaSpMat(j["_sparse_data_mat"]) ),
    _minmax          ( j["_minmax"].get<std::vector<double>>() )
{ }

void Data::setCacheCholesky (const arma::mat& xtx)
{
  try {
    _mat_cache = std::make_pair("cholesky", arma::chol(xtx));
  } catch (const std::exception& e) {
    std::string msg = "From data object '" + _data_identifier + "': Trying cholesky decomposition of XtX." + std::string(e.what());
    throw std::runtime_error(msg);
  }
}

void Data::setCacheInverse (const arma::mat& xtx)
{
  try {
    _mat_cache = std::make_pair("inverse", arma::inv(xtx));
  } catch (const std::exception& e) {
    std::string msg = "From data object '" + _data_identifier + "': Trying to calculate inverse of XtX." +  std::string(e.what());
    throw msg;
  }
}

void Data::setCacheIdentity (const arma::mat& X)
{
  _mat_cache = std::make_pair("identity", X);
}

void Data::setCacheCustom (const std::string ctype, const arma::mat& X)
{
  _mat_cache = std::make_pair(ctype, X);
}

void Data::setDenseData  (const arma::mat& X)    { _use_sparse = false; _data_mat = X; }
void Data::setSparseData (const arma::sp_mat& X) { _use_sparse = true; _sparse_data_mat = X; }

void Data::setCache (const std::string cache_type, const arma::mat& xtx)
{
  std::vector<std::string> choices{ "cholesky", "inverse", "identity" };
  helper::assertChoice(cache_type, choices);

  if (cache_type == "cholesky") setCacheCholesky(xtx);
  if (cache_type == "inverse")  setCacheInverse(xtx);
  if (cache_type == "identity") setCacheIdentity(xtx);
  if (cache_type == "custom")   setCacheCustom(cache_type, xtx);
}

void Data::setIndexVector (const arma::uvec& idx)
{
  _use_binning = true;
  _bin_idx = idx;
}

void Data::setMinMax (const std::vector<double>& minmax)
{
  _minmax = minmax;
}

std::string Data::getType () const
{
  return _type;
}

std::string Data::getDataIdentifier () const
{
  return _data_identifier;
}

std::pair<std::string, arma::mat> Data::getCache () const
{
  return _mat_cache;
}

std::string Data::getCacheType () const
{
  return _mat_cache.first;
}

arma::mat   Data::getCacheMat  () const
{
  return _mat_cache.second;
}

arma::mat Data::getDenseData () const
{
  if (_use_sparse) {
    arma::mat out(_sparse_data_mat);
    return out;
  } else {
    return _data_mat;
  }
}

arma::sp_mat        Data::getSparseData    () const { return _sparse_data_mat; }
arma::uvec          Data::getBinningIndex  () const { return _bin_idx; }
bool                Data::usesSparseMatrix () const { return _use_sparse; }
bool                Data::usesBinning      () const { return _use_binning; }
std::vector<double> Data::getMinMax        () const { return _minmax; }

json Data::baseToJson (const std::string cln, const bool rm_data) const
{
  arma::mat zero(1, 1, arma::fill::zeros);
  arma::uvec one(1, arma::fill::ones);
  json jdata, jdata_sparse, jmcache, jbin_idx;
  if (rm_data) {
    jdata = saver::armaMatToJson(zero);
    jdata_sparse = saver::armaSpMatToJson(arma::sp_mat(zero));
    jmcache = saver::armaMatToJson(zero);
    jbin_idx = saver::armaUvecToJson(one);
  } else {
    jdata = saver::armaMatToJson(_data_mat);
    jdata_sparse = saver::armaSpMatToJson(_sparse_data_mat);
    jmcache = saver::armaMatToJson(_mat_cache.second);
    jbin_idx = saver::armaUvecToJson(_bin_idx);
  }

  json j = {
    { "Class", cln },
    { "_type", _type },
    { "_data_identifier", _data_identifier },
    { "_mat_cache", {
      { "type", _mat_cache.first },
      { "mat",  jmcache }
    }},
    { "_use_sparse",      _use_sparse },
    { "_use_binning",     _use_binning },
    { "_data_mat",        jdata },
    { "_bin_idx",         jbin_idx },
    { "_sparse_data_mat", jdata_sparse },
    { "_minmax",          _minmax }
  };
  return j;
}

// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

InMemoryData::InMemoryData (const std::string data_identifier)
  : Data::Data ( std::string(data_identifier), std::string("in_memory") )
{ }

InMemoryData::InMemoryData (const std::string data_identifier, const arma::mat& raw_data)
  : Data::Data ( data_identifier, std::string("in_memory"), raw_data )
{ }

InMemoryData::InMemoryData (const std::string data_identifier, const arma::sp_mat& raw_sp_data)
  : Data::Data ( data_identifier, std::string("in_memory"), raw_sp_data )
{ }

InMemoryData::InMemoryData (const json& j)
  : Data::Data ( j )
{ }

arma::mat InMemoryData::getData () const
{
  return Data::getDenseData();
}

unsigned int InMemoryData::getNObs () const
{
  if (_use_sparse) {
    return _sparse_data_mat.n_cols;
  } else {
    return _data_mat.n_rows;
  }
}

unsigned int InMemoryData::getNCols () const
{
  if (_use_sparse) {
    return _sparse_data_mat.n_rows;
  } else {
    return _data_mat.n_cols;
  }
}

json InMemoryData::toJson (const bool rm_data) const
{
  return Data::baseToJson("InMemoryData", rm_data);
}

InMemoryData::~InMemoryData () {}

// BinnedData:
// ------------------------------
BinnedData::BinnedData (const std::string data_identifier)
  : Data::Data ( std::string(data_identifier), std::string("binned") )
{ }

BinnedData::BinnedData (const std::string data_identifier, const unsigned int bin_root, const arma::vec& x, const arma::vec& x_bins)
  : Data::Data ( data_identifier, std::string("binned") ),
    _bin_root  ( bin_root )
{
  _use_binning = bin_root > 0;
  _bin_idx     = binning::calculateIndexVectorLin(x, x_bins);
}

BinnedData::BinnedData (const json& j)
  : Data::Data ( j ),
    _bin_root  ( j["_bin_root"] )
{ }

arma::mat BinnedData::getData () const
{
  return Data::getDenseData();
}

unsigned int BinnedData::getNObs () const
{
  if (_use_sparse) {
    return _sparse_data_mat.n_cols;
  } else {
    return _data_mat.n_rows;
  }
}

unsigned int BinnedData::getNCols () const
{
  if (_use_sparse) {
    return _sparse_data_mat.n_rows;
  } else {
    return _data_mat.n_cols;
  }
}

json BinnedData::toJson (const bool rm_data) const
{
  json j = Data::baseToJson("BinnedData", rm_data);
  j["_bin_root"] = _bin_root;

  return j;
}


// CategoricalDataRaw:
// ---------------------------------

CategoricalDataRaw::CategoricalDataRaw (const std::string data_identifier, const std::vector<std::string>& raw_data)
  : Data::Data ( std::string(data_identifier), std::string("categorical") ),
    _raw_data  ( raw_data )
{ }

CategoricalDataRaw::CategoricalDataRaw (const json& j)
  : Data::Data ( json(j) ),
    _raw_data  ( j["_raw_data"].get<std::vector<std::string>>() )
{ }

arma::mat CategoricalDataRaw::getData () const {
  throw std::logic_error("Raw categorical data does not contain a numerical representation, call '$getRawData()' instead");
   //1 x 1 dummy  matrix:
  return Data::getDenseData();
}

std::vector<std::string> CategoricalDataRaw::getRawData () const
{
  return _raw_data;
}

unsigned int CategoricalDataRaw::getNObs () const
{
  return _raw_data.size();
}

unsigned int CategoricalDataRaw::getNCols () const
{
  return 1;
}

json CategoricalDataRaw::toJson (const bool rm_data) const
{
  json j = Data::baseToJson("CategoricalDataRaw", rm_data);
  if (rm_data) {
    std::vector<std::string> empty;
    empty.push_back("<REMOVED>");
    j["_raw_data"] = empty;
  } else {
    j["_raw_data"] = _raw_data;
  }

  return j;
}


} // namespace data
