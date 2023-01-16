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

#include "data.h"

namespace data
{

Data::Data (const std::string data_identifier)
  : _data_identifier ( data_identifier )
{ }

Data::Data (const std::string data_identifier, const arma::mat& data_mat)
  : _data_identifier ( data_identifier ),
    _data_mat        ( data_mat )
{ }

Data::Data (const std::string data_identifier, const arma::sp_mat& sparse_data_mat)
  : _data_identifier ( data_identifier ),
    _use_sparse      ( true ),
    _sparse_data_mat ( sparse_data_mat )
{ }

Data::Data (const json& j)
  : _data_identifier ( j["_data_identifier"] ),
    _mat_cache       (std::make_pair(
      j["_mat_cache"]["type"],
      saver::jsonToArmaMat(j["_mat_cache"]["mat"])
    )),
    _use_sparse      ( j["_use_sparse"] ),
    _use_binning     ( j["_use_binning"] ),
    _data_mat        ( saver::jsonToArmaMat(j["_data_mat"]) ),
    _bin_idx         ( saver::jsonToArmaUvec(j["_bin_idx"]) ),
    _sparse_data_mat ( saver::jsonToArmaSpMat(j["_sparse_data_mat"]) )
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

std::string Data::getDataIdentifier () const { return _data_identifier; }
std::pair<std::string, arma::mat> Data::getCache () const { return _mat_cache; }
std::string Data::getCacheType () const { return _mat_cache.first; }
arma::mat   Data::getCacheMat  () const { return _mat_cache.second; }

arma::mat Data::getDenseData () const
{
  if (_use_sparse) {
    arma::mat out(_sparse_data_mat);
    return out;
  } else {
    return _data_mat;
  }
}

arma::sp_mat Data::getSparseData    () const { return _sparse_data_mat; }
arma::uvec   Data::getBinningIndex  () const { return _bin_idx; }
bool         Data::usesSparseMatrix () const { return _use_sparse; }
bool         Data::usesBinning      () const { return _use_binning; }

json Data::baseToJson (const std::string cln) const
{
  json j = {
    {"Class", cln},

    {"_data_identifier", _data_identifier},
    {"_mat_cache", {
      {"type", _mat_cache.first},
      {"mat", saver::armaMatToJson(_mat_cache.second)}
    }},
    {"_use_sparse",      _use_sparse},
    {"_use_binning",     _use_binning},
    {"_data_mat",        saver::armaMatToJson(_data_mat)},
    {"_bin_idx",         saver::armaUvecToJson(_bin_idx)},
    {"_sparse_data_mat", saver::armaSpMatToJson(_sparse_data_mat)}
  };
  return j;
}

// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

InMemoryData::InMemoryData (const std::string data_identifier)
  : Data::Data ( std::string(data_identifier) )
{ }

InMemoryData::InMemoryData (const std::string data_identifier, const arma::mat& raw_data)
  : Data::Data ( data_identifier, raw_data )
{ }

InMemoryData::InMemoryData (const std::string data_identifier, const arma::sp_mat& raw_sp_data)
  : Data::Data ( data_identifier, raw_sp_data )
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

json InMemoryData::toJson () const
{
  return Data::baseToJson("InMemoryData");
}

InMemoryData::~InMemoryData () {}

// BinnedData:
// ------------------------------
BinnedData::BinnedData (const std::string data_identifier)
  : Data::Data ( std::string(data_identifier) )
{ }

BinnedData::BinnedData (const std::string data_identifier, const unsigned int bin_root, const arma::vec& x, const arma::vec& x_bins)
  : Data::Data ( data_identifier),
    _bin_root  ( bin_root )
{
  _use_binning = bin_root > 0;
  _bin_idx     = binning::calculateIndexVector(x, x_bins);
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

json BinnedData::toJson () const
{
  json j = Data::baseToJson("BinnedData");
  j["_bin_root"] = _bin_root;

  return j;
}


// CategoricalDataRaw:
// ---------------------------------

CategoricalDataRaw::CategoricalDataRaw (const std::string data_identifier, const std::vector<std::string>& raw_data)
  : Data::Data ( std::string(data_identifier) ),
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

json CategoricalDataRaw::toJson () const
{
  json j = Data::baseToJson("CategoricalDataRaw");
  j["_raw_data"] = _raw_data;

  return j;
}


} // namespace data
