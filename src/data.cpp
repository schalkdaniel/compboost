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

Data::Data (const std::string data_identifier) : _data_identifier ( data_identifier ) { }

Data::Data (const std::string data_identifier, const arma::mat& data_mat)
  : _data_identifier ( data_identifier ),
    _data_mat        ( data_mat )
{ }

Data::Data (const std::string data_identifier, const arma::sp_mat& sparse_data_mat)
  : _data_identifier ( data_identifier ),
    _use_sparse      ( true ),
    _sparse_data_mat ( sparse_data_mat )
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
void Data::setPenaltyMat (const arma::mat& D)    { _penalty_mat = D; }

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

arma::mat    Data::getPenaltyMat    () const { return _penalty_mat; }
arma::sp_mat Data::getSparseData    () const { return _sparse_data_mat; }
arma::uvec   Data::getBinningIndex  () const { return _bin_idx; }
bool         Data::usesSparseMatrix () const { return _use_sparse; }
bool         Data::usesBinning      () const { return _use_binning; }

// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

InMemoryData::InMemoryData (const std::string data_identifier)
  : Data::Data ( data_identifier )
{ }

InMemoryData::InMemoryData (const std::string data_identifier, const arma::mat& raw_data)
  : Data::Data ( data_identifier, raw_data )
{ }

InMemoryData::InMemoryData (const std::string data_identifier, const arma::sp_mat& raw_sp_data)
  : Data::Data ( data_identifier, raw_sp_data )
{ }

// void InMemoryData::setData (const arma::mat& transformed_data) { data_mat = transformed_data; }
// Todo! Autotransform sparse to dense and ALWAYS return a dense matrix
arma::mat InMemoryData::getData () const { return Data::getDenseData(); }

unsigned int InMemoryData::getNObs () const {
  if (_use_sparse) {
    return _sparse_data_mat.n_cols;
  } else {
    return _data_mat.n_rows;
  }
}

InMemoryData::~InMemoryData () {}


// BinnedData:
// ------------------------------
BinnedData::BinnedData (const std::string data_identifier)
  : Data ( data_identifier )
{ }

BinnedData::BinnedData (const std::string data_identifier, const unsigned int bin_root, const arma::vec& x, const arma::vec& x_bins)
  : Data         ( data_identifier),
    _bin_root    ( bin_root )
{
  _use_binning = bin_root > 0;
  _bin_idx     = binning::calculateIndexVector(x, x_bins);
}

arma::mat  BinnedData::getData         () const { return Data::getDenseData(); }
//bool       BinnedData::usesBinning     () const { return _use_binning; }

//void BinnedData::setBinRoot (const unsigned int& bin_root)
//{
  //_bin_root = bin_root;
  //_use_binning = bin_root > 0;
//}


unsigned int BinnedData::getNObs () const {
  return _bin_idx.n_rows;
}


// CategoricalDataRaw:
// ---------------------------------

CategoricalDataRaw::CategoricalDataRaw (const std::string data_identifier, const std::vector<std::string>& raw_data)
  : Data      ( data_identifier ),
    _raw_data ( raw_data )
{ }

arma::mat CategoricalDataRaw::getData () const {
  throw std::logic_error("Raw categorical data does not contain a numerical representation, call '$getRawData()' instead");
   //1 x 1 dummy  matrix:
  return Data::getDenseData();
}

std::vector<std::string> CategoricalDataRaw::getRawData () const { return _raw_data; };

unsigned int CategoricalDataRaw::getNObs () const {
  return _raw_data.size();
}

} // namespace data
