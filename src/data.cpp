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
  _mat_cache = std::make_pair("cholesky", arma::chol(xtx));
}

void Data::setCacheInverse (const arma::mat& xtx)
{
  _mat_cache = std::make_pair("inverse", arma::inv(xtx));
}

void Data::setCacheIdentity (const arma::mat& X)
{
  _mat_cache = std::make_pair("identity", X);
}

void Data::setDenseData  (const arma::mat& X)    { _data_mat = X; }
void Data::setSparseData (const arma::sp_mat& X) { _sparse_data_mat = X; }

void Data::setCache (const std::string cache_type, const arma::mat& xtx)
{
  std::vector<std::string> choices{ "cholesky", "inverse", "identity" };
  helper::assertChoice(cache_type, choices);

  if (cache_type == "cholesky") setCacheCholesky(xtx);
  if (cache_type == "inverse")  setCacheInverse(xtx);
  if (cache_type == "identity") setCacheIdentity(xtx);
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

arma::sp_mat Data::getSparseData () const { return _sparse_data_mat; }
bool Data::usesSparseMatrix () const { return _use_sparse; }


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
arma::mat InMemoryData::getData () const { return Data::getDenseData(); }

InMemoryData::~InMemoryData () {}


// BinnedData:
// ------------------------------
BinnedData::BinnedData (const std::string data_identifier)
  : Data ( data_identifier )
{ }

BinnedData::BinnedData (const std::string data_identifier, const unsigned int bin_root, const arma::vec& x, const arma::vec& x_bins)
  : Data         ( data_identifier),
    _use_binning ( true ),
    _bin_root    ( bin_root ),
    _bin_idx     ( binning::calculateIndexVector(x, x_bins) )
{ }

arma::mat  BinnedData::getData         () const { return Data::getDenseData(); }
arma::uvec BinnedData::getBinningIndex () const { return _bin_idx; }
bool       BinnedData::usesBinning     () const { return _use_binning; }


// PSplineData:
// -------------------------------

PSplineData::PSplineData (const std::string data_identifier, const unsigned int degree, const arma::mat& knots, const arma::mat& penalty_mat)
  : BinnedData   ( data_identifier ),
    _degree      ( degree ),
    _knots       ( knots ),
    _penalty_mat ( penalty_mat ),
    _range_min   ( knots(degree) ),
    _range_max   ( knots(knots.n_rows - degree - 1) )
{ }

PSplineData::PSplineData (const std::string data_identifier, const unsigned int degree, const arma::mat& knots, const arma::mat& penalty_mat,
  const unsigned int bin_root, const arma::vec& x, const arma::vec& x_bins)
  : BinnedData ( data_identifier, bin_root, x, x_bins ),
    _degree ( degree ),
    _knots ( knots ),
    _penalty_mat ( penalty_mat ),
    _range_min ( knots(degree) ),
    _range_max ( knots(knots.n_rows - degree - 1) )
{ }

arma::mat    PSplineData::filterKnotRange (const arma::mat& x) const { return splines::filterKnotRange(x, _range_min, _range_max); }
arma::mat    PSplineData::getKnots        () const { return _knots; }
arma::mat    PSplineData::getPenaltyMat   () const { return _penalty_mat; }
unsigned int PSplineData::getDegree       () const { return _degree; }


// CategoricalBinaryData:
// ---------------------------------

CategoricalBinaryData::CategoricalBinaryData (const std::string data_identifier, const arma::uvec& idx)
  : Data            ( data_identifier ),
    _idx            ( idx ),
    _xtx_inv_scalar ( 1 / (double)(idx.size()-1) )
{ }

arma::mat    CategoricalBinaryData::getData      () const { return Data::getDenseData(); }
arma::uvec   CategoricalBinaryData::getIndex     () const { return _idx; }
unsigned int CategoricalBinaryData::getIndex     (const unsigned int i) const { return _idx(i); }
double       CategoricalBinaryData::getXtxScalar () const { return _xtx_inv_scalar; }

CategoricalBinaryData::~CategoricalBinaryData () {}


} // namespace data
