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

arma::sp_mat Data::getSparseData    () const { return _sparse_data_mat; }
bool         Data::usesSparseMatrix () const { return _use_sparse; }


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


// CategoricalData:
// ---------------------------------

typedef std::map<std::string, unsigned int> map_dict;

CategoricalData::CategoricalData (const std::string data_identifier, const std::vector<std::string>& chr_classes)
  : Data ( data_identifier )
{
  std::string        chr_class;
  unsigned int       int_class;
  arma::urowvec      temp_classes(chr_classes.size(), arma::fill::zeros);
  map_dict::iterator it;


  for (unsigned int i = 0; i < chr_classes.size(); i++) {
    chr_class = chr_classes.at(i);
    it = _dictionary.find(chr_class);
    if (it == _dictionary.end()) {
      int_class = _dictionary.size();
      _dictionary.insert(std::pair<std::string, unsigned int>(chr_class, int_class));
    } else {
      int_class = it->second;
    }
    temp_classes(i) = int_class;
  }
  _classes = temp_classes;
}

arma::mat CategoricalData::getData () const
{
  // No conversion from urowvec -> mat, therefore, convert to std::vector and then to mat:
  std::vector<unsigned int> temp = arma::conv_to<std::vector<unsigned int>>::from(_classes);
  return arma::conv_to<arma::mat>::from(temp);
}

map_dict CategoricalData::getDictionary () const
{
  return _dictionary;
}

arma::urowvec CategoricalData::getClasses () const
{
  return _classes;
}

void CategoricalData::initRidgeData (const double df)
{
  _df = df;
  unsigned int   nrows = _classes.n_cols;

  arma::urowvec  row_idx = arma::linspace<arma::urowvec>(0, nrows-1, nrows);
  arma::vec      fill(nrows, arma::fill::ones);

  // Initialize sparse data matrix as (transposed) binary matrix (p x n).
  // Switching row_idx and col_idx gives the transposed p x n matrix:
  arma::umat locations = arma::join_cols(_classes, row_idx);
  Data::setSparseData(arma::sp_mat(locations, fill));

  double penalty;
  if (df == 0) {
    penalty = 0;
  } else {
    penalty = Data::getSparseData().n_rows / df - 1;
  }
  arma::vec temp_XtX_inv = 1 / (arma::diagvec(Data::getSparseData() * Data::getSparseData().t()) + penalty);
  Data::setCache("identity", temp_XtX_inv);

  _is_used_as_target = true;
}

void CategoricalData::initRidgeData ()
{
  initRidgeData(0);
}

arma::mat CategoricalData::dictionaryInsert (const std::vector<std::string>& classes, const arma::vec& insertion) const
{
  arma::mat out(classes.size(), insertion.n_cols, arma::fill::zeros);
  for (unsigned int i = 0; i < classes.size(); i++) {
    auto it = _dictionary.find(classes.at(i));
    if (it != _dictionary.end()) {
      out.row(i) = insertion.row(it->second);
    }
  }
  return out;
}


// CategoricalDataRaw:
// ---------------------------------

CategoricalDataRaw::CategoricalDataRaw (const std::string data_identifier, const std::vector<std::string>& raw_data)
  : Data      ( data_identifier ),
    _raw_data ( raw_data )
{ }

arma::mat CategoricalDataRaw::getData () const {
  throw std::logic_error("Raw categorical data does not contain a numerical representation, call '$getRawData()' instead");
  // 1 x 1 dummy  matrix:
  return Data::getDenseData();
}

std::vector<std::string> CategoricalDataRaw::getRawData () const { return _raw_data; };

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
