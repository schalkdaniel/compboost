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

Data::Data () {}
Data::Data (const std::string data_identifier) : data_identifier ( data_identifier ) { }
void Data::setSparseData (const arma::sp_mat& X)
{
  use_sparse = true;
  sparse_data_mat = X;
}
void Data::setCache (const std::string cache_type, const arma::mat& xtx)
{
  std::vector<std::string> choices{ "cholesky", "inverse" };
  helper::assertChoice(cache_type, choices);

  if (cache_type == "cholesky") setCacheCholesky(xtx);
  if (cache_type == "inverse") setCacheInverse(xtx);
}
void Data::setCacheCholesky (const arma::mat& xtx)
{
  mat_cache = std::make_pair("cholesky", arma::chol(xtx));
}
void Data::setCacheInverse (const arma::mat& xtx)
{
  mat_cache = std::make_pair("inverse", arma::inv(xtx));
}
std::pair<std::string, arma::mat> Data::getCachedMat () const { return mat_cache; }
void Data::setDataIdentifier (const std::string& new_data_identifier) { data_identifier = new_data_identifier; }
std::string Data::getDataIdentifier () const { return data_identifier; }
bool Data::usesSparseMatrix () const { return use_sparse; }

// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

InMemoryData::InMemoryData () {}

InMemoryData::InMemoryData (const arma::mat& raw_data, const std::string& data_identifier0)
{
  data_mat = raw_data;
  data_identifier = data_identifier0;
}

InMemoryData::InMemoryData (const arma::mat& raw_data, const std::string& data_identifier0, const bool use_sparse0)
{
  use_sparse = use_sparse0;
  arma::sp_mat temp_sparse(raw_data);
  sparse_data_mat = temp_sparse;
  data_identifier = data_identifier0;
}

void InMemoryData::setData (const arma::mat& transformed_data) { data_mat = transformed_data; }
arma::mat InMemoryData::getData () const { return data_mat; }

InMemoryData::~InMemoryData () {}


// BinnedData:
// ------------------------------
BinnedData::BinnedData (const std::string data_identifier)
  : Data ( data_identifier )
{ }

BinnedData::BinnedData (const std::string data_identifier, const unsigned int bin_root)
  : Data ( data_identifier ),
    use_binning ( true ),
    bin_root ( bin_root )
{ }

bool BinnedData::usesBinning () const
{
  return use_binning;
}
void BinnedData::setIndexVector (const arma::vec& x, const arma::vec& x_bins)
{
  bin_idx = binning::calculateIndexVector(x, x_bins);
}
void BinnedData::setData (const arma::mat& transformed_data) { data_mat = transformed_data; }
arma::mat BinnedData::getData () const
{
  if (usesSparseMatrix()) {
    // std::cout << "Use sparse matrices" << std::endl;
    arma::mat out (sparse_data_mat.t());
    return out;
  } else {
    // std::cout << "Use dense matrices" << std::endl;
    return getData();
  }

}

// CategoricalBinaryData:
// ---------------------------------

CategoricalBinaryData::CategoricalBinaryData (const arma::uvec& idx)
  : idx ( idx ),
    xtx_inv_scalar ( 1 / (double)(idx.size()-1) )
{ }

void CategoricalBinaryData::setData (const arma::mat& transformed_data) { data_mat = transformed_data; }

arma::mat CategoricalBinaryData::getData () const { return data_mat; }

CategoricalBinaryData::~CategoricalBinaryData () {}


// PSplineData:
// -------------------------------

PSplineData::PSplineData (const std::string data_identifier, const unsigned int degree, const arma::mat& knots, const arma::mat& penalty_mat)
  : BinnedData ( data_identifier ),
    degree ( degree ),
    _knots ( knots ),
    penalty_mat ( penalty_mat ),
    _range_min ( knots(degree) ),
    _range_max ( knots(knots.n_rows - degree - 1) )
{ }

PSplineData::PSplineData (const std::string data_identifier, const unsigned int degree, const arma::mat& knots, const arma::mat& penalty_mat, const unsigned int bin_root)
  : BinnedData ( data_identifier, bin_root ),
    degree ( degree ),
    _knots ( knots ),
    penalty_mat ( penalty_mat ),
    _range_min ( knots(degree) ),
    _range_max ( knots(knots.n_rows - degree - 1) )
{ }

arma::mat PSplineData::filterKnotRange (const arma::mat& x) const { return splines::filterKnotRange(x, _range_min, _range_max); }
arma::mat PSplineData::getKnots () const { return _knots; }
} // namespace data
