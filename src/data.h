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

#ifndef DATA_H_
#define DATA_H_

#include "RcppArmadillo.h"
#include "binning.h"
#include "splines.h"
#include "helper.h"

#include <vector>

namespace data
{

// -------------------------------------------------------------------------- //
// Abstract 'Data' class:
// -------------------------------------------------------------------------- //

class Data
{
private:
  const std::string   _data_identifier  = "";
  const bool          _use_sparse       = false;

  arma::mat     _data_mat         = arma::mat (1, 1, arma::fill::zeros);
  arma::sp_mat  _sparse_data_mat;

  std::pair<std::string, arma::mat> _mat_cache;

  // Private functions
  void setCacheCholesky (const arma::mat&);
  void setCacheInverse  (const arma::mat&);
  void setCacheIdentity (const arma::mat&);

protected:
  Data (const std::string);
  Data (const std::string, const arma::mat&);
  Data (const std::string, const arma::sp_mat&);

public:
  // Virtual functions
  virtual arma::mat getData () const = 0;

  // Getter/Setter
  std::string                       getDataIdentifier () const;
  std::pair<std::string, arma::mat> getCache          () const;
  std::string                       getCacheType      () const;
  arma::mat                         getCacheMat       () const;
  arma::mat                         getDenseData      () const;
  arma::sp_mat                      getSparseData     () const;
  bool                              usesSparseMatrix  () const;

  void setDenseData   (const arma::mat&);
  void setSparseData  (const arma::sp_mat&);
  void setCache       (const std::string, const arma::mat&);

  // Destructor
  virtual ~Data () {};
};


// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

class InMemoryData : public Data
{
public:
  InMemoryData (const std::string);
  InMemoryData (const std::string, const arma::mat&);
  InMemoryData (const std::string, const arma::sp_mat&);

  // void setData (const arma::mat&);
  arma::mat getData() const;

  // Destructor
  ~InMemoryData ();
};


// BinnedData:
// ------------------------------

class BinnedData : public Data
{
private:
  const arma::uvec    _bin_idx;
  const bool          _use_binning = false;
  const unsigned int  _bin_root = 1;

protected:
  BinnedData (const std::string);
  BinnedData (const std::string, const unsigned int, const arma::vec&, const arma::vec&);

public:
  arma::mat  getData         () const;
  arma::uvec getBinningIndex () const;
  bool       usesBinning     () const;

  //void setIndexVector (const arma::vec&, const arma::vec&);
  //void setData        (const arma::mat&);
};


// PSplineData:
// -------------------------------

class PSplineData : public BinnedData
{
private:
  const unsigned int  _degree;
  const arma::mat     _knots;
  const arma::mat     _penalty_mat;
  const double        _range_min;
  const double        _range_max;

public:
  PSplineData (const std::string, const unsigned int, const arma::mat&,
    const arma::mat&);
  PSplineData (const std::string, const unsigned int, const arma::mat&,
    const arma::mat&, const unsigned int, const arma::vec&, const arma::vec&);

  arma::mat    filterKnotRange (const arma::mat&) const;
  arma::mat    getKnots        ()                 const;
  arma::mat    getPenaltyMat   ()                 const;
  unsigned int getDegree       ()                 const;
};


// CategoricalData:
// ----------------------------

typedef std::map<std::string, unsigned int> map_dict;
class CategoricalData : public Data
{
private:
  map_dict       _dictionary;
  arma::urowvec  _classes;

  double  _df = 0;
  bool    _is_used_as_target = false;

public:
  CategoricalData (const std::string, const std::vector<std::string>&);

  arma::mat  getData()       const;
  map_dict   getDictionary() const;

  void       initRidgeData    (const double);
  void       initRidgeData    ();
  arma::mat  dictionaryInsert (const std::vector<std::string>&, const arma::vec&) const;
};


// CategoricalDataRaw:
// ----------------------------

class CategoricalDataRaw : public Data
{
private:
  std::vector<std::string> _raw_data;

public:
  CategoricalDataRaw (const std::string, const std::vector<std::string>&);

  arma::mat                getData    () const;
  std::vector<std::string> getRawData () const;
};


// CategoricalBinaryData:
// ----------------------------

class CategoricalBinaryData : public Data
{
private:
  const arma::uvec  _idx;
  const double      _xtx_inv_scalar;

public:
  CategoricalBinaryData (const std::string, const arma::uvec&);

  arma::mat    getData      ()                   const;
  arma::uvec   getIndex     ()                   const;
  unsigned int getIndex     (const unsigned int) const;
  double       getXtxScalar ()                   const;

  // Destructor
  ~CategoricalBinaryData ();
};


} // namespace data

#endif // DATA_H_
