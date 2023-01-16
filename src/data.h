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

#include <utility>

#include "RcppArmadillo.h"
#include "binning.h"
#include "splines.h"
#include "helper.h"
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

namespace data
{

// -------------------------------------------------------------------------- //
// Abstract 'Data' class:
// -------------------------------------------------------------------------- //

class Data
{
private:
  const std::string _data_identifier = "";

  std::pair<std::string, arma::mat> _mat_cache;

  // Private functions
  void setCacheCholesky (const arma::mat&);
  void setCacheInverse  (const arma::mat&);
  void setCacheIdentity (const arma::mat&);

protected:
  bool          _use_sparse  = false;
  bool          _use_binning = false;
  arma::mat     _data_mat    = arma::mat (1, 1, arma::fill::zeros);
  arma::uvec    _bin_idx;
  arma::sp_mat  _sparse_data_mat;

  Data (const std::string);
  Data (const std::string, const arma::mat&);
  Data (const std::string, const arma::sp_mat&);
  Data (const json&);

public:
  // Virtual functions
  virtual arma::mat    getData  () const = 0;
  virtual unsigned int getNObs  () const = 0;
  virtual unsigned int getNCols () const = 0;
  virtual json         toJson   () const = 0;

  // Getter/Setter
  std::string                       getDataIdentifier () const;
  std::pair<std::string, arma::mat> getCache          () const;
  std::string                       getCacheType      () const;
  arma::mat                         getCacheMat       () const;
  arma::mat                         getDenseData      () const;
  arma::sp_mat                      getSparseData     () const;
  arma::uvec                        getBinningIndex   () const;
  bool                              usesSparseMatrix  () const;
  bool                              usesBinning       () const;

  void setDenseData   (const arma::mat&);
  void setSparseData  (const arma::sp_mat&);
  void setCache       (const std::string, const arma::mat&);
  void setCacheCustom (const std::string, const arma::mat&);
  void setIndexVector (const arma::uvec&);
  json baseToJson     (const std::string) const;

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
  InMemoryData (const json&);

  arma::mat getData     () const;
  unsigned int getNObs  () const;
  unsigned int getNCols () const;
  json         toJson   () const;

  // Destructor
  ~InMemoryData ();
};


// BinnedData:
// ------------------------------

class BinnedData : public Data
{
private:
  //arma::uvec    _bin_idx;
  //bool          _use_binning = false;
  unsigned int  _bin_root = 1;

public:
  BinnedData (const std::string);
  BinnedData (const std::string, const unsigned int, const arma::vec&, const arma::vec&);
  BinnedData (const json&);

  arma::mat    getData  () const;
  unsigned int getNObs  () const;
  unsigned int getNCols () const;
  json         toJson   () const;
};


// CategoricalDataRaw:
// ----------------------------

class CategoricalDataRaw : public Data
{
private:
  std::vector<std::string> _raw_data;

public:
  CategoricalDataRaw (const std::string, const std::vector<std::string>&);
  CategoricalDataRaw (const json&);

  arma::mat                getData    () const;
  unsigned int             getNObs    () const;
  unsigned int             getNCols   () const;
  std::vector<std::string> getRawData () const;
  json                     toJson     () const;
};


} // namespace data

#endif // DATA_H_
