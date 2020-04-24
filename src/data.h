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

namespace data
{

// -------------------------------------------------------------------------- //
// Abstract 'Data' class:
// -------------------------------------------------------------------------- //

class Data
{
protected:

  std::string data_identifier = "";
  bool use_sparse = false;

public:
  Data(const std::string);


  // Declare the data stuff public that every class can access the data
  // it needs:

  // Initialize with zeros and null pointer. Idea: The real data object which is
  // used for modelling is stored in the data_mat. This should only be the case
  // for the data target. The data source gets a pointer to e.g. the column of
  // a data.frame:

  /// Dense matrix for design matrix (accessed by getData and setData and directly)
  arma::mat data_mat = arma::mat (1, 1, arma::fill::zeros);

  /// Sparse matrix for design matrix (directly accessible)
  arma::sp_mat sparse_data_mat;

  // Some spline specific data stuff (of course they can be used for other
  // classes to):

  /// This is way to speed up the algorithm (nicked from the mboost guys)
  /// Generally we calculate \f$X^T X\f$ once and reuse this in every iteration.
  arma::mat XtX_inv;
  std::pair<std::string, arma::mat> mat_cache;

  /// String to indicate what is used in the mat cache, is it the inverse or
  /// the Cholesky decomposition (default):
  // const std::string cache_type = "cholesky";

  /// Flag if binning should be used:
  //bool bin_use_binning = false;

  /// In case of binning we store the vector of indexes here:
  //arma::uvec bin_index_vec;

  // Member functions:
  Data ();

  /// Set the main data (design matrix)
  virtual void setData (const arma::mat&) = 0;
  void setSparseData (const arma::sp_mat&);

  /// Get the design matrix
  virtual arma::mat getData () const = 0;

  void setCache (const std::string, const arma::mat&);
  void setCacheCholesky (const arma::mat&);
  void setCacheInverse (const arma::mat&);
  std::pair<std::string, arma::mat> getCachedMat () const;

  void setDataIdentifier (const std::string&);
  std::string getDataIdentifier () const;
  bool usesSparseMatrix () const;

  virtual
    ~Data () {};
};


// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

// This one does nothing special, just takes the data and use the transformed
// one as train data.

class InMemoryData : public Data
{
public:

  // Empty constructor for data target
  InMemoryData ();

  // Classical way via data matrix:
  InMemoryData (const arma::mat&, const std::string&);

  // Define sparse matrix:
  InMemoryData (const arma::mat&, const std::string&, const bool);

  void setData (const arma::mat&);
  arma::mat getData() const;

  ~InMemoryData ();
};

class CategoricalBinaryData : public Data
{
public:
  CategoricalBinaryData (const arma::uvec&);

  /// The index is a special format to save the binary data. For a feature (0, 1, 0, 0, 1) we
  /// save the locations (in C++ counting) as (1, 4) and the length of the vector (5) in one
  /// concatinated vector (1, 4, 5):
  arma::uvec idx;
  double xtx_inv_scalar;

  void setData (const arma::mat&);
  arma::mat getData () const;
  /// Get the idx without the last element which represents the number of observations:

  ~CategoricalBinaryData ();
};

class BinnedData : public Data
{
public:
  BinnedData (const std::string);
  BinnedData (const std::string, const unsigned int);

  /// In case of binning we store the vector of indexes here:
  arma::uvec bin_idx;

  // Order of binning:
  const unsigned int bin_root = 1;

  bool usesBinning () const;
  void setIndexVector (const arma::vec&, const arma::vec&);
  void setData (const arma::mat&);
  arma::mat getData () const;

private:
  // Member used for binning:
  const bool use_binning = false;

};

class PSplineData : public BinnedData
{
private:
  const arma::mat _knots;
  const double _range_min;
  const double _range_max;


public:
  PSplineData (const unsigned int, const arma::mat&, const arma::mat&);
  PSplineData (const std::string, const unsigned int, const arma::mat&, const arma::mat&);
  PSplineData (const std::string, const unsigned int, const arma::mat&, const arma::mat&,  const unsigned int);

  /// Degree of splines
  const unsigned int degree;

  /// Penalty matrix (directly accessible)
  arma::mat penalty_mat;

  /// Number of :inner knots
  // const unsigned int n_knots;

  /// Regularization parameter
  //const double penalty;

  /// Order of differences used for penalty matrix
  //const unsigned int differences;

  arma::mat filterKnotRange(const arma::mat&) const;
  arma::mat getKnots () const;
};



} // namespace data

#endif // DATA_H_
