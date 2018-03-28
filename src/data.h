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
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with Compboost. If not, see <http://www.gnu.org/licenses/>.
//
// This file contains:
// -------------------
//
//   
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
// ========================================================================== //

#ifndef DATA_H_
#define DATA_H_

#include "RcppArmadillo.h"

namespace data 
{

// -------------------------------------------------------------------------- //
// Abstract 'Data' class:
// -------------------------------------------------------------------------- //

class Data
{
protected:
  
  std::string data_identifier = "";
  std::string data_type = "ordinary";
  
public:
  
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
  
  
  const arma::mat* data_mat_ptr = NULL;
  
  // Some spline specific data stuff (of course they can be used for other
  // classes to):
  
  /// Penalty matrix (directly accesible)
  arma::mat penalty_mat;
  
  /// Vector of knots (directky accesible)
  arma::vec knots;
  
  /// Upper and lower boundary for values, values which are out of range
  /// have a basis of zeros (directly accesible)
  arma::vec knot_boundaries;
  
  /// This is way to speed up the algorithm (nicked from the mboost guys)
  /// Genrally we calculate \f$X^T X\f$ once and reuse this in every iteration.
  arma::mat XtX_inv;
  
  // Member functions:
  Data ();
  
  /// Set the main data (design matrix)
  virtual void setData (const arma::mat&) = 0;
  
  /// Get the design matrix 
  virtual arma::mat getData () const = 0;
  
  void setDataIdentifier (const std::string&);
  std::string getDataIdentifier () const;
  
  void setDataType (const std::string&);
  
  virtual 
    ~Data () { };
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
  
  // // Colvec to refer directly to R vectors or data frame columns:
  // InMemoryData (const arma::vec&, const std::string&);
  
  // Classical way via data matrix:
  InMemoryData (const arma::mat&, const std::string&);
  
  void setData (const arma::mat&);
  arma::mat getData() const;
  
  ~InMemoryData ();
  
};


// SparseInMemoryData:
// -----------------------

// This one does nothing special, just takes the data and use the transformed
// one as train data.

/**
 * \class SparseInMemoryData
 * 
 * \brief Data class to store sparse matrices
 * 
 * This class does basically the same as the `InMemoryData` class but stores
 * 
 */

// class SparseInMemoryData : public Data
// {
// private:
//   
//   
//   arma::mat data_mat = arma::mat (1, 1, arma::fill::zeros);
//   const arma::mat* data_mat_ptr = NULL;
//   
// public:
//   
//   // Empty constructor for data target
//   InMemoryData ();
//   
//   // // Colvec to refer directly to R vectors or data frame columns:
//   // InMemoryData (const arma::vec&, const std::string&);
//   
//   // Classical way via data matrix:
//   InMemoryData (const arma::mat&, const std::string&);
//   
//   void setData (const arma::mat&);
//   arma::mat getData() const;
//   
//   ~InMemoryData ();
//   
// };

} // namespace data

#endif // DATA_H_