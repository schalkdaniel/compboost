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

#ifndef BASELEARNER_H_
#define BASELEARNER_H_

#include <RcppArmadillo.h>
#include <memory>
#include <string>

#include "data.h"
#include "response.h"
#include "splines.h"
#include "binning.h"

namespace blearner {

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

class Baselearner
{
public:

  virtual void train (const arma::mat&) = 0;
  arma::mat getParameter () const;

  virtual arma::mat predict () const = 0;
  virtual arma::mat predict (std::shared_ptr<data::Data>) const = 0;

  // Specify how the data has to be transformed. E. g. for splines a mapping
  // to the higher dimension space. The overloading function with the
  // arma mat as parameter is used for newdata:
  virtual arma::mat instantiateData (const arma::mat&) const = 0;

  // Clone function (in some places needed e.g. "optimizer.cpp") and a copy
  // function which is called by clone to avoid copy and pasting of the
  // protected members:
  void copyMembers (const arma::mat&, const std::string&, std::shared_ptr<data::Data>);
  virtual Baselearner* clone () = 0;

  // Within 'setData' the pointer will be setted, while 'instantiateData'
  // overwrite the object on which 'data_ptr' points. This guarantees that
  // the data is just stored once in the factory and then called by reference
  // within the baselearner:
  void setData (std::shared_ptr<data::Data>);
  // arma::mat getData () const;

  // Get data identifier stored within the data object:
  std::string getDataIdentifier () const;

  // Set and get identifier of a specific baselearner (this is unique):
  void setIdentifier (const std::string&);
  std::string getIdentifier () const;

  // Set and get baselearner type (this can be the same for multiple
  // baselearner e.g. linear baselearner for variable x1 and x2).
  // This one is setted by the factory which later creates the objects:
  void setBaselearnerType (const std::string&);
  std::string getBaselearnerType () const;

  // Destructor:
  virtual ~Baselearner ();

protected:

  // Members which should be directly accessible through the child classes:
  arma::mat parameter;
  std::string blearner_identifier;
  std::string blearner_type;
  std::shared_ptr<data::Data> sh_ptr_data;
  // std::string data_identifier;
};

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------


// This baselearner trains a linear model without intercept and covariable
// x^degree:

class BaselearnerPolynomial : public Baselearner
{
private:
  unsigned int degree;
  bool intercept;

public:
  // (data pointer, data identifier, baselearner identifier, degree)
  BaselearnerPolynomial (std::shared_ptr<data::Data>, const std::string&, const unsigned int&, const bool&);

  Baselearner* clone ();

  // arma::mat instantiateData ();
  arma::mat instantiateData (const arma::mat&) const;

  void train (const arma::mat&);
  arma::mat predict () const;
  arma::mat predict (std::shared_ptr<data::Data>) const;

  ~BaselearnerPolynomial ();

};

// BaselearnerPSpline:
// -----------------------

/**
 * \class BaselearnerPSpline
 *
 * \brief P-Spline Baselearner
 *
 * This class implements the P-Spline baselearners. We have used de Boors
 * algorithm (from the Nurbs Book) to create the basis. The penalty parameter
 * can be specified directly or by using the degrees of freedom. If you are
 * using the degrees of freedom insteat of the penalty parameter, then this is
 * transformed to a penalty parameter using the Demmler-Reinsch
 * Orthogonalization.
 *
 * Please note, that this baselearner is just the dummy object. The most
 * functionality is done while creating the data target which contains the
 * most object which are used here.
 *
 */

class BaselearnerPSpline : public Baselearner
{
private:

  /// Degree of polynomial functions as base models
  unsigned int degree;

  /// Number of inner knots
  unsigned int n_knots;

  /// Penalty parameter
  double penalty;

  /// Differences of penalty matrix
  unsigned int differences;

  /// Flag if sparse matrices should be used:
  const bool use_sparse_matrices;

public:
  /// Default constructor of `BaselearnerPSpline` class
  BaselearnerPSpline (std::shared_ptr<data::Data>, const std::string&, const unsigned int&,
    const unsigned int&, const double&, const unsigned int&, const bool&);

  /// Clean copy of baselearner
  Baselearner* clone ();

  /// Instantiate data matrix (design matrix)
  arma::mat instantiateData (const arma::mat&) const;

  /// Training of a baselearner
  void train (const arma::mat&);

  /// Predict on training data
  arma::mat predict () const;

  /// Predict on newdata
  arma::mat predict (std::shared_ptr<data::Data>) const;


  /// Destructor
  ~BaselearnerPSpline ();

};

// BaselearnerCustom:
// -----------------------

// This class can be used to define custom baselearner in R and expose thi
// to the c++ class:

class BaselearnerCustom : public Baselearner
{
private:

  SEXP model;

  // R functions for a custom baselearner:
  Rcpp::Function instantiateDataFun;
  Rcpp::Function trainFun;
  Rcpp::Function predictFun;
  Rcpp::Function extractParameter;

public:

  // (data pointer, data identifier, baselearner identifier, R function for
  // data instantiation, R function for training, R function for prediction,
  // R function to extract parameter):
  BaselearnerCustom (std::shared_ptr<data::Data>, const std::string&, Rcpp::Function,
    Rcpp::Function, Rcpp::Function, Rcpp::Function);

  // Copy constructor:
  Baselearner* clone ();

  // arma::mat instantiateData ();
  arma::mat instantiateData (const arma::mat&) const;

  void train (const arma::mat&);
  arma::mat predict () const;
  arma::mat predict (std::shared_ptr<data::Data>) const;

  ~BaselearnerCustom ();

};

// BaselearnerCustom:
// -----------------------

// This is a  bit tricky. The key is that we store the cpp functions as
// pointer. Therefore we can go with R and use the XPtr class of Rcpp to
// give the pointer as SEXP. To try a working example see
// "tutorial/stages_of_custom_learner.html".

// Please note, that the result of the train function should be a matrix
// containing the estimated parameter.

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
typedef arma::mat (*trainFunPtr) (const arma::mat& y, const arma::mat& X);
typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);

class BaselearnerCustomCpp : public Baselearner
{
private:

  // Cpp functions for a custom baselearner:
  instantiateDataFunPtr instantiateDataFun;
  trainFunPtr trainFun;
  predictFunPtr predictFun;

public:

  // (data pointer, data identifier, baselearner identifier, R function for
  // data instantiation, R function for training, R function for prediction,
  // R function to extract parameter):
  BaselearnerCustomCpp (std::shared_ptr<data::Data>, const std::string&, SEXP, SEXP, SEXP);

  // Copy constructor:
  Baselearner* clone ();

  // arma::mat instantiateData ();
  arma::mat instantiateData (const arma::mat&) const;

  void train (const arma::mat&);
  arma::mat predict () const;
  arma::mat predict (std::shared_ptr<data::Data>) const;

  ~BaselearnerCustomCpp ();
};

} // namespace blearner

#endif // BASELEARNER_H_
