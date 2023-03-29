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
// it under the terms of the LGPL-3 License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// LGPL-3 License for more details. You should have received a copy of
// the license along with compboost.
//
// =========================================================================== #

#ifndef INIT_H_
#define INIT_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <string>
#include <memory>
#include <map>

#include "data.h"
#include "tensors.h"
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

namespace init {

typedef std::shared_ptr<data::Data> sdata;
typedef std::shared_ptr<data::BinnedData> sbindata;

struct PolynomialAttributes {
  double df;
  double penalty;
  arma::mat penalty_mat;
  unsigned int degree;
  bool use_intercept;
  unsigned int bin_root;

  PolynomialAttributes ();
  PolynomialAttributes (const unsigned int, const bool);
  PolynomialAttributes (const json&);

  json toJson () const;
};

struct PSplineAttributes {
  double df;
  double penalty;
  arma::mat penalty_mat;
  unsigned int degree;
  unsigned int n_knots;
  unsigned int differences;
  bool use_sparse_matrices;
  unsigned int bin_root;
  arma::mat knots;

  PSplineAttributes ();
  PSplineAttributes (const json&);

  json toJson () const;
};

struct RidgeAttributes {
  double df;
  double penalty;
  arma::mat penalty_mat;
  std::map<std::string, unsigned int> dictionary;

  RidgeAttributes ();
  RidgeAttributes (const json&);

  json toJson () const;
};

struct BinaryAttributes {
  std::string cls;

  BinaryAttributes ();
  BinaryAttributes (const json&);

  json toJson () const;
};

struct TensorAttributes {
  double penalty;

  TensorAttributes ();
  TensorAttributes (const json&);

  json toJson () const;
};

struct CenteredAttributes {
  arma::mat rotation;

  CenteredAttributes ();
  CenteredAttributes (const json&);

  json toJson () const;
};

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
typedef arma::mat (*trainFunPtr)           (const arma::mat& y, const arma::mat& X);
typedef arma::mat (*predictFunPtr)         (const arma::mat& newdata, const arma::mat& parameter);

struct CustomCppAttributes {
  instantiateDataFunPtr instantiateDataFun;
  trainFunPtr trainFun;
  predictFunPtr predictFun;
  json toJson () const;
};

sbindata initPolynomialData (const sdata&, const std::shared_ptr<PolynomialAttributes>&);
sbindata initPSplineData    (const sdata&, const std::shared_ptr<PSplineAttributes>&);
sdata initRidgeData         (const sdata&, const std::shared_ptr<RidgeAttributes>&);
sdata initBinaryData        (const sdata&, const std::shared_ptr<BinaryAttributes>&);
sdata initTensorData        (const sdata&, const sdata&);
//sdata initCenteredData      (const sbindata&,const std::shared_ptr<CenteredAttributes>&);
sbindata initCenteredData   (const sdata&,const std::shared_ptr<CenteredAttributes>&);
sdata initCustomData        (const sdata&, Rcpp::Function);
sdata initCustomCppData     (const sdata&, const std::shared_ptr<CustomCppAttributes>&);

} // init

#endif // INIT_H_
