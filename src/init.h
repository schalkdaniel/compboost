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

#ifndef INIT_H_
#define INIT_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <string>
#include <memory>
#include <map>

#include "data.h"
#include "tensors.h"

namespace init {

typedef std::shared_ptr<data::Data> sdata;
typedef std::shared_ptr<data::BinnedData> sbindata;

struct PolynomialAttributes {
  unsigned int degree;
  bool use_intercept;
  unsigned int bin_root;
  PolynomialAttributes () {};
  PolynomialAttributes (const unsigned int _degree, const bool _use_intercept)
    : degree ( _degree ), use_intercept ( _use_intercept ) {};
};
struct PSplineAttributes {
  unsigned int degree;
  unsigned int n_knots;
  double penalty;
  double df;
  unsigned int differences;
  bool use_sparse_matrices;
  unsigned int bin_root;
  arma::mat knots;
};
struct RidgeAttributes {
  std::map<std::string, unsigned int> dictionary;
};
struct BinaryAttributes {
  std::string cls;
};
struct CenteredAttributes {
  arma::mat rotation;
};
//struct CustomAttributes {
  //Rcpp::Function instantiateDataFun;
  //Rcpp::Function trainFun;
  //Rcpp::Function predictFun;
  //Rcpp::Function extractParameter;
//};

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
typedef arma::mat (*trainFunPtr) (const arma::mat& y, const arma::mat& X);
typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);

struct CustomCppAttributes {
 instantiateDataFunPtr instantiateDataFun;
 trainFunPtr trainFun;
 predictFunPtr predictFun;
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
