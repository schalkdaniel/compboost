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
//   Implementation of the "BaselearnerFactory".
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München

//   https://www.compstat.statistik.uni-muenchen.de
//
// =========================================================================== #

#include "baselearner_factory.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

arma::mat BaselearnerFactory::getData () const
{
  return data_target->getData();
}

std::string BaselearnerFactory::getDataIdentifier () const
{
  return data_target->getDataIdentifier();
}

std::string BaselearnerFactory::getBaselearnerType() const
{
  return blearner_type;
}

void BaselearnerFactory::initializeDataObjects (data::Data* data_source0,
  data::Data* data_target0)
{
  data_source = data_source0;
  data_target = data_target0;
  
  // Make sure that the data identifier is setted correctly:
  data_target->setDataIdentifier(data_source->getDataIdentifier());
  
  // Get the data of the source, transform it and write it into the target:
  data_target->setData(instantiateData(data_source->getData()));
}

BaselearnerFactory::~BaselearnerFactory () {};

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// PolynomialBlearner:
// -----------------------

PolynomialBlearnerFactory::PolynomialBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data_source0, data::Data* data_target0, const unsigned int& degree, 
  const bool& intercept)
  : degree ( degree ),
    intercept ( intercept )
{
  blearner_type = blearner_type0;
  
  data_source = data_source0;
  data_target = data_target0;
  
  // Make sure that the data identifier is setted correctly:
  data_target->setDataIdentifier(data_source->getDataIdentifier());
  
  // Get the data of the source, transform it and write it into the target:
  data_target->setData(instantiateData(data_source->getData()));
  data_target->XtX_inv = arma::inv(data_target->getData().t() * data_target->getData());
  
  // blearner_type = blearner_type + " with degree " + std::to_string(degree);
}

blearner::Baselearner* PolynomialBlearnerFactory::createBaselearner (const std::string& identifier)
{
  blearner::Baselearner* blearner_obj;
  
  // Create new polynomial baselearner. This one will be returned by the 
  // factory:
  blearner_obj = new blearner::PolynomialBlearner(data_target, identifier, degree, intercept);
  blearner_obj->setBaselearnerType(blearner_type);
  
  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->instantiateData();
  //   
  //   is_data_instantiated = true;
  //   
  //   // update baselearner type:
  //   blearner_type = blearner_type + " with degree " + std::to_string(degree);
  // }
  return blearner_obj;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat PolynomialBlearnerFactory::instantiateData (const arma::mat& newdata)
{
  arma::mat temp = arma::pow(newdata, degree);
  if (intercept) {
    arma::mat temp_intercept(temp.n_rows, 1, arma::fill::ones);
    temp = join_rows(temp_intercept, temp);
  }
  return temp;
}



// PSplineBlearner:
// -----------------------

/**
 * \brief Default constructor of class `PSplineBleanrerFactory`
 * 
 * The PSpline constructor has some important tasks which are:
 *   - Set the knots
 *   - Initialize the data (knots must be setted prior)
 *   - Compute and store penalty matrix 
 * 
 * \param blearner_type0 `std::string` Name of the baselearner type (setted by
 *   the Rcpp Wrapper classes in `compboost_modules.cpp`)
 * \param data_source `data::Data*` Source of the data
 * \param data_target `data::Data*` Object to store the transformed data source
 * \param degree `unsigned int` Polynomial degree of the splines
 * \param n_knots `unsigned int` Number of inner knots used 
 * \param penalty `double` Regularization parameter `penalty = 0` yields
 *   b splines while a bigger penalty forces the splines into a global
 *   polynomial form.
 * \param differences `unsigned int` Number of differences used for the 
 *   penalty matrix.
 */

PSplineBlearnerFactory::PSplineBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data_source0, data::Data* data_target0, const unsigned int& degree, 
  const unsigned int& n_knots, const double& penalty, const unsigned int& differences)
  : degree ( degree ),
    n_knots ( n_knots ),
    penalty ( penalty ),
    differences ( differences )
{
  blearner_type = blearner_type0;
  // Set data, data identifier and the data_mat (dense at this stage)
  data_source = data_source0;
  data_target = data_target0;
  
  if (data_source->getData().n_cols > 1) {
    Rcpp::stop("Given data must have just one column!");
  }

  // Initialize knots:
  data_target->knots = createKnots(data_source->getData(), n_knots, degree);
  
  // Additionally set the penalty matrix:
  data_target->penalty_mat = penaltyMat(n_knots + (degree + 1), differences);
  
  // Make sure that the data identifier is setted correctly:
  data_target->setDataIdentifier(data_source->getDataIdentifier());
  
  // Get the data of the source, transform it and write it into the target:
  data_target->setData(instantiateData(data_source->getData()));

  data_target->XtX_inv = arma::inv(data_target->getData().t() * data_target->getData() + penalty * data_target->penalty_mat);
  
  // Set blearner type:
  // blearner_type = blearner_type + " with degree " + std::to_string(degree);
}

/**
 * \brief Create new `PSplineBlearner` object
 * 
 * \param identifier `std::string` identifier of that specific baselearner object
 */
blearner::Baselearner* PSplineBlearnerFactory::createBaselearner (const std::string& identifier)
{
  blearner::Baselearner* blearner_obj;
  
  // Create new polynomial baselearner. This one will be returned by the 
  // factory:
  blearner_obj = new blearner::PSplineBlearner(data_target, identifier, degree,
    n_knots, penalty, differences);
  blearner_obj->setBaselearnerType(blearner_type);
  
  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->instantiateData();
  //   
  //   is_data_instantiated = true;
  //   
  //   // update baselearner type:
  //   blearner_type = blearner_type + " with degree " + std::to_string(degree);
  // }
  return blearner_obj;
}

/**
 * \brief Instantiate data matrix (design matrix)
 * 
 * This function is ment to create the design matrix which is then stored
 * within the data object. This should be done just once and then reused all
 * the time.
 * 
 * Note that this function sets the `data_mat` object of the data object!
 * 
 * \param newdata `arma::mat` Input data which is transformed to the design matrix
 * 
 * \returns `arma::mat` of transformed data
 */
arma::mat PSplineBlearnerFactory::instantiateData (const arma::mat& newdata)
{
  // Data object has to be created prior! That means that data_ptr must have
  // initialized knots, and penalty matrix!
  return createBasis (newdata, degree, data_target->knots);
}

// CustomBlearner:
// -----------------------

CustomBlearnerFactory::CustomBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data_source, data::Data* data_target, Rcpp::Function instantiateDataFun, 
  Rcpp::Function trainFun, Rcpp::Function predictFun, Rcpp::Function extractParameter)
  : instantiateDataFun ( instantiateDataFun ),
    trainFun ( trainFun ),
    predictFun ( predictFun ),
    extractParameter ( extractParameter )
{
  blearner_type = blearner_type0;
  initializeDataObjects(data_source, data_target);
}

blearner::Baselearner *CustomBlearnerFactory::createBaselearner (const std::string &identifier)
{
  blearner::Baselearner *blearner_obj;
  
  blearner_obj = new blearner::CustomBlearner(data_target, identifier, 
    instantiateDataFun, trainFun, predictFun, extractParameter);
  
  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->instantiateData();
  //   
  //   is_data_instantiated = true;
  // }
  return blearner_obj;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat CustomBlearnerFactory::instantiateData (const arma::mat& newdata)
{
  Rcpp::NumericMatrix out = instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}

// CustomCppBlearner:
// -----------------------

CustomCppBlearnerFactory::CustomCppBlearnerFactory (const std::string& blearner_type0, 
  data::Data* data_source, data::Data* data_target, SEXP instantiateDataFun, 
  SEXP trainFun, SEXP predictFun)
  : instantiateDataFun ( instantiateDataFun ),
    trainFun ( trainFun ),
    predictFun ( predictFun )
{
  blearner_type = blearner_type0;
  initializeDataObjects(data_source, data_target);
}

blearner::Baselearner* CustomCppBlearnerFactory::createBaselearner (const std::string& identifier)
{
  blearner::Baselearner* blearner_obj;
  
  blearner_obj = new blearner::CustomCppBlearner(data_target, identifier, 
    instantiateDataFun, trainFun, predictFun);
  
  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = blearner_obj->instantiateData();
  //   
  //   is_data_instantiated = true;
  // }
  return blearner_obj;
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat CustomCppBlearnerFactory::instantiateData (const arma::mat& newdata)
{
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun);
  instantiateDataFunPtr instantiateDataFun0 = *myTempInstantiation;
  
  return instantiateDataFun0(newdata);
}

} // namespace blearnerfactory
