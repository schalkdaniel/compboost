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

#include "baselearner_factory.h"

namespace blearnerfactory {

// -------------------------------------------------------------------------- //
// Abstract 'BaselearnerFactory' class:
// -------------------------------------------------------------------------- //

// arma::mat BaselearnerFactory::getData () const
// {
//   return data_target->getData();
// }

std::string BaselearnerFactory::getDataIdentifier () const
{
  return data_target->getDataIdentifier();
}

std::string BaselearnerFactory::getBaselearnerType() const
{
  return blearner_type;
}

void BaselearnerFactory::initializeDataObjects (std::shared_ptr<data::Data> data_source0,
  std::shared_ptr<data::Data> data_target0)
{
  data_source = data_source0;
  data_target = data_target0;

  // Make sure that the data identifier is setted correctly:
  data_target->setDataIdentifier(data_source->getDataIdentifier());

  // Get the data of the source, transform it and write it into the target:
  data_target->setData(instantiateData(data_source->getData()));
}

BaselearnerFactory::~BaselearnerFactory () {}

// -------------------------------------------------------------------------- //
// BaselearnerFactory implementations:
// -------------------------------------------------------------------------- //

// BaselearnerPolynomial:
// -----------------------

BaselearnerPolynomialFactory::BaselearnerPolynomialFactory (const std::string& blearner_type0,
  std::shared_ptr<data::Data> data_source0, std::shared_ptr<data::Data> data_target0, const unsigned int& degree,
  const bool& intercept)
  : degree ( degree ),
    intercept ( intercept )
{
  blearner_type = blearner_type0;

  data_source = data_source0;
  data_target = data_target0;

  // Make sure that the data identifier is setted correctly:
  data_target->setDataIdentifier(data_source->getDataIdentifier());

  // Prepare computation of intercept and slope of an ordinary linear regression:
  if (data_source->getData().n_cols == 1) {
    // Store centered x values for faster computation:
    data_target->setData(arma::pow(data_source->getData(), degree));

    // Hack to store some properties which are reused over and over again:
    arma::mat temp_mat(1, 2, arma::fill::zeros);

    if (intercept) {
      temp_mat(0,0) = arma::as_scalar(arma::mean(data_target->getData()));
    }
    temp_mat(0,1) = arma::as_scalar(arma::sum(arma::pow(data_target->getData() - temp_mat(0,0), 2)));
    data_target->XtX_inv = temp_mat;

  } else {
    // Get the data of the source, transform it and write it into the target:
    data_target->setData(instantiateData(data_source->getData()));
    data_target->XtX_inv = arma::inv(data_target->getData().t() * data_target->getData());
  }

  // blearner_type = blearner_type + " with degree " + std::to_string(degree);
}

std::shared_ptr<blearner::Baselearner> BaselearnerPolynomialFactory::createBaselearner (const std::string& identifier)
{
  std::shared_ptr<blearner::Baselearner>  sh_ptr_blearner = std::make_shared<blearner::BaselearnerPolynomial>(data_target, identifier, degree, intercept);
  sh_ptr_blearner->setBaselearnerType(blearner_type);

  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = sh_ptr_blearner->instantiateData();
  //
  //   is_data_instantiated = true;
  //
  //   // update baselearner type:
  //   blearner_type = blearner_type + " with degree " + std::to_string(degree);
  // }
  return sh_ptr_blearner;
}

/**
 * \brief Data getter which always returns an arma::mat
 *
 * This function is important to have a unified interface to access the data
 * matrices. Especially for predicting we have to get the data of each factory
 * as dense matrix. This is a huge drawback in terms of memory usage. Therefore,
 * this function should only be used to get temporary matrices which are deleted
 * when they run out of scope to reduce memory load. Also note that there is a
 * dispatch with the getData() function of the Data objects which are mostly
 * called internally.
 *
 * \returns `arma::mat` of data used for modelling a single base-learner
 */
arma::mat BaselearnerPolynomialFactory::getData () const
{
  // In the case of p = 1 we have to treat the getData() function differently
  // due to the saved and already transformed data without intercept. This
  // is annoying but improves performance of the fitting process.
  if (data_target->getData().n_cols == 1) {
    if (intercept) {
      return instantiateData(arma::pow(data_target->getData(), 1/degree));
    } else {
      return data_target->getData();
    }
  } else {
    return data_target->getData();
  }
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerPolynomialFactory::instantiateData (const arma::mat& newdata) const
{
  arma::mat temp = arma::pow(newdata, degree);
  if (intercept) {
    arma::mat temp_intercept(temp.n_rows, 1, arma::fill::ones);
    temp = join_rows(temp_intercept, temp);
  }
  return temp;
}

// BaselearnerPSpline:
// -----------------------

/**
 * \brief Default constructor of class `PSplineBlearnerFactory`
 *
 * The PSpline constructor has some important tasks which are:
 *   - Set the knots
 *   - Initialize the data (knots must be setted prior)
 *   - Compute and store penalty matrix
 *
 * \param blearner_type0 `std::string` Name of the baselearner type (setted by
 *   the Rcpp Wrapper classes in `compboost_modules.cpp`)
 * \param data_source `std::shared_ptr<data::Data>` Source of the data
 * \param data_target `std::shared_ptr<data::Data>` Object to store the transformed data source
 * \param degree `unsigned int` Polynomial degree of the splines
 * \param n_knots `unsigned int` Number of inner knots used
 * \param penalty `double` Regularization parameter `penalty = 0` yields
 *   b splines while a bigger penalty forces the splines into a global
 *   polynomial form.
 * \param differences `unsigned int` Number of differences used for the
 *   penalty matrix.
 */

BaselearnerPSplineFactory::BaselearnerPSplineFactory (const std::string& blearner_type0,
  std::shared_ptr<data::Data> data_source0, std::shared_ptr<data::Data> data_target0, const unsigned int& degree,
  const unsigned int& n_knots, const double& penalty, const unsigned int& differences,
  const bool& use_sparse_matrices)
  : degree ( degree ),
    n_knots ( n_knots ),
    penalty ( penalty ),
    differences ( differences ),
    use_sparse_matrices ( use_sparse_matrices )
{
  blearner_type = blearner_type0;
  // Set data, data identifier and the data_mat (dense at this stage)
  data_source = data_source0;
  data_target = data_target0;

  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (data_source->getData().n_cols > 1) {
      Rcpp::stop("Given data should have just one column.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }
  // Initialize knots:
  data_target->knots = splines::createKnots(data_source->getData(), n_knots, degree);

  // Additionally set the penalty matrix:
  data_target->penalty_mat = splines::penaltyMat(n_knots + (degree + 1), differences);

  // Make sure that the data identifier is setted correctly:
  data_target->setDataIdentifier(data_source->getDataIdentifier());

  // Get the data of the source, transform it and write it into the target. This needs some explanations:
  //   - If we use sparse matrices we want to store the sparse matrix into the sparse data matrix member of
  //     the data object. This also requires to adopt getData() for that purpose.
  //   - To get some (very) nice speed ups we store the transposed matrix not the standard one. This also
  //     affects how the training in baselearner.cpp is done. Nevertheless, this speed up things dramatically.
  if (use_sparse_matrices) {
    data_target->sparse_data_mat = splines::createSparseSplineBasis (data_source->getData(), degree, data_target->knots).t();
    data_target->XtX_inv = arma::inv(data_target->sparse_data_mat * data_target->sparse_data_mat.t() + penalty * data_target->penalty_mat);
  } else {
    data_target->setData(instantiateData(data_source->getData()));
    data_target->XtX_inv = arma::inv(data_target->getData().t() * data_target->getData() + penalty * data_target->penalty_mat);
  }
}

/**
 * \brief Create new `BaselearnerPSpline` object
 *
 * \param identifier `std::string` identifier of that specific baselearner object
 */
std::shared_ptr<blearner::Baselearner> BaselearnerPSplineFactory::createBaselearner (const std::string& identifier)
{
  // Create new polynomial baselearner. This one will be returned by the
  // factory:
  std::shared_ptr<blearner::Baselearner>  sh_ptr_blearner = std::make_shared<blearner::BaselearnerPSpline>(data_target, identifier, degree,
    n_knots, penalty, differences, use_sparse_matrices);
  sh_ptr_blearner->setBaselearnerType(blearner_type);

  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = sh_ptr_blearner->instantiateData();
  //
  //   is_data_instantiated = true;
  //
  //   // update baselearner type:
  //   blearner_type = blearner_type + " with degree " + std::to_string(degree);
  // }
  return sh_ptr_blearner;
}

/**
 * \brief Data getter which always returns an arma::mat
 *
 * This function is important to have a unified interface to access the data
 * matrices. Especially for predicting we have to get the data of each factory
 * as dense matrix. This is a huge drawback in terms of memory usage. Therefore,
 * this function should only be used to get temporary matrices which are deleted
 * when they run out of scope to reduce memory load. Also note that there is a
 * dispatch with the getData() function of the Data objects which are mostly
 * called internally.
 *
 * \returns `arma::mat` of data used for modelling a single base-learner
 */
arma::mat BaselearnerPSplineFactory::getData () const
{
  if (use_sparse_matrices) {
    // std::cout << "Use sparse matrices" << std::endl;
    arma::mat out (data_target->sparse_data_mat.t());
    return out;
  } else {
    // std::cout << "Use dense matrices" << std::endl;
    return data_target->getData();
  }
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
arma::mat BaselearnerPSplineFactory::instantiateData (const arma::mat& newdata) const
{
  arma::vec knots = data_target->knots;

  // check if the new data matrix contains value which are out of range:
  double range_min = knots[degree];                   // minimal value from original data
  double range_max = knots[n_knots + degree + 1];     // maximal value from original data

  arma::mat temp = splines::filterKnotRange(newdata, range_min, range_max, data_target->getDataIdentifier());

  // Data object has to be created prior! That means that data_ptr must have
  // initialized knots, and penalty matrix!
  arma::mat out = splines::createSplineBasis (temp, degree, data_target->knots);
  return out;
}

// BaselearnerCustom:
// -----------------------

BaselearnerCustomFactory::BaselearnerCustomFactory (const std::string& blearner_type0,
  std::shared_ptr<data::Data> data_source, std::shared_ptr<data::Data> data_target, Rcpp::Function instantiateDataFun,
  Rcpp::Function trainFun, Rcpp::Function predictFun, Rcpp::Function extractParameter)
  : instantiateDataFun ( instantiateDataFun ),
    trainFun ( trainFun ),
    predictFun ( predictFun ),
    extractParameter ( extractParameter )
{
  blearner_type = blearner_type0;
  initializeDataObjects(data_source, data_target);
}

std::shared_ptr<blearner::Baselearner> BaselearnerCustomFactory::createBaselearner (const std::string &identifier)
{
  std::shared_ptr<blearner::Baselearner> sh_ptr_blearner = std::make_shared<blearner::BaselearnerCustom>(data_target, identifier,
    instantiateDataFun, trainFun, predictFun, extractParameter);
  sh_ptr_blearner->setBaselearnerType(blearner_type);

  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = sh_ptr_blearner->instantiateData();
  //
  //   is_data_instantiated = true;
  // }
  return sh_ptr_blearner;
}

/**
 * \brief Data getter which always returns an arma::mat
 *
 * This function is important to have a unified interface to access the data
 * matrices. Especially for predicting we have to get the data of each factory
 * as dense matrix. This is a huge drawback in terms of memory usage. Therefore,
 * this function should only be used to get temporary matrices which are deleted
 * when they run out of scope to reduce memory load. Also note that there is a
 * dispatch with the getData() function of the Data objects which are mostly
 * called internally.
 *
 * \returns `arma::mat` of data used for modelling a single base-learner
 */
arma::mat BaselearnerCustomFactory::getData () const
{
  return data_target->getData();
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCustomFactory::instantiateData (const arma::mat& newdata) const
{
  Rcpp::NumericMatrix out = instantiateDataFun(newdata);
  return Rcpp::as<arma::mat>(out);
}

// BaselearnerCustomCpp:
// -----------------------

BaselearnerCustomCppFactory::BaselearnerCustomCppFactory (const std::string& blearner_type0,
  std::shared_ptr<data::Data> data_source, std::shared_ptr<data::Data> data_target, SEXP instantiateDataFun,
  SEXP trainFun, SEXP predictFun)
  : instantiateDataFun ( instantiateDataFun ),
    trainFun ( trainFun ),
    predictFun ( predictFun )
{
  blearner_type = blearner_type0;
  initializeDataObjects(data_source, data_target);
}

std::shared_ptr<blearner::Baselearner> BaselearnerCustomCppFactory::createBaselearner (const std::string& identifier)
{
  std::shared_ptr<blearner::Baselearner> sh_ptr_blearner = std::make_shared<blearner::BaselearnerCustomCpp>(data_target, identifier,
    instantiateDataFun, trainFun, predictFun);
  sh_ptr_blearner->setBaselearnerType(blearner_type);

  // // Check if the data is already set. If not, run 'instantiateData' from the
  // // baselearner:
  // if (! is_data_instantiated) {
  //   data = sh_ptr_blearner->instantiateData();
  //
  //   is_data_instantiated = true;
  // }
  return sh_ptr_blearner;
}

/**
 * \brief Data getter which always returns an arma::mat
 *
 * This function is important to have a unified interface to access the data
 * matrices. Especially for predicting we have to get the data of each factory
 * as dense matrix. This is a huge drawback in terms of memory usage. Therefore,
 * this function should only be used to get temporary matrices which are deleted
 * when they run out of scope to reduce memory load. Also note that there is a
 * dispatch with the getData() function of the Data objects which are mostly
 * called internally.
 *
 * \returns `arma::mat` of data used for modelling a single base-learner
 */
arma::mat BaselearnerCustomCppFactory::getData () const
{
  return data_target->getData();
}

// Transform data. This is done twice since it makes the prediction
// of the whole compboost object so much easier:
arma::mat BaselearnerCustomCppFactory::instantiateData (const arma::mat& newdata) const
{
  Rcpp::XPtr<instantiateDataFunPtr> myTempInstantiation (instantiateDataFun);
  instantiateDataFunPtr instantiateDataFun0 = *myTempInstantiation;

  return instantiateDataFun0(newdata);
}


} // namespace blearnerfactory
