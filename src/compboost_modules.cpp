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
// Written by:
// -----------
//
//   Daniel Schalk
//   Department of Statistics
//   Ludwig-Maximilians-University Munich
//   Ludwigstrasse 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
//   Contact
//   e: contact@danielschalk.com
//   w: danielschalk.com
//
// =========================================================================== #

#ifndef COMPBOOST_MODULES_CPP_
#define COMPBOOST_MODULES_CPP_

#include "compboost.h"
#include "baselearner_factory.h"
#include "baselearner_factory_list.h"
#include "loss.h"
#include "data.h"

// -------------------------------------------------------------------------- //
//                                   DATA                                     //
// -------------------------------------------------------------------------- //

class DataWrapper
{
public:
  DataWrapper () {}

  data::Data* getDataObj () { return obj; }

  virtual ~DataWrapper () { delete obj; }

protected:
  data::Data* obj;

};

//' In memory data class to store data in RAM
//'
//' \code{InMemoryData} creates an data object which can be used as source or
//' target object within the base-learner factories of \code{compboost}. The
//' convention to initialize target data is to call the constructor without
//' any arguments.
//'
//' @format \code{\link{S4}} object.
//' @name InMemoryData
//'
//' @section Usage:
//' \preformatted{
//' InMemoryData$new()
//' InMemoryData$new(data.mat, data.identifier)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data.mat} [\code{matrix}]}{
//'   Matrix containing the source data. This source data is later transformed
//'   to obtain the design matrix a base-learner uses for training.
//' }
//' \item{\code{data.identifier} [\code{character(1)}]}{
//'   The name for the data specified in \code{data.mat}. Note that it is
//'   important to have the same data names for train and evaluation data.
//' }
//' }
//'
//'
//' @section Details:
//'   The \code{data.mat} needs to suits the base-learner. For instance, the
//'   spline base-learner does just take a one column matrix since there are
//'   just one dimensional splines till now. Additionally, using the polynomial
//'   base-learner the \code{data.mat} is used to control if a intercept should
//'   be fitted or not by adding a column containing just ones. It is also
//'   possible to add other columns to estimate multiple features
//'   simultaneously. Anyway, this is not recommended in terms of unbiased
//'   features selection.
//'
//'   The \code{data.mat} and \code{data.identifier} of a target data object
//'   is set automatically by passing the source and target object to the
//'   desired factory. \code{getData()} can then be used to access the
//'   transformed data of the target object.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classdata_1_1_in_memory_data.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{method extract the \code{data.mat} from the data object.}
//' \item{\code{getIdentifier()}}{method to extract the used name from the data object.}
//' }
//' @examples
//' # Sample data:
//' data.mat = cbind(1:10)
//'
//' # Create new data object:
//' data.obj = InMemoryData$new(data.mat, "my.data.name")
//'
//' # Get data and identifier:
//' data.obj$getData()
//' data.obj$getIdentifier()
//'
//' @export InMemoryData
class InMemoryDataWrapper : public DataWrapper
{

// Solve this copying issue:
// https://github.com/schalkdaniel/compboost/issues/123
private:
  arma::vec data_vec = arma::vec (1, arma::fill::zeros);
  arma::mat data_mat = arma::mat (1, 1, arma::fill::zeros);

public:

  // Set data type in constructors to
  //   - arma::mat    -> const arma::mat&
  //   - arma::vec    -> const arma::vec&
  //   - std::string  -> const std::string &
  // crashes the compilation?

  InMemoryDataWrapper ()
  {
    obj = new data::InMemoryData ();
  }

  // Rcpp doesn't detect if vector or matrix if using arma::vec and arma::mat:
  // InMemoryDataWrapper (Rcpp::NumericVector data0, std::string data_identifier)
  // {
  //   data_vec = Rcpp::as<arma::vec>(data0);
  //   Rcpp::Rcout << "Vector Initializer" << std::endl;
  //   obj = new data::InMemoryData (data_vec, data_identifier);
  // }

  InMemoryDataWrapper (arma::mat data0, std::string data_identifier)
  {
    data_mat = data0;

    obj = new data::InMemoryData (data_mat, data_identifier);
  }
  arma::mat getData () const
  {
    return obj->getData();
  }
  std::string getIdentifier () const
  {
    return obj->getDataIdentifier();
  }
};



RCPP_EXPOSED_CLASS(DataWrapper)
RCPP_MODULE (data_module)
{
  using namespace Rcpp;

  class_<DataWrapper> ("Data")
    .constructor ("Create Data class")
  ;

  class_<InMemoryDataWrapper> ("InMemoryData")
    .derives<DataWrapper> ("Data")

    .constructor ()
    // .constructor<Rcpp::NumericVector, std::string> ()
    .constructor<arma::mat, std::string> ()

    .method("getData",       &InMemoryDataWrapper::getData, "Get data")
    .method("getIdentifier", &InMemoryDataWrapper::getIdentifier, "Get the data identifier")
  ;
}


// -------------------------------------------------------------------------- //
//                         BASELEARNER FACTORIES                               //
// -------------------------------------------------------------------------- //

// Abstract class. This one is given to the factory list. The factory list then
// handles all factories equally. It does not differ between a polynomial or
// custom factory:
class BaselearnerFactoryWrapper
{
public:

  blearnerfactory::BaselearnerFactory* getFactory () { return obj; }
  virtual ~BaselearnerFactoryWrapper () { delete obj; }

protected:

  blearnerfactory::BaselearnerFactory* obj;
  // blearner::Baselearner* test_obj;
};


//' Base-learner factory to make polynomial regression
//'
//' \code{BaselearnerPolynomial} creates a polynomial base-learner factory
//'  object which can be registered within a base-learner list and then used
//'  for training.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerPolynomial
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerPolynomial$new(data_source, data_target, degree, intercept)
//' BaselearnerPolynomial$new(data_source, data_target, blearner_type, degree, intercept)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
//' }
//' \item{\code{data_target} [\code{Data} Object]}{
//'   Data object which gets the transformed source data.
//' }
//' \item{\code{degree} [\code{integer(1)}]}{
//'   This argument is used for transforming the source data. Each element is
//'   taken to the power of the \code{degree} argument.
//' }
//' \item{\code{intercept} [\code{logical(1)}]}{
//'   Indicating whether an intercept should be added or not. Default is set to TRUE.
//' }
//' }
//'
//'
//' @section Details:
//'   The polynomial base-learner factory takes any matrix which the user wants
//'   to pass the number of columns indicates how much parameter are estimated.
//'   Note that the intercept isn't added by default. To get an intercept add a
//'   column of ones to the source data matrix.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classblearnerfactory_1_1_polynomial_blearner_factory.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Get the data matrix of the target data which is used
//'   for modeling.}
//' \item{\code{transformData(X)}}{Transform a data matrix as defined within the
//'   factory. The argument has to be a matrix with one column.}
//' \item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
//' }
//' @examples
//' # Sample data:
//' data.mat = cbind(1:10)
//'
//' # Create new data object:
//' data.source = InMemoryData$new(data.mat, "my.data.name")
//' data.target1 = InMemoryData$new()
//' data.target2 = InMemoryData$new()
//'
//' # Create new linear base-learner factory:
//' lin.factory = BaselearnerPolynomial$new(data.source, data.target1, 
//'   degree = 2, intercept = FALSE)
//' lin.factory.int = BaselearnerPolynomial$new(data.source, data.target2, 
//'   degree = 2, intercept = TRUE)
//'
//' # Get the transformed data:
//' lin.factory$getData()
//' lin.factory.int$getData()
//'
//' # Summarize factory:
//' lin.factory$summarizeFactory()
//'
//' # Transform data manually:
//' lin.factory$transformData(data.mat)
//' lin.factory.int$transformData(data.mat)
//'
//' @export BaselearnerPolynomial
class BaselearnerPolynomialFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  const unsigned int degree;
  bool intercept;

public:

  BaselearnerPolynomialFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    const unsigned int& degree, bool intercept)
    : degree ( degree ),
      intercept ( intercept )
  {
    std::string blearner_type_temp = "polynomial_degree_" + std::to_string(degree);

    obj = new blearnerfactory::BaselearnerPolynomialFactory(blearner_type_temp, data_source.getDataObj(),
      data_target.getDataObj(), degree, intercept);
  }

  BaselearnerPolynomialFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    const std::string& blearner_type, const unsigned int& degree, bool intercept)
    : degree ( degree ),
      intercept ( intercept )
  {
    obj = new blearnerfactory::BaselearnerPolynomialFactory(blearner_type, data_source.getDataObj(),
      data_target.getDataObj(), degree, intercept);
  }

  arma::mat getData () { return obj->getData(); }
  std::string getDataIdentifier () { return obj->getDataIdentifier(); }
  std::string getBaselearnerType () { return obj->getBaselearnerType(); }

  arma::mat transformData (const arma::mat& newdata)
  {
    return obj->instantiateData(newdata);
  }

  void summarizeFactory ()
  {
    if (degree == 1) {
      Rcpp::Rcout << "Linear base-learner factory:" << std::endl;
    }
    if (degree == 2) {
      Rcpp::Rcout << "Quadratic base-learner factory:" << std::endl;
    }
    if (degree == 3) {
      Rcpp::Rcout << "Cubic base-learner factory:" << std::endl;
    }
    if (degree > 3) {
      Rcpp::Rcout << "Polynomial base-learner of degree " << degree << " factory:" << std::endl;
    }
    Rcpp::Rcout << "\t- Name of the used data: " << obj->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << obj->getBaselearnerType() << std::endl;
  }
};

//' Base-learner factory to do non-parametric B or P-spline regression
//'
//' \code{BaselearnerPSpline} creates a spline base-learner factory
//'  object which can be registered within a base-learner list and then used
//'  for training.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerPSpline
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerPSpline$new(data_source, data_target, degree, n_knots, penalty,
//'   differences)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
//' }
//' \item{\code{data_target} [\code{Data} Object]}{
//'   Data object which gets the transformed source data.
//' }
//' \item{\code{degree} [\code{integer(1)}]}{
//'   Degree of the spline functions to interpolate the knots.
//' }
//' \item{\code{n_knots} [\code{integer(1)}]}{
//'   Number of \strong{inner knots}. To prevent weird behavior on the edges
//'   the inner knots are expanded by \eqn{\mathrm{degree} - 1} additional knots.
//' }
//' \item{\code{penalty} [\code{numeric(1)}]}{
//'   Positive numeric value to specify the penalty parameter. Setting the
//'   penalty to 0 ordinary B-splines are used for the fitting.
//' }
//' \item{\code{differences} [\code{integer(1)}]}{
//'   The number of differences which are penalized. A higher value leads to
//'   smoother curves.
//' }
//' }
//'
//' @section Details:
//'   The data matrix of the source data is restricted to have just one column.
//'   The spline bases are created for this single feature. Multidimensional
//'   splines are not supported at the moment.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classblearnerfactory_1_1_p_spline_blearner_factory.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Get the data matrix of the target data which is used
//'   for modeling.}
//' \item{\code{transformData(X)}}{Transform a data matrix as defined within the
//'   factory. The argument has to be a matrix with one column.}
//' \item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
//' }
//' @examples
//' # Sample data:
//' data.mat = cbind(1:10)
//' y = sin(1:10)
//'
//' # Create new data object:
//' data.source = InMemoryData$new(data.mat, "my.data.name")
//' data.target = InMemoryData$new()
//'
//' # Create new linear base-learner:
//' spline.factory = BaselearnerPSpline$new(data.source, data.target,
//'   degree = 3, n_knots = 4, penalty = 2, differences = 2)
//'
//' # Get the transformed data:
//' spline.factory$getData()
//'
//' # Summarize factory:
//' spline.factory$summarizeFactory()
//'
//' # Transform data manually:
//' spline.factory$transformData(data.mat)
//'
//' @export BaselearnerPSpline
class BaselearnerPSplineFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  const unsigned int degree;

public:

  BaselearnerPSplineFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    const unsigned int& degree, const unsigned int& n_knots, const double& penalty,
    const unsigned int& differences)
    : degree ( degree )
  {
    std::string blearner_type_temp = "spline_degree_" + std::to_string(degree);
    
    obj = new blearnerfactory::BaselearnerPSplineFactory(blearner_type_temp, data_source.getDataObj(),
       data_target.getDataObj(), degree, n_knots, penalty, differences, TRUE);      

  }

  BaselearnerPSplineFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    const std::string& blearner_type, const unsigned int& degree, 
    const unsigned int& n_knots, const double& penalty, const unsigned int& differences)
    : degree ( degree )
  {
    obj = new blearnerfactory::BaselearnerPSplineFactory(blearner_type, data_source.getDataObj(),
      data_target.getDataObj(), degree, n_knots, penalty, differences, TRUE);

  }

  arma::mat getData () { return obj->getData(); }
  std::string getDataIdentifier () { return obj->getDataIdentifier(); }
  std::string getBaselearnerType () { return obj->getBaselearnerType(); }

  arma::mat transformData (const arma::mat& newdata)
  {
    return obj->instantiateData(newdata);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Spline factory of degree" << " " << std::to_string(degree) << std::endl;
    Rcpp::Rcout << "\t- Name of the used data: " << obj->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << obj->getBaselearnerType() << std::endl;
  }
};

//' Create custom base-learner factory by using R functions.
//'
//' \code{BaselearnerCustom} creates a custom base-learner factory by
//'   setting custom \code{R} functions. This factory object can be registered
//'   within a base-learner list and then used for training.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerCustom
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCustom$new(data_source, data_target, instantiateData, train,
//'   predict, extractParameter)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
//' }
//' \item{\code{data_target} [\code{Data} Object]}{
//'   Data object which gets the transformed source data.
//' }
//' \item{\code{instantiateData} [\code{function}]}{
//'   \code{R} function to transform the source data. For details see the
//'   \code{Details}.
//' }
//' \item{\code{train} [\code{function}]}{
//'   \code{R} function to train the base-learner on the target data. For
//'   details see the \code{Details}.
//' }
//' \item{\code{predict} [\code{function}]}{
//'   \code{R} function to predict on the object returned by \code{train}.
//'   For details see the \code{Details}.
//' }
//' \item{\code{extractParameter} [\code{function}]}{
//'   \code{R} function to extract the parameter of the object returned by
//'   \code{train}. For details see the \code{Details}.
//' }
//' }
//'
//' @section Details:
//'   The function must have the following structure:
//'
//'   \code{instantiateData(X) { ... return (X.trafo) }} With a matrix argument
//'   \code{X} and a matrix as return object.
//'
//'   \code{train(y, X) { ... return (SEXP) }} With a vector argument \code{y}
//'   and a matrix argument \code{X}. The target data is used in \code{X} while
//'   \code{y} contains the response. The function can return any \code{R}
//'   object which is stored within a \code{SEXP}.
//'
//'   \code{predict(model, newdata) { ... return (prediction) }} The returned
//'   object of the \code{train} function is passed to the \code{model}
//'   argument while \code{newdata} contains a new matrix used for predicting.
//'
//'   \code{extractParameter() { ... return (parameters) }} Again, \code{model}
//'   contains the object returned by \code{train}. The returned object must be
//'   a matrix containing the estimated parameter. If no parameter should be
//'   estimated one can return \code{NA}.
//'
//'   For an example see the \code{Examples}.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classblearnerfactory_1_1_custom_blearner_factory.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Get the data matrix of the target data which is used
//'   for modeling.}
//' \item{\code{transformData(X)}}{Transform a data matrix as defined within the
//'   factory. The argument has to be a matrix with one column.}
//' \item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
//' }
//' @examples
//' # Sample data:
//' data.mat = cbind(1, 1:10)
//' y = 2 + 3 * 1:10
//'
//' # Create new data object:
//' data.source = InMemoryData$new(data.mat, "my.data.name")
//' data.target = InMemoryData$new()
//'
//' instantiateDataFun = function (X) {
//'   return(X)
//' }
//' # Ordinary least squares estimator:
//' trainFun = function (y, X) {
//'   return(solve(t(X) %*% X) %*% t(X) %*% y)
//' }
//' predictFun = function (model, newdata) {
//'   return(as.matrix(newdata %*% model))
//' }
//' extractParameter = function (model) {
//'   return(as.matrix(model))
//' }
//'
//' # Create new custom linear base-learner factory:
//' custom.lin.factory = BaselearnerCustom$new(data.source, data.target,
//'   instantiateDataFun, trainFun, predictFun, extractParameter)
//'
//' # Get the transformed data:
//' custom.lin.factory$getData()
//'
//' # Summarize factory:
//' custom.lin.factory$summarizeFactory()
//'
//' # Transform data manually:
//' custom.lin.factory$transformData(data.mat)
//'
//' @export BaselearnerCustom
class BaselearnerCustomFactoryWrapper : public BaselearnerFactoryWrapper
{
public:

  BaselearnerCustomFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    Rcpp::Function instantiateDataFun, Rcpp::Function trainFun,
    Rcpp::Function predictFun, Rcpp::Function extractParameter)
  {
    obj = new blearnerfactory::BaselearnerCustomFactory("custom", data_source.getDataObj(),
      data_target.getDataObj(), instantiateDataFun, trainFun, predictFun, extractParameter);
  }

  BaselearnerCustomFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    const std::string& blearner_type, Rcpp::Function instantiateDataFun, Rcpp::Function trainFun,
    Rcpp::Function predictFun, Rcpp::Function extractParameter)
  {
    obj = new blearnerfactory::BaselearnerCustomFactory(blearner_type, data_source.getDataObj(),
      data_target.getDataObj(), instantiateDataFun, trainFun, predictFun, extractParameter);
  }

  arma::mat getData () { return obj->getData(); }
  std::string getDataIdentifier () { return obj->getDataIdentifier(); }
  std::string getBaselearnerType () { return obj->getBaselearnerType(); }

  arma::mat transformData (const arma::mat& newdata)
  {
    return obj->instantiateData(newdata);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Custom base-learner Factory:" << std::endl;

    Rcpp::Rcout << "\t- Name of the used data: " << obj->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << obj->getBaselearnerType() << std::endl;
  }
};

//' Create custom cpp base-learner factory by using cpp functions and external
//' pointer.
//'
//' \code{BaselearnerCustomCpp} creates a custom base-learner factory by
//'   setting custom \code{C++} functions. This factory object can be registered
//'   within a base-learner list and then used for training.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerCustomCpp
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCustomCpp$new(data_source, data_target, instantiate_data_ptr,
//'   train_ptr, predict_ptr)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
//' }
//' \item{\code{data_target} [\code{Data} Object]}{
//'   Data object which gets the transformed source data.
//' }
//' \item{\code{instantiate_data_ptr} [\code{externalptr}]}{
//'   External pointer to the \code{C++} instantiate data function.
//' }
//' \item{\code{train_ptr} [\code{externalptr}]}{
//'   External pointer to the \code{C++} train function.
//' }
//' \item{\code{predict_ptr} [\code{externalptr}]}{
//'   External pointer to the \code{C++} predict function.
//' }
//' }
//'
//' @section Details:
//'   For an example see the extending compboost vignette or the function
//'   \code{getCustomCppExample}.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classblearnerfactory_1_1_custom_cpp_blearner_factory.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Get the data matrix of the target data which is used
//'   for modeling.}
//' \item{\code{transformData(X)}}{Transform a data matrix as defined within the
//'   factory. The argument has to be a matrix with one column.}
//' \item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
//' }
//' @examples
//' \donttest{
//' # Sample data:
//' data.mat = cbind(1, 1:10)
//' y = 2 + 3 * 1:10
//'
//' # Create new data object:
//' data.source = InMemoryData$new(data.mat, "my.data.name")
//' data.target = InMemoryData$new()
//'
//' # Source the external pointer exposed by using XPtr:
//' Rcpp::sourceCpp(code = getCustomCppExample(silent = TRUE))
//'
//' # Create new linear base-learner:
//' custom.cpp.factory = BaselearnerCustomCpp$new(data.source, data.target,
//'   dataFunSetter(), trainFunSetter(), predictFunSetter())
//'
//' # Get the transformed data:
//' custom.cpp.factory$getData()
//'
//' # Summarize factory:
//' custom.cpp.factory$summarizeFactory()
//'
//' # Transform data manually:
//' custom.cpp.factory$transformData(data.mat)
//' }
//' @export BaselearnerCustomCpp
class BaselearnerCustomCppFactoryWrapper : public BaselearnerFactoryWrapper
{
public:

  BaselearnerCustomCppFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    SEXP instantiateDataFun, SEXP trainFun, SEXP predictFun)
  {
    obj = new blearnerfactory::BaselearnerCustomCppFactory("custom_cpp", data_source.getDataObj(),
      data_target.getDataObj(), instantiateDataFun, trainFun, predictFun);
  }

  BaselearnerCustomCppFactoryWrapper (DataWrapper& data_source, DataWrapper& data_target,
    const std::string& blearner_type, SEXP instantiateDataFun, SEXP trainFun, SEXP predictFun)
  {
    obj = new blearnerfactory::BaselearnerCustomCppFactory(blearner_type, data_source.getDataObj(),
      data_target.getDataObj(), instantiateDataFun, trainFun, predictFun);
  }

  arma::mat getData () { return obj->getData(); }
  std::string getDataIdentifier () { return obj->getDataIdentifier(); }
  std::string getBaselearnerType () { return obj->getBaselearnerType(); }

  arma::mat transformData (const arma::mat& newdata)
  {
    return obj->instantiateData(newdata);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Custom cpp base-learner Factory:" << std::endl;

    Rcpp::Rcout << "\t- Name of the used data: " << obj->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << obj->getBaselearnerType() << std::endl;
  }
};


// // Boiler code to check for constructors with same argument length and decide
// // on input parameter type which one to take. See:
// //   http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2011-May/002300.html
// bool fun1( SEXP* args, int nargs){
//      if (nargs != 4) { return false; }
//      if ((TYPEOF(args[3]) == INTSXP) | (TYPEOF(args[3]) == REALSXP)) { return true; }
//      return false;
// }
// bool fun2( SEXP* args, int nargs){
//      if (nargs != 4) { return false; }
//      if (TYPEOF(args[3]) == LGLSXP) { return true; }
//      return false;
// }

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(BaselearnerFactoryWrapper)
RCPP_MODULE (baselearner_factory_module)
{
  using namespace Rcpp;

  class_<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor ("Create BaselearnerFactory class")
  ;

  class_<BaselearnerPolynomialFactoryWrapper> ("BaselearnerPolynomial")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    // .constructor<DataWrapper&, DataWrapper&, unsigned int> ()
    // .constructor<DataWrapper&, DataWrapper&, std::string, unsigned int> ("", &fun1)
    .constructor<DataWrapper&, DataWrapper&, unsigned int, bool> () // ("", &fun2)
    .constructor<DataWrapper&, DataWrapper&, std::string, unsigned int, bool> ()
     .method("getData",          &BaselearnerPolynomialFactoryWrapper::getData, "Get the data which the factory uses")
     .method("transformData",     &BaselearnerPolynomialFactoryWrapper::transformData, "Transform newdata corresponding to polynomial learner")
     .method("summarizeFactory", &BaselearnerPolynomialFactoryWrapper::summarizeFactory, "Sumamrize Factory")
  ;

  class_<BaselearnerPSplineFactoryWrapper> ("BaselearnerPSpline")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, DataWrapper&, unsigned int, unsigned int, double, unsigned int> ()
    .constructor<DataWrapper&, DataWrapper&, std::string, unsigned int, unsigned int, double, unsigned int> ()
    .method("getData",          &BaselearnerPSplineFactoryWrapper::getData, "Get design matrix")
    .method("transformData",    &BaselearnerPSplineFactoryWrapper::transformData, "Compute spline basis for new data")
    .method("summarizeFactory", &BaselearnerPSplineFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

  class_<BaselearnerCustomFactoryWrapper> ("BaselearnerCustom")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, DataWrapper&, Rcpp::Function, Rcpp::Function, Rcpp::Function, Rcpp::Function> ()
    .constructor<DataWrapper&, DataWrapper&, std::string, Rcpp::Function, Rcpp::Function, Rcpp::Function, Rcpp::Function> ()
     .method("getData",          &BaselearnerCustomFactoryWrapper::getData, "Get the data which the factory uses")
     .method("transformData",    &BaselearnerCustomFactoryWrapper::transformData, "Transform data")
     .method("summarizeFactory", &BaselearnerCustomFactoryWrapper::summarizeFactory, "Sumamrize Factory")
  ;

  class_<BaselearnerCustomCppFactoryWrapper> ("BaselearnerCustomCpp")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, DataWrapper&, SEXP, SEXP, SEXP> ()
    .constructor<DataWrapper&, DataWrapper&, std::string, SEXP, SEXP, SEXP> ()
     .method("getData",          &BaselearnerCustomCppFactoryWrapper::getData, "Get the data which the factory uses")
     .method("transformData",    &BaselearnerCustomCppFactoryWrapper::transformData, "Transform data")
     .method("summarizeFactory", &BaselearnerCustomCppFactoryWrapper::summarizeFactory, "Sumamrize Factory")
  ;
}



// -------------------------------------------------------------------------- //
//                              BASELEARNERLIST                               //
// -------------------------------------------------------------------------- //

//' Base-learner factory list to define the set of base-learners
//'
//' \code{BlearnerFactoryList} creates an object in which base-learner factories
//' can be registered. This object can then be passed to compboost as set of
//' base-learner which is used by the optimizer to get the new best
//' base-learner.
//'
//' @format \code{\link{S4}} object.
//' @name BlearnerFactoryList
//'
//' @section Usage:
//' \preformatted{
//' BlearnerFactoryList$new()
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classblearnerlist_1_1_baselearner_factory_list.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{registerFactory(BaselearnerFactory)}}{Takes a object of the
//'   class \code{BaseLearnerFactory} and adds this factory to the set of
//'   base-learner.}
//' \item{\code{printRegisteredFactories()}}{Get all registered factories.}
//' \item{\code{clearRegisteredFactories()}}{Remove all registered factories.
//'   Note that the factories are not deleted, just removed from the map.}
//' \item{\code{getModelFrame()}}{Get each target data matrix parsed to one
//'   big matrix.}
//' \item{\code{getNumberOfRegisteredFactories()}}{Get the number of registered
//'   factories.}
//' }
//' @examples
//' # Sample data:
//' data.mat = cbind(1:10)
//'
//' # Create new data object:
//' data.source = InMemoryData$new(data.mat, "my.data.name")
//' data.target1 = InMemoryData$new()
//' data.target2 = InMemoryData$new()
//'
//' lin.factory = BaselearnerPolynomial$new(data.source, data.target1, 1, TRUE)
//' poly.factory = BaselearnerPolynomial$new(data.source, data.target2, 2, TRUE)
//'
//' # Create new base-learner list:
//' my.bl.list = BlearnerFactoryList$new()
//'
//' # Register factories:
//' my.bl.list$registerFactory(lin.factory)
//' my.bl.list$registerFactory(poly.factory)
//'
//' # Get registered factories:
//' my.bl.list$printRegisteredFactories()
//'
//' # Get all target data matrices in one big matrix:
//' my.bl.list$getModelFrame()
//'
//' # Clear list:
//' my.bl.list$clearRegisteredFactories()
//'
//' # Get number of registered factories:
//' my.bl.list$getNumberOfRegisteredFactories()
//'
//' @export BlearnerFactoryList
class BlearnerFactoryListWrapper
{
private:

  blearnerlist::BaselearnerFactoryList obj;

public:

  void registerFactory (BaselearnerFactoryWrapper& my_factory_to_register)
  {
    std::string factory_id = my_factory_to_register.getFactory()->getDataIdentifier() + "_" + my_factory_to_register.getFactory()->getBaselearnerType();
    obj.registerBaselearnerFactory(factory_id, my_factory_to_register.getFactory());
  }

  void printRegisteredFactories ()
  {
    obj.printRegisteredFactories();
  }

  void clearRegisteredFactories ()
  {
    obj.clearMap();
  }

  blearnerlist::BaselearnerFactoryList* getFactoryList ()
  {
    return &obj;
  }

  Rcpp::List getModelFrame ()
  {
    std::pair<std::vector<std::string>, arma::mat> raw_frame = obj.getModelFrame();

    return Rcpp::List::create(
      Rcpp::Named("colnames")    = raw_frame.first,
      Rcpp::Named("model.frame") = raw_frame.second
    );
  }

  unsigned int getNumberOfRegisteredFactories ()
  {
    return obj.getMap().size();
  }

  std::vector<std::string> getRegisteredFactoryNames ()
  {
    return obj.getRegisteredFactoryNames();
  }

  // Nothing needs to be done since we allocate the object on the stack
  ~BlearnerFactoryListWrapper () {}
};

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(BlearnerFactoryListWrapper)
RCPP_MODULE (baselearner_list_module)
{
  using  namespace Rcpp;

  class_<BlearnerFactoryListWrapper> ("BlearnerFactoryList")
    .constructor ()
    .method("registerFactory", &BlearnerFactoryListWrapper::registerFactory, "Register new factory")
    .method("printRegisteredFactories", &BlearnerFactoryListWrapper::printRegisteredFactories, "Print all registered factories")
    .method("clearRegisteredFactories", &BlearnerFactoryListWrapper::clearRegisteredFactories, "Clear factory map")
    .method("getModelFrame", &BlearnerFactoryListWrapper::getModelFrame, "Get the data used for modeling")
    .method("getNumberOfRegisteredFactories", &BlearnerFactoryListWrapper::getNumberOfRegisteredFactories, "Get number of registered factories. Main purpose is for testing.")
    .method("getRegisteredFactoryNames", &BlearnerFactoryListWrapper::getRegisteredFactoryNames, "Get names of registered factories")
  ;
}

// -------------------------------------------------------------------------- //
//                                    LOSS                                    //
// -------------------------------------------------------------------------- //

class LossWrapper
{
public:

  loss::Loss* getLoss () { return obj; }
  virtual ~LossWrapper () { delete obj; }

protected:

  loss::Loss* obj;
};

//' Quadratic loss for regression tasks.
//'
//' This loss can be used for regression with \eqn{y \in \mathrm{R}}.
//'
//' \strong{Loss Function:}
//' \deqn{
//'   L(y, f(x)) = \frac{1}{2}( y - f(x))^2
//' }
//' \strong{Gradient:}
//' \deqn{
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = f(x) - y
//' }
//' \strong{Initialization:}
//' \deqn{
//'   \hat{f}^{[0]}(x) = \mathrm{arg~min}{c\in\mathrm{R}}{\mathrm{arg~min}}\ \frac{1}{n}\sum\limits_{i=1}^n
//'   L\left(y^{(i)}, c\right) = \bar{y}
//' }
//'
//' @format \code{\link{S4}} object.
//' @name LossQuadratic
//'
//' @section Usage:
//' \preformatted{
//' LossQuadratic$new()
//' LossQuadratic$new(offset)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{offset} [\code{numeric(1)}]}{
//'   Numerical value which can be used to set a custom offset. If so, this
//'   value is returned instead of the loss optimal initialization.
//' }
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classloss_1_1_quadratic_loss.html}.
//'
//' @examples
//'
//' # Create new loss object:
//' quadratic.loss = LossQuadratic$new()
//' quadratic.loss
//'
//' @export LossQuadratic
class LossQuadraticWrapper : public LossWrapper
{
public:
  LossQuadraticWrapper () { obj = new loss::LossQuadratic(); }
  LossQuadraticWrapper (double custom_offset) { obj = new loss::LossQuadratic(custom_offset); }
};

//' Absolute loss for regression tasks.
//'
//' This loss can be used for regression with \eqn{y \in \mathrm{R}}.
//'
//' \strong{Loss Function:}
//' \deqn{
//'   L(y, f(x)) = | y - f(x)|
//' }
//' \strong{Gradient:}
//' \deqn{
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = \mathrm{sign}( f(x) - y)
//' }
//' \strong{Initialization:}
//' \deqn{
//'   \hat{f}^{[0]}(x) = \mathrm{arg~min}_{c\in R}\ \frac{1}{n}\sum\limits_{i=1}^n
//'   L(y^{(i)}, c) = \mathrm{median}(y)
//' }
//'
//' @format \code{\link{S4}} object.
//' @name LossAbsolute
//'
//' @section Usage:
//' \preformatted{
//' LossAbsolute$new()
//' LossAbsolute$new(offset)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{offset} [\code{numeric(1)}]}{
//'   Numerical value which can be used to set a custom offset. If so, this
//'   value is returned instead of the loss optimal initialization.
//' }
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classloss_1_1_absolute_loss.html}.
//'
//' @examples
//'
//' # Create new loss object:
//' absolute.loss = LossAbsolute$new()
//' absolute.loss
//'
//' @export LossAbsolute
class LossAbsoluteWrapper : public LossWrapper
{
public:
  LossAbsoluteWrapper () { obj = new loss::LossAbsolute(); }
  LossAbsoluteWrapper (double custom_offset) { obj = new loss::LossAbsolute(custom_offset); }
};

//' 0-1 Loss for binary classification derived of the binomial distribution
//'
//' This loss can be used for binary classification. The coding we have chosen
//' here acts on
//' \eqn{y \in \{-1, 1\}}.
//'
//' \strong{Loss Function:}
//' \deqn{
//'   L(y, f(x)) = \log(1 + \mathrm{exp}(-2yf(x)))
//' }
//' \strong{Gradient:}
//' \deqn{
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = - \frac{y}{1 + \mathrm{exp}(2yf)}
//' }
//' \strong{Initialization:}
//' \deqn{
//'   \hat{f}^{[0]}(x) = \frac{1}{2}\mathrm{log}(p / (1 - p))
//' }
//' with
//' \deqn{
//'   p = \frac{1}{n}\sum\limits_{i=1}^n\mathrm{1}_{\{y^{(i)} = 1\}}
//' }
//'
//' @format \code{\link{S4}} object.
//' @name LossBinomial
//'
//' @section Usage:
//' \preformatted{
//' LossBinomial$new()
//' LossBinomial$new(offset)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{offset} [\code{numeric(1)}]}{
//'   Numerical value which can be used to set a custom offset. If so, this
//'   value is returned instead of the loss optimal initialization.
//' }
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classloss_1_1_binomial_loss.html}.
//'
//' @examples
//' 
//' # Create new loss object:
//' bin.loss = LossBinomial$new()
//' bin.loss
//'
//' @export LossBinomial
class LossBinomialWrapper : public LossWrapper
{
public:
  LossBinomialWrapper () { obj = new loss::LossBinomial(); }
  LossBinomialWrapper (double custom_offset) { obj = new loss::LossBinomial(custom_offset); }
};

//' Create LossCustom by using R functions.
//'
//' \code{LossCustom} creates a custom loss by using
//' \code{Rcpp::Function} to set \code{R} functions.
//'
//' @format \code{\link{S4}} object.
//' @name LossCustom
//'
//' @section Usage:
//' \preformatted{
//' LossCustom$new(lossFun, gradientFun, initFun)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{lossFun} [\code{function}]}{
//'   \code{R} function to calculate the loss. For details see the
//'   \code{Details}.
//' }
//' \item{\code{gradientFun} [\code{function}]}{
//'   \code{R} function to calculate the gradient. For details see the
//'   \code{Details}.
//' }
//' \item{\code{initFun} [\code{function}]}{
//'   \code{R} function to calculate the constant initialization. For
//'   details see the \code{Details}.
//' }
//' }
//'
//' @section Details:
//'   The functions must have the following structure:
//'
//'   \code{lossFun(truth, prediction) { ... return (loss) }} With a vector
//'   argument \code{truth} containing the real values and a vector of
//'   predictions \code{prediction}. The function must return a vector
//'   containing the loss for each component.
//'
//'   \code{gradientFun(truth, prediction) { ... return (grad) }} With a vector
//'   argument \code{truth} containing the real values and a vector of
//'   predictions \code{prediction}. The function must return a vector
//'   containing the gradient of the loss for each component.
//'
//'   \code{initFun(truth) { ... return (init) }} With a vector
//'   argument \code{truth} containing the real values. The function must
//'   return a numeric value containing the offset for the constant
//'   initialization.
//'
//'   For an example see the \code{Examples}.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classloss_1_1_custom_loss.html}.
//'
//' @examples
//'
//' # Loss function:
//' myLoss = function (true.values, prediction) {
//'   return (0.5 * (true.values - prediction)^2)
//' }
//' # Gradient of loss function:
//' myGradient = function (true.values, prediction) {
//'   return (prediction - true.values)
//' }
//' # Constant initialization:
//' myConstInit = function (true.values) {
//'   return (mean(true.values))
//' }
//'
//' # Create new custom quadratic loss:
//' my.loss = LossCustom$new(myLoss, myGradient, myConstInit)
//'
//' @export LossCustom
class LossCustomWrapper : public LossWrapper
{
public:
  LossCustomWrapper (Rcpp::Function lossFun, Rcpp::Function gradientFun,
    Rcpp::Function initFun)
  {
    obj = new loss::LossCustom(lossFun, gradientFun, initFun);
  }
};

//' Create custom cpp losses by using cpp functions and external pointer.
//'
//' \code{LossCustomCpp} creates a custom loss by using
//' \code{Rcpp::XPtr} to set \code{C++} functions.
//'
//' @format \code{\link{S4}} object.
//' @name LossCustomCpp
//'
//' @section Usage:
//' \preformatted{
//' LossCustomCpp$new(loss_ptr, grad_ptr, const_init_ptr)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{loss_ptr} [\code{externalptr}]}{
//'   External pointer to the \code{C++} loss function.
//' }
//' \item{\code{grad_ptr} [\code{externalptr}]}{
//'   External pointer to the \code{C++} gradient function.
//' }
//' \item{\code{const_init_ptr} [\code{externalptr}]}{
//'   External pointer to the \code{C++} constant initialization function.
//' }
//' }
//'
//' @section Details:
//'   For an example see the extending compboost vignette or the function
//'   \code{getCustomCppExample(example = "loss")}.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classloss_1_1_custom_cpp_loss.html}.
//'
//' @examples
//' \donttest{
//' # Load loss functions:
//' Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE))
//'
//' # Create new custom quadratic loss:
//' my.cpp.loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter())
//' }
//' @export LossCustomCpp
class LossCustomCppWrapper : public LossWrapper
{
public:
  LossCustomCppWrapper (SEXP loss_ptr, SEXP grad_ptr, SEXP const_init_ptr)
  {
    obj = new loss::LossCustomCpp(loss_ptr, grad_ptr, const_init_ptr);
  }
};

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(LossWrapper)
RCPP_MODULE (loss_module)
{
  using namespace Rcpp;

  class_<LossWrapper> ("Loss")
    .constructor ()
  ;

  class_<LossQuadraticWrapper> ("LossQuadratic")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
  ;

  class_<LossAbsoluteWrapper> ("LossAbsolute")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
  ;

  class_<LossBinomialWrapper> ("LossBinomial")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
  ;

  class_<LossCustomWrapper> ("LossCustom")
    .derives<LossWrapper> ("Loss")
    .constructor<Rcpp::Function, Rcpp::Function, Rcpp::Function> ()
  ;

  class_<LossCustomCppWrapper> ("LossCustomCpp")
    .derives<LossWrapper> ("Loss")
    .constructor<SEXP, SEXP, SEXP> ()
  ;
}

// -------------------------------------------------------------------------- //
//                                  LOGGER                                    //
// -------------------------------------------------------------------------- //

// Logger classes:
// ---------------

class LoggerWrapper
{
public:

  LoggerWrapper () {};

  logger::Logger* getLogger ()
  {
    return obj;
  }

  std::string getLoggerId ()
  {
    return logger_id;
  }

  virtual ~LoggerWrapper () { delete obj; }

protected:
  logger::Logger* obj;
  std::string logger_id;
};


//' Logger class to log the current iteration
//'
//' This class seems to be useless, but it gives more control about the algorithm
//' and doesn't violate the idea of object programming here. Additionally, it is
//' quite convenient to have this class instead of tracking the iteration at any
//' stage of the fitting within the compboost object as another vector.
//'
//' @format \code{\link{S4}} object.
//' @name LoggerIteration
//'
//' @section Usage:
//' \preformatted{
//' LoggerIterationWrapper$new(use_as_stopper, max_iterations)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{use_as_stopper} [\code{logical(1)}]}{
//'   Boolean to indicate if the logger should also be used as stopper.
//' }
//' \item{\code{max_iterations} [\code{integer(1)}]}{
//'   If the logger is used as stopper this argument defines the maximal
//'   iterations.
//' }
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classlogger_1_1_iteration_logger.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{summarizeLogger()}}{Summarize the logger object.}
//' }
//' @examples
//' # Define logger:
//' log.iters = LoggerIteration$new(FALSE, 100)
//'
//' # Summarize logger:
//' log.iters$summarizeLogger()
//'
//' @export LoggerIteration
class LoggerIterationWrapper : public LoggerWrapper
{

private:
  unsigned int max_iterations;
  bool use_as_stopper;

public:
  LoggerIterationWrapper (bool use_as_stopper, unsigned int max_iterations)
    : max_iterations ( max_iterations ),
      use_as_stopper ( use_as_stopper )
  {
    obj = new logger::LoggerIteration (use_as_stopper, max_iterations);
    logger_id = " iterations";
  }

  void summarizeLogger ()
  {
    Rcpp::Rcout << "Iteration logger:" << std::endl;
    Rcpp::Rcout << "\t- Maximal iterations: " << max_iterations << std::endl;
    Rcpp::Rcout << "\t- Use logger as stopper: " << use_as_stopper << std::endl;
  }
};

//' Logger class to log the inbag risk
//'
//' This class logs the inbag risk for a specific loss function. It is also
//' possible to use custom losses to log performance measures. For details
//' see the use case or extending compboost vignette.
//'
//' @format \code{\link{S4}} object.
//' @name LoggerInbagRisk
//'
//' @section Usage:
//' \preformatted{
//' LoggerInbagRisk$new(use_as_stopper, used_loss, eps_for_break)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{use_as_stopper} [\code{logical(1)}]}{
//'   Boolean to indicate if the logger should also be used as stopper.
//' }
//' \item{\code{used_loss} [\code{Loss} object]}{
//'   The loss used to calculate the empirical risk by taking the mean of the
//'   returned defined loss within the loss object.
//' }
//' \item{\code{eps_for_break} [\code{numeric(1)}]}{
//'   This argument is used if the loss is also used as stopper. If the relative
//'   improvement of the logged inbag risk falls above this boundary the stopper
//'   returns \code{TRUE}.
//' }
//' }
//'
//' @section Details:
//'
//' This logger computes the risk for the given training data
//' \eqn{\mathcal{D} = \{(x^{(i)},\ y^{(i)})\ |\ i \in \{1, \dots, n\}\}}
//' and stores it into a vector. The empirical risk \eqn{\mathcal{R}} for
//' iteration \eqn{m} is calculated by:
//' \deqn{
//'   \mathcal{R}_\mathrm{emp}^{[m]} = \frac{1}{n}\sum\limits_{i = 1}^n L(y^{(i)}, \hat{f}^{[m]}(x^{(i)}))
//' }
//'
//' \strong{Note:}
//' \itemize{
//'   \item
//'     If \eqn{m=0} than \eqn{\hat{f}} is just the offset.
//'
//'   \item
//'     The implementation to calculate \eqn{\mathcal{R}_\mathrm{emp}^{[m]}} is
//'     done in two steps:
//'       \enumerate{
//'        \item
//'          Calculate vector \code{risk_temp} of losses for every observation for
//'          given response \eqn{y^{(i)}} and prediction \eqn{\hat{f}^{[m]}(x^{(i)})}.
//'
//'        \item
//'          Average over \code{risk_temp}.
//'      }
//'    }
//'    This procedure ensures, that it is possible to e.g. use the AUC or any
//'    arbitrary performance measure for risk logging. This gives just one
//'    value for \code{risk_temp} and therefore the average equals the loss
//'    function. If this is just a value (like for the AUC) then the value is
//'    returned.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classlogger_1_1_inbag_risk_logger.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//'   \item{\code{summarizeLogger()}}{Summarize the logger object.}
//' }
//' @examples
//' # Used loss:
//' log.bin = LossBinomial$new()
//'
//' # Define logger:
//' log.inbag.risk = LoggerInbagRisk$new(FALSE, log.bin, 0.05)
//'
//' # Summarize logger:
//' log.inbag.risk$summarizeLogger()
//'
//' @export LoggerInbagRisk
class LoggerInbagRiskWrapper : public LoggerWrapper
{

private:
  double eps_for_break;
  bool use_as_stopper;

public:
  LoggerInbagRiskWrapper (bool use_as_stopper, LossWrapper& used_loss, double eps_for_break)
    : eps_for_break ( eps_for_break ),
      use_as_stopper ( use_as_stopper)
  {
    obj = new logger::LoggerInbagRisk (use_as_stopper, used_loss.getLoss(), eps_for_break);
    logger_id = "inbag.risk";
  }

  void summarizeLogger ()
  {
    Rcpp::Rcout << "Inbag risk logger:" << std::endl;
    if (use_as_stopper) {
      Rcpp::Rcout << "\t- Epsylon used to stop algorithm: " << eps_for_break << std::endl;
    }
    Rcpp::Rcout << "\t- Use logger as stopper: " << use_as_stopper;
  }
};

//' Logger class to log the out of bag risk
//'
//' This class logs the out of bag risk for a specific loss function. It is
//' also possible to use custom losses to log performance measures. For details
//' see the use case or extending compboost vignette.
//'
//' @format \code{\link{S4}} object.
//' @name LoggerOobRisk
//'
//' @section Usage:
//' \preformatted{
//' LoggerOobRisk$new(use_as_stopper, used_loss, eps_for_break, oob_data,
//'   oob_response)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{use_as_stopper} [\code{logical(1)}]}{
//'   Boolean to indicate if the logger should also be used as stopper.
//' }
//' \item{\code{used_loss} [\code{Loss} object]}{
//'   The loss used to calculate the empirical risk by taking the mean of the
//'   returned defined loss within the loss object.
//' }
//' \item{\code{eps_for_break} [\code{numeric(1)}]}{
//'   This argument is used if the loss is also used as stopper. If the relative
//'   improvement of the logged inbag risk falls above this boundary the stopper
//'   returns \code{TRUE}.
//' }
//' \item{\code{oob_data} [\code{list}]}{
//'   A list which contains data source objects which corresponds to the
//'   source data of each registered factory. The source data objects should
//'   contain the out of bag data. This data is then used to calculate the
//'   prediction in each step.
//' }
//' \item{\code{oob_response} [\code{numeric}]}{
//'   Vector which contains the response for the out of bag data given within
//'   the \code{list}.
//' }
//' }
//'
//' @section Details:
//'
//' This logger computes the risk for a given new dataset
//' \eqn{\mathcal{D}_\mathrm{oob} = \{(x^{(i)},\ y^{(i)})\ |\ i \in I_\mathrm{oob}\}}
//' and stores it into a vector. The OOB risk \eqn{\mathcal{R}_\mathrm{oob}} for
//' iteration \eqn{m} is calculated by:
//' \deqn{
//'   \mathcal{R}_\mathrm{oob}^{[m]} = \frac{1}{|\mathcal{D}_\mathrm{oob}|}\sum\limits_{(x,y) \in \mathcal{D}_\mathrm{oob}}
//'   L(y, \hat{f}^{[m]}(x))
//' }
//'
//' \strong{Note:}
//'   \itemize{
//'
//'   \item
//'     If \eqn{m=0} than \eqn{\hat{f}} is just the offset.
//'
//'   \item
//'     The implementation to calculate \eqn{\mathcal{R}_\mathrm{emp}^{[m]}} is
//'     done in two steps:
//'       \enumerate{
//'
//'       \item
//'         Calculate vector \code{risk_temp} of losses for every observation for
//'         given response \eqn{y^{(i)}} and prediction \eqn{\hat{f}^{[m]}(x^{(i)})}.
//'
//'       \item
//'         Average over \code{risk_temp}.
//'      }
//'    }
//'
//'    This procedure ensures, that it is possible to e.g. use the AUC or any
//'    arbitrary performance measure for risk logging. This gives just one
//'    value for \eqn{risk_temp} and therefore the average equals the loss
//'    function. If this is just a value (like for the AUC) then the value is
//'    returned.
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classlogger_1_1_oob_risk_logger.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{summarizeLogger()}}{Summarize the logger object.}
//' }
//' @examples
//' # Define data:
//' X1 = cbind(1:10)
//' X2 = cbind(10:1)
//' data.source1 = InMemoryData$new(X1, "x1")
//' data.source2 = InMemoryData$new(X2, "x2")
//'
//' oob.list = list(data.source1, data.source2)
//'
//' set.seed(123)
//' y.oob = rnorm(10)
//'
//' # Used loss:
//' log.bin = LossBinomial$new()
//'
//' # Define logger:
//' log.oob.risk = LoggerOobRisk$new(FALSE, log.bin, 0.05, oob.list, y.oob)
//'
//' # Summarize logger:
//' log.oob.risk$summarizeLogger()
//'
//' @export LoggerOobRisk
class LoggerOobRiskWrapper : public LoggerWrapper
{

private:
  double eps_for_break;
  bool use_as_stopper;

public:
  LoggerOobRiskWrapper (bool use_as_stopper, LossWrapper& used_loss, double eps_for_break,
    Rcpp::List oob_data, arma::vec oob_response)
  {
    std::map<std::string, data::Data*> oob_data_map;

    // Be very careful with the wrappers. For instance: doing something like
    // temp = oob_data[i] within the loop will force temp to call its destructor
    // when it runs out of scope. This will trigger the destructor of the
    // underlying data class which deletes the data needed for logging.
    // Therefore, the system crashes!
    //
    // Additionally auto would be more convenient but doesn't really make
    // things better since we need the conversion from SEXP to DataWrapper*.
    for (unsigned int i = 0; i < oob_data.size(); i++) {

      // Get data wrapper as pointer to prevent the underlying object from
      // destruction:
      DataWrapper* temp = oob_data[i];

      // Get the real data pointer:
      oob_data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();

    }

    obj = new logger::LoggerOobRisk (use_as_stopper, used_loss.getLoss(), eps_for_break,
      oob_data_map, oob_response);
    logger_id = "oob.risk";
  }

  void summarizeLogger ()
  {
    Rcpp::Rcout << "Out of bag risk logger:" << std::endl;
    if (use_as_stopper) {
      Rcpp::Rcout << "\t- Epsylon used to stop algorithm: " << eps_for_break << std::endl;
    }
    Rcpp::Rcout << "\t- Use logger as stopper: " << use_as_stopper;
  }
};

//' Logger class to log the elapsed time
//'
//' This class just logs the elapsed time. This should be very handy if one
//' wants to run the algorithm for just 2 hours and see how far he comes within
//' that time. There are three time units available for logging:
//' \itemize{
//'   \item minutes
//'   \item seconds
//'   \item microseconds
//' }
//'
//' @format \code{\link{S4}} object.
//' @name LoggerTime
//'
//' @section Usage:
//' \preformatted{
//' LoggerTime$new(use_as_stopper, max_time, time_unit)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{use_as_stopper} [\code{logical(1)}]}{
//'   Boolean to indicate if the logger should also be used as stopper.
//' }
//' \item{\code{max_time} [\code{integer(1)}]}{
//'   If the logger is used as stopper this argument contains the maximal time
//'   which are available to train the model.
//' }
//' \item{\code{time_unit} [\code{character(1)}]}{
//'   Character to specify the time unit. Possible choices are \code{minutes},
//'   \code{seconds} or \code{microseconds}
//' }
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classlogger_1_1_time_logger.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{summarizeLogger()}}{Summarize the logger object.}
//' }
//' @examples
//' # Define logger:
//' log.time = LoggerTime$new(FALSE, 20, "minutes")
//'
//' # Summarize logger:
//' log.time$summarizeLogger()
//'
//' @export LoggerTime
class LoggerTimeWrapper : public LoggerWrapper
{

private:
  bool use_as_stopper;
  unsigned int max_time;
  std::string time_unit;

public:
  LoggerTimeWrapper (bool use_as_stopper, unsigned int max_time,
    std::string time_unit)
    : use_as_stopper ( use_as_stopper ),
      max_time ( max_time ),
      time_unit ( time_unit )
  {
    obj = new logger::LoggerTime (use_as_stopper, max_time, time_unit);
    logger_id = "time." + time_unit;
  }

  void summarizeLogger ()
  {
    Rcpp::Rcout << "Time logger:" << std::endl;
    if (use_as_stopper) {
      Rcpp::Rcout << "\t- Stop algorithm if " << max_time << " " << time_unit << " are over" << std::endl;
    }
    Rcpp::Rcout << "\t- Tracked time unit: " << time_unit << std::endl;
  }
};



// Logger List:
// ------------

//' Logger list class to collect all loggers
//'
//' This class is meant to define all logger which should be used to track the
//' progress of the algorithm.
//'
//' @format \code{\link{S4}} object.
//' @name LoggerList
//'
//' @section Usage:
//' \preformatted{
//' LoggerList$new()
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classloggerlist_1_1_logger_list.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{clearRegisteredLogger()}}{Removes all registered logger
//'   from the list. The used logger are not deleted, just removed from the
//'   map.}
//' \item{\code{getNamesOfRegisteredLogger()}}{Returns the registered logger
//'   names as character vector.}
//' \item{\code{getNumberOfRegisteredLogger()}}{Returns the number of registered
//'   logger as integer.}
//' \item{\code{printRegisteredLogger()}}{Prints all registered logger.}
//' \item{\code{registerLogger(logger.id, logger)}}{Includes a new \code{logger}
//'   into the logger list with the \code{logger.id} as key.}
//' }
//' @examples
//' # Define logger:
//' log.iters = LoggerIteration$new(TRUE, 100)
//' log.time = LoggerTime$new(FALSE, 20, "minutes")
//'
//' # Create logger list:
//' logger.list = LoggerList$new()
//'
//' # Register new loggeR:
//' logger.list$registerLogger("iteration", log.iters)
//' logger.list$registerLogger("time", log.time)
//'
//' # Print registered logger:
//' logger.list$printRegisteredLogger()
//'
//' # Important: The keys has to be unique:
//' logger.list$registerLogger("iteration", log.iters)
//'
//' # Still just two logger:
//' logger.list$printRegisteredLogger()
//'
//' # Remove all logger:
//' logger.list$clearRegisteredLogger()
//'
//' # Get number of registered logger:
//' logger.list$getNumberOfRegisteredLogger()
//'
//' @export LoggerList
class LoggerListWrapper
{
private:

  loggerlist::LoggerList* obj = new loggerlist::LoggerList();

public:

  LoggerListWrapper () {};

  loggerlist::LoggerList* getLoggerList ()
  {
    return obj;
  }

  void registerLogger (const std::string& logger_id, LoggerWrapper& logger_wrapper)
  {
    obj->registerLogger(logger_id, logger_wrapper.getLogger());
  }

  void printRegisteredLogger ()
  {
    obj->printRegisteredLogger();
  }

  void clearRegisteredLogger ()
  {
    obj->clearMap();
  }

  unsigned int getNumberOfRegisteredLogger ()
  {
    return obj->getMap().size();
  }

  std::vector<std::string> getNamesOfRegisteredLogger ()
  {
    std::vector<std::string> out;
    for (auto& it : obj->getMap()) {
      out.push_back(it.first);
    }
    return out;
  }

  virtual ~LoggerListWrapper () { delete obj; }
};

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(LoggerWrapper)
RCPP_EXPOSED_CLASS(LoggerListWrapper)

RCPP_MODULE(logger_module)
{
  using namespace Rcpp;

  class_<LoggerWrapper> ("Logger")
    .constructor ()
  ;

  class_<LoggerIterationWrapper> ("LoggerIteration")
    .derives<LoggerWrapper> ("Logger")
    .constructor<bool, unsigned int> ()
    .method("summarizeLogger", &LoggerIterationWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerInbagRiskWrapper> ("LoggerInbagRisk")
    .derives<LoggerWrapper> ("Logger")
    .constructor<bool, LossWrapper&, double> ()
    .method("summarizeLogger", &LoggerInbagRiskWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerOobRiskWrapper> ("LoggerOobRisk")
    .derives<LoggerWrapper> ("Logger")
    .constructor<bool, LossWrapper&, double, Rcpp::List, arma::vec> ()
    .method("summarizeLogger", &LoggerOobRiskWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerTimeWrapper> ("LoggerTime")
    .derives<LoggerWrapper> ("Logger")
    .constructor<bool, unsigned int, std::string> ()
    .method("summarizeLogger", &LoggerTimeWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerListWrapper> ("LoggerList")
    .constructor ()
    .method("registerLogger", &LoggerListWrapper::registerLogger, "Register Logger")
    .method("printRegisteredLogger", &LoggerListWrapper::printRegisteredLogger, "Print registered logger")
    .method("clearRegisteredLogger", &LoggerListWrapper::clearRegisteredLogger, "Clear registered logger")
    .method("getNumberOfRegisteredLogger", &LoggerListWrapper::getNumberOfRegisteredLogger, "Get number of registered logger. Mainly for testing.")
    .method("getNamesOfRegisteredLogger",  &LoggerListWrapper::getNamesOfRegisteredLogger, "Get names of registered logger. Mainly for testing.")
  ;
}


// -------------------------------------------------------------------------- //
//                                 OPTIMIZER                                  //
// -------------------------------------------------------------------------- //

class OptimizerWrapper
{
public:
  OptimizerWrapper () {};
  optimizer::Optimizer* getOptimizer () { return obj; }

  virtual ~OptimizerWrapper () { delete obj; }

protected:
  optimizer::Optimizer* obj;
};

//' Greedy Optimizer
//'
//' This class defines a new object for the greedy optimizer. The optimizer
//' just calculates for each base-learner the sum of squared errors and returns
//' the base-learner with the smallest SSE.
//'
//' @format \code{\link{S4}} object.
//' @name OptimizerCoordinateDescent
//'
//' @section Usage:
//' \preformatted{
//' OptimizerCoordinateDescent$new()
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classoptimizer_1_1_greedy_optimizer.html}.
//'
//' @examples
//'
//' # Define optimizer:
//' optimizer = OptimizerCoordinateDescent$new()
//'
//' @export OptimizerCoordinateDescent
class OptimizerCoordinateDescent : public OptimizerWrapper
{
public:
  OptimizerCoordinateDescent () { obj = new optimizer::OptimizerCoordinateDescent(); }

  // Rcpp::List testOptimizer (arma::vec& response, BlearnerFactoryListWrapper factory_list)
  // {
  //   std::string temp_str = "test run";
  //   blearner::Baselearner* blearner_test = obj->findBestBaselearner(temp_str, response, factory_list.getFactoryList()->getMap());

  //   Rcpp::List out = Rcpp::List::create(
  //     Rcpp::Named("selected.learner") = blearner_test->getIdentifier(),
  //     Rcpp::Named("parameter")        = blearner_test->getParameter()
  //   );

  //   delete blearner_test;

  //   return out;
  // }
};

RCPP_EXPOSED_CLASS(OptimizerWrapper)
RCPP_MODULE(optimizer_module)
{
  using namespace Rcpp;

  class_<OptimizerWrapper> ("Optimizer")
    .constructor ()
  ;

  class_<OptimizerCoordinateDescent> ("OptimizerCoordinateDescent")
    .derives<OptimizerWrapper> ("Optimizer")
    .constructor ()
  ;
}


// -------------------------------------------------------------------------- //
//                                 COMPBOOST                                  //
// -------------------------------------------------------------------------- //

//' Main Compboost Class
//'
//' This class collects all parts such as the factory list or the used logger
//' and passes them to \code{C++}. On the \code{C++} side is then the main
//' algorithm.
//'
//' @format \code{\link{S4}} object.
//' @name Compboost_internal
//'
//' @section Usage:
//' \preformatted{
//' Compboost$new(response, learning_rate, stop_if_all_stopper_fulfilled,
//'   factory_list, loss, logger_list, optimizer)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{response} [\code{numeric}]}{
//'   Vector of the true values which should be modeled.
//' }
//' \item{\code{learning_rate} [\code{numeric(1)}]}{
//'   The learning rate which is used to shrink the parameter in each iteration.
//' }
//' \item{\code{stop_if_all_stopper_fulfilled} [\code{logical(1)}]}{
//'   Boolean to indicate which stopping strategy is used. If \code{TRUE} then
//'   the algorithm stops if all registered logger stopper are fulfilled.
//' }
//' \item{\code{factory_list} [\code{BlearnerFactoryList} object]}{
//'   List of base-learner factories from which one base-learner is selected
//'   in each iteration by using the
//' }
//' \item{\code{loss} [\code{Loss} object]}{
//'   The loss which should be used to calculate the pseudo residuals in each
//'   iteration.
//' }
//' \item{\code{logger_list} [\code{LoggerList} object]}{
//'   The list with all registered logger which are used to track the algorithm.
//' }
//' \item{\code{optimizer} [\code{Optimizer} object]}{
//'   The optimizer which is used to select in each iteration one good
//'   base-learner.
//' }
//' }
//'
//' @section Details:
//'
//'   This class is a wrapper around the pure \code{C++} implementation. To see
//'   the functionality of the \code{C++} class visit
//'   \url{https://schalkdaniel.github.io/compboost/cpp_man/html/classcboost_1_1_compboost.html}.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{train(trace)}}{Initial training of the model. The integer
//'   argument \code{trace} indicates if the logger progress should be printed
//'   or not and if so trace indicates which iterations should be printed.}
//' \item{\code{continueTraining(trace, logger_list)}}{Continue the training
//'   by using an additional \code{logger_list}. The retraining is stopped if
//'   the first logger says that the algorithm should be stopped.}
//' \item{\code{getPrediction()}}{Get the inbag prediction which is done during
//'   the fitting process.}
//' \item{\code{getSelectedBaselearner()}}{Returns a character vector of how
//'   the base-learner are selected.}
//' \item{\code{getLoggerData()}}{Returns a list of all logged data. If the
//'   algorithm is retrained, then the list contains for each training one
//'   element.}
//' \item{\code{getEstimatedParameter()}}{Returns a list with the estimated
//'   parameter for base-learner which was selected at least once.}
//' \item{\code{getParameterAtIteration(k)}}{Calculates the prediction at the
//'   iteration \code{k}.}
//' \item{\code{getParameterMatrix()}}{Calculates a matrix where row \code{i}
//'   includes the parameter at iteration \code{i}. There are as many rows
//'   as done iterations.}
//' \item{\code{isTrained()}}{This function returns just a boolean value which
//'   indicates if the initial training was already done.}
//' \item{\code{predict(newdata)}}{Prediction on new data organized within a
//'   list of source data objects. It is important that the names of the source
//'   data objects matches those one that were used to define the factories.}
//' \item{\code{predictAtIteration(newdata, k)}}{Prediction on new data by using
//'   another iteration \code{k}.}
//' \item{\code{setToIteration(k)}}{Set the whole model to another iteration
//'   \code{k}. After calling this function all other elements such as the
//'   parameters or the prediction are calculated corresponding to \code{k}.}
//' \item{\code{summarizeCompboost()}}{Summarize the \code{Compboost} object.}
//' }
//' @examples
//'
//' # Some data:
//' df = mtcars
//' df$mpg.cat = ifelse(df$mpg > 20, 1, -1)
//'
//' # # Create new variable to check the polynomial base-learner with degree 2:
//' # df$hp2 = df[["hp"]]^2
//'
//' # Data for the baselearner are matrices:
//' X.hp = as.matrix(df[["hp"]])
//' X.wt = as.matrix(df[["wt"]])
//'
//' # Target variable:
//' y = df[["mpg.cat"]]
//'
//' data.source.hp = InMemoryData$new(X.hp, "hp")
//' data.source.wt = InMemoryData$new(X.wt, "wt")
//'
//' data.target.hp1 = InMemoryData$new()
//' data.target.hp2 = InMemoryData$new()
//' data.target.wt1 = InMemoryData$new()
//' data.target.wt2 = InMemoryData$new()
//'
//' # List for oob logging:
//' oob.data = list(data.source.hp, data.source.wt)
//'
//' # List to test prediction on newdata:
//' test.data = oob.data
//'
//' # Factories:
//' linear.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target.hp1, 1, TRUE)
//' linear.factory.wt = BaselearnerPolynomial$new(data.source.wt, data.target.wt1, 1, TRUE)
//' quadratic.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target.hp2, 2, TRUE)
//' spline.factory.wt = BaselearnerPSpline$new(data.source.wt, data.target.wt2, 3, 10, 2, 2)
//'
//' # Create new factory list:
//' factory.list = BlearnerFactoryList$new()
//'
//' # Register factories:
//' factory.list$registerFactory(linear.factory.hp)
//' factory.list$registerFactory(linear.factory.wt)
//' factory.list$registerFactory(quadratic.factory.hp)
//' factory.list$registerFactory(spline.factory.wt)
//'
//' # Define loss:
//' loss.bin = LossBinomial$new()
//'
//' # Define optimizer:
//' optimizer = OptimizerCoordinateDescent$new()
//'
//' ## Logger
//'
//' # Define logger. We want just the iterations as stopper but also track the
//' # time, inbag risk and oob risk:
//' log.iterations  = LoggerIteration$new(TRUE, 500)
//' log.time        = LoggerTime$new(FALSE, 500, "microseconds")
//' log.inbag       = LoggerInbagRisk$new(FALSE, loss.bin, 0.05)
//' log.oob         = LoggerOobRisk$new(FALSE, loss.bin, 0.05, oob.data, y)
//'
//' # Define new logger list:
//' logger.list = LoggerList$new()
//'
//' # Register the logger:
//' logger.list$registerLogger(" iteration.logger", log.iterations)
//' logger.list$registerLogger("time.logger", log.time)
//' logger.list$registerLogger("inbag.binomial", log.inbag)
//' logger.list$registerLogger("oob.binomial", log.oob)
//'
//' # Run compboost:
//' # --------------
//'
//' # Initialize object:
//' cboost = Compboost_internal$new(
//'   response      = y,
//'   learning_rate = 0.05,
//'   stop_if_all_stopper_fulfilled = FALSE,
//'   factory_list = factory.list,
//'   loss         = loss.bin,
//'   logger_list  = logger.list,
//'   optimizer    = optimizer
//' )
//'
//' # Train the model (we want to print the trace):
//' cboost$train(trace = 50)
//' cboost
//'
//' # Get estimated parameter:
//' cboost$getEstimatedParameter()
//'
//' # Get trace of selected base-learner:
//' cboost$getSelectedBaselearner()
//'
//' # Set to iteration 200:
//' cboost$setToIteration(200)
//'
//' # Get new parameter values:
//' cboost$getEstimatedParameter()
//'
//' @export Compboost_internal
class CompboostWrapper
{
public:

  // Set data type in constructor to
  //   - arma::vec -> const arma::vec&
  //   - double    -> const double &
  //   - bool      -> const bool&
  // crashes the compilation?
  CompboostWrapper (arma::vec response, double learning_rate,
    bool stop_if_all_stopper_fulfilled, BlearnerFactoryListWrapper& factory_list,
    LossWrapper& loss, LoggerListWrapper& logger_list, OptimizerWrapper& optimizer)
  {

    learning_rate0 = learning_rate;
    used_logger    = logger_list.getLoggerList();
    used_optimizer = optimizer.getOptimizer();
    blearner_list_ptr = factory_list.getFactoryList();

    obj = new cboost::Compboost(response, learning_rate0, stop_if_all_stopper_fulfilled,
      used_optimizer, loss.getLoss(), used_logger, *blearner_list_ptr);
  }

  // Member functions
  void train (unsigned int trace)
  {
    obj->trainCompboost(trace);
    is_trained = true;
  }

  void continueTraining (unsigned int trace, LoggerListWrapper& logger_list)
  {
    obj->continueTraining(logger_list.getLoggerList(), trace);
  }

  arma::vec getPrediction (bool as_response)
  {
    return obj->getPrediction(as_response);
  }

  std::vector<std::string> getSelectedBaselearner ()
  {
    return obj->getSelectedBaselearner();
  }

  Rcpp::List getLoggerData ()
  {
    Rcpp::List out_list;

    for (auto& it : obj->getLoggerList()) {
      out_list[it.first] = Rcpp::List::create(
        Rcpp::Named("logger.names") = it.second->getLoggerData().first,
        Rcpp::Named("logger.data")  = it.second->getLoggerData().second
      );
    }
    if (out_list.size() == 1) {
      return out_list[0];
    } else {
      return out_list;
    }
  }

  Rcpp::List getEstimatedParameter ()
  {
    std::map<std::string, arma::mat> parameter = obj->getParameter();

    Rcpp::List out;

    for (auto &it : parameter) {
      out[it.first] = it.second;
    }
    return out;
  }

  Rcpp::List getParameterAtIteration (unsigned int k)
  {
    std::map<std::string, arma::mat> parameter = obj->getParameterOfIteration(k);

    Rcpp::List out;

    for (auto &it : parameter) {
      out[it.first] = it.second;
    }
    return out;
  }

  Rcpp::List getParameterMatrix ()
  {
    std::pair<std::vector<std::string>, arma::mat> out_pair = obj->getParameterMatrix();

    return Rcpp::List::create(
      Rcpp::Named("parameter.names")   = out_pair.first,
      Rcpp::Named("parameter.matrix")  = out_pair.second
    );
  }

  arma::vec predict (Rcpp::List& newdata, bool as_response)
  {
    std::map<std::string, data::Data*> data_map;

    // Create data map (see line 780, same applies here):
    for (unsigned int i = 0; i < newdata.size(); i++) {

      // Get data wrapper:
      DataWrapper* temp = newdata[i];

      // Get the real data pointer:
      data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();

    }
    return obj->predict(data_map, as_response);
  }

  arma::vec predictAtIteration (Rcpp::List& newdata, unsigned int k, bool as_response)
  {
    std::map<std::string, data::Data*> data_map;

    // Create data map (see line 780, same applies here):
    for (unsigned int i = 0; i < newdata.size(); i++) {

      // Get data wrapper:
      DataWrapper* temp = newdata[i];

      // Get the real data pointer:
      data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();

    }
    return obj->predictionOfIteration(data_map, k, as_response);
  }

  void summarizeCompboost ()
  {
    obj->summarizeCompboost();
  }

  bool isTrained ()
  {
    return is_trained;
  }

  double getOffset ()
  {
    return obj->getOffset();
  }

  std::vector<double> getRiskVector ()
  {
    return obj->getRiskVector();
  }

  void setToIteration (const unsigned int& k)
  {
    obj->setToIteration(k);
  }

  // Destructor:
  ~CompboostWrapper ()
  {
    delete obj;
  }

private:

  blearnerlist::BaselearnerFactoryList* blearner_list_ptr;
  loggerlist::LoggerList* used_logger;
  optimizer::Optimizer* used_optimizer;
  cboost::Compboost* obj;
  arma::mat* eval_data;

  unsigned int max_iterations;
  double learning_rate0;

  bool is_trained = false;
};


RCPP_EXPOSED_CLASS(CompboostWrapper)
RCPP_MODULE (compboost_module)
{
  using namespace Rcpp;

  class_<CompboostWrapper> ("Compboost_internal")
    .constructor<arma::vec, double, bool, BlearnerFactoryListWrapper&, LossWrapper&, LoggerListWrapper&, OptimizerWrapper&> ()
    .method("train", &CompboostWrapper::train, "Run componentwise boosting")
    .method("continueTraining", &CompboostWrapper::continueTraining, "Continue Training")
    .method("getPrediction", &CompboostWrapper::getPrediction, "Get prediction")
    .method("getSelectedBaselearner", &CompboostWrapper::getSelectedBaselearner, "Get vector of selected base-learner")
    .method("getLoggerData", &CompboostWrapper::getLoggerData, "Get data of the used logger")
    .method("getEstimatedParameter", &CompboostWrapper::getEstimatedParameter, "Get the estimated paraemter")
    .method("getParameterAtIteration", &CompboostWrapper::getParameterAtIteration, "Get the estimated parameter for iteration k < iter.max")
    .method("getParameterMatrix", &CompboostWrapper::getParameterMatrix, "Get matrix of all estimated parameter in each iteration")
    .method("predict", &CompboostWrapper::predict, "Predict newdata")
    .method("predictAtIteration", &CompboostWrapper::predictAtIteration, "Predict newdata for iteration k < iter.max")
    .method("summarizeCompboost",    &CompboostWrapper::summarizeCompboost, "Sumamrize compboost object.")
    .method("isTrained", &CompboostWrapper::isTrained, "Status of algorithm if it is already trained.")
    .method("setToIteration", &CompboostWrapper::setToIteration, "Set state of the model to a given iteration")
    .method("getOffset", &CompboostWrapper::getOffset, "Get offset.")
    .method("getRiskVector", &CompboostWrapper::getRiskVector, "Get the risk vector.")
  ;
}

#endif // COMPBOOST_MODULES_CPP_
