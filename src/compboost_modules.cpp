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
// ========================================================================== //

#ifndef COMPBOOST_MODULES_CPP_
#define COMPBOOST_MODULES_CPP_

#include "compboost.h"
#include "baselearner_factory.h"
#include "baselearner_factory_list.h"
#include "loss.h"
#include "data.h"
#include "helper.h"
#include "optimizer.h"
#include "response.h"


// -------------------------------------------------------------------------- //
//                                   DATA                                     //
// -------------------------------------------------------------------------- //

class DataWrapper
{
public:
  DataWrapper () {}
  virtual std::shared_ptr<data::Data> getDataObj () { return sh_ptr_data; }
  virtual ~DataWrapper () {}

protected:
  std::shared_ptr<data::Data> sh_ptr_data;
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
//' InMemoryData$new(data_mat, data_identifier)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_mat} [\code{matrix}]}{
//'   Matrix containing the source data. This source data is later transformed
//'   to obtain the design matrix a base-learner uses for training.
//' }
//' \item{\code{data_identifier} [\code{character(1)}]}{
//'   The name for the data specified in \code{data_mat}. Note that it is
//'   important to have the same data names for train and evaluation data.
//' }
//' }
//'
//'
//' @section Details:
//'   The \code{data_mat} needs to suits the base-learner. For instance, the
//'   spline base-learner does just take a one column matrix since there are
//'   just one dimensional splines at the moment.
//'
//'   The \code{data_mat} and \code{data_identifier} of a target data object
//'   is set automatically by passing the source and target object to a
//'   factory. \code{getData()} can then be used to access the
//'   transformed data of the target object.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{}
//' \item{\code{getIdentifier()}}{}
//' }
//' @examples
//' # Sample data:
//' data_mat = cbind(1:10)
//'
//' # Create new data object:
//' data_obj = InMemoryData$new(data_mat, "my_data_name")
//'
//' # Get data and identifier:
//' data_obj$getData()
//' data_obj$getIdentifier()
//'
//' @export InMemoryData
class InMemoryDataWrapper : public DataWrapper
{

// Solve this copying issue:
// https://github.com/schalkdaniel/compboost/issues/123

public:

  InMemoryDataWrapper ()
  {
    sh_ptr_data = std::make_shared<data::InMemoryData>("");
  }

  InMemoryDataWrapper (arma::mat data_mat, std::string data_identifier)
  {
    // data_mat = data0;
    sh_ptr_data = std::make_shared<data::InMemoryData>(data_identifier, data_mat);
  }

  InMemoryDataWrapper (arma::mat data_mat, std::string data_identifier, bool use_sparse)
  {
    // data_mat = data0;
    arma::sp_mat temp_sp_mat(data_mat);
    sh_ptr_data = std::make_shared<data::InMemoryData>(data_identifier, temp_sp_mat);
  }

  arma::mat getData () const
  {
    return sh_ptr_data->getDenseData();
  }


  std::string getIdentifier () const
  {
    return sh_ptr_data->getDataIdentifier();
  }
};


//' Data class for character variables
//'
//' \code{CategoricalDataRaw} creates an data object which can be used as source
//' object to instantiate categorical base learner. In contrast to \code{CategoricalData}
//' the data are stored as raw categorical vector.
//'
//' @format \code{\link{S4}} object.
//' @name CategoricalDataRaw
//'
//' @section Usage:
//' \preformatted{
//' CategoricalDataRaw$new(x, data_identifier)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{x} [\code{character}]}{
//'   Character vector containing the classes.
//' }
//' \item{\code{data_identifier} [\code{character(1)}]}{
//'   The name for the data specified in \code{data_mat}. Note that it is
//'   important to have the same data names for train and evaluation data.
//' }
//' }
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Throws error because no representation is calculated.}
//' \item{\code{getRawData()}}{Get raw character data.}
//' \item{\code{getIdentifier()}}{Get data identifier.}
//' }
//' @examples
//' # Sample data:
//' x = sample(c("one","two", "three"), 20, TRUE)
//'
//' # Create new data object:
//' data_obj = CategoricalDataRaw$new(x, "cat_raw")
//'
//' # Get data and identifier:
//' data_obj$getRawData()
//' data_obj$getIdentifier()
//'
//' @export CategoricalDataRaw
class CategoricalDataRawWrapper : public DataWrapper
{
private:
  std::shared_ptr<data::CategoricalDataRaw> _sh_ptr_rawcdata;

public:

  CategoricalDataRawWrapper (Rcpp::StringVector classes, std::string data_identifier)
  {
    std::vector<std::string> str_classes = Rcpp::as< std::vector<std::string> >(classes);
    _sh_ptr_rawcdata = std::make_shared<data::CategoricalDataRaw>(data_identifier, str_classes);
  }

  std::shared_ptr<data::CategoricalDataRaw> getCDataRawPtr () const { return _sh_ptr_rawcdata; }
  std::shared_ptr<data::Data> getDataObj () { return _sh_ptr_rawcdata; }

  arma::mat getData () const
  {
    return _sh_ptr_rawcdata->getData();
  }

  std::string getIdentifier () const
  {
    return _sh_ptr_rawcdata->getDataIdentifier();
  }

  std::vector<std::string> getRawData () const
  {
    return _sh_ptr_rawcdata->getRawData();
  }
};





RCPP_EXPOSED_CLASS(DataWrapper)
//RCPP_EXPOSED_CLASS(CategoricalDataWrapper)
RCPP_EXPOSED_CLASS(CategoricalDataRawWrapper)
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
    .constructor<arma::mat, std::string, bool> ()

    .method("getData",       &InMemoryDataWrapper::getData, "Get data")
    .method("getIdentifier", &InMemoryDataWrapper::getIdentifier, "Get the data identifier")
  ;

  class_<CategoricalDataRawWrapper> ("CategoricalDataRaw")
    .derives<DataWrapper> ("Data")

    .constructor<Rcpp::StringVector, std::string> ()

    .method("getData",       &CategoricalDataRawWrapper::getData, "Get data")
    .method("getRawData",    &CategoricalDataRawWrapper::getRawData, "Get raw data")
    .method("getIdentifier", &CategoricalDataRawWrapper::getIdentifier, "Get the data identifier")
  ;


}


// -------------------------------------------------------------------------- //
//                         BASELEARNER FACTORIES                              //
// -------------------------------------------------------------------------- //

// Abstract class. This one is given to the factory list. The factory list then
// handles all factories equally. It does not differ between a polynomial or
// a custom factory:
class BaselearnerFactoryWrapper
{
public:
  std::shared_ptr<blearnerfactory::BaselearnerFactory> getFactory () { return sh_ptr_blearner_factory; }
  virtual ~BaselearnerFactoryWrapper () {}

  arma::mat getData () { return sh_ptr_blearner_factory->getData(); }
  std::string getDataIdentifier () { return sh_ptr_blearner_factory->getDataIdentifier(); }
  std::string getBaselearnerType () { return sh_ptr_blearner_factory->getBaselearnerType(); }
  //arma::mat transformData (const arma::mat& newdata) { return sh_ptr_blearner_factory->instantiateData(newdata); }

  std::string getFeatureName () const { return sh_ptr_blearner_factory->getDataIdentifier(); }

protected:
  std::shared_ptr<blearnerfactory::BaselearnerFactory> sh_ptr_blearner_factory;
};


//' Base-learner factory for polynomial regression
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
//' BaselearnerPolynomial$new(data_source, list(degree, intercept))
//' BaselearnerPolynomial$new(data_source, blearner_type, list(degree, intercept))
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
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
//' data_mat = cbind(1:10)
//'
//' # Create new data object:
//' data_source = InMemoryData$new(data_mat, "my_data_name")
//'
//' # Create new linear base-learner factory:
//' lin_factory = BaselearnerPolynomial$new(data_source,
//'   list(degree = 2, intercept = FALSE))
//' lin_factory_int = BaselearnerPolynomial$new(data_source,
//'   list(degree = 2, intercept = TRUE))
//'
//' # Get the transformed data:
//' lin_factory$getData()
//' lin_factory_int$getData()
//'
//' # Summarize factory:
//' lin_factory$summarizeFactory()
//'
//' # Transform data manually:
//' lin_factory$transformData(data_mat)
//' lin_factory_int$transformData(data_mat)
//'
//' @export BaselearnerPolynomial
class BaselearnerPolynomialFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  Rcpp::List internal_arg_list = Rcpp::List::create(
    Rcpp::Named("degree") = 1,
    Rcpp::Named("intercept") = true
  );

public:

  BaselearnerPolynomialFactoryWrapper (DataWrapper& data_source,
    Rcpp::List arg_list)
  {
    // Match defaults with custom arguments:
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, TRUE);

    // We need to converse the SEXP from the element to an integer:
    int degree = internal_arg_list["degree"];

    std::string blearner_type_temp = "polynomial_degree_" + std::to_string(degree);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPolynomialFactory>(blearner_type_temp, data_source.getDataObj(),
       internal_arg_list["degree"], internal_arg_list["intercept"]);
  }

  BaselearnerPolynomialFactoryWrapper (DataWrapper& data_source,
    const std::string& blearner_type, Rcpp::List arg_list)
  {
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, TRUE);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPolynomialFactory>(blearner_type, data_source.getDataObj(),
      internal_arg_list["degree"], internal_arg_list["intercept"]);
  }

  void summarizeFactory ()
  {
    // We need to converse the SEXP from the element to an integer:
    int degree = internal_arg_list["degree"];

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
    Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
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
//' BaselearnerPSpline$new(data_source, list(degree, n_knots, penalty,
//'   differences, df))
//' }
//'
//' @section arguments:
//' \describe{
//' \item{\code{data_source} [\code{data} object]}{
//'   data object which contains the source data.
//' }
//' \item{\code{degree} [\code{integer(1)}]}{
//'   degree of the spline functions to interpolate the knots.
//' }
//' \item{\code{n_knots} [\code{integer(1)}]}{
//'   number of \strong{inner knots}. to prevent weird behavior on the edges
//'   the inner knots are expanded by \eqn{\mathrm{degree} - 1} additional knots.
//' }
//' \item{\code{penalty} [\code{numeric(1)}]}{
//'   positive numeric value to specify the penalty parameter. setting the
//'   penalty to 0 ordinary b-splines are used for the fitting.
//' }
//' \item{\code{differences} [\code{integer(1)}]}{
//'   the number of differences which are penalized. a higher value leads to
//'   smoother curves.
//' }
//' \item{\code{bin_root} [\code{integer(1)}]}{
//'   If set to a value greater than zero, binning is applied and reduces the number of used
//'   x values to n^(1/bin_root) equidistant points. If you want to use binning we suggest
//'   to set \code{bin_root = 2}.
//' }
//' }
//'
//' @section Details:
//'   If \code{bin_root > 0} the original feature is discretized to on an equidistant grid on
//'   \eqn{[\min(x),\max(x)]} with \eqn{\sqrt{n}} points. The fitting is then done by
//'   using weights per new data point.
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
//' data_mat = cbind(1:10)
//' y = sin(1:10)
//'
//' # Create new data object:
//' data_source = InMemoryData$new(data_mat, "my_data_name")
//'
//' # Create new linear base-learner:
//' spline_factory = BaselearnerPSpline$new(data_source,
//'   list(degree = 3, n_knots = 4, penalty = 2, differences = 2))
//'
//' # Get the transformed data:
//' spline_factory$getData()
//'
//' # Summarize factory:
//' spline_factory$summarizeFactory()
//'
//' # Transform data manually:
//' spline_factory$transformData(data_mat)
//'
//' @export BaselearnerPSpline
class BaselearnerPSplineFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  Rcpp::List internal_arg_list = Rcpp::List::create(
    Rcpp::Named("degree") = 3,
    Rcpp::Named("n_knots") = 20,
    Rcpp::Named("penalty") = 2,
    Rcpp::Named("df") = 0,
    Rcpp::Named("differences") = 2,
    Rcpp::Named("bin_root") = 0,
    Rcpp::Named("cache_type") = "inverse"
  );

public:

  BaselearnerPSplineFactoryWrapper (DataWrapper& data_source, Rcpp::List arg_list)
  {
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, true);

    // We need to converse the SEXP from the element to an integer:
    int degree = internal_arg_list["degree"];

    std::string blearner_type_temp = "spline_degree_" + std::to_string(degree);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPSplineFactory>(blearner_type_temp, data_source.getDataObj(),
      internal_arg_list["degree"], internal_arg_list["n_knots"], internal_arg_list["penalty"], internal_arg_list["df"], internal_arg_list["differences"], true,
      internal_arg_list["bin_root"], internal_arg_list["cache_type"]);
  }

  BaselearnerPSplineFactoryWrapper (DataWrapper& data_source, const std::string& blearner_type, Rcpp::List arg_list)
  {
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, true);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPSplineFactory>(blearner_type, data_source.getDataObj(),
      internal_arg_list["degree"], internal_arg_list["n_knots"], internal_arg_list["penalty"], internal_arg_list["df"], internal_arg_list["differences"], true,
      internal_arg_list["bin_root"], internal_arg_list["cache_type"]);
  }

  void summarizeFactory ()
  {
    // We need to converse the SEXP from the element to an integer:
    int degree = internal_arg_list["degree"];

    Rcpp::Rcout << "Spline factory of degree" << " " << std::to_string(degree) << std::endl;
    Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
  }
};

//' Base-learner factory to make regression using tensor products
//'
//' \code{BaselearnerTensor} creates a combined base-learner factory
//'  object which can be registered within a base-learner list and then used
//'  for training.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerTensor
//'
//' @export BaselearnerTensor
class BaselearnerTensorFactoryWrapper : public BaselearnerFactoryWrapper
{
private:

public:

  BaselearnerTensorFactoryWrapper (BaselearnerFactoryWrapper& blearner1, BaselearnerFactoryWrapper& blearner2, std::string blc)
  {
    // We need to converse the SEXP from the element to an integer:
    std::string blearner_type_temp = blc;

    std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner1 = blearner1.getFactory();
    std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner2 = blearner2.getFactory();

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerTensorFactory>(blearner_type_temp, ptr_blearner1,
      ptr_blearner2);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
  }
};

//' Base-learner factory to make regression using centered base learner
//'
//' \code{BaselearnerCentered} creates a base learner factory which is centered
//'  using another base learner.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerCentered
//'
//' @export BaselearnerCentered
class BaselearnerCenteredFactoryWrapper : public BaselearnerFactoryWrapper
{
private:

public:
  BaselearnerCenteredFactoryWrapper (BaselearnerFactoryWrapper& blearner1, BaselearnerFactoryWrapper& blearner2, std::string blc)
  {
    // We need to converse the SEXP from the element to an integer:
    std::string blearner_type_temp = blc;

    std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner1 = blearner1.getFactory();
    std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner2 = blearner2.getFactory();

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCenteredFactory>(blearner_type_temp, ptr_blearner1,
      ptr_blearner2);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
  }
  arma::mat getRotation ()
  {
    return std::static_pointer_cast<blearnerfactory::BaselearnerCenteredFactory>(sh_ptr_blearner_factory)->getRotation();
  }
};




//' Base-learner factory for categorical feature using Ridge penalty
//'
//' \code{BaselearnerCategoricalRidge} can be used to estimate effects of  categorical
//' features. The categories are included as in the linear model by using a binary matrix.
//' The Ridge penalty enables unbiased feature selection by setting the penalty corresponding
//' to degree of freedoms.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerCategoricalRidge
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCategoricalRidge$new(data_source, list(df))
//' }
//'
//' @section arguments:
//' \describe{
//' \item{\code{data_source} [\code{data} object]}{
//'   data object which contains the source data.
//' }
//' }
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Get the data matrix of the target data which is used
//'   for modeling.}
//' \item{\code{transformData(X)}}{This class does is not allowed to transform data. This is due to the internal structure.}
//' \item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
//' }
//' @examples
//' # Sample data:
//' x = sample(c(0,1), 20, TRUE)
//' data_mat = cbind(x)
//'
//' @export BaselearnerCategoricalRidge
class BaselearnerCategoricalRidgeFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  Rcpp::List internal_arg_list = Rcpp::List::create(
    Rcpp::Named("df") = .0
  );

public:
  BaselearnerCategoricalRidgeFactoryWrapper (const CategoricalDataRawWrapper& cdata_source, Rcpp::List arg_list)
  {
    std::string blearner_type_temp = cdata_source.getCDataRawPtr()->getDataIdentifier();
    std::shared_ptr<data::CategoricalDataRaw> temp = cdata_source.getCDataRawPtr();

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCategoricalRidgeFactory>(blearner_type_temp, temp, internal_arg_list["df"]);
  }

  BaselearnerCategoricalRidgeFactoryWrapper (const CategoricalDataRawWrapper& cdata_source, std::string blearner_type, Rcpp::List arg_list)
  {
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, true);
    std::shared_ptr<data::CategoricalDataRaw> temp = cdata_source.getCDataRawPtr();
    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCategoricalRidgeFactory>(blearner_type, temp, internal_arg_list["df"]);
}

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Categorical base-learner of category " << sh_ptr_blearner_factory->getDataIdentifier() << std::endl;
  }
};


//' Base-learner factory for categorical feature on a binary base-learner basis
//'
//' \code{BaselearnerCategoricalBinary} can be used to estimate effects of one category of a categorical
//' feature. The base-learner gets the data as index vector of the observations assigned to the group.
//' For example, if the vector is (a, a, b, b, a, b), then the index vector is (1, 2, 5) for group a.
//' This reduces memory load and fasten the fitting process.
//'
//' @format \code{\link{S4}} object.
//' @name BaselearnerCategoricalBinary
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCategoricalBinary$new(data_source, list(n_obs))
//' }
//'
//' @section arguments:
//' \describe{
//' \item{\code{data_source} [\code{data} object]}{
//'   data object of class \code{CategoricalData} which contains the source data.
//' }
//' }
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{getData()}}{Get the data matrix of the target data which is used
//'   for modeling.}
//' \item{\code{transformData(X)}}{Transform a data matrix as defined within the
//'   factory. The argument has to be a matrix with one column. In case of the categorical
//'   binary base-learner this is the index of non-zero elements concatinated with the
//'   number of observations. This helps to fully reconstruct the original feature by using less memory. This also speed up computation time.}
//' \item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
//' }
//' @examples
//' # Sample data:
//' x = sample(c("pos","neg"), 20, TRUE)
//'
//' # Create new data object:
//' data_source = CategoricalData$new(x, "pos")
//'
//' # Create new linear base-learner:
//' cat_factory = BaselearnerCategoricalBinary$new(data_source, "pos")
//'
//' # Get the transformed data as stored for internal use:
//' cat_factory$getData()
//'
//' # Summarize factory:
//' cat_factory$summarizeFactory()
//'
//' @export BaselearnerCategoricalBinary
class BaselearnerCategoricalBinaryFactoryWrapper : public BaselearnerFactoryWrapper
{

public:

  BaselearnerCategoricalBinaryFactoryWrapper (const CategoricalDataRawWrapper& data_source, std::string cls)
  {
    std::string blearner_type_temp = data_source.getCDataRawPtr()->getDataIdentifier() + "_" + cls;

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCategoricalBinaryFactory>(blearner_type_temp, cls, data_source.getCDataRawPtr());
  }

  BaselearnerCategoricalBinaryFactoryWrapper (const CategoricalDataRawWrapper& data_source, std::string cls, std::string blearner_type)
  {
    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCategoricalBinaryFactory>(blearner_type, cls, data_source.getCDataRawPtr());
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Categorical factory of category " << sh_ptr_blearner_factory->getDataIdentifier() << std::endl;
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
//' BaselearnerCustom$new(data_source, list(instantiate_fun,
//'   train_fun, predict_fun, param_fun))
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
//' }
//' \item{\code{instantiate_fun} [\code{function}]}{
//'   \code{R} function to transform the source data. For details see the
//'   \code{Details}.
//' }
//' \item{\code{train_fun} [\code{function}]}{
//'   \code{R} function to train the base-learner on the target data. For
//'   details see the \code{Details}.
//' }
//' \item{\code{predict_fun} [\code{function}]}{
//'   \code{R} function to predict on the object returned by \code{train}.
//'   For details see the \code{Details}.
//' }
//' \item{\code{param_fun} [\code{function}]}{
//'   \code{R} function to extract the parameter of the object returned by
//'   \code{train}. For details see the \code{Details}.
//' }
//' }
//'
//' @section Details:
//'   The function must have the following structure:
//'
//'   \code{instantiateData(X) { ... return (X_trafo) }} With a matrix argument
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
//' data_mat = cbind(1, 1:10)
//' y = 2 + 3 * 1:10
//'
//' # Create new data object:
//' data_source = InMemoryData$new(data_mat, "my_data_name")
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
//' custom_lin_factory = BaselearnerCustom$new(data_source,
//'   list(instantiate_fun = instantiateDataFun, train_fun = trainFun,
//'     predict_fun = predictFun, param_fun = extractParameter))
//'
//' # Get the transformed data:
//' custom_lin_factory$getData()
//'
//' # Summarize factory:
//' custom_lin_factory$summarizeFactory()
//'
//' # Transform data manually:
//' custom_lin_factory$transformData(data_mat)
//'
//' @export BaselearnerCustom
class BaselearnerCustomFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  Rcpp::List internal_arg_list = Rcpp::List::create(
    Rcpp::Named("instantiate_fun") = 0,
    Rcpp::Named("train_fun") = 0,
    Rcpp::Named("predict_fun") = 0,
    Rcpp::Named("param_fun") = 0
  );

public:

  BaselearnerCustomFactoryWrapper (DataWrapper& data_source,
    Rcpp::List arg_list)
  {
    // Don't check argument types since we don't have a function placeholder for the default list:
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, FALSE);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCustomFactory>("custom", data_source.getDataObj(),
      internal_arg_list["instantiate_fun"], internal_arg_list["train_fun"],
      internal_arg_list["predict_fun"], internal_arg_list["param_fun"]);
  }

  BaselearnerCustomFactoryWrapper (DataWrapper& data_source,
    const std::string& blearner_type, Rcpp::List arg_list)
  {
    // Don't check argument types since we don't have a function placeholder for the default list:
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, FALSE);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCustomFactory>(blearner_type, data_source.getDataObj(),
      internal_arg_list["instantiate_fun"], internal_arg_list["train_fun"],
      internal_arg_list["predict_fun"], internal_arg_list["param_fun"]);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Custom base-learner Factory:" << std::endl;

    Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
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
//' BaselearnerCustomCpp$new(data_source, list(instantiate_ptr,
//'   train_ptr, predict_ptr))
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{data_source} [\code{Data} Object]}{
//'   Data object which contains the source data.
//' }
//' \item{\code{instantiate_ptr} [\code{externalptr}]}{
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
//' data_mat = cbind(1, 1:10)
//' y = 2 + 3 * 1:10
//'
//' # Create new data object:
//' data_source = InMemoryData$new(data_mat, "my_data_name")
//'
//' # Source the external pointer exposed by using XPtr:
//' Rcpp::sourceCpp(code = getCustomCppExample(silent = TRUE))
//'
//' # Create new linear base-learner:
//' custom_cpp_factory = BaselearnerCustomCpp$new(data_source,
//'   list(instantiate_ptr = dataFunSetter(), train_ptr = trainFunSetter(),
//'     predict_ptr = predictFunSetter()))
//'
//' # Get the transformed data:
//' custom_cpp_factory$getData()
//'
//' # Summarize factory:
//' custom_cpp_factory$summarizeFactory()
//'
//' # Transform data manually:
//' custom_cpp_factory$transformData(data_mat)
//' }
//' @export BaselearnerCustomCpp
class BaselearnerCustomCppFactoryWrapper : public BaselearnerFactoryWrapper
{
private:
  Rcpp::List internal_arg_list = Rcpp::List::create(
    Rcpp::Named("instantiate_ptr") = 0,
    Rcpp::Named("train_ptr") = 0,
    Rcpp::Named("predict_ptr") = 0
  );

public:

  BaselearnerCustomCppFactoryWrapper (DataWrapper& data_source,
    Rcpp::List arg_list)
  {
    // Don't check argument types since we don't have a function placeholder for the default list:
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, FALSE);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCustomCppFactory>("custom_cpp", data_source.getDataObj(),
      internal_arg_list["instantiate_ptr"], internal_arg_list["train_ptr"],
      internal_arg_list["predict_ptr"]);
  }

  BaselearnerCustomCppFactoryWrapper (DataWrapper& data_source,
    const std::string& blearner_type, Rcpp::List arg_list)
  {
    // Don't check argument types since we don't have a function placeholder for the default list:
    internal_arg_list = helper::argHandler(internal_arg_list, arg_list, FALSE);

    sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCustomCppFactory>(blearner_type, data_source.getDataObj(),
      internal_arg_list["instantiate_ptr"], internal_arg_list["train_ptr"],
      internal_arg_list["predict_ptr"]);
  }

  void summarizeFactory ()
  {
    Rcpp::Rcout << "Custom cpp base-learner Factory:" << std::endl;
    Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataIdentifier() << std::endl;
    Rcpp::Rcout << "\t- Factory creates the following base-learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
  }
};

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(BaselearnerFactoryWrapper)
RCPP_MODULE (baselearner_factory_module)
{
  using namespace Rcpp;

  class_<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor ("Create BaselearnerFactory class")

    .method("getData",        &BaselearnerFactoryWrapper::getData, "Get the data used within the learner")
    //.method("transformData",  &BaselearnerFactoryWrapper::transformData, "Transform data to the dataset used within the learner")
    .method("getFeatureName", &BaselearnerFactoryWrapper::getFeatureName, "Get name of the feature used for the base-learner")
  ;

  class_<BaselearnerPolynomialFactoryWrapper> ("BaselearnerPolynomial")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerPolynomialFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

 class_<BaselearnerCategoricalRidgeFactoryWrapper> ("BaselearnerCategoricalRidge")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<const CategoricalDataRawWrapper&, Rcpp::List> ()
    .constructor<const CategoricalDataRawWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerCategoricalRidgeFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

 class_<BaselearnerCategoricalBinaryFactoryWrapper> ("BaselearnerCategoricalBinary")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<const CategoricalDataRawWrapper&, std::string> ()
    .constructor<const CategoricalDataRawWrapper&, std::string, std::string> ()

    .method("summarizeFactory", &BaselearnerCategoricalBinaryFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

  class_<BaselearnerCenteredFactoryWrapper> ("BaselearnerCentered")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&, BaselearnerFactoryWrapper&, std::string> ()

    .method("summarizeFactory", &BaselearnerCenteredFactoryWrapper::summarizeFactory, "Summarize Factory")
    .method("getRotation",      &BaselearnerCenteredFactoryWrapper::getRotation, "Get rotation matrix for centering")
  ;

  class_<BaselearnerTensorFactoryWrapper> ("BaselearnerTensor")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&, BaselearnerFactoryWrapper&, std::string> ()

    .method("summarizeFactory", &BaselearnerTensorFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

  class_<BaselearnerPSplineFactoryWrapper> ("BaselearnerPSpline")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerPSplineFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

  class_<BaselearnerCustomFactoryWrapper> ("BaselearnerCustom")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerCustomFactoryWrapper::summarizeFactory, "Summarize Factory")
  ;

  class_<BaselearnerCustomCppFactoryWrapper> ("BaselearnerCustomCpp")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerCustomCppFactoryWrapper::summarizeFactory, "Summarize Factory")
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
//' data_mat = cbind(1:10)
//'
//' # Create new data object:
//' data_source = InMemoryData$new(data_mat, "my_data_name")
//'
//' lin_factory = BaselearnerPolynomial$new(data_source,
//'   list(degree = 1, intercept = TRUE))
//' poly_factory = BaselearnerPolynomial$new(data_source,
//'   list(degree = 2, intercept = TRUE))
//'
//' # Create new base-learner list:
//' my_bl_list = BlearnerFactoryList$new()
//'
//' # Register factories:
//' my_bl_list$registerFactory(lin_factory)
//' my_bl_list$registerFactory(poly_factory)
//'
//' # Get registered factories:
//' my_bl_list$printRegisteredFactories()
//'
//' # Get all target data matrices in one big matrix:
//' my_bl_list$getModelFrame()
//'
//' # Clear list:
//' my_bl_list$clearRegisteredFactories()
//'
//' # Get number of registered factories:
//' my_bl_list$getNumberOfRegisteredFactories()
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
      Rcpp::Named("model_frame") = raw_frame.second
    );
  }

  unsigned int getNumberOfRegisteredFactories ()
  {
    return obj.getFactoryMap().size();
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

  std::shared_ptr<loss::Loss> getLoss () { return sh_ptr_loss; }
  arma::mat calculatePseudoResiduals (const arma::mat& response, const arma::mat& pred) const {
    return sh_ptr_loss->calculatePseudoResiduals(response, pred);
  }
  virtual ~LossWrapper () {}


protected:
  std::shared_ptr<loss::Loss> sh_ptr_loss;
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
//' @examples
//'
//' # Create new loss object:
//' quadratic_loss = LossQuadratic$new()
//' quadratic_loss
//'
//' @export LossQuadratic
class LossQuadraticWrapper : public LossWrapper
{
public:
  LossQuadraticWrapper () { sh_ptr_loss = std::make_shared<loss::LossQuadratic>(); }
  LossQuadraticWrapper (double custom_offset) { sh_ptr_loss = std::make_shared<loss::LossQuadratic>(custom_offset); }
  LossQuadraticWrapper (arma::mat custom_offset, bool temp) { sh_ptr_loss = std::make_shared<loss::LossQuadratic>(custom_offset); }
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
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = -\mathrm{sign}(y - f(x))
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
//' @examples
//'
//' # Create new loss object:
//' absolute_loss = LossAbsolute$new()
//' absolute_loss
//'
//' @export LossAbsolute
class LossAbsoluteWrapper : public LossWrapper
{
public:
  LossAbsoluteWrapper () { sh_ptr_loss = std::make_shared<loss::LossAbsolute>(); }
  LossAbsoluteWrapper (double custom_offset) { sh_ptr_loss = std::make_shared<loss::LossAbsolute>(custom_offset); }
};

//' Quantile loss for regression tasks.
//'
//' This loss can be used for regression with \eqn{y \in \mathrm{R}}.
//'
//' \strong{Loss Function:}
//' \deqn{
//'   L(y, f(x)) = h| y - f(x)|
//' }
//' \strong{Gradient:}
//' \deqn{
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = -h\mathrm{sign}( y - f(x))
//' }
//' \strong{Initialization:}
//' \deqn{
//'   \hat{f}^{[0]}(x) = \mathrm{arg~min}_{c\in R}\ \frac{1}{n}\sum\limits_{i=1}^n
//'   L(y^{(i)}, c) = \mathrm{quantile}(y, q)
//' }
//'
//' @format \code{\link{S4}} object.
//' @name LossQuantile
//'
//' @section Usage:
//' \preformatted{
//' LossAbsolute$new()
//' LossAbsolute$new(quantile)
//' LossAbsolute$new(offset, quantile)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{offset} [\code{numeric(1)}]}{
//'   Numerical value which can be used to set a custom offset. If so, this
//'   value is returned instead of the loss optimal initialization.
//' }
//' \item{\code{quantile} [\code{numeric(1)}]}{
//'   Numerical value between 0 and 1 indicating the quantile used for boosting.
//' }
//' }
//'
//' @examples
//'
//' # Create new loss object:
//' quadratic_loss = LossQuadratic$new()
//' quadratic_loss
//'
//' @export LossQuantile
class LossQuantileWrapper : public LossWrapper
{
public:
  LossQuantileWrapper () { sh_ptr_loss = std::make_shared<loss::LossQuantile>(0.5); }
  LossQuantileWrapper (double quantile) { sh_ptr_loss = std::make_shared<loss::LossQuantile>(quantile); }
  LossQuantileWrapper (double custom_offset, double quantile) { sh_ptr_loss = std::make_shared<loss::LossQuantile>(custom_offset, quantile); }

  double getQuantile () const { return std::static_pointer_cast<loss::LossQuantile>(sh_ptr_loss)->getQuantile(); }
};


//' Huber loss for regression tasks.
//'
//' This loss can be used for regression with \eqn{y \in \mathrm{R}}.
//'
//' \strong{Loss Function:}
//' \deqn{
//'   L(y, f(x)) = 0.5(y - f(x))^2 \ \ \mathrm{if} \ \ |y - f(x)| < d
//' }
//' \deqn{
//'   L(y, f(x)) = d|y - f(x)| - 0.5d^2 \ \ \mathrm{otherwise}
//' }
//' \strong{Gradient:}
//' \deqn{
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = f(x) - y \ \ \mathrm{if} \ \ |y - f(x)| < d
//' }
//' \deqn{
//'   \frac{\delta}{\delta f(x)}\ L(y, f(x)) = -d\mathrm{sign}(y - f(x)) \ \ \mathrm{otherwise}
//' }
//'
//' @format \code{\link{S4}} object.
//' @name LossHuber
//'
//' @section Usage:
//' \preformatted{
//' LossHuber$new()
//' LossHuber$new(delta)
//' LossHuber$new(offset, delta)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{offset} [\code{numeric(1)}]}{
//'   Numerical value which can be used to set a custom offset. If so, this
//'   value is returned instead of the loss optimal initialization.
//' }
//' \item{\code{delta} [\code{numeric(1)}]}{
//'   Numerical value greater than 0 to specify the interval around 0 for the quadratic error measuring.
//'   Default is 1.
//' }
//' }
//'
//' @examples
//'
//' # Create new loss object:
//' huber_loss = LossHuber$new()
//' huber_loss
//'
//' @export LossHuber
class LossHuberWrapper : public LossWrapper
{
public:
  LossHuberWrapper () { sh_ptr_loss = std::make_shared<loss::LossHuber>(1); }
  LossHuberWrapper (double delta) { sh_ptr_loss = std::make_shared<loss::LossHuber>(delta); }
  LossHuberWrapper (double custom_offset, double delta) { sh_ptr_loss = std::make_shared<loss::LossHuber>(custom_offset, delta); }

  double getDelta () const { return std::static_pointer_cast<loss::LossHuber>(sh_ptr_loss)->getDelta(); }
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
//' @examples
//'
//' # Create new loss object:
//' bin_loss = LossBinomial$new()
//' bin_loss
//'
//' @export LossBinomial
class LossBinomialWrapper : public LossWrapper
{
public:
  LossBinomialWrapper () { sh_ptr_loss = std::make_shared<loss::LossBinomial>(); }
  LossBinomialWrapper (double custom_offset) { sh_ptr_loss = std::make_shared<loss::LossBinomial>(custom_offset); }
  LossBinomialWrapper (arma::mat custom_offset, bool temp) { sh_ptr_loss = std::make_shared<loss::LossBinomial>(custom_offset); }};

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
//' @examples
//'
//' # Loss function:
//' myLoss = function (true_values, prediction) {
//'   return (0.5 * (true_values - prediction)^2)
//' }
//' # Gradient of loss function:
//' myGradient = function (true_values, prediction) {
//'   return (prediction - true_values)
//' }
//' # Constant initialization:
//' myConstInit = function (true_values) {
//'   return (mean(true_values))
//' }
//'
//' # Create new custom quadratic loss:
//' my_loss = LossCustom$new(myLoss, myGradient, myConstInit)
//'
//' @export LossCustom
class LossCustomWrapper : public LossWrapper
{
public:
  LossCustomWrapper (Rcpp::Function lossFun, Rcpp::Function gradientFun,
    Rcpp::Function initFun)
  {
    sh_ptr_loss = std::make_shared<loss::LossCustom>(lossFun, gradientFun, initFun);
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
//' @examples
//' \donttest{
//' # Load loss functions:
//' Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE))
//'
//' # Create new custom quadratic loss:
//' my_cpp_loss = LossCustomCpp$new(lossFunSetter(), gradFunSetter(), constInitFunSetter())
//' }
//' @export LossCustomCpp
class LossCustomCppWrapper : public LossWrapper
{
public:
  LossCustomCppWrapper (SEXP loss_ptr, SEXP grad_ptr, SEXP const_init_ptr)
  {
    sh_ptr_loss = std::make_shared<loss::LossCustomCpp>(loss_ptr, grad_ptr, const_init_ptr);
  }
};



// //' Find loss optimal constant
// //'
// //' This function optimizes the empirical risk for an arbitrary loss function and
// //' a constant as predictor.
// //'
// //' @param truth [\code{matrix}]\cr
// //'   Matrix of true values passed to the loss.
// //' @param loss [\code{Loss Object}]\cr
// //'   The \code{S4 Loss} object which we like to optimize.
// //' @param lower_bound [\code{numeric(1)}]\cr
// //'   Lower bound to restrict the search space.
// //' @param upper_bound [\code{numeric(1)}]\cr
// //'   Upper bound to restrict the search space.
// //' @return \code{numeric(1)} Loss optimal constant.
// //' @examples
// //' X = cbind(1, iris$Petal.Length, iris$Sepal.Length)
// //' loss = LossQuadratic$new()
// //'
// //' findOptimalLossConstant(X, loss, min(X), max(X))
// //' @export findOptimalLossConstant
// double findOptimalLossConstant (const arma::mat& truth, LossWrapper& loss,
//   const double& lower_bound, const double& upper_bound)
// {
//   return lossoptim::findOptimalLossConstant(truth, loss.getLoss(), lower_bound, upper_bound);
// }



// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(LossWrapper)
RCPP_MODULE (loss_module)
{
  using namespace Rcpp;

  class_<LossWrapper> ("Loss")
    .constructor ()
    .method("calculatePseudoResiduals", &LossWrapper::calculatePseudoResiduals)
  ;

  class_<LossQuadraticWrapper> ("LossQuadratic")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <arma::mat, bool> ()
  ;

  class_<LossAbsoluteWrapper> ("LossAbsolute")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
  ;

  class_<LossQuantileWrapper> ("LossQuantile")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <double, double> ()
    .method("getQuantile", &LossQuantileWrapper::getQuantile)
  ;

  class_<LossHuberWrapper> ("LossHuber")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <double, double> ()
    .method("getDelta", &LossHuberWrapper::getDelta)
  ;

  class_<LossBinomialWrapper> ("LossBinomial")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <arma::mat, bool> ()
  ;

  class_<LossCustomWrapper> ("LossCustom")
    .derives<LossWrapper> ("Loss")
    .constructor<Rcpp::Function, Rcpp::Function, Rcpp::Function> ()
  ;

  class_<LossCustomCppWrapper> ("LossCustomCpp")
    .derives<LossWrapper> ("Loss")
    .constructor<SEXP, SEXP, SEXP> ()
  ;

  // function("findOptimalLossConstant", &findOptimalLossConstant);
}


// -------------------------------------------------------------------------- //
//                             RESPONSE CLASSES                               //
// -------------------------------------------------------------------------- //


class ResponseWrapper
{
public:
  ResponseWrapper () {}

  std::shared_ptr<response::Response> getResponseObj () { return sh_ptr_response; }

  std::string getTargetName () const { return sh_ptr_response->getTargetName(); }
  arma::mat getResponse () const { return sh_ptr_response->getResponse(); }
  arma::mat getWeights () const { return sh_ptr_response->getWeights(); }
  arma::mat getPrediction () const { return sh_ptr_response->getPredictionScores(); }
  arma::mat getPredictionTransform () const { return sh_ptr_response->getPredictionTransform(); }
  arma::mat getPredictionResponse () const { return sh_ptr_response->getPredictionResponse(); }
  void filter (const arma::uvec& idx) const { sh_ptr_response->filter(idx - 1); } // +1 to shift from R to C++ index
  double calculateEmpiricalRisk (LossWrapper& loss) const { return sh_ptr_response->calculateEmpiricalRisk(loss.getLoss()); }

protected:
  std::shared_ptr<response::Response> sh_ptr_response;
};

//' Create response object for regression.
//'
//' \code{ResponseRegr} creates a response object that are used as target during the
//' fitting process.
//'
//' @format \code{\link{S4}} object.
//' @name ResponseRegr
//'
//' @section Usage:
//' \preformatted{
//' ResponseRegr$new(target_name, response)
//' ResponseRegr$new(target_name, response, weights)
//' }
//'
//' @examples
//'
//' response_regr = ResponseRegr$new("target", cbind(rnorm(10)))
//' response_regr$getResponse()
//' response_regr$getTargetName()
//'
//' @export ResponseRegr
class ResponseRegrWrapper : public ResponseWrapper
{
public:

  ResponseRegrWrapper () {
    Rcpp::stop("Cannot initialize empty response object. See `?ResponseRegr` for help.");
  }

  ResponseRegrWrapper (std::string target_name, arma::mat response)
  {
    sh_ptr_response = std::make_shared<response::ResponseRegr>(target_name, response);
  }
  ResponseRegrWrapper (std::string target_name, arma::mat response, arma::mat weights)
  {
    sh_ptr_response = std::make_shared<response::ResponseRegr>(target_name, response, weights);
  }
};

//' Create response object for binary classification.
//'
//' \code{ResponseBinaryClassif} creates a response object that are used as target during the
//' fitting process.
//'
//' @format \code{\link{S4}} object.
//' @name ResponseBinaryClassif
//'
//' @section Usage:
//' \preformatted{
//' ResponseBinaryClassif$new(target_name, pos_class, response)
//' ResponseBinaryClassif$new(target_name, pos_class, response, weights)
//' }
//'
//' @examples
//'
//' response_binary = ResponseBinaryClassif$new("target", "A", sample(c("A", "B"), 10, TRUE))
//' response_binary$getResponse()
//' response_binary$getPrediction()
//' response_binary$getPredictionTransform() # Applies sigmoid to prediction scores
//' response_binary$getPredictionResponse()  # Categorizes depending on the transformed predictions
//' response_binary$getTargetName()
//' response_binary$setThreshold(0.7)
//' response_binary$getThreshold()
//' response_binary$getPositiveClass()
//'
//' @export ResponseBinaryClassif
class ResponseBinaryClassifWrapper : public ResponseWrapper
{
public:

  ResponseBinaryClassifWrapper () {
    Rcpp::stop("Cannot initialize empty response object. See `?ResponseBinaryClassif` for help.");
  }

  ResponseBinaryClassifWrapper (std::string target_name, std::string pos_class, std::vector<std::string> response)
  {
    sh_ptr_response = std::make_shared<response::ResponseBinaryClassif>(target_name, pos_class, response);
  }
  ResponseBinaryClassifWrapper (std::string target_name, std::string pos_class, std::vector<std::string> response, arma::mat weights)
  {
    sh_ptr_response = std::make_shared<response::ResponseBinaryClassif>(target_name, pos_class, response, weights);
  }

  double getThreshold () const
  {
    return std::static_pointer_cast<response::ResponseBinaryClassif>(sh_ptr_response)->getThreshold();
  }
  void setThreshold (double thresh)
  {
    std::static_pointer_cast<response::ResponseBinaryClassif>(sh_ptr_response)->setThreshold(thresh);
  }
  std::string getPositiveClass () const
  {
    return std::static_pointer_cast<response::ResponseBinaryClassif>(sh_ptr_response)->getPositiveClass();
  }
  std::map<std::string, unsigned int> getClassTable () const
  {
    return std::static_pointer_cast<response::ResponseBinaryClassif>(sh_ptr_response)->getClassTable();
  }
};

RCPP_EXPOSED_CLASS(ResponseWrapper)
RCPP_MODULE (response_module)
{
  using namespace Rcpp;

  class_<ResponseWrapper> ("Response")
    .constructor ("Create Response class")

    .method("getTargetName",          &ResponseWrapper::getTargetName, "Get the name of the target variable")
    .method("getResponse",            &ResponseWrapper::getResponse, "Get the original response")
    .method("getWeights",             &ResponseWrapper::getWeights, "Get the weights")
    .method("getPrediction",          &ResponseWrapper::getPrediction, "Get prediction scores")
    .method("getPredictionTransform", &ResponseWrapper::getPredictionTransform, "Get transformed prediction scores")
    .method("getPredictionResponse",  &ResponseWrapper::getPredictionResponse, "Get transformed prediction as response")
    .method("filter",                 &ResponseWrapper::filter, "Filter response elements")
    .method("calculateEmpiricalRisk", &ResponseWrapper::calculateEmpiricalRisk, "Calculates the empirical list given a specific loss")
  ;

  class_<ResponseRegrWrapper> ("ResponseRegr")
    .derives<ResponseWrapper> ("Response")

    .constructor ()
    .constructor<std::string, arma::mat> ()
    .constructor<std::string, arma::mat, arma::mat> ()
  ;

  class_<ResponseBinaryClassifWrapper> ("ResponseBinaryClassif")
    .derives<ResponseWrapper> ("Response")

    .constructor ()
    .constructor<std::string, std::string, std::vector<std::string>> ()
    .constructor<std::string, std::string, std::vector<std::string>, arma::mat> ()

    .method("getThreshold",           &ResponseBinaryClassifWrapper::getThreshold, "Get threshold used to transform scores to labels")
    .method("setThreshold",           &ResponseBinaryClassifWrapper::setThreshold, "Set threshold used to transform scores to labels")
    .method("getPositiveClass",       &ResponseBinaryClassifWrapper::getPositiveClass, "Get string of the positive class")
    .method("getClassTable",          &ResponseBinaryClassifWrapper::getClassTable, "Get table of response used for modeling")
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

  std::shared_ptr<logger::Logger> getLogger ()
  {
    return sh_ptr_logger;
  }

  std::string getLoggerId ()
  {
    return logger_id;
  }

  virtual ~LoggerWrapper () {}

protected:
  std::shared_ptr<logger::Logger> sh_ptr_logger;
  std::string logger_id;
};


//' Logger class to log the current iteration
//'
//' @format \code{\link{S4}} object.
//' @name LoggerIteration
//'
//' @section Usage:
//' \preformatted{
//' LoggerIterationWrapper$new(logger_id, use_as_stopper, max_iterations)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{logger_id} [\code{character(1)}]}{
//'   Unique identifier of the logger.
//' }
//' \item{\code{use_as_stopper} [\code{logical(1)}]}{
//'   Boolean to indicate if the logger should also be used as stopper.
//' }
//' \item{\code{max_iterations} [\code{integer(1)}]}{
//'   If the logger is used as stopper this argument defines the maximal
//'   iterations.
//' }
//' }
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
//' log_iters = LoggerIteration$new("iterations", FALSE, 100)
//'
//' # Summarize logger:
//' log_iters$summarizeLogger()
//'
//' @export LoggerIteration
class LoggerIterationWrapper : public LoggerWrapper
{

private:
  unsigned int max_iterations;
  bool use_as_stopper;

public:
  LoggerIterationWrapper (std::string logger_id0, bool use_as_stopper, unsigned int max_iterations)
    : max_iterations ( max_iterations ),
      use_as_stopper ( use_as_stopper )
  {
    logger_id = logger_id0;
    sh_ptr_logger = std::make_shared<logger::LoggerIteration>(logger_id, use_as_stopper, max_iterations);
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
//' LoggerInbagRisk$new(logger_id, use_as_stopper, used_loss, eps_for_break, patience)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{logger_id} [\code{character(1)}]}{
//'   Unique identifier of the logger.
//' }
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
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//'   \item{\code{summarizeLogger()}}{Summarize the logger object.}
//' }
//' @examples
//' # Used loss:
//' log_bin = LossBinomial$new()
//'
//' # Define logger:
//' log_inbag_risk = LoggerInbagRisk$new("inbag", FALSE, log_bin, 0.05, 5)
//'
//' # Summarize logger:
//' log_inbag_risk$summarizeLogger()
//'
//' @export LoggerInbagRisk
class LoggerInbagRiskWrapper : public LoggerWrapper
{

private:
  double eps_for_break;
  bool use_as_stopper;

public:
  LoggerInbagRiskWrapper (std::string logger_id0, bool use_as_stopper, LossWrapper& used_loss, double eps_for_break,
    unsigned int patience)
    : eps_for_break ( eps_for_break ),
      use_as_stopper ( use_as_stopper)
  {
    logger_id = logger_id0;
    sh_ptr_logger = std::make_shared<logger::LoggerInbagRisk>(logger_id, use_as_stopper, used_loss.getLoss(), eps_for_break, patience);
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
//' LoggerOobRisk$new(logger_id, use_as_stopper, used_loss, eps_for_break,
//'   patience, oob_data, oob_response)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{logger_id} [\code{character(1)}]}{
//'   Unique identifier of the logger.
//' }
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
//' data_source1 = InMemoryData$new(X1, "x1")
//' data_source2 = InMemoryData$new(X2, "x2")
//'
//' oob_list = list(data_source1, data_source2)
//'
//' set.seed(123)
//' y_oob = rnorm(10)
//'
//' # Used loss:
//' log_bin = LossBinomial$new()
//'
//' # Define response object of oob data:
//' oob_response = ResponseRegr$new("oob_response", as.matrix(y_oob))
//'
//' # Define logger:
//' log_oob_risk = LoggerOobRisk$new("oob", FALSE, log_bin, 0.05, 5, oob_list, oob_response)
//'
//' # Summarize logger:
//' log_oob_risk$summarizeLogger()
//'
//' @export LoggerOobRisk
class LoggerOobRiskWrapper : public LoggerWrapper
{

private:
  double eps_for_break;
  bool use_as_stopper;

public:
  LoggerOobRiskWrapper (std::string logger_id0, bool use_as_stopper, LossWrapper& used_loss, double eps_for_break,
    unsigned int patience, Rcpp::List oob_data, ResponseWrapper& oob_response)
  {
    std::map<std::string, std::shared_ptr<data::Data>> oob_data_map;

    // Be very careful with the wrappers. For instance: doing something like
    // temp = oob_data[i] within the loop will force temp to call its destructor
    // when it runs out of the scope. This will trigger the destructor of the
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

    logger_id = logger_id0;
    sh_ptr_logger = std::make_shared<logger::LoggerOobRisk>(logger_id, use_as_stopper, used_loss.getLoss(), eps_for_break,
      patience, oob_data_map, oob_response.getResponseObj());
  }

  void summarizeLogger ()
  {
    Rcpp::Rcout << "Out of bag risk logger:" << std::endl;
    if (use_as_stopper) {
      Rcpp::Rcout << "\t- Epsilon used to stop algorithm: " << eps_for_break << std::endl;
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
//' LoggerTime$new(logger_id, use_as_stopper, max_time, time_unit)
//' }
//'
//' @section Arguments:
//' \describe{
//' \item{\code{logger_id} [\code{character(1)}]}{
//'   Unique identifier of the logger.
//' }
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
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' \describe{
//' \item{\code{summarizeLogger()}}{Summarize the logger object.}
//' }
//' @examples
//' # Define logger:
//' log_time = LoggerTime$new("time_minutes", FALSE, 20, "minutes")
//'
//' # Summarize logger:
//' log_time$summarizeLogger()
//'
//' @export LoggerTime
class LoggerTimeWrapper : public LoggerWrapper
{

private:
  bool use_as_stopper;
  unsigned int max_time;
  std::string time_unit;

public:
  LoggerTimeWrapper (std::string logger_id0, bool use_as_stopper, unsigned int max_time,
    std::string time_unit)
    : use_as_stopper ( use_as_stopper ),
      max_time ( max_time ),
      time_unit ( time_unit )
  {
    // logger_id = logger_id0 + "." + time_unit;
    logger_id = logger_id0;
    sh_ptr_logger = std::make_shared<logger::LoggerTime>(logger_id, use_as_stopper, max_time, time_unit);
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
//' \item{\code{registerLogger(logger)}}{Includes a new \code{logger}
//'   into the logger list with the \code{logger_id} as key.}
//' }
//' @examples
//' # Define logger:
//' log_iters = LoggerIteration$new("iteration", TRUE, 100)
//' log_time = LoggerTime$new("time", FALSE, 20, "minutes")
//'
//' # Create logger list:
//' logger_list = LoggerList$new()
//'
//' # Register new loggeR:
//' logger_list$registerLogger(log_iters)
//' logger_list$registerLogger(log_time)
//'
//' # Print registered logger:
//' logger_list$printRegisteredLogger()
//'
//' # Remove all logger:
//' logger_list$clearRegisteredLogger()
//'
//' # Get number of registered logger:
//' logger_list$getNumberOfRegisteredLogger()
//'
//' @export LoggerList
class LoggerListWrapper
{
private:
  std::shared_ptr<loggerlist::LoggerList> sh_ptr_loggerlist = std::make_shared<loggerlist::LoggerList>();

public:
  LoggerListWrapper () {};

  std::shared_ptr<loggerlist::LoggerList> getLoggerList ()
  {
    return sh_ptr_loggerlist;
  }

  void registerLogger (LoggerWrapper& logger_wrapper)
  {
    sh_ptr_loggerlist->registerLogger(logger_wrapper.getLogger());
  }

  void printRegisteredLogger ()
  {
    sh_ptr_loggerlist->printRegisteredLogger();
  }

  void clearRegisteredLogger ()
  {
    sh_ptr_loggerlist->clearMap();
  }

  unsigned int getNumberOfRegisteredLogger ()
  {
    return sh_ptr_loggerlist->getLoggerMap().size();
  }

  std::vector<std::string> getNamesOfRegisteredLogger ()
  {
    std::vector<std::string> out;
    for (auto& it : sh_ptr_loggerlist->getLoggerMap()) {
      out.push_back(it.first);
    }
    return out;
  }

  virtual ~LoggerListWrapper () {}
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
    .constructor<std::string, bool, unsigned int> ()
    .method("summarizeLogger", &LoggerIterationWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerInbagRiskWrapper> ("LoggerInbagRisk")
    .derives<LoggerWrapper> ("Logger")
    .constructor<std::string, bool, LossWrapper&, double, unsigned int> ()
    .method("summarizeLogger", &LoggerInbagRiskWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerOobRiskWrapper> ("LoggerOobRisk")
    .derives<LoggerWrapper> ("Logger")
    .constructor<std::string, bool, LossWrapper&, double, unsigned int, Rcpp::List, ResponseWrapper&> ()
    .method("summarizeLogger", &LoggerOobRiskWrapper::summarizeLogger, "Summarize logger")
  ;

  class_<LoggerTimeWrapper> ("LoggerTime")
    .derives<LoggerWrapper> ("Logger")
    .constructor<std::string, bool, unsigned int, std::string> ()
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
  std::shared_ptr<optimizer::Optimizer> getOptimizer () { return sh_ptr_optimizer; }

  virtual ~OptimizerWrapper () {}

protected:
  std::shared_ptr<optimizer::Optimizer> sh_ptr_optimizer;
};

//' Coordinate Descent
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
//' @examples
//'
//' # Define optimizer:
//' optimizer = OptimizerCoordinateDescent$new()
//'
//' @export OptimizerCoordinateDescent
class OptimizerCoordinateDescent : public OptimizerWrapper
{
public:

  OptimizerCoordinateDescent () {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCoordinateDescent>();
  }

  OptimizerCoordinateDescent (unsigned int num_threads) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCoordinateDescent>(num_threads);
  }
};

//' Coordinate Descent with line search
//'
//' This class defines a new object which is used to conduct Coordinate Descent with line search.
//' The optimizer just calculates for each base-learner the sum of squared error and returns
//' the base-learner with the smallest SSE. In addition, this optimizer computes
//' a line search to find the optimal step size in each iteration.
//'
//' @format \code{\link{S4}} object.
//' @name OptimizerCoordinateDescentLineSearch
//'
//' @section Usage:
//' \preformatted{
//' OptimizerCoordinateDescentLineSearch$new()
//' }
//'
//' @examples
//'
//' # Define optimizer:
//' optimizer = OptimizerCoordinateDescentLineSearch$new()
//'
//' @export OptimizerCoordinateDescentLineSearch
class OptimizerCoordinateDescentLineSearch : public OptimizerWrapper
{
public:
  OptimizerCoordinateDescentLineSearch () {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCoordinateDescentLineSearch>();
  }
  OptimizerCoordinateDescentLineSearch (unsigned int num_threads) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCoordinateDescentLineSearch>(num_threads);
  }
  std::vector<double> getStepSize() { return sh_ptr_optimizer->getStepSize(); }
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
    .constructor <unsigned int> ()
  ;

  class_<OptimizerCoordinateDescentLineSearch> ("OptimizerCoordinateDescentLineSearch")
    .derives<OptimizerWrapper> ("Optimizer")
    .constructor ()
    .constructor <unsigned int> ()
    .method("getStepSize", &OptimizerCoordinateDescentLineSearch::getStepSize, "Get vector of step sizes")
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
//' df$mpg_cat = ifelse(df$mpg > 20, "high", "low")
//'
//' # # Create new variable to check the polynomial base-learner with degree 2:
//' # df$hp2 = df[["hp"]]^2
//'
//' # Data for the baselearner are matrices:
//' X_hp = as.matrix(df[["hp"]])
//' X_wt = as.matrix(df[["wt"]])
//'
//' # Target variable:
//' response = ResponseBinaryClassif$new("mpg_cat", "high", df[["mpg_cat"]])
//'
//' data_source_hp = InMemoryData$new(X_hp, "hp")
//' data_source_wt = InMemoryData$new(X_wt, "wt")
//'
//' # List for oob logging:
//' oob_data = list(data_source_hp, data_source_wt)
//'
//' # List to test prediction on newdata:
//' test_data = oob_data
//'
//' # Factories:
//' linear_factory_hp = BaselearnerPolynomial$new(data_source_hp,
//'   list(degree = 1, intercept = TRUE))
//' linear_factory_wt = BaselearnerPolynomial$new(data_source_wt,
//'   list(degree = 1, intercept = TRUE))
//' quadratic_factory_hp = BaselearnerPolynomial$new(data_source_hp,
//'   list(degree = 2, intercept = TRUE))
//' spline_factory_wt = BaselearnerPSpline$new(data_source_wt,
//'   list(degree = 3, n_knots = 10, penalty = 2, differences = 2))
//'
//' # Create new factory list:
//' factory_list = BlearnerFactoryList$new()
//'
//' # Register factories:
//' factory_list$registerFactory(linear_factory_hp)
//' factory_list$registerFactory(linear_factory_wt)
//' factory_list$registerFactory(quadratic_factory_hp)
//' factory_list$registerFactory(spline_factory_wt)
//'
//' # Define loss:
//' loss_bin = LossBinomial$new()
//'
//' # Define optimizer:
//' optimizer = OptimizerCoordinateDescent$new()
//'
//' ## Logger
//'
//' # Define logger. We want just the iterations as stopper but also track the
//' # time, inbag risk and oob risk:
//' log_iterations  = LoggerIteration$new(" iteration_logger", TRUE, 500)
//' log_time        = LoggerTime$new("time_logger", FALSE, 500, "microseconds")
//'
//' # Define new logger list:
//' logger_list = LoggerList$new()
//'
//' # Register the logger:
//' logger_list$registerLogger(log_iterations)
//' logger_list$registerLogger(log_time)
//'
//' # Run compboost:
//' # --------------
//'
//' # Initialize object:
//' cboost = Compboost_internal$new(
//'   response      = response,
//'   learning_rate = 0.05,
//'   stop_if_all_stopper_fulfilled = FALSE,
//'   factory_list = factory_list,
//'   loss         = loss_bin,
//'   logger_list  = logger_list,
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
//' cboost$setToIteration(200, 30)
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
  CompboostWrapper (ResponseWrapper& response, double learning_rate,
    bool stop_if_all_stopper_fulfilled, BlearnerFactoryListWrapper& factory_list,
    LossWrapper& loss, LoggerListWrapper& logger_list, OptimizerWrapper& optimizer)
  {

    learning_rate0     =  learning_rate;
    sh_ptr_loggerlist  =  logger_list.getLoggerList();
    sh_ptr_optimizer   =  optimizer.getOptimizer();
    blearner_list_ptr  =  factory_list.getFactoryList();

    std::unique_ptr<cboost::Compboost> unique_ptr_cboost_temp(new cboost::Compboost(response.getResponseObj(), learning_rate0,
      stop_if_all_stopper_fulfilled, sh_ptr_optimizer, loss.getLoss(), sh_ptr_loggerlist, *blearner_list_ptr));
    unique_ptr_cboost = std::move(unique_ptr_cboost_temp);
  }

  // Member functions
  void train (unsigned int trace)
  {
    unique_ptr_cboost->trainCompboost(trace);
    is_trained = true;
  }

  void continueTraining (unsigned int trace) { unique_ptr_cboost->continueTraining(trace); }
  arma::vec getPrediction (bool as_response) { return unique_ptr_cboost->getPrediction(as_response); }
  std::vector<std::string> getSelectedBaselearner () { return unique_ptr_cboost->getSelectedBaselearner(); }

  Rcpp::List getLoggerData ()
  {
    Rcpp::List out_list;

    out_list["logger_data"] = Rcpp::List::create(
      Rcpp::Named("logger_names") = unique_ptr_cboost->getLoggerList()->getLoggerData().first,
      Rcpp::Named("logger_data")  = unique_ptr_cboost->getLoggerList()->getLoggerData().second
    );
    return out_list[0];
  }

  Rcpp::List getEstimatedParameter ()
  {
    std::map<std::string, arma::mat> parameter = unique_ptr_cboost->getParameter();

    Rcpp::List out;

    for (auto &it : parameter) {
      out[it.first] = it.second;
    }
    return out;
  }

  Rcpp::List getParameterAtIteration (unsigned int k)
  {
    std::map<std::string, arma::mat> parameter = unique_ptr_cboost->getParameterOfIteration(k);

    Rcpp::List out;

    for (auto &it : parameter) {
      out[it.first] = it.second;
    }
    return out;
  }

  Rcpp::List getParameterMatrix ()
  {
    std::pair<std::vector<std::string>, arma::mat> out_pair = unique_ptr_cboost->getParameterMatrix();

    return Rcpp::List::create(
      Rcpp::Named("parameter_names")   = out_pair.first,
      Rcpp::Named("parameter_matrix")  = out_pair.second
    );
  }

  arma::vec predict (Rcpp::List& newdata, bool as_response)
  {
    std::map<std::string, std::shared_ptr<data::Data>> data_map;

    for (unsigned int i = 0; i < newdata.size(); i++) {
      DataWrapper* temp = newdata[i];
      data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();
    }
    return unique_ptr_cboost->predict(data_map, as_response);
  }

  void summarizeCompboost () { unique_ptr_cboost->summarizeCompboost(); }
  bool isTrained () { return is_trained; }
  arma::mat getOffset () { return unique_ptr_cboost->getOffset(); }
  std::vector<double> getRiskVector () { return unique_ptr_cboost->getRiskVector(); }
  void setToIteration (const unsigned int& k, const unsigned int& trace) { unique_ptr_cboost->setToIteration(k, trace); }

  ~CompboostWrapper () {}

private:

  std::unique_ptr<cboost::Compboost> unique_ptr_cboost;

  blearnerlist::BaselearnerFactoryList* blearner_list_ptr;
  std::shared_ptr<loggerlist::LoggerList> sh_ptr_loggerlist;
  std::shared_ptr<optimizer::Optimizer> sh_ptr_optimizer;

  unsigned int max_iterations;
  double learning_rate0;

  bool is_trained = false;
};


RCPP_EXPOSED_CLASS(CompboostWrapper)
RCPP_MODULE (compboost_module)
{
  using namespace Rcpp;

  class_<CompboostWrapper> ("Compboost_internal")
    .constructor<ResponseWrapper&, double, bool, BlearnerFactoryListWrapper&, LossWrapper&, LoggerListWrapper&, OptimizerWrapper&> ()
    .method("train", &CompboostWrapper::train, "Run component-wise boosting")
    .method("continueTraining", &CompboostWrapper::continueTraining, "Continue Training")
    .method("getPrediction", &CompboostWrapper::getPrediction, "Get prediction")
    .method("getSelectedBaselearner", &CompboostWrapper::getSelectedBaselearner, "Get vector of selected base-learner")
    .method("getLoggerData", &CompboostWrapper::getLoggerData, "Get data of the used logger")
    .method("getEstimatedParameter", &CompboostWrapper::getEstimatedParameter, "Get the estimated parameter")
    .method("getParameterAtIteration", &CompboostWrapper::getParameterAtIteration, "Get the estimated parameter for iteration k < iter_max")
    .method("getParameterMatrix", &CompboostWrapper::getParameterMatrix, "Get matrix of all estimated parameter in each iteration")
    .method("predict", &CompboostWrapper::predict, "Predict new data")
    .method("summarizeCompboost",    &CompboostWrapper::summarizeCompboost, "Summarize compboost object.")
    .method("isTrained", &CompboostWrapper::isTrained, "Status of algorithm if it is already trained.")
    .method("setToIteration", &CompboostWrapper::setToIteration, "Set state of the model to a given iteration")
    .method("getOffset", &CompboostWrapper::getOffset, "Get offset.")
    .method("getRiskVector", &CompboostWrapper::getRiskVector, "Get the risk vector.")
  ;
}


#endif // COMPBOOST_MODULES_CPP_
