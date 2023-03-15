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

#include <memory>

#include "compboost.h"
#include "baselearner_factory.h"
#include "baselearner_factory_list.h"
#include "loss.h"
#include "data.h"
#include "helper.h"
#include "optimizer.h"
#include "response.h"
#include "saver.h"
#include "init.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

// -------------------------------------------------------------------------- //
//                                   DATA                                     //
// -------------------------------------------------------------------------- //

class DataWrapper
{
  public:
    DataWrapper ()
      : sh_ptr_data ( std::make_shared<data::InMemoryData>(std::string("temp")) )
    { }

    DataWrapper (std::shared_ptr<data::Data> ds)
      : sh_ptr_data ( ds )
    { }

    virtual std::shared_ptr<data::Data> getDataObj ()
    {
      return sh_ptr_data;
    }

    std::string getDataType () const
    {
      return sh_ptr_data->getType();
    }

    virtual ~DataWrapper ()
    { }

  protected:
    std::shared_ptr<data::Data> sh_ptr_data;
};

//' @title Store data in RAM
//'
//' @description
//' This data container stores a vector as it is in the RAM and makes it
//' accessible for [Compboost].
//'
//' @format [S4] object.
//' @name InMemoryData
//'
//' @section Usage:
//' \preformatted{
//' InMemoryData$new()
//' InMemoryData$new(data_mat, data_identifier)
//' InMemoryData$new(data_mat, data_identifier, use_sparse)
//' }
//'
//' @param data_mat (`matrix()`)\cr
//' The data matrix.
//' @param data_identifier (`character(1)`)\cr
//' Data id, e.g. a feature name.
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$getData()`: `() -> matrix()`
//' * `$getIdentifier()`: `() -> character(1)`
//' @template section-data-base-methods
//'
//' @examples
//' # Sample data:
//' data_mat = cbind(rnorm(10))
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
  public:
    InMemoryDataWrapper ()
    {
      sh_ptr_data = std::make_shared<data::InMemoryData>(std::string(""));
    }

    InMemoryDataWrapper (DataWrapper& dw)
      : DataWrapper::DataWrapper(std::static_pointer_cast<data::InMemoryData>(dw.getDataObj()))
    { }

    InMemoryDataWrapper (arma::mat data_mat, std::string data_identifier)
    {
      sh_ptr_data = std::make_shared<data::InMemoryData>(data_identifier, data_mat);
    }

    InMemoryDataWrapper (arma::mat data_mat, std::string data_identifier, bool use_sparse)
    {
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


//' @title Data class for categorical variables
//'
//' @description
//' [CategoricalDataRaw] creates an data object which can be used as source
//' object to instantiate categorical base learner.
//'
//' @format [S4] object.
//' @name CategoricalDataRaw
//'
//' @section Usage:
//' \preformatted{
//' CategoricalDataRaw$new(x, data_identifier)
//' }
//'
//' @param x (`character()`)\cr
//' Categorical vector.
//' @param data_identifier (`character(1)`)\cr
//' Data id, e.g. a feature name.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$getData()`: `() -> stop()`\cr Throws error because no representation is calculated.
//' * `$getRawData()`: `() -> character()`
//' * `$getIdentifier()`: `() -> character(1)`
//' @template section-data-base-methods
//'
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

    CategoricalDataRawWrapper (DataWrapper& dw)
      : _sh_ptr_rawcdata ( std::static_pointer_cast<data::CategoricalDataRaw>(dw.getDataObj()) )
    {
      sh_ptr_data = _sh_ptr_rawcdata;
    }

    CategoricalDataRawWrapper (Rcpp::StringVector classes, std::string data_identifier)
    {
      std::vector<std::string> str_classes = Rcpp::as< std::vector<std::string> >(classes);
      _sh_ptr_rawcdata = std::make_shared<data::CategoricalDataRaw>(data_identifier, str_classes);
    }

    std::shared_ptr<data::CategoricalDataRaw> getCDataRawPtr () const
    {
      return _sh_ptr_rawcdata;
    }

    std::shared_ptr<data::Data> getDataObj ()
    {
      return _sh_ptr_rawcdata;
    }

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
RCPP_EXPOSED_CLASS(CategoricalDataRawWrapper)
RCPP_MODULE (data_module)
{
  using namespace Rcpp;

  class_<DataWrapper> ("Data")
    .constructor ("Create Data class")
    .method("getDataType", &DataWrapper::getDataType)
  ;

  class_<InMemoryDataWrapper> ("InMemoryData")
    .derives<DataWrapper> ("Data")

    .constructor ()
    .constructor<DataWrapper&> ()
    .constructor<arma::mat, std::string> ()
    .constructor<arma::mat, std::string, bool> ()

    .method("getData",       &InMemoryDataWrapper::getData)
    .method("getIdentifier", &InMemoryDataWrapper::getIdentifier)
  ;

  class_<CategoricalDataRawWrapper> ("CategoricalDataRaw")
    .derives<DataWrapper> ("Data")

    .constructor<DataWrapper&> ()
    .constructor<Rcpp::StringVector, std::string> ()

    .method("getData",       &CategoricalDataRawWrapper::getData)
    .method("getRawData",    &CategoricalDataRawWrapper::getRawData)
    .method("getIdentifier", &CategoricalDataRawWrapper::getIdentifier)
  ;
}


// -------------------------------------------------------------------------- //
//                         BASELEARNER FACTORIES                              //
// -------------------------------------------------------------------------- //

// Common API for all derived classes:
class BaselearnerFactoryWrapper
{
  public:

    BaselearnerFactoryWrapper () {}
    BaselearnerFactoryWrapper (std::shared_ptr<blearnerfactory::BaselearnerFactory> bl) : sh_ptr_blearner_factory ( bl ) {}

    std::shared_ptr<blearnerfactory::BaselearnerFactory> getFactory () { return sh_ptr_blearner_factory; }

    virtual ~BaselearnerFactoryWrapper () {}

    std::string              getModelName       () const { return sh_ptr_blearner_factory->getBaseModelName(); }
    arma::mat                getData            () const { return sh_ptr_blearner_factory->getData(); }
    arma::vec                getDF              () const { return sh_ptr_blearner_factory->getDF(); }
    arma::vec                getPenalty         () const { return sh_ptr_blearner_factory->getPenalty(); }
    arma::mat                getPenaltyMat      () const { return sh_ptr_blearner_factory->getPenaltyMat(); }
    std::string              getBaselearnerType () const { return sh_ptr_blearner_factory->getBaselearnerType(); }
    std::string              getBaselearnerId   () const { return sh_ptr_blearner_factory->getFactoryId(); }
    std::vector<double>      getMinMax          () const { return sh_ptr_blearner_factory->getMinMax(); }
    std::vector<std::string> getDataIdentifier  () const { return sh_ptr_blearner_factory->getDataIdentifier(); } // Duplicated?
    std::vector<std::string> getFeatureName     () const { return sh_ptr_blearner_factory->getDataIdentifier(); }

    std::map<std::string, std::vector<std::string>> getValueNames () const { return sh_ptr_blearner_factory->getValueNames(); }

    std::shared_ptr<data::Data> transform (Rcpp::List& newdata) const
    {
      std::map<std::string, std::shared_ptr<data::Data>> data_map;

      for (unsigned int i = 0; i < newdata.size(); i++) {
        DataWrapper* temp = newdata[i];
        data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();
      }

      return sh_ptr_blearner_factory->instantiateData(data_map);
    }

  protected:
    std::shared_ptr<blearnerfactory::BaselearnerFactory> sh_ptr_blearner_factory;
};


//' @title Polynomial base learner
//'
//' @description
//' `[BaselearnerPolynomial]` creates a polynomial base learner object.
//' The base learner takes one feature and calculates the polynomials (with
//' intercept) \eqn{1 + x + x^2 + \dots + x^d} for a given degree \eqn{d}.
//'
//' @format [S4] object.
//' @name BaselearnerPolynomial
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerPolynomial$new(data_source, list(degree, intercept, bin_root))
//' BaselearnerPolynomial$new(data_source, blearner_type, list(degree, intercept, bin_root))
//' }
//'
//' @param data_source ([InMemoryData]) \cr
//' Data object which contains the raw data (see \code{?InMemoryData}).
//' @param blearner_type (`character(1)`) \cr
//' Type of the base learner (if not specified, `blearner_type = paste0("poly", d)` is used).
//' The unique id of the base learner is defined by appending `blearner_type` to
//' the feature name: `paste0(data_source$getIdentifier(), "_", blearner_type)`.
//' @param degree (`integer(1)`)\cr
//' Polynomial degree.
//' @param intercept (`logical(1)`)\cr
//' Polynomial degree.
//' @template param-bin_root
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
//' @examples
//' # Sample data:
//' x = runif(100)
//' y = 1 + 2*x + rnorm(100, 0, 0.2)
//' dat = data.frame(x, y)
//'
//' # S4 wrapper
//'
//' # Create new data object, a matrix is required as input:
//' data_mat = cbind(x)
//' data_source = InMemoryData$new(data_mat, "my_data_name")
//'
//' # Create new linear base learner factory:
//' bl_lin = BaselearnerPolynomial$new(data_source,
//'   list(degree = 1))
//' bl_cub = BaselearnerPolynomial$new(data_source,
//'   list(intercept = FALSE, degree = 3, bin_root = 2))
//'
//' # Get the transformed data:
//' head(bl_lin$getData())
//' head(bl_cub$getData())
//'
//' # Summarize factory:
//' bl_lin$summarizeFactory()
//'
//' # Transform "new data":
//' newdata = list(InMemoryData$new(cbind(rnorm(5)), "my_data_name"))
//' bl_lin$transformData(newdata)
//' bl_cub$transformData(newdata)
//'
//' # R6 wrapper
//'
//' cboost_lin = Compboost$new(dat, "y")
//' cboost_lin$addBaselearner("x", "lin", BaselearnerPolynomial, degree = 1)
//' cboost_lin$train(100, 0)
//'
//' cboost_cub = Compboost$new(dat, "y")
//' cboost_cub$addBaselearner("x", "cubic", BaselearnerPolynomial,
//'   intercept = FALSE, degree = 3, bin_root = 2)
//' cboost_cub$train(100, 0)
//'
//' # Access base learner directly from the API (n = sqrt(100) = 10 with binning):
//' head(cboost_lin$baselearner_list$x_lin$factory$getData())
//' cboost_cub$baselearner_list$x_cubic$factory$getData()
//'
//' gg_lin = plotPEUni(cboost_lin, "x")
//' gg_cub = plotPEUni(cboost_cub, "x")
//'
//' library(ggplot2)
//' library(patchwork)
//'
//' (gg_lin | gg_cub) &
//'   geom_point(data = dat, aes(x = x, y = y - c(cboost_lin$offset)), alpha = 0.2)
//' @export BaselearnerPolynomial
class BaselearnerPolynomialFactoryWrapper : public BaselearnerFactoryWrapper
{
  private:
    Rcpp::List internal_arg_list = Rcpp::List::create(
      Rcpp::Named("degree") = 1,
      Rcpp::Named("intercept") = true,
      Rcpp::Named("bin_root") = 0);

  public:
    BaselearnerPolynomialFactoryWrapper (BaselearnerFactoryWrapper& blf)
      : BaselearnerFactoryWrapper::BaselearnerFactoryWrapper (
          std::static_pointer_cast<blearnerfactory::BaselearnerPolynomialFactory>(blf.getFactory()) )
    { }

    BaselearnerPolynomialFactoryWrapper (DataWrapper& data_source, Rcpp::List arg_list)
    {
      // Match defaults with custom arguments:
      internal_arg_list = helper::argHandler(internal_arg_list, arg_list, TRUE);

      // We need to converse the SEXP from the element to an integer:
      int degree = internal_arg_list["degree"];

      std::string blearner_type_temp = "poly" + std::to_string(degree);

      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPolynomialFactory>(blearner_type_temp, data_source.getDataObj(),
         internal_arg_list["degree"], internal_arg_list["intercept"], internal_arg_list["bin_root"]);
    }

    BaselearnerPolynomialFactoryWrapper (DataWrapper& data_source,
      const std::string& blearner_type, Rcpp::List arg_list)
    {
      internal_arg_list = helper::argHandler(internal_arg_list, arg_list, TRUE);

      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPolynomialFactory>(blearner_type, data_source.getDataObj(),
        internal_arg_list["degree"], internal_arg_list["intercept"], internal_arg_list["bin_root"]);
    }

    void summarizeFactory ()
    {
      // We need to converse the SEXP from the element to an integer:
      int degree = internal_arg_list["degree"];

      if (degree == 1) {
        Rcpp::Rcout << "Linear base learner factory:" << std::endl;
      }
      if (degree == 2) {
        Rcpp::Rcout << "Quadratic base learner factory:" << std::endl;
      }
      if (degree == 3) {
        Rcpp::Rcout << "Cubic base learner factory:" << std::endl;
      }
      if (degree > 3) {
        Rcpp::Rcout << "Polynomial base learner of degree " << degree << " factory:" << std::endl;
      }
      Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataSource()->getDataIdentifier() << std::endl;
      Rcpp::Rcout << "\t- Factory creates the following base learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
    }

    Rcpp::List transformData (Rcpp::List& newdata) const {
      std::shared_ptr<data::Data> dout = transform(newdata);
      auto mout = Rcpp::List::create(
        Rcpp::Named("design") = std::static_pointer_cast<data::BinnedData>(dout)->getDenseData());
      return mout;
    }

    Rcpp::List getMeta () const {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerPolynomialFactory>(sh_ptr_blearner_factory);
      Rcpp::List lout = Rcpp::List::create(
        Rcpp::Named("df") = fp->_attributes->df,
        Rcpp::Named("penalty") = fp->_attributes->penalty,
        Rcpp::Named("penalty_mat") = fp->_attributes->penalty_mat,
        Rcpp::Named("degree") = fp->_attributes->degree,
        Rcpp::Named("bin_root") = fp->_attributes->bin_root);

      return lout;
    }

};


//' @title Non-parametric B or P-spline base learner
//'
//' @description
//' [BaselearnerPSpline] creates a spline base learner object.
//' The object calculates the B-spline basis functions and in the case
//' of P-splines also the penalty. Instead of defining the penalty
//' term directly, one should consider to restrict the flexibility by
//' setting the degrees of freedom.
//'
//' @format [S4] object.
//' @name BaselearnerPSpline
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerPSpline$new(data_source, list(degree, n_knots, penalty, differences, df, bin_root))
//' BaselearnerPSpline$new(data_source, blearner_type, list(degree, n_knots, penalty, differences, df, bin_root))
//' }
//'
//' @param data_source ([InMemoryData]) \cr
//' Data object which contains the raw data (see `?InMemoryData`).
//' @param blearner_type (`character(1)`) \cr
//' Type of the base learner (if not specified, `blearner_type = "spline"` is used).
//' The unique id of the base learner is defined by appending `blearner_type` to
//' the feature name: `paste0(data_source$getIdentifier(), "_", blearner_type)`.
//' @param degree (`integer(1)`)\cr
//' Degree of the piecewise polynomial (default `degree = 3` for cubic splines).
//' @param n_knots (`integer(1)`)\cr
//' Number of inner knots (default `n_knots = 20`). The inner knots are expanded by
//' `degree - 1` additional knots at each side to prevent unstable behavior on the edges.
//' @param penalty (`numeric(1)`)\cr
//' Penalty term for P-splines (default `penalty = 2`). Set to zero for B-splines.
//' @param differences (`integer(1)`)\cr
//' The number of differences to are penalized. A higher value leads to smoother curves.
//' @template param-df
//' @template param-bin_root
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
//' @section Details:
//' The data matrix is instantiated as transposed sparse matrix due to performance
//' reasons. The member function `$getData()` accounts for that while  `$transformData()`
//' returns the raw data matrix as p x n matrix.
//'
//' @examples
//' # Sample data:
//' x = runif(100, 0, 10)
//' y = sin(x) + rnorm(100, 0, 0.2)
//' dat = data.frame(x, y)
//'
//' # S4 wrapper
//'
//' # Create new data object, a matrix is required as input:
//' data_mat = cbind(x)
//' data_source = InMemoryData$new(data_mat, "my_data_name")
//'
//' # Create new linear base learner factory:
//' bl_sp_df2 = BaselearnerPSpline$new(data_source,
//'   list(n_knots = 10, df = 2, bin_root = 2))
//' bl_sp_df5 = BaselearnerPSpline$new(data_source,
//'   list(n_knots = 15, df = 5))
//'
//' # Get the transformed data:
//' dim(bl_sp_df2$getData())
//' dim(bl_sp_df5$getData())
//'
//' # Summarize factory:
//' bl_sp_df2$summarizeFactory()
//'
//' # Get full meta data such as penalty term or matrix as well as knots:
//' str(bl_sp_df2$getMeta())
//' bl_sp_df2$getPenalty()
//' bl_sp_df5$getPenalty() # The penalty here is smaller due to more flexibility
//'
//' # Transform "new data":
//' newdata = list(InMemoryData$new(cbind(rnorm(5)), "my_data_name"))
//' bl_sp_df2$transformData(newdata)
//' bl_sp_df5$transformData(newdata)
//'
//' # R6 wrapper
//'
//' cboost_df2 = Compboost$new(dat, "y")
//' cboost_df2$addBaselearner("x", "sp", BaselearnerPSpline,
//'   n_knots = 10, df = 2, bin_root = 2)
//' cboost_df2$train(200, 0)
//'
//' cboost_df5 = Compboost$new(dat, "y")
//' cboost_df5$addBaselearner("x", "sp", BaselearnerPSpline,
//'   n_knots = 15, df = 5)
//' cboost_df5$train(200, 0)
//'
//' # Access base learner directly from the API (n = sqrt(100) = 10 with binning):
//' str(cboost_df2$baselearner_list$x_sp$factory$getData())
//' str(cboost_df5$baselearner_list$x_sp$factory$getData())
//'
//' gg_df2 = plotPEUni(cboost_df2, "x")
//' gg_df5 = plotPEUni(cboost_df5, "x")
//'
//' library(ggplot2)
//' library(patchwork)
//'
//' (gg_df2 | gg_df5) &
//'   geom_point(data = dat, aes(x = x, y = y - c(cboost_df2$offset)), alpha = 0.2)
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
      Rcpp::Named("cache_type") = "cholesky"
    );

  public:
    BaselearnerPSplineFactoryWrapper (BaselearnerFactoryWrapper& blf)
      : BaselearnerFactoryWrapper::BaselearnerFactoryWrapper (
          std::static_pointer_cast<blearnerfactory::BaselearnerPSplineFactory>(blf.getFactory()) )
    { }

    BaselearnerPSplineFactoryWrapper (DataWrapper& data_source, Rcpp::List arg_list)
    {
      internal_arg_list = helper::argHandler(internal_arg_list, arg_list, true);

      // We need to converse the SEXP from the element to an integer:
      int degree = internal_arg_list["degree"];

      std::string blearner_type_temp = "spline_degree_" + std::to_string(degree);

      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerPSplineFactory>(blearner_type_temp,
        data_source.getDataObj(), internal_arg_list["degree"], internal_arg_list["n_knots"], internal_arg_list["penalty"],
        internal_arg_list["df"], internal_arg_list["differences"], true, internal_arg_list["bin_root"],
        internal_arg_list["cache_type"]);
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
      Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataSource()->getDataIdentifier() << std::endl;
      Rcpp::Rcout << "\t- Factory creates the following base learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
    }

    Rcpp::List transformData (Rcpp::List& newdata) const {
      std::shared_ptr<data::Data> dout = transform(newdata);

      auto smat = std::static_pointer_cast<data::BinnedData>(dout)->getSparseData();
      smat = arma::trans(smat); // This step is required, otherwise Rcpp complains about not knowing the data type. So
                                // this: `Rcpp::List::create(Rcpp::Named("design") = smat.t());` is not possible.
      auto mout = Rcpp::List::create(Rcpp::Named("design") = smat);
      return mout;
    }

    Rcpp::List getMeta () const {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerPSplineFactory>(sh_ptr_blearner_factory);
      Rcpp::List lout = Rcpp::List::create(
        Rcpp::Named("df") = fp->_attributes->df,
        Rcpp::Named("penalty") = fp->_attributes->penalty,
        Rcpp::Named("penalty_mat") = fp->_attributes->penalty_mat,
        Rcpp::Named("degree") = fp->_attributes->degree,
        Rcpp::Named("n_knots") = fp->_attributes->n_knots,
        Rcpp::Named("differences") = fp->_attributes->differences,
        Rcpp::Named("bin_root") = fp->_attributes->bin_root,
        Rcpp::Named("knots") = fp->_attributes->knots);

      return lout;
    }

};

//' @title Row-wise tensor product base learner
//'
//' @description
//' This class combines base learners. The base learner is defined by a data matrix
//' calculated as row-wise tensor product of the two data matrices given in the
//' base learners to combine.
//'
//' @format [S4] object.
//' @name BaselearnerTensor
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerTensor$new(blearner1, blearner2, blearner_type)
//' BaselearnerTensor$new(blearner1, blearner2, blearner_type, anisotrop)
//' }
//'
//' @param blearner1 (`Baselearner*`)\cr
//' First base learner.
//' @param blearner2 (`Baselearner*`)\cr
//' Second base learner.
//' @param blearner_type (`character(1)`) \cr
//' Type of the base learner (if not specified, `blearner_type = "spline"` is used).
//' The unique id of the base learner is defined by appending `blearner_type` to
//' the feature name:
//' `paste0(blearner1$getDataSource()getIdentifier(), "_",
//'    blearner2$getDataSource()getIdentifier(), "_", blearner_type)`.
//' @param anisotrop (`logical(1)`)\cr
//' Defines how the penalty is added up. If `anisotrop = TRUE`, the marginal effects of the
//' are penalized as defined in the underlying factories. If `anisotrop = FALSE`, an isotropic
//' penalty is used, which means that both directions gets penalized equally.
//'
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
//' @examples
//' # Sample data:
//' x1 = runif(100, 0, 10)
//' x2 = runif(100, 0, 10)
//' y = sin(x1) * cos(x2) + rnorm(100, 0, 0.2)
//' dat = data.frame(x1, x2, y)
//'
//' # S4 wrapper
//'
//' # Create new data object, a matrix is required as input:
//' ds1 = InMemoryData$new(cbind(x1), "x1")
//' ds2 = InMemoryData$new(cbind(x2), "x2")
//'
//' # Create new linear base learner factory:
//' bl1 = BaselearnerPSpline$new(ds1, "sp", list(n_knots = 10, df = 5))
//' bl2 = BaselearnerPSpline$new(ds2, "sp", list(n_knots = 10, df = 5))
//'
//' tensor = BaselearnerTensor$new(bl1, bl2, "row_tensor")
//'
//' # Get the transformed data:
//' dim(tensor$getData())
//'
//' # Get full meta data such as penalty term or matrix as well as knots:
//' str(tensor$getMeta())
//'
//' # Transform "new data":
//' newdata = list(InMemoryData$new(cbind(runif(5)), "x1"),
//'   InMemoryData$new(cbind(runif(5)), "x2"))
//' str(tensor$transformData(newdata))
//'
//' # R6 wrapper
//'
//' cboost = Compboost$new(dat, "y")
//' cboost$addTensor("x1", "x2", df = 5)
//' cboost$train(50, 0)
//'
//' table(cboost$getSelectedBaselearner())
//' plotTensor(cboost, "x1_x2_tensor")
//' @export BaselearnerTensor
class BaselearnerTensorFactoryWrapper : public BaselearnerFactoryWrapper
{
  private:
    BaselearnerFactoryWrapper bl1;
    BaselearnerFactoryWrapper bl2;

  public:
    BaselearnerTensorFactoryWrapper (BaselearnerFactoryWrapper& blf)
      : BaselearnerFactoryWrapper::BaselearnerFactoryWrapper (
          std::static_pointer_cast<blearnerfactory::BaselearnerTensorFactory>(blf.getFactory()) )
    { }

    BaselearnerTensorFactoryWrapper (BaselearnerFactoryWrapper& blearner1, BaselearnerFactoryWrapper& blearner2, std::string blearner_type)
      : bl1 ( blearner1 ),
        bl2 ( blearner2 )
    {
      // We need to converse the SEXP from the element to an integer:
      std::string blearner_type_temp = blearner_type;

      std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner1 = blearner1.getFactory();
      std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner2 = blearner2.getFactory();

      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerTensorFactory>(blearner_type_temp, ptr_blearner1, ptr_blearner2);
    }

    BaselearnerTensorFactoryWrapper (BaselearnerFactoryWrapper& blearner1, BaselearnerFactoryWrapper& blearner2, std::string blearner_type, bool anisotrop)
    {
      // We need to converse the SEXP from the element to an integer:
      std::string blearner_type_temp = blearner_type;

      std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner1 = blearner1.getFactory();
      std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner2 = blearner2.getFactory();

      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerTensorFactory>(blearner_type_temp, ptr_blearner1,
        ptr_blearner2, anisotrop);
    }

    void summarizeFactory ()
    {
      Rcpp::Rcout << "\t- Factory creates the following base learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
    }

    Rcpp::List transformData (Rcpp::List& newdata) const {
      auto dout = transform(newdata);
      Rcpp::List mout;
      if (dout->usesSparseMatrix()) {
        auto smat = dout->getSparseData();
        smat = arma::trans(smat); // This step is required, otherwise Rcpp complains about not knowing the data type. So
                                  // this: `Rcpp::List::create(Rcpp::Named("design") = smat.t());` is not possible.
        mout = Rcpp::List::create(Rcpp::Named("design") = smat);
      } else  {
        mout = Rcpp::List::create(Rcpp::Named("design") = dout->getDenseData());
      }
      return mout;
    }

    Rcpp::List getMeta () const {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerTensorFactory>(sh_ptr_blearner_factory);
      Rcpp::List lout = Rcpp::List::create(
        Rcpp::Named("df") = fp->getDF(),
        Rcpp::Named("penalty") = fp->getPenalty(),
        Rcpp::Named("penalty_mat") = fp->getPenaltyMat());
        //Rcpp::Named("meta_bl1") = bl1.getMeta());

      return lout;
    }
};

//' @title Centering a base learner by another one
//'
//' @description
//' This base learner subtracts the effect of two base learners (usually defined
//' on the same feature). By subtracting the effects, one is not able to predict
//' the other one. This becomes handy for decomposing effects into, e.g., a
//' linear and non-linear component in which the non-linear component
//' is not capable to capture the linear part and hence is selected after
//' the linear effect is estimated.
//'
//' @format [S4] object.
//' @name BaselearnerCentered
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' * `$getRotation()`: `() -> matrix()`
//' @template section-bl-base-methods
//'
//' @examples
//' # Sample data:
//' x = runif(100, 0, 10)
//' y = 2 * sin(x) + 2 * x + rnorm(100, 0, 0.5)
//' dat = data.frame(x, y)
//'
//' # S4 wrapper
//'
//' # Create new data object, a matrix is required as input:
//' data_mat = cbind(x)
//' data_source = InMemoryData$new(data_mat, "x")
//'
//' # Prerequisite: Create a linear and spline base learner:
//' bl_lin = BaselearnerPolynomial$new(data_source,
//'   list(degree = 1, intercept = TRUE))
//' bl_sp = BaselearnerPSpline$new(data_source,
//'   list(n_knots = 15, df = 5))
//'
//' # Now, subtract the linear effect from the spline:
//' bl_ctr = BaselearnerCentered$new(bl_sp, bl_lin, "ctr")
//'
//' # Recognize, that the data matrix of this base learner has
//' # `nrow(bl_sp$getData()) - ncol(bl_lin$getData())` columns:
//' dim(bl_ctr$getData())
//' str(bl_ctr$getMeta())
//'
//' # The data matrix is created by rotating the spline data matrix:
//' all.equal(t(bl_sp$getData()) %*% bl_ctr$getRotation(), bl_ctr$getData())
//'
//' # Transform "new data". Internally, the basis of the spline is build and
//' # then rotated by the rotation matrix to subtract the linear part:
//' newdata = list(InMemoryData$new(cbind(rnorm(5)), "x"))
//' bl_ctr$transformData(newdata)
//'
//' # R6 wrapper
//'
//' # Compboost has a wrapper called `$addComponents()` that automatically
//' cboost = Compboost$new(dat, "y")
//'
//' # creates and adds the linear base learner and a centered base learner
//' # as above (the `...` args are passed to `BaselearnerPSpline$new():
//' cboost$addComponents("x", n_knots = 10, df = 5, bin_root = 2)
//'
//' # Note that we have used binning to save memory, hence the data matrix
//' # is reduced to 10 observations:
//' dim(cboost$baselearner_list$x_x_spline_centered$factory$getData())
//'
//' cboost$train(200, 0)
//'
//' library(ggplot2)
//'
//' plotPEUni(cboost, "x") +
//'   geom_point(data = dat, aes(x = x, y = y - c(cboost$offset)), alpha = 0.2)
//' @export BaselearnerCentered
class BaselearnerCenteredFactoryWrapper : public BaselearnerFactoryWrapper
{
  private:
    BaselearnerFactoryWrapper bl1;
    BaselearnerFactoryWrapper bl2;

  public:
    BaselearnerCenteredFactoryWrapper (BaselearnerFactoryWrapper& blf)
      : BaselearnerFactoryWrapper::BaselearnerFactoryWrapper (
          std::static_pointer_cast<blearnerfactory::BaselearnerCenteredFactory>(blf.getFactory()) )
    { }

    BaselearnerCenteredFactoryWrapper (BaselearnerFactoryWrapper& blearner1, BaselearnerFactoryWrapper& blearner2, std::string blearner_type)
      : bl1 ( blearner1 ),
        bl2 ( blearner2 )
    {
      // We need to converse the SEXP from the element to an integer:
      std::string blearner_type_temp = blearner_type;

      std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner1 = blearner1.getFactory();
      std::shared_ptr<blearnerfactory::BaselearnerFactory> ptr_blearner2 = blearner2.getFactory();

      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCenteredFactory>(blearner_type_temp, ptr_blearner1,
        ptr_blearner2);
    }

    void summarizeFactory ()
    {
      Rcpp::Rcout << "\t- Factory creates the following base learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
    }
    arma::mat getRotation ()
    {
      return std::static_pointer_cast<blearnerfactory::BaselearnerCenteredFactory>(sh_ptr_blearner_factory)->getRotation();
    }

    Rcpp::List transformData (Rcpp::List& newdata) const {
      auto dout = transform(newdata);

      auto mout = Rcpp::List::create(
        Rcpp::Named("design") = std::static_pointer_cast<data::BinnedData>(dout)->getDenseData());
      return mout;
    }

    Rcpp::List getMeta () const {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerCenteredFactory>(sh_ptr_blearner_factory);
      Rcpp::List lout = Rcpp::List::create(
        Rcpp::Named("df") = fp->getDF(),
        Rcpp::Named("penalty") = fp->getPenalty(),
        Rcpp::Named("penalty_mat") = fp->getPenaltyMat(),
        Rcpp::Named("rotation") = fp->getRotation());

      return lout;
    }
};


//' @title One-hot encoded base learner for a categorical feature
//'
//' @description
//' This base learner can be used to estimate effects of categorical
//' features. The classes are included similar as in the linear model by
//' using a one-hot encoded data matrix. Additionally, a Ridge penalty
//' allows unbiased feature selection.
//'
//' @format [S4] object.
//' @name BaselearnerCategoricalRidge
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCategoricalRidge$new(data_source, list(df))
//' BaselearnerCategoricalRidge$new(data_source, blearner_type, list(df))
//' }
//'
//' @param data_source [CategoricalDataRaw]\cr
//' Data container of the raw categorical feature.
//' @param blearner_type (`character(1)`) \cr
//' Type of the base learner (if not specified, `blearner_type = "ridge"` is used).
//' The unique id of the base learner is defined by appending `blearner_type` to
//' the feature name: `paste0(data_source$getIdentifier(), "_", blearner_type)`.
//' @template param-df
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
//' @examples
//' # Sample data:
//' x = sample(c("one","two"), 20, TRUE)
//' y = c(one = 0.8, two = -1.2)[x] + rnorm(20, 0, 0.2)
//' dat = data.frame(x, y)
//'
//' # S4 API:
//' ds = CategoricalDataRaw$new(x, "cat")
//' bl = BaselearnerCategoricalRidge$new(ds, list(df = 1))
//'
//' bl$getData()
//' bl$summarizeFactory()
//'
//' bl$getData()
//' bl$summarizeFactory()
//' bl$transformData(list(ds))
//' bl$getBaselearnerId()
//'
//' # R6 API:
//' cboost = Compboost$new(dat, "y")
//' cboost$addBaselearner("x", "binary", BaselearnerCategoricalRidge)
//' cboost$train(100, 0)
//' table(cboost$getSelectedBaselearner())
//' plotPEUni(cboost, "x", individual = FALSE)
//'
//' @export BaselearnerCategoricalRidge
class BaselearnerCategoricalRidgeFactoryWrapper : public BaselearnerFactoryWrapper
{
  private:
    Rcpp::List internal_arg_list = Rcpp::List::create(
      Rcpp::Named("df") = .0
    );

  public:
    BaselearnerCategoricalRidgeFactoryWrapper (BaselearnerFactoryWrapper& blf)
      : BaselearnerFactoryWrapper::BaselearnerFactoryWrapper (
          std::static_pointer_cast<blearnerfactory::BaselearnerCategoricalRidgeFactory>(blf.getFactory()) )
    { }

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
      Rcpp::Rcout << "Categorical base learner of category " << sh_ptr_blearner_factory->getDataSource()->getDataIdentifier() << std::endl;
    }

    std::map<std::string, unsigned int> getDictionary () const
    {
      return std::static_pointer_cast<blearnerfactory::BaselearnerCategoricalRidgeFactory>(sh_ptr_blearner_factory)->getDictionary();
    }

    Rcpp::List transformData (Rcpp::List& newdata) const {
      auto dout = transform(newdata);

      auto smat = dout->getSparseData();
      smat = arma::trans(smat); // This step is required, otherwise Rcpp complains about not knowing the data type. So
                                // this: `Rcpp::List::create(Rcpp::Named("design") = smat.t());` is not possible.
      auto mout = Rcpp::List::create(Rcpp::Named("design") = smat);
      return mout;
    }

    Rcpp::List getMeta () const {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerCategoricalRidgeFactory>(sh_ptr_blearner_factory);
      Rcpp::List lout = Rcpp::List::create(
        Rcpp::Named("df") = fp->getDF(),
        Rcpp::Named("penalty") = fp->getPenalty(),
        Rcpp::Named("penalty_mat") = fp->getPenaltyMat(),
        Rcpp::Named("dictionary") = fp->getDictionary());

      return lout;
    }
};


//' @title Base learner to encode one single class of a categorical feature
//'
//' @description
//' This class create a one-column one-hot encoded data matrix with ones at
//' `x == class_name` and zero otherwise.
//'
//' @format [S4] object.
//' @name BaselearnerCategoricalBinary
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCategoricalBinary$new(data_source, class_name)
//' BaselearnerCategoricalBinary$new(data_source, class_name, blearner_type)
//' }
//'
//' @param data_source [CategoricalDataRaw]\cr
//' The raw data object. Must be an object generated by [CategoricalDataRaw].
//' @param class_name (`character(1)`)\cr
//' The class for which a binary vector is created as data representation.
//' @param blearner_type (`character(1)`) \cr
//' Type of the base learner (if not specified, `blearner_type = "binary"` is used).
//' The unique id of the base learner is defined by appending `blearner_type` to
//' the feature name: `paste0(data_source$getIdentifier(), "_", class_name, "_", blearner_type)`.
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
//' @examples
//' # Sample data:
//' x = sample(c("one","two"), 20, TRUE)
//' y = c(one = 0.8, two = -1.2)[x] + rnorm(20, 0, 0.2)
//' dat = data.frame(x, y)
//'
//' # S4 API:
//' ds = CategoricalDataRaw$new(x, "cat")
//' bl = BaselearnerCategoricalBinary$new(ds, "one")
//'
//' bl$getData()
//' bl$summarizeFactory()
//' bl$transformData(list(ds))
//' bl$getBaselearnerId()
//'
//' # R6 API:
//' cboost = Compboost$new(dat, "y")
//' cboost$addBaselearner("x", "binary", BaselearnerCategoricalBinary)
//' cboost$train(500, 0)
//' table(cboost$getSelectedBaselearner())
//' plotPEUni(cboost, "x", individual = FALSE)
//' @export BaselearnerCategoricalBinary
class BaselearnerCategoricalBinaryFactoryWrapper : public BaselearnerFactoryWrapper
{
  public:
    BaselearnerCategoricalBinaryFactoryWrapper (BaselearnerFactoryWrapper& blf)
      : BaselearnerFactoryWrapper::BaselearnerFactoryWrapper (
          std::static_pointer_cast<blearnerfactory::BaselearnerCategoricalBinaryFactory>(blf.getFactory()) )
    { }

    BaselearnerCategoricalBinaryFactoryWrapper (const CategoricalDataRawWrapper& data_source, std::string cls)
    {
      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCategoricalBinaryFactory>("binary", cls, data_source.getCDataRawPtr());
    }

    BaselearnerCategoricalBinaryFactoryWrapper (const CategoricalDataRawWrapper& data_source, std::string cls, std::string blearner_type)
    {
      sh_ptr_blearner_factory = std::make_shared<blearnerfactory::BaselearnerCategoricalBinaryFactory>(blearner_type, cls, data_source.getCDataRawPtr());
    }

    void summarizeFactory ()
    {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerCategoricalBinaryFactory>(sh_ptr_blearner_factory);
      Rcpp::Rcout << "Categorical base learner of feature " <<
        fp->getDataSource()->getDataIdentifier() << " and category " <<
        fp->_attributes->cls << std::endl;
    }

    Rcpp::List transformData (Rcpp::List& newdata) const {
      auto dout = transform(newdata);

      auto smat = dout->getSparseData();
      smat = arma::trans(smat); // This step is required, otherwise Rcpp complains about not knowing the data type. So
                                // this: `Rcpp::List::create(Rcpp::Named("design") = smat.t());` is not possible.
      auto mout = Rcpp::List::create(Rcpp::Named("design") = smat);
      return mout;
    }

    Rcpp::List getMeta () const {
      auto fp = std::static_pointer_cast<blearnerfactory::BaselearnerCategoricalBinaryFactory>(sh_ptr_blearner_factory);
      Rcpp::List lout = Rcpp::List::create(Rcpp::Named("class") = fp->_attributes->cls);

      return lout;
    }
};


//' @title Custom base learner using `R` functions.
//'
//' @description
//' This class defines a custom base learner factory by
//' passing `R` functions for instantiation, fitting, and predicting.
//'
//' @format [S4] object.
//' @name BaselearnerCustom
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCustom$new(data_source, list(instantiate_fun,
//'   train_fun, predict_fun, param_fun))
//' }
//'
//' @template param-data_source
//' @param instantiate_fun (`function`)\cr
//' `R` function to transform the source data.
//' @param train_fun (`function`)\cr
//' `R` function to train the base learner on the target data.
//' @param predict_fun (`function`)\cr
//' `R` function to predict on the object returned by `train_fun`.
//' @param param_fun (`function`)\cr
//' `R` function to extract the parameter of the object returned by `train`.
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
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
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
//' # Create new custom linear base learner factory:
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
      Rcpp::Rcout << "Custom base learner Factory:" << std::endl;

      Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataSource()->getDataIdentifier() << std::endl;
      Rcpp::Rcout << "\t- Factory creates the following base learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
    }
};

/* CURRENTLY NOT INCLUDED BECAUSE OF BUGS THAT CAUSES SEGFAULTS.
 * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//' @title Custom base learner using `C++` functions.
//'
//' @description
//' This class defines a custom base learner factory by
//' passing pointers to `C++` functions for instantiation,
//' fitting, and predicting.
//'
//' @format [S4] object.
//' @name BaselearnerCustomCpp
//'
//' @section Usage:
//' \preformatted{
//' BaselearnerCustomCpp$new(data_source, list(instantiate_ptr, train_ptr, predict_ptr))
//' }
//'
//' @template param-data_source
//' @param instantiate_ptr (`externalptr`)\cr
//' External pointer to the `C++` instantiate data function.
//' @param train_ptr (`externalptr`)\cr
//' External pointer to the `C++` train function.
//' @param predict_ptr (`externalptr`)\cr
//' External pointer to the `C++` predict function.
//'
//' @section Details:
//' For an example see the extending compboost vignette or the function
//' [getCustomCppExample()].
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeFactory()`: `() -> ()`
//' * `$transfromData(newdata)`: `list(InMemoryData) -> matrix()`
//' * `$getMeta()`: `() -> list()`
//' @template section-bl-base-methods
//'
//' @examples
//' \dontrun{
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
//' # Create new linear base learner:
//' custom_cpp_factory = BaselearnerCustomCpp$new(data_source,
//'   list(instantiate_ptr = dataFunSetter(), train_ptr = trainFunSetter(),
//'     predict_ptr = predictFunSetter()))
//'
//' # Get the transformed data:
//' custom_cpp_factory$getData()
//'
//' # Summarize factory:
//' custom_cpp_factory$summarizeFactory()
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
      Rcpp::Rcout << "Custom cpp base learner Factory:" << std::endl;
      Rcpp::Rcout << "\t- Name of the used data: " << sh_ptr_blearner_factory->getDataSource()->getDataIdentifier() << std::endl;
      Rcpp::Rcout << "\t- Factory creates the following base learner: " << sh_ptr_blearner_factory->getBaselearnerType() << std::endl;
    }
};
*/

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(BaselearnerFactoryWrapper)
RCPP_MODULE (baselearner_factory_module)
{
  using namespace Rcpp;

  class_<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor ("Create BaselearnerFactory class")

    .method("getData",          &BaselearnerFactoryWrapper::getData)
    .method("getDF",            &BaselearnerFactoryWrapper::getDF)
    .method("getPenalty",       &BaselearnerFactoryWrapper::getPenalty)
    .method("getPenaltyMat",    &BaselearnerFactoryWrapper::getPenaltyMat)
    .method("getFeatureName",   &BaselearnerFactoryWrapper::getFeatureName)
    .method("getModelName",     &BaselearnerFactoryWrapper::getModelName)
    .method("getBaselearnerId", &BaselearnerFactoryWrapper::getBaselearnerId)
    .method("getMinMax",        &BaselearnerFactoryWrapper::getMinMax)
    .method("getValueNames",    &BaselearnerFactoryWrapper::getValueNames)
  ;

  class_<BaselearnerPolynomialFactoryWrapper> ("BaselearnerPolynomial")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&> ()
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerPolynomialFactoryWrapper::summarizeFactory)
    .method("transformData",    &BaselearnerPolynomialFactoryWrapper::transformData)
    .method("getMeta",          &BaselearnerPolynomialFactoryWrapper::getMeta)
  ;

 class_<BaselearnerCategoricalRidgeFactoryWrapper> ("BaselearnerCategoricalRidge")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&> ()
    .constructor<const CategoricalDataRawWrapper&, Rcpp::List> ()
    .constructor<const CategoricalDataRawWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerCategoricalRidgeFactoryWrapper::summarizeFactory)
    .method("getDictionary",    &BaselearnerCategoricalRidgeFactoryWrapper::getDictionary)
    .method("transformData",    &BaselearnerCategoricalRidgeFactoryWrapper::transformData)
    .method("getMeta",          &BaselearnerCategoricalRidgeFactoryWrapper::getMeta)
  ;

 class_<BaselearnerCategoricalBinaryFactoryWrapper> ("BaselearnerCategoricalBinary")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&> ()
    .constructor<const CategoricalDataRawWrapper&, std::string> ()
    .constructor<const CategoricalDataRawWrapper&, std::string, std::string> ()

    .method("summarizeFactory", &BaselearnerCategoricalBinaryFactoryWrapper::summarizeFactory)
    .method("transformData",    &BaselearnerCategoricalBinaryFactoryWrapper::transformData)
    .method("getMeta",          &BaselearnerCategoricalBinaryFactoryWrapper::getMeta)
  ;

  class_<BaselearnerCenteredFactoryWrapper> ("BaselearnerCentered")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&> ()
    .constructor<BaselearnerFactoryWrapper&, BaselearnerFactoryWrapper&, std::string> ()

    .method("summarizeFactory", &BaselearnerCenteredFactoryWrapper::summarizeFactory)
    .method("getRotation",      &BaselearnerCenteredFactoryWrapper::getRotation)
    .method("transformData",    &BaselearnerCenteredFactoryWrapper::transformData)
    .method("getMeta",          &BaselearnerCenteredFactoryWrapper::getMeta)
  ;

  class_<BaselearnerTensorFactoryWrapper> ("BaselearnerTensor")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&> ()
    .constructor<BaselearnerFactoryWrapper&, BaselearnerFactoryWrapper&, std::string> ()
    .constructor<BaselearnerFactoryWrapper&, BaselearnerFactoryWrapper&, std::string, bool> ()

    .method("summarizeFactory", &BaselearnerTensorFactoryWrapper::summarizeFactory)
    .method("transformData",    &BaselearnerTensorFactoryWrapper::transformData)
    .method("getMeta",          &BaselearnerTensorFactoryWrapper::getMeta)
  ;

  class_<BaselearnerPSplineFactoryWrapper> ("BaselearnerPSpline")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<BaselearnerFactoryWrapper&> ()
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerPSplineFactoryWrapper::summarizeFactory)
    .method("transformData",    &BaselearnerPSplineFactoryWrapper::transformData)
    .method("getMeta",          &BaselearnerPSplineFactoryWrapper::getMeta)
  ;

  // CUSTOM BASE LEARNERS
  // ------------------------------------------------------------------------
  class_<BaselearnerCustomFactoryWrapper> ("BaselearnerCustom")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerCustomFactoryWrapper::summarizeFactory)
  ;
  /*
  class_<BaselearnerCustomCppFactoryWrapper> ("BaselearnerCustomCpp")
    .derives<BaselearnerFactoryWrapper> ("Baselearner")
    .constructor<DataWrapper&, Rcpp::List> ()
    .constructor<DataWrapper&, std::string, Rcpp::List> ()

    .method("summarizeFactory", &BaselearnerCustomCppFactoryWrapper::summarizeFactory)
  ;
  */
}



// -------------------------------------------------------------------------- //
//                              BASELEARNERLIST                               //
// -------------------------------------------------------------------------- //

//' Base learner factory list to define the set of base learners
//'
//' \code{BlearnerFactoryList} creates an object in which base learner factories
//' can be registered. This object can then be passed to compboost as set of
//' base learner which is used by the optimizer to get the new best
//' base learner.
//'
//' @format [S4] object.
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
//'   base learner.}
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
//' # Create new base learner list:
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
    std::shared_ptr<blearnerlist::BaselearnerFactoryList> obj;

  public:

    BlearnerFactoryListWrapper ()
      : obj ( std::make_shared<blearnerlist::BaselearnerFactoryList>() )
    { }

    BlearnerFactoryListWrapper (std::shared_ptr<blearnerlist::BaselearnerFactoryList> bll)
      : obj ( bll )
    { }

    void registerFactory (BaselearnerFactoryWrapper& my_factory_to_register)
    {
      std::string factory_id = my_factory_to_register.getFactory()->getFactoryId();
      obj->registerBaselearnerFactory(factory_id, my_factory_to_register.getFactory());
    }

    void rmFactory (const std::string factory_id)
    {
      obj->rmBaselearnerFactory(factory_id);
    }

    void printRegisteredFactories ()
    {
      obj->printRegisteredFactories();
    }

    void clearRegisteredFactories ()
    {
      obj->clearMap();
    }

    std::shared_ptr<blearnerlist::BaselearnerFactoryList> getFactoryList ()
    {
      return obj;
    }

    Rcpp::List getModelFrame ()
    {
      std::pair<std::vector<std::string>, arma::mat> raw_frame = obj->getModelFrame();

      return Rcpp::List::create(
        Rcpp::Named("colnames")    = raw_frame.first,
        Rcpp::Named("model_frame") = raw_frame.second
      );
    }

    unsigned int getNumberOfRegisteredFactories () { return obj->getFactoryMap().size(); }
    std::vector<std::string> getRegisteredFactoryNames () { return obj->getRegisteredFactoryNames(); }
    std::vector<std::string> getDataNames () { return obj->getDataNames(); }

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
    .method("rmFactory", &BlearnerFactoryListWrapper::rmFactory, "Remove factory")
    .method("printRegisteredFactories", &BlearnerFactoryListWrapper::printRegisteredFactories, "Print all registered factories")
    .method("clearRegisteredFactories", &BlearnerFactoryListWrapper::clearRegisteredFactories, "Clear factory map")
    .method("getModelFrame", &BlearnerFactoryListWrapper::getModelFrame, "Get the data used for modeling")
    .method("getNumberOfRegisteredFactories", &BlearnerFactoryListWrapper::getNumberOfRegisteredFactories, "Get number of registered factories. Main purpose is for testing.")
    .method("getRegisteredFactoryNames", &BlearnerFactoryListWrapper::getRegisteredFactoryNames, "Get names of registered factories")
    .method("getDataNames", &BlearnerFactoryListWrapper::getDataNames, "Get names of data of registered factories")
  ;
}

// -------------------------------------------------------------------------- //
//                                    LOSS                                    //
// -------------------------------------------------------------------------- //

class LossWrapper
{
  public:
    LossWrapper ()
    { }

    LossWrapper (std::shared_ptr<loss::Loss> l)
      : sh_ptr_loss ( l )
    { }

    std::shared_ptr<loss::Loss> getLoss () {
      return sh_ptr_loss;
    }

    arma::mat calculatePseudoResiduals (const arma::mat& response, const arma::mat& pred) const
    {
      return sh_ptr_loss->calculatePseudoResiduals(response, pred);
    }

    arma::mat loss (const arma::mat& response, const arma::mat& pred) const
    {
      return sh_ptr_loss->loss(response, pred);
    }

    arma::mat gradient (const arma::mat& response, const arma::mat& pred) const
    {
      return sh_ptr_loss->gradient(response, pred);
    }

    arma::mat constInit (const arma::mat& response) const
    {
      return sh_ptr_loss->constantInitializer(response);
    }

    std::string getLossType ()
    {
      return sh_ptr_loss->getType();
    }

    std::shared_ptr<loss::Loss> getLossObj () const
    {
      return sh_ptr_loss;
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
//' @format [S4] object.
//' @name LossQuadratic
//'
//' @section Usage:
//' \preformatted{
//' LossQuadratic$new()
//' LossQuadratic$new(offset)
//' }
//'
//' @template section-loss-base-methods
//' @template param-offset
//'
//' @examples
//' # Create new loss object:
//' quadratic_loss = LossQuadratic$new()
//' quadratic_loss
//'
//' @export LossQuadratic
class LossQuadraticWrapper : public LossWrapper
{
  public:
    LossQuadraticWrapper ()
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossQuadratic>() )
    { }

    LossQuadraticWrapper (double custom_offset)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossQuadratic>(custom_offset) )
    { }

    LossQuadraticWrapper (arma::mat custom_offset, bool temp)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossQuadratic>(custom_offset) )
    { }

    LossQuadraticWrapper (LossWrapper l, bool b1, bool b2)
      : LossWrapper::LossWrapper ( std::static_pointer_cast<loss::LossQuadratic>(l.getLossObj()) )
    { }
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
//' @format [S4] object.
//' @name LossAbsolute
//'
//' @section Usage:
//' \preformatted{
//' LossAbsolute$new()
//' LossAbsolute$new(offset)
//' }
//'
//' @template section-loss-base-methods
//' @template param-offset
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
    LossAbsoluteWrapper ()
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossAbsolute>() )
    { }

    LossAbsoluteWrapper (double custom_offset)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossAbsolute>(custom_offset) )
    { }

    LossAbsoluteWrapper (LossWrapper l, bool b1, bool b2)
      : LossWrapper::LossWrapper ( std::static_pointer_cast<loss::LossAbsolute>(l.getLossObj()) )
    { }
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
//' @format [S4] object.
//' @name LossQuantile
//'
//' @section Usage:
//' \preformatted{
//' LossAbsolute$new()
//' LossAbsolute$new(quantile)
//' LossAbsolute$new(offset, quantile)
//' }
//'
//' @template section-loss-base-methods
//' @template param-offset
//' @param quantile (`numeric(1)`)\cr
//' Numerical value between 0 and 1 that defines the quantile that is modeled.
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
    LossQuantileWrapper ()
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossQuantile>(0.5) )
    { }

    LossQuantileWrapper (double quantile)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossQuantile>(quantile) )
    { }

    LossQuantileWrapper (double custom_offset, double quantile)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossQuantile>(custom_offset, quantile) )
    { }

    LossQuantileWrapper (LossWrapper l, bool b1, bool b2)
      : LossWrapper::LossWrapper ( std::static_pointer_cast<loss::LossQuantile>(l.getLossObj()) )
    { }

    double getQuantile () const
    {
      return std::static_pointer_cast<loss::LossQuantile>(sh_ptr_loss)->getQuantile();
    }
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
//' @format [S4] object.
//' @name LossHuber
//'
//' @section Usage:
//' \preformatted{
//' LossHuber$new()
//' LossHuber$new(delta)
//' LossHuber$new(offset, delta)
//' }
//'
//' @template section-loss-base-methods
//' @template param-offset
//' @param delta (`numeric(1)`)\cr
//' Numerical value greater than 0 to specify the interval around 0 for the
//' quadratic error measuring (default `delta = 1`).
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
    LossHuberWrapper ()
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossHuber>(1) )
    { }

    LossHuberWrapper (double delta)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossHuber>(delta) )
    { }

    LossHuberWrapper (double custom_offset, double delta)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossHuber>(custom_offset, delta) )
    { }

    LossHuberWrapper (LossWrapper l, bool b1, bool b2)
      : LossWrapper::LossWrapper ( std::static_pointer_cast<loss::LossHuber>(l.getLossObj()) )
    { }

    double getDelta () const
    {
      return std::static_pointer_cast<loss::LossHuber>(sh_ptr_loss)->getDelta();
    }
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
//' @format [S4] object.
//' @name LossBinomial
//'
//' @section Usage:
//' \preformatted{
//' LossBinomial$new()
//' LossBinomial$new(offset)
//' }
//'
//' @template section-loss-base-methods
//' @template param-offset
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
    LossBinomialWrapper ()
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossBinomial>() )
    { }

    LossBinomialWrapper (double custom_offset)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossBinomial>(custom_offset) )
    { }

    LossBinomialWrapper (arma::mat custom_offset, bool temp)
      : LossWrapper::LossWrapper ( std::make_shared<loss::LossBinomial>(custom_offset) )
    { }

    LossBinomialWrapper (LossWrapper l, bool b1, bool b2)
      : LossWrapper::LossWrapper ( std::static_pointer_cast<loss::LossBinomial>(l.getLossObj()) )
    { }
};


//' Create LossCustom by using R functions.
//'
//' \code{LossCustom} creates a custom loss by using
//' \code{Rcpp::Function} to set \code{R} functions.
//'
//' @format [S4] object.
//' @name LossCustom
//'
//' @section Usage:
//' \preformatted{
//' LossCustom$new(lossFun, gradientFun, initFun)
//' }
//'
//' @template section-loss-base-methods
//' @param lossFun (`function`)\cr
//' `R` function to calculate the loss.
//' @param gradientFun (`function`)\cr
//' `R` function to calculate the gradient.
//' @param initFun (`function`)\cr
//' `R` function to calculate the constant initialization.
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

/* CURRENTLY NOT INCLUDED BECAUSE OF BUGS THAT CAUSES SEGFAULTS.
 * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//' @title Custom loss using `C++` functions.
//'
//' @description
//' \code{LossCustomCpp} creates a custom loss by using
//' \code{Rcpp::XPtr} to set \code{C++} functions.
//'
//' @format [S4] object.
//' @name LossCustomCpp
//'
//' @section Usage:
//' \preformatted{
//' LossCustomCpp$new(loss_ptr, grad_ptr, const_init_ptr)
//' }
//'
//' @param loss_ptr (`externalptr`)\cr
//' External pointer to the \code{C++} loss function.
//' @param grad_ptr (`externalptr`)\cr
//' External pointer to the \code{C++} gradient function.
//' @param const_init_ptr (`externalptr`)\cr
//' External pointer to the \code{C++} constant initialization function.
//'
//' @examples
//' \dontrun{
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
    LossCustomCppWrapper (const SEXP& loss_ptr, const SEXP& grad_ptr, const SEXP& const_init_ptr)
    {
      sh_ptr_loss = std::make_shared<loss::LossCustomCpp>(loss_ptr, grad_ptr, const_init_ptr);
    }
};
*
*
*/

// Expose abstract BaselearnerWrapper class and define modules:
RCPP_EXPOSED_CLASS(LossWrapper)
RCPP_MODULE (loss_module)
{
  using namespace Rcpp;

  class_<LossWrapper> ("Loss")
    .constructor ()
    .method("calculatePseudoResiduals", &LossWrapper::calculatePseudoResiduals)
    .method ("getLossType", &LossWrapper::getLossType)
    .method ("loss", &LossWrapper::loss)
    .method ("gradient", &LossWrapper::gradient)
    .method ("constInit", &LossWrapper::constInit)
  ;

  class_<LossQuadraticWrapper> ("LossQuadratic")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <arma::mat, bool> ()
    .constructor <LossWrapper, bool, bool> ()
  ;

  class_<LossAbsoluteWrapper> ("LossAbsolute")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <LossWrapper, bool, bool> ()
  ;

  class_<LossQuantileWrapper> ("LossQuantile")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <double, double> ()
    .constructor <LossWrapper, bool, bool> ()
    .method("getQuantile", &LossQuantileWrapper::getQuantile)
  ;

  class_<LossHuberWrapper> ("LossHuber")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <double, double> ()
    .constructor <LossWrapper, bool, bool> ()
    .method("getDelta", &LossHuberWrapper::getDelta)
  ;

  class_<LossBinomialWrapper> ("LossBinomial")
    .derives<LossWrapper> ("Loss")
    .constructor ()
    .constructor <double> ()
    .constructor <arma::mat, bool> ()
    .constructor <LossWrapper, bool, bool> ()
  ;

  class_<LossCustomWrapper> ("LossCustom")
    .derives<LossWrapper> ("Loss")
    .constructor<Rcpp::Function, Rcpp::Function, Rcpp::Function> ()
  ;

  /*
  class_<LossCustomCppWrapper> ("LossCustomCpp")
    .derives<LossWrapper> ("Loss")
    .constructor<SEXP, SEXP, SEXP> ()
  ;
  */
}


// -------------------------------------------------------------------------- //
//                             RESPONSE CLASSES                               //
// -------------------------------------------------------------------------- //


class ResponseWrapper
{
  public:
    ResponseWrapper () {}
    ResponseWrapper (std::shared_ptr<response::Response> r) : sh_ptr_response ( r ) {}

    std::shared_ptr<response::Response> getResponseObj () { return sh_ptr_response; }

    std::string getTargetName () const { return sh_ptr_response->getTargetName(); }
    std::string getResponseType () const { return sh_ptr_response->getTaskIdentifier(); }
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
//' @format [S4] object.
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

//' @name ResponseRegr
//' @title Response class for regression tasks.
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

    ResponseRegrWrapper (ResponseWrapper r)
      : ResponseWrapper::ResponseWrapper(std::static_pointer_cast<response::ResponseRegr>(r.getResponseObj()))
    { }
};

//' Create response object for binary classification.
//'
//' \code{ResponseBinaryClassif} creates a response object that are used as target during the
//' fitting process.
//'
//' @format [S4] object.
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

    //ResponseBinaryClassifWrapper (std::string file)
    //{
      //json j = saver::jsonLoader(file);
      //sh_ptr_response = response::jsonToResponse(j["_sh_ptr_response"]);
    //}

    ResponseBinaryClassifWrapper (ResponseWrapper r)
      : ResponseWrapper::ResponseWrapper(std::static_pointer_cast<response::ResponseBinaryClassif>(r.getResponseObj()))
    { }

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
    .method("getResponseType",        &ResponseWrapper::getResponseType, "Get the original response")
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
    .constructor<ResponseWrapper> ()
  ;

  class_<ResponseBinaryClassifWrapper> ("ResponseBinaryClassif")
    .derives<ResponseWrapper> ("Response")

    .constructor ()
    .constructor<std::string, std::string, std::vector<std::string>> ()
    .constructor<std::string, std::string, std::vector<std::string>, arma::mat> ()
    .constructor<ResponseWrapper> ()

    .method("getThreshold",           &ResponseBinaryClassifWrapper::getThreshold, "Get threshold used to transform scores to labels")
    .method("setThreshold",           &ResponseBinaryClassifWrapper::setThreshold, "Set threshold used to transform scores to labels")
    .method("getPositiveClass",       &ResponseBinaryClassifWrapper::getPositiveClass, "Get string of the positive class")
    .method("getClassTable",          &ResponseBinaryClassifWrapper::getClassTable, "Get table of response used for modeling")
  ;
}



// -------------------------------------------------------------------------- //
//                                  LOGGER                                    //
// -------------------------------------------------------------------------- //

class LoggerWrapper
{
  public:

    LoggerWrapper () {}

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
//' @format [S4] object.
//' @name LoggerIteration
//'
//' @section Usage:
//' \preformatted{
//' LoggerIterationWrapper$new(logger_id, use_as_stopper, max_iterations)
//' }
//'
//' @template param-logger_id
//' @template param-use_as_stopper
//' @param max_iterations (`integer(1)`)\cr
//' If the logger is used as stopper this argument defines the maximal iterations.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeLogger()`: `() -> ()`
//'
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
    LoggerIterationWrapper () {
      Rcpp::stop("Cannot create empty logger");
    }

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

//' @title Log the train risk.
//'
//' @description
//' This class logs the train risk for a specific loss function.
//'
//' @format [S4] object.
//' @name LoggerInbagRisk
//'
//' @section Usage:
//' \preformatted{
//' LoggerInbagRisk$new(logger_id, use_as_stopper, loss, eps_for_break, patience)
//' }
//'
//' @template param-logger_id
//' @template param-use_as_stopper
//' @template param-loss
//' @param eps_for_break (`numeric(1)`)\cr
//' This argument becomes active if the loss is also used as stopper. If the relative
//' improvement of the logged inbag risk falls above this boundary the stopper
//' returns `TRUE`.
//' @template param-patience
//'
//' @section Details:
//' This logger computes the risk for the training data
//' \eqn{\mathcal{D} = \{(x^{(i)},\ y^{(i)})\ |\ i \in \{1, \dots, n\}\}}
//' and stores it into a vector. The empirical risk \eqn{\mathcal{R}_\mathrm{emp}} for
//' iteration \eqn{m} is calculated by:
//' \deqn{
//'   \mathcal{R}_\mathrm{emp}^{[m]} = \frac{1}{n}\sum\limits_{i = 1}^n L(y^{(i)}, \hat{f}^{[m]}(x^{(i)}))
//' }
//' __Note:__
//' * If \eqn{m=0} than \eqn{\hat{f}} is just the offset.
//' * The implementation to calculate \eqn{\mathcal{R}_\mathrm{emp}^{[m]}} is done in two steps:
//'   1. Calculate vector \code{risk_temp} of losses for every observation for
//'      given response \eqn{y^{(i)}} and prediction \eqn{\hat{f}^{[m]}(x^{(i)})}.
//'   2. Average over \code{risk_temp}.
//'
//'   This procedure ensures, that it is possible to e.g. use the AUC or any
//'   arbitrary performance measure for risk logging. This gives just one
//'   value for \code{risk_temp} and therefore the average equals the loss
//'   function. If this is just a value (like for the AUC) then the value is
//'   returned.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeLogger()`: `() -> ()`
//'
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
    LoggerInbagRiskWrapper () {
      Rcpp::stop("Cannot create empty logger");
    }

    LoggerInbagRiskWrapper (std::string logger_id0, bool use_as_stopper, LossWrapper& loss, double eps_for_break,
      unsigned int patience)
      : eps_for_break ( eps_for_break ),
        use_as_stopper ( use_as_stopper)
    {
      logger_id = logger_id0;
      sh_ptr_logger = std::make_shared<logger::LoggerInbagRisk>(logger_id, use_as_stopper, loss.getLoss(), eps_for_break, patience);
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

//' @title Log the validation/test/out-of-bag risk
//'
//' @description
//' This class logs the out of bag risk for a specific loss function.
//'
//' @format [S4] object.
//' @name LoggerOobRisk
//'
//' @section Usage:
//' \preformatted{
//' LoggerOobRisk$new(logger_id, use_as_stopper, loss, eps_for_break,
//'   patience, oob_data, oob_response)
//' }
//'
//' @template param-logger_id
//' @template param-use_as_stopper
//' @template param-loss
//' @param eps_for_break (`numeric(1)`)\cr
//' This argument is used if the loss is also used as stopper. If the relative
//' improvement of the logged inbag risk falls above this boundary the stopper
//' returns `TRUE`.
//' @template param-patience
//' @param oob_data (`list()`)\cr
//' A list which contains data source objects which corresponds to the
//' source data of each registered factory. The source data objects should
//' contain the out of bag data. This data is then used to calculate the
//' prediction in each step.
//' @param oob_response ([ResponseRegr] | [ResponseBinaryClassif])\cr
//' The response object used for the predictions on the validation data.
//'
//' @section Details:
//' This logger computes the risk for a given new dataset
//' \eqn{\mathcal{D}_\mathrm{oob} = \{(x^{(i)},\ y^{(i)})\ |\ i \in I_\mathrm{oob}\}}
//' and stores it into a vector. The OOB risk \eqn{\mathcal{R}_\mathrm{oob}} for
//' iteration \eqn{m} is calculated by:
//' \deqn{
//'   \mathcal{R}_\mathrm{oob}^{[m]} = \frac{1}{|\mathcal{D}_\mathrm{oob}|}\sum\limits_{(x,y) \in \mathcal{D}_\mathrm{oob}}
//'   L(y, \hat{f}^{[m]}(x))
//' }
//' __Note:__
//' * If \eqn{m=0} than \eqn{\hat{f}} is just the offset.
//' * The implementation to calculate \eqn{\mathcal{R}_\mathrm{emp}^{[m]}} is done in two steps:
//'   1. Calculate vector \code{risk_temp} of losses for every observation for
//'      given response \eqn{y^{(i)}} and prediction \eqn{\hat{f}^{[m]}(x^{(i)})}.
//'   2. Average over \code{risk_temp}.
//'
//'   This procedure ensures, that it is possible to e.g. use the AUC or any
//'   arbitrary performance measure for risk logging. This gives just one
//'   value for \eqn{risk_temp} and therefore the average equals the loss
//'   function. If this is just a value (like for the AUC) then the value is
//'   returned.
//'
//' @section Fields:
//'   This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeLogger()`: `() -> ()`
//'
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
    LoggerOobRiskWrapper () {
      Rcpp::stop("Cannot create empty logger");
    }

    LoggerOobRiskWrapper (std::string logger_id0, bool use_as_stopper, LossWrapper& loss, double eps_for_break,
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
      sh_ptr_logger = std::make_shared<logger::LoggerOobRisk>(logger_id, use_as_stopper, loss.getLoss(), eps_for_break,
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

//' @title Log the runtime
//'
//' @description
//' This class logs the runtime of the algorithm. The logger also can be used
//' to stop the algorithm after a defined time budget. The available time units are:
//' * minutes
//' * seconds
//' * microseconds
//'
//' @format [S4] object.
//' @name LoggerTime
//'
//' @section Usage:
//' \preformatted{
//' LoggerTime$new(logger_id, use_as_stopper, max_time, time_unit)
//' }
//'
//' @template param-logger_id
//' @template param-use_as_stopper
//' @param max_time (`integer(1)`)\cr
//' If the logger is used as stopper this argument contains the maximal time
//' which are available to train the model.
//' @param time_unit (`character(1)`)\cr
//' The unit in which the time is measured. Choices are `minutes`,
//' `seconds` or `microseconds`.
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$summarizeLogger()`: `() -> ()`
//'
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
    LoggerTimeWrapper () {
      Rcpp::stop("Cannot create empty logger");
    }

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



//' @title Collect loggers
//'
//' @description
//' This class collects all loggers that are used in the algorithm and
//' takes care about stopping strategies and tracing.
//'
//' @format [S4] object.
//' @name LoggerList
//'
//' @section Usage:
//' \preformatted{
//' LoggerList$new()
//' }
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$registerLogger()`: `Logger* -> ()`
//' * `$printRegisteredLogger()`: `() -> ()`
//' * `$clearRegisteredLogger()`: `() -> ()`
//' * `$getNumberOfRegisteredLogger()`: `() -> integer(1)`
//' * `$getNamesOfRegisteredLogger()`: `() -> character()`
//' * `$isStopper()`: `() -> logical()`
//'
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
    LoggerListWrapper ()
    { }

    LoggerListWrapper (std::shared_ptr<loggerlist::LoggerList> ll)
      : sh_ptr_loggerlist ( ll )
    { }

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

    std::map<std::string, bool> isStopper () const
    {
      std::map<std::string, bool> out;
      for (auto& it : sh_ptr_loggerlist->getLoggerMap()) {
        out[it.first] = it.second->isStopper();
      }
      return out;
    }

    virtual ~LoggerListWrapper () {}
};

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
    .constructor ()
    .constructor<std::string, bool, unsigned int> ()
    .method("summarizeLogger", &LoggerIterationWrapper::summarizeLogger)
  ;

  class_<LoggerInbagRiskWrapper> ("LoggerInbagRisk")
    .derives<LoggerWrapper> ("Logger")
    .constructor ()
    .constructor<std::string, bool, LossWrapper&, double, unsigned int> ()
    .method("summarizeLogger", &LoggerInbagRiskWrapper::summarizeLogger)
  ;

  class_<LoggerOobRiskWrapper> ("LoggerOobRisk")
    .derives<LoggerWrapper> ("Logger")
    .constructor ()
    .constructor<std::string, bool, LossWrapper&, double, unsigned int, Rcpp::List, ResponseWrapper&> ()
    .method("summarizeLogger", &LoggerOobRiskWrapper::summarizeLogger)
  ;

  class_<LoggerTimeWrapper> ("LoggerTime")
    .derives<LoggerWrapper> ("Logger")
    .constructor ()
    .constructor<std::string, bool, unsigned int, std::string> ()
    .method("summarizeLogger", &LoggerTimeWrapper::summarizeLogger)
  ;

  class_<LoggerListWrapper> ("LoggerList")
    .constructor ()
    .method("registerLogger", &LoggerListWrapper::registerLogger)
    .method("printRegisteredLogger", &LoggerListWrapper::printRegisteredLogger)
    .method("clearRegisteredLogger", &LoggerListWrapper::clearRegisteredLogger)
    .method("getNumberOfRegisteredLogger", &LoggerListWrapper::getNumberOfRegisteredLogger)
    .method("getNamesOfRegisteredLogger",  &LoggerListWrapper::getNamesOfRegisteredLogger)
    .method("isStopper",  &LoggerListWrapper::isStopper)
  ;
}


// -------------------------------------------------------------------------- //
//                                 OPTIMIZER                                  //
// -------------------------------------------------------------------------- //

class OptimizerWrapper
{
  public:
    OptimizerWrapper ()
    { }

    OptimizerWrapper (std::shared_ptr<optimizer::Optimizer> op)
      : sh_ptr_optimizer ( op )
    { }

    std::shared_ptr<optimizer::Optimizer> getOptimizer ()
    {
      return sh_ptr_optimizer;
    }

    std::string getOptimizerType () const
    {
      return sh_ptr_optimizer->getType();
    }

    virtual ~OptimizerWrapper () {}

  protected:
    std::shared_ptr<optimizer::Optimizer> sh_ptr_optimizer;
};

//' @title Coordinate descent
//'
//' @description
//' This class defines a new object to conduct gradient descent in function space.
//' Because of the component-wise structure, this is more like a block-wise
//' coordinate descent.
//'
//' @format [S4] object.
//' @name OptimizerCoordinateDescent
//'
//' @section Usage:
//' \preformatted{
//' OptimizerCoordinateDescent$new()
//' OptimizerCoordinateDescent$new(ncores)
//' }
//'
//' @template param-ncores
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$getOptimizerType()`: `() -> character(1)`
//' * `$getStepSize()`: `() -> numeric()`
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

    // Include bool arguments to have a unique constructor that can be used by the RCPP modules:
    OptimizerCoordinateDescent (OptimizerWrapper op, bool b1)
      : OptimizerWrapper::OptimizerWrapper ( std::static_pointer_cast<optimizer::OptimizerCoordinateDescent>(op.getOptimizer()) )
    { }
};

//' @title Coordinate descent with cosine annealing
//'
//' @description
//' Same as [OptimizerCoordinateDescent] but with a cosine annealing scheduler to
//' adjust the learning rate during the fitting process.
//'
//' @format [S4] object.
//' @name OptimizerCosineAnnealing
//'
//' @section Usage:
//' \preformatted{
//' OptimizerCosineAnnealing$new()
//' OptimizerCosineAnnealing$new(ncores)
//' OptimizerCosineAnnealing$new(nu_min, nu_max, cycles, anneal_iter_max, cycles)
//' OptimizerCosineAnnealing$new(nu_min, nu_max, cycles, anneal_iter_max, cycles, ncores)
//' }
//'
//' @template param-ncores
//' @param nu_min (`numeric(1)`)\cr
//' Minimal learning rate.
//' @param nu_max (`numeric(1)`)\cr
//' Maximal learning rate.
//' @param cycles (`integer(1)`)\cr
//' Number of annealing cycles form `nu_max` to `nu_min` between 1 and anneal_`anneal_iter_max`.
//' `anneal_iter_max (`integer(1)`)\cr
//' Maximal number of iterations for which the annealing is conducted. `nu_min` is used as
//' fixed learning rate after `anneal_iter_max`.
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$getOptimizerType()`: `() -> character(1)`
//' * `$getStepSize()`: `() -> numeric()`
//'
//' @examples
//'
//' # Define optimizer:
//' optimizer = OptimizerCosineAnnealing$new()
//'
//' @export OptimizerCosineAnnealing
class OptimizerCosineAnnealing : public OptimizerWrapper
{
public:
  OptimizerCosineAnnealing () {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCosineAnnealing>();
  }
  OptimizerCosineAnnealing (unsigned int num_threads) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCosineAnnealing>(num_threads);
  }
  OptimizerCosineAnnealing (double nu_min, double nu_max, unsigned int cycles, unsigned int anneal_iter_max) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCosineAnnealing>(nu_min, nu_max, cycles, anneal_iter_max, 1);
  }
  OptimizerCosineAnnealing (double nu_min, double nu_max, unsigned int cycles, unsigned int anneal_iter_max,
      unsigned int num_threads) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerCosineAnnealing>(nu_min, nu_max, cycles, anneal_iter_max,
      num_threads);
  }
  // Include bool arguments to have a unique constructor that can be used by the RCPP modules:
  OptimizerCosineAnnealing (OptimizerWrapper op, bool b1)
    : OptimizerWrapper::OptimizerWrapper ( std::static_pointer_cast<optimizer::OptimizerCosineAnnealing>(op.getOptimizer()) )
  { }

  std::vector<double> getStepSize() { return sh_ptr_optimizer->getStepSize(); }
};

//' @title Coordinate descent with line search
//'
//' @description
//' Same as [OptimizerCoordinateDescent] but with a line search in each iteration.
//'
//' @format [S4] object.
//' @name OptimizerCoordinateDescentLineSearch
//'
//' @section Usage:
//' \preformatted{
//' OptimizerCoordinateDescentLineSearch$new()
//' OptimizerCoordinateDescentLineSearch$new(ncores)
//' }
//'
//' @template param-ncores
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$getOptimizerType()`: `() -> character(1)`
//' * `$getStepSize()`: `() -> numeric()`
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
  // Include bool arguments to have a unique constructor that can be used by the RCPP modules:
  OptimizerCoordinateDescentLineSearch (OptimizerWrapper op, bool b1)
    : OptimizerWrapper::OptimizerWrapper ( std::static_pointer_cast<optimizer::OptimizerCoordinateDescentLineSearch>(op.getOptimizer()) )
  { }
  std::vector<double> getStepSize() { return sh_ptr_optimizer->getStepSize(); }
};


//' @title Nesterovs momentum
//'
//' @description
//' This class defines a new object to conduct Nesterovs momentum in function space.
//'
//' @format [S4] object.
//' @name OptimizerAGBM
//'
//' @section Usage:
//' \preformatted{
//' OptimizerAGBM$new(momentum)
//' OptimizerAGBM$new(momentum, ncores)
//' }
//'
//' @template param-ncores
//' @param momentum (`numeric(1)`)\cr
//' Momentum term used to accelerate the fitting process. If chosen large, the algorithm trains
//' faster but also tends to overfit faster.
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$getOptimizerType()`: `() -> character(1)`
//' * `$getStepSize()`: `() -> numeric()`
//' * `$getMomentumParameter()`: `() -> numeric(1)`
//' * `$getSelectedMomentumBaselearner()`: `() -> character()`
//' * `$getParameterMatrix()`: `() -> list(matrix()`
//' * `$getErrorCorrectedPseudoResiduals()`: `() -> matrix()`
//'
//' @examples
//'
//' optimizer = OptimizerAGBM$new(0.1)
//'
//' @export OptimizerAGBM
class OptimizerAGBM: public OptimizerWrapper
{
public:
  OptimizerAGBM () {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerAGBM>(0.1);
  }
  OptimizerAGBM (double momentum) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerAGBM>(momentum);
  }
  OptimizerAGBM (double momentum, unsigned int num_threads) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerAGBM>(momentum, num_threads);
  }
  OptimizerAGBM (double momentum, unsigned int acc_iters, unsigned int num_threads) {
    sh_ptr_optimizer = std::make_shared<optimizer::OptimizerAGBM>(momentum, acc_iters, num_threads);
  }
  // Include bool arguments to have a unique constructor that can be used by the RCPP modules:
  OptimizerAGBM (OptimizerWrapper op, bool b1, bool b2, bool b3)
    : OptimizerWrapper::OptimizerWrapper ( std::static_pointer_cast<optimizer::OptimizerAGBM>(op.getOptimizer()) )
  { }


  std::map<std::string, arma::mat> getMomentumParameter ()
  {
    std::map<std::string, arma::mat> param_map = std::static_pointer_cast<optimizer::OptimizerAGBM>(sh_ptr_optimizer)->getMomentumParameter();
    return param_map;
  }
  std::vector<std::string> getSelectedMomentumBaselearner ()
  {
    std::vector<std::string> out  = std::static_pointer_cast<optimizer::OptimizerAGBM>(sh_ptr_optimizer)->getSelectedMomentumBaselearner();
    return out;
  }
  Rcpp::List getParameterMatrix () const
  {
    std::pair<std::vector<std::string>, arma::mat> out_pair = std::static_pointer_cast<optimizer::OptimizerAGBM>(sh_ptr_optimizer)->getParameterMatrix();

    return Rcpp::List::create(
      Rcpp::Named("parameter_names")   = out_pair.first,
      Rcpp::Named("parameter_matrix")  = out_pair.second
    );
  }
  std::vector<double> getStepSize() { return sh_ptr_optimizer->getStepSize(); }

  arma::mat getErrorCorrectedPseudoResiduals()
  {
    return std::static_pointer_cast<optimizer::OptimizerAGBM>(sh_ptr_optimizer)->getErrorCorrectedPseudoResiduals();
  }
};


RCPP_EXPOSED_CLASS(OptimizerWrapper)
RCPP_MODULE(optimizer_module)
{
  using namespace Rcpp;

  class_<OptimizerWrapper> ("Optimizer")
    .constructor ()
    .method("getOptimizerType", &OptimizerWrapper::getOptimizerType)
  ;

  class_<OptimizerCoordinateDescent> ("OptimizerCoordinateDescent")
    .derives<OptimizerWrapper> ("Optimizer")
    .constructor ()
    .constructor <unsigned int> ()
    .constructor <OptimizerWrapper, bool> ()
  ;

  class_<OptimizerCosineAnnealing> ("OptimizerCosineAnnealing")
    .derives<OptimizerWrapper> ("Optimizer")
    .constructor ()
    .constructor <unsigned int> ()
    .constructor <double, double, unsigned int, unsigned int> ()
    .constructor <double, double, unsigned int, unsigned int, unsigned int> ()
    .constructor <OptimizerWrapper, bool> ()
    .method("getStepSize", &OptimizerCosineAnnealing::getStepSize)
  ;

  class_<OptimizerCoordinateDescentLineSearch> ("OptimizerCoordinateDescentLineSearch")
    .derives<OptimizerWrapper> ("Optimizer")
    .constructor ()
    .constructor <unsigned int> ()
    .constructor <OptimizerWrapper, bool> ()
    .method("getStepSize", &OptimizerCoordinateDescentLineSearch::getStepSize)
  ;

  class_<OptimizerAGBM> ("OptimizerAGBM")
    .derives<OptimizerWrapper> ("Optimizer")
    .constructor  ()
    .constructor <double> ()
    .constructor <double, unsigned int> ()
    .constructor <double, unsigned int, unsigned int> ()
    .constructor <OptimizerWrapper, bool, bool, bool> ()
    .method("getMomentumParameter", &OptimizerAGBM::getMomentumParameter)
    .method("getSelectedMomentumBaselearner", &OptimizerAGBM::getSelectedMomentumBaselearner)
    .method("getStepSize", &OptimizerAGBM::getStepSize)
    .method("getParameterMatrix", &OptimizerAGBM::getParameterMatrix)
    .method("getErrorCorrectedPseudoResiduals", &OptimizerAGBM::getErrorCorrectedPseudoResiduals)
  ;
}


// -------------------------------------------------------------------------- //
//                                 COMPBOOST                                  //
// -------------------------------------------------------------------------- //

//' @title Internal Compboost Class
//'
//' This class is the raw `C++` pendant and still at a very high-level.
//' It is the base for the [Compboost] [R6] class and provides
//' many convenient wrapper to access data and execute methods by calling
//' the `C++` methods.
//'
//' @format [S4] object.
//' @name Compboost_internal
//'
//' @section Usage:
//' \preformatted{
//' Compboost$new(response, learning_rate, stop_if_all_stopper_fulfilled,
//'   factory_list, loss, logger_list, optimizer)
//' }
//'
//' @param oob_response ([ResponseRegr] | [ResponseBinaryClassif])\cr
//' The response object containing the target variable.
//' @param learning_rate (`numeric(1)`)\cr
//' The learning rate.
//' @param stop_if_all_stopper_fulfilled (`logical(1)`)\cr
//' Boolean to indicate which stopping strategy is used. If `TRUE`,
//' the algorithm stops if the conditions of all loggers for stopping apply.
//' @param factory_list ([BlearnerFactoryList])\cr
//'   List of base learner factories from which one base learner is selected
//'   in each iteration by using the
//' @template param-loss
//' @param logger_list ([LoggerList])\cr
//' The [LoggerList] object with all loggers.
//' @template param-optimizer
//'
//' @section Fields:
//' This class doesn't contain public fields.
//'
//' @section Methods:
//' * `$train()`: `() -> ()`
//' * `$continueTraining()`: `() -> ()`
//' * `$getLearningRate()`: `() -> numeric(1)`
//' * `$getPrediction()`: `() -> matrix()`
//' * `$getSelectedBaselearner()`: `() -> character()`
//' * `$getLoggerData()`: `() -> list(character(), matrix())`
//' * `$getEstimatedParameter()`: `() -> list(matrix())`
//' * `$getParameterAtIteration()`: `() -> list(matrix())`
//' * `$getParameterMatrix()`: `() -> matrix()`
//' * `$predictFactoryTrainData()`: `() -> matrix()`
//' * `$predictFactoryNewData()`: `list(Data*) -> matrix()`
//' * `$predictIndividualTrainData()`: `() -> list(matrix())` Get the linear contribution of each base learner for the training data.
//' * `$predictIndividual()`: `list(Data*) -> list(matrix())` Get the linear contribution of each base learner for new data.
//' * `$predict()`: `list(Data*), logical(1) -> matrix()`
//' * `$summarizeCompboost()`: `() -> ()`
//' * `$isTrained()`: `() -> logical(1)`
//' * `$setToIteration()`: `() -> ()`
//' * `$saveJson()`: `() -> ()`
//' * `$getOffset()`: `() -> numeric(1) | matrix()`
//' * `$getRiskVector()`: `() -> numeric()`
//' * `$getResponse()`: `() -> Response*`
//' * `$getOptimizer()`: `() -> Optimizer*`
//' * `$getLoss()`: `() -> Loss*`
//' * `$getLoggerList()`: `() -> LoggerList`
//' * `$getBaselearnerList()`: `() -> BlearnerFactoryList`
//' * `$useGlobalStopping()`: `() -> logical(1)*`
//' * `$getFactoryMap()`: `() -> list(Baselearner*)`
//' * `$getDataMap()`: `() -> list(Data*)`
//' @examples
//'
//' # Some data:
//' df = mtcars
//' df$mpg_cat = ifelse(df$mpg > 20, "high", "low")
//'
//' # # Create new variable to check the polynomial base learner with degree 2:
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
//' # Get trace of selected base learner:
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
  private:
    std::unique_ptr<cboost::Compboost> unique_ptr_cboost;

    std::shared_ptr<blearnerlist::BaselearnerFactoryList> sh_ptr_blearner_list;
    std::shared_ptr<loggerlist::LoggerList> sh_ptr_loggerlist;
    std::shared_ptr<optimizer::Optimizer> sh_ptr_optimizer;

    double learning_rate0;
    bool is_trained = false;

  public:
    CompboostWrapper (ResponseWrapper& response, double learning_rate,
      bool stop_if_all_stopper_fulfilled, BlearnerFactoryListWrapper& factory_list,
      LossWrapper& loss, LoggerListWrapper& logger_list, OptimizerWrapper& optimizer)
    {
      learning_rate0       =  learning_rate;
      sh_ptr_loggerlist    =  logger_list.getLoggerList();
      sh_ptr_optimizer     =  optimizer.getOptimizer();
      sh_ptr_blearner_list =  factory_list.getFactoryList();

      unique_ptr_cboost = std::make_unique<cboost::Compboost>(response.getResponseObj(), learning_rate0,
        stop_if_all_stopper_fulfilled, sh_ptr_optimizer, loss.getLoss(), sh_ptr_loggerlist, sh_ptr_blearner_list);
    }

    CompboostWrapper (const std::string file)
    {
      unique_ptr_cboost = std::make_unique<cboost::Compboost>(file);

      sh_ptr_blearner_list = unique_ptr_cboost->getBaselearnerList();
      sh_ptr_loggerlist = unique_ptr_cboost->getLoggerList();
      sh_ptr_optimizer = unique_ptr_cboost->getOptimizer();
      learning_rate0 = unique_ptr_cboost->getLearningRate();

      is_trained = true;
    }

    // Member functions
    void train (unsigned int trace)
    {
      unique_ptr_cboost->trainCompboost(trace);
      is_trained = true;
    }

    void continueTraining (unsigned int trace)
    {
      unique_ptr_cboost->continueTraining(trace);
    }

    arma::vec getPrediction (bool as_response)
    {
      return unique_ptr_cboost->getPrediction(as_response);
    }

    double getLearningRate () const
    {
      return unique_ptr_cboost->getLearningRate();
    }

    std::vector<std::string> getSelectedBaselearner ()
    {
      return unique_ptr_cboost->getSelectedBaselearner();
    }

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


    arma::mat predictFactoryTrainData (const std::string& factory_id) { return unique_ptr_cboost->predictFactory(factory_id); }

    arma::mat predictFactoryNewData (const std::string& factory_id, const Rcpp::List& newdata) {
      std::map<std::string, std::shared_ptr<data::Data>> data_map;

      for (unsigned int i = 0; i < newdata.size(); i++) {
        DataWrapper* temp = newdata[i];
        data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();
      }

      return unique_ptr_cboost->predictFactory(factory_id, data_map);
    }

    std::map<std::string, arma::mat> predictIndividualTrainData () { return unique_ptr_cboost->predictIndividual(); }

    std::map<std::string, arma::mat> predictIndividual (Rcpp::List& newdata)
    {
      std::map<std::string, std::shared_ptr<data::Data>> data_map;

      for (unsigned int i = 0; i < newdata.size(); i++) {
        DataWrapper* temp = newdata[i];
        data_map[ temp->getDataObj()->getDataIdentifier() ] = temp->getDataObj();
      }
      return unique_ptr_cboost->predictIndividual(data_map);
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

    void summarizeCompboost () const { unique_ptr_cboost->summarizeCompboost(); }
    bool isTrained () const { return is_trained; }
    void setToIteration (const unsigned int& k, const unsigned int& trace) { unique_ptr_cboost->setToIteration(k, trace); }

    void saveJson(std::string file, bool rm_data) { unique_ptr_cboost->saveJson(file, rm_data); }

    arma::mat getOffset () const { return unique_ptr_cboost->getOffset(); }
    std::vector<double> getRiskVector () const { return unique_ptr_cboost->getRiskVector(); }

    ResponseWrapper getResponse () const
    {
      ResponseWrapper out(unique_ptr_cboost->getResponse());
      return out;
    }

    OptimizerWrapper getOptimizer () const
    {
      OptimizerWrapper out(unique_ptr_cboost->getOptimizer());
      return out;
    }

    LossWrapper getLoss () const
    {
      LossWrapper out(unique_ptr_cboost->getLoss());
      return out;
    }

    LoggerListWrapper getLoggerList () const
    {
      LoggerListWrapper out(unique_ptr_cboost->getLoggerList());
      return out;
    }

    BlearnerFactoryListWrapper getBaselearnerList () const
    {
      BlearnerFactoryListWrapper out(unique_ptr_cboost->getBaselearnerList());
      return out;
    }

    bool useGlobalStopping () const
    {
      return unique_ptr_cboost->useGlobalStopping();
    }

    std::map<std::string, BaselearnerFactoryWrapper> getFactoryMap () const
    {
      std::map<std::string, BaselearnerFactoryWrapper> out;
      auto fmt = unique_ptr_cboost->getBaselearnerList()->getFactoryMap();
      for (auto& it : fmt) {
        out[it.first] = BaselearnerFactoryWrapper(it.second);
      }
      return out;
    }

    std::map<std::string, DataWrapper> getDataMap () const
    {
      std::map<std::string, DataWrapper> out;
      std::vector<std::shared_ptr<data::Data>> dvec;
      for (auto& it : unique_ptr_cboost->getBaselearnerList()->getFactoryMap()) {
        dvec = it.second->getVecDataSource();
        for (auto& it : dvec) {
          out[it->getDataIdentifier()] = DataWrapper(it);
        }
      }
      return out;
    }

    ~CompboostWrapper () {}
};


RCPP_EXPOSED_CLASS(CompboostWrapper)
RCPP_MODULE (compboost_module)
{
  using namespace Rcpp;

  class_<CompboostWrapper> ("Compboost_internal")
    .constructor<ResponseWrapper&, double, bool, BlearnerFactoryListWrapper&, LossWrapper&, LoggerListWrapper&, OptimizerWrapper&> ()
    .constructor<std::string> ()

    .method("train",                      &CompboostWrapper::train)
    .method("continueTraining",           &CompboostWrapper::continueTraining)
    .method("getLearningRate",            &CompboostWrapper::getLearningRate)
    .method("getPrediction",              &CompboostWrapper::getPrediction)
    .method("getSelectedBaselearner",     &CompboostWrapper::getSelectedBaselearner)
    .method("getLoggerData",              &CompboostWrapper::getLoggerData)
    .method("getEstimatedParameter",      &CompboostWrapper::getEstimatedParameter)
    .method("getParameterAtIteration",    &CompboostWrapper::getParameterAtIteration)
    .method("getParameterMatrix",         &CompboostWrapper::getParameterMatrix)
    .method("predictFactoryTrainData",    &CompboostWrapper::predictFactoryTrainData)
    .method("predictFactoryNewData",      &CompboostWrapper::predictFactoryNewData)
    .method("predictIndividualTrainData", &CompboostWrapper::predictIndividualTrainData)
    .method("predictIndividual",          &CompboostWrapper::predictIndividual)
    .method("predict",                    &CompboostWrapper::predict)
    .method("summarizeCompboost",         &CompboostWrapper::summarizeCompboost)
    .method("isTrained",                  &CompboostWrapper::isTrained)
    .method("setToIteration",             &CompboostWrapper::setToIteration)
    .method("saveJson",                   &CompboostWrapper::saveJson)
    .method("getOffset",                  &CompboostWrapper::getOffset)
    .method("getRiskVector",              &CompboostWrapper::getRiskVector)
    .method("getResponse",                &CompboostWrapper::getResponse)
    .method("getOptimizer",               &CompboostWrapper::getOptimizer)
    .method("getLoss",                    &CompboostWrapper::getLoss)
    .method("getLoggerList",              &CompboostWrapper::getLoggerList)
    .method("getBaselearnerList",         &CompboostWrapper::getBaselearnerList)
    .method("useGlobalStopping",          &CompboostWrapper::useGlobalStopping)
    .method("getFactoryMap",              &CompboostWrapper::getFactoryMap)
    .method("getDataMap",                 &CompboostWrapper::getDataMap)
  ;
}

#endif // COMPBOOST_MODULES_CPP_
