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
//   The "LinearWrapper" class which is exported to R by using the Rcpp
//   modules. For a tutorial see
//   <http://dirk.eddelbuettel.com/code/rcpp/Rcpp-modules.pdf>.
//
//   This one is just to play around with! It doesn't have any real usecase!!!
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
// =========================================================================== #

#include "baselearner_factory.h"
#include "baselearner_list.h"

class BaselearnerWrapper
{
  private:
  
    blearner::Baselearner *obj;
    blearnerfactory::BaselearnerFactory *factory_obj;
  
  public:
    
    // Baselearnerlist for all the baselearner:
    static blearnerlist::BaselearnerList *blearner_factory_list;
    
    // Constructors
    // -----------------
    
    // Polynomial baselearner:
    BaselearnerWrapper (std::string identifier, arma::mat data, unsigned int degree)
    {
      factory_obj = new blearnerfactory::BaselearnerFactory("polynomial", data);
      
      // Initialize baselearner:
      obj = factory_obj->CreateBaselearner(identifier, degree);
    }
    
    // Custom baselearner:
    BaselearnerWrapper (std::string identifier, arma::mat data,
      Rcpp::Function instantiateDataFun, Rcpp::Function trainFun, 
      Rcpp::Function predictFun, Rcpp::Function extractParameter)
    {
      // The custom baselearner have a predefined type 'custom':
      factory_obj = new blearnerfactory::BaselearnerFactory("custom", data);
      
      // Initialize baselearner:
      obj = factory_obj->CreateBaselearner(identifier, instantiateDataFun, trainFun, 
        predictFun, extractParameter);
    }
  
    // Member functions
    // ------------------
    
    void train (arma::vec &response)
    {
      obj->train(response);
    }
    
    std::string GetIdentifier ()
    {
      return obj->GetIdentifier();
    }
    
    arma::mat GetParameter ()
    {
      return obj->GetParameter();
    }
    
    std::string GetBaselearnerType ()
    {
      return factory_obj->GetBaselearnerType();
    }
    
    arma::mat predict ()
    {
      return obj->predict();
    }
    
    arma::mat GetData () 
    {
      return obj->GetData();
    }
    
    // Register Factory in 'BaselearnerList':
    void RegisterFactory (std::string id)
    {
      blearner_factory_list->RegisterBaselearnerFactory (factory_obj->GetBaselearnerType() + " - " + id, *factory_obj);
    }
};

blearnerlist::BaselearnerList *BaselearnerWrapper::blearner_factory_list = new blearnerlist::BaselearnerList();

//' @title Print Registered Factorys
//'
//' @description This function prints all registered factorys to the screen.
//'   This is just ment to be a control function to see which factorys are
//'   already initialized. 
//' @export
// [[Rcpp::export]]
void printRegisteredFactorys ()
{
  BaselearnerWrapper::blearner_factory_list->PrintRegisteredFactorys();
}
//' @title Clear the existing Hash Map of 'BaselearnerFactorys'
//'
//' @description This function delets all registered factorys. This may be
//'   necessary for fitting a new object.
//' @export
// [[Rcpp::export]]
void clearRegisteredFactorys ()
{
  BaselearnerWrapper::blearner_factory_list->ClearMap();
  std::cout << "Clear all registered factorys!" << std::endl;
}


// --------------------------------------------------------------------------- #
// Rcpp module:
// --------------------------------------------------------------------------- #

RCPP_MODULE(baselearner_module) {

  using namespace Rcpp;

  class_<BaselearnerWrapper> ("BaselearnerWrapper")

  .constructor <std::string, arma::mat, unsigned int> ("Initialize BaselearnerWrapper (Baselearner) object")
  .constructor <std::string, arma::mat, Rcpp::Function, Rcpp::Function, Rcpp::Function, Rcpp::Function> ("Initialize BaselearnerWrapper (Baselearner) object with custom baselearner")
  
  .method ("train",                 &BaselearnerWrapper::train, "Train a linear baselearner")
  .method ("GetIdentifier",         &BaselearnerWrapper::GetIdentifier, "Get Identifier of baselearner")
  .method ("GetBaselearnerType",    &BaselearnerWrapper::GetBaselearnerType, "Get the type of the baselearner")
  .method ("GetParameter",          &BaselearnerWrapper::GetParameter, "Get the parameter of the learner")
  .method ("predict",               &BaselearnerWrapper::predict, "Predict a linear baselearner")
  .method ("GetData",               &BaselearnerWrapper::GetData, "Get the data")
  .method ("RegisterFactory",       &BaselearnerWrapper::RegisterFactory, "Register a factory")
  ;
}
