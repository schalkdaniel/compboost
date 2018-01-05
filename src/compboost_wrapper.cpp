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
//   Wrapper of the "Baselearner" and "Compboost" class. Those are in one file
//   since we have a static "BaselearnerList" object which should be used by
//   both wrapper. This classes are exposed within this class using the
//   Rcpp modules.
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

#include "compboost.h"
#include "baselearner_factory.h"
#include "baselearner_list.h"


// -------------------------------------------------------------------------- //
//                  Wrapper of the baselearner class                          //
// -------------------------------------------------------------------------- //

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
  BaselearnerWrapper (std::string identifier, arma::mat data, std::string data_identifier, 
    unsigned int degree)
  {
    factory_obj = new blearnerfactory::PolynomialFactory("polynomial", data, data_identifier, degree);
    
    // Initialize baselearner:
    obj = factory_obj->CreateBaselearner(identifier);
  }
  
  // Custom baselearner:
  BaselearnerWrapper (std::string identifier, arma::mat data, std::string data_identifier,
    Rcpp::Function instantiateDataFun, Rcpp::Function trainFun, Rcpp::Function predictFun, 
    Rcpp::Function extractParameter)
  {
    // The custom baselearner have a predefined type 'custom':
    factory_obj = new blearnerfactory::CustomFactory("custom", data, data_identifier,
      instantiateDataFun, trainFun, predictFun, extractParameter);
    
    // Initialize baselearner:
    obj = factory_obj->CreateBaselearner(identifier);
  }
  
  // Member functions
  // ------------------
  
  // This ones just calles the member functions of the baselearner object:
  
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
    
    // std::cout << "My data identifier is: " << obj->GetDataIdentifier() << std::endl;
    // std::cout << std::endl;
    
    return obj->GetData();
  }
  
  // Register Factory in 'BaselearnerList':
  void RegisterFactory (std::string id)
  {
    std::string factory_registry = factory_obj->GetBaselearnerType() + 
      " with id " + id + " of variable " + factory_obj->GetDataIdentifier();
    
    blearner_factory_list->RegisterBaselearnerFactory (factory_registry, factory_obj);
  }
};

// Instantiate static BaselearnerList object:
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

//' @title Optimize over registered Baselearner
//'
//' @description This function train every baselearner within the registered
//'   list and returns a list of specifica about the best baselearner since
//'   we are not supposed to give back a not exported C++ class in R.
//' @param pseudo_residuals [\code{numeric}] \cr
//'   Pseudo residuals to train the baselearner.
//' @return [\code{list}] \cr
//'   List of specifica about the best baselearner.
//' @export
// [[Rcpp::export]]
Rcpp::List getBestBaselearner (arma::vec &pseudo_residuals)
{
  optimizer::Optimizer *opt = new optimizer::Greedy (*BaselearnerWrapper::blearner_factory_list);
  
  std::string temp_string = "test run";
  
  blearner::Baselearner *blearner = opt->FindBestBaselearner(temp_string, pseudo_residuals);
  
  return Rcpp::List::create(
    blearner->GetIdentifier(), 
    blearner->GetParameter()
  );
}


// -------------------------------------------------------------------------- //
//                       Wrapper of Compboost Class                           //
// -------------------------------------------------------------------------- //


// IN PROGRESS!


class CompboostWrapper
{
  public:

    // Constructor
    CompboostWrapper (arma::vec response, unsigned int max_iterations0, 
      double learning_rate0) {
      
      std::cout << "Create Compbost!" << std::endl;
      
      max_iterations = max_iterations0;
      std::cout << "Max iterations are setted" << std::endl;
      
      learning_rate = learning_rate0;
      std::cout << "Learning rate is setted" << std::endl;
      
      optimizer::Optimizer* used_optimizer = new optimizer::Greedy(*BaselearnerWrapper::blearner_factory_list);
      std::cout << "New Optimizer is created!" << std::endl;
      
      loss::Loss* used_loss = new loss::Quadratic();
      std::cout << "New Loss is created!" << std::endl;
      
      std::chrono::system_clock::time_point init_time;
      init_time = std::chrono::high_resolution_clock::now();
      std::cout << "New time was instantiated" << std::endl;
      
      loggerlist::LoggerList* used_logger = new loggerlist::LoggerList(*eval_data, init_time, 4);
      std::cout << "New loggerlist was instantiated" << std::endl;
      
      logger::Logger* log_iterations = new logger::LogIteration(max_iterations);
      std::cout << "New logger (Greed) was instantiated" << std::endl;
      
      used_logger->RegisterLogger("iterations", log_iterations);
      std::cout << "Logger was registered" << std::endl;
      
      obj = new cboost::Compboost(response, learning_rate, false, used_optimizer, used_loss, used_logger);
      std::cout << "Compboost object was declared" << std::endl;
    }

    // Member functions
    void Train () {
      obj->TrainCompboost();
    }
    // arma::vec Predict () {
    //   return obj->PredictEnsemble();
    // }

  private:

    cboost::Compboost* obj;
    arma::mat* eval_data;
    unsigned int max_iterations;
    double learning_rate;
};

// -------------------------------------------------------------------------- //
//                            Rcpp Modules                                    //
// -------------------------------------------------------------------------- //

// Expose Baselearner:
// -------------------

RCPP_MODULE(baselearner_module) {
  
  using namespace Rcpp;
  
  class_<BaselearnerWrapper> ("BaselearnerWrapper")
    
    .constructor <std::string, arma::mat, std::string, unsigned int> ("Initialize BaselearnerWrapper (Baselearner) object")
    .constructor <std::string, arma::mat, std::string, Rcpp::Function, Rcpp::Function, Rcpp::Function, Rcpp::Function> ("Initialize BaselearnerWrapper (Baselearner) object with custom baselearner")
    
    .method ("train",                 &BaselearnerWrapper::train, "Train a linear baselearner")
    .method ("GetIdentifier",         &BaselearnerWrapper::GetIdentifier, "Get Identifier of baselearner")
    .method ("GetBaselearnerType",    &BaselearnerWrapper::GetBaselearnerType, "Get the type of the baselearner")
    .method ("GetParameter",          &BaselearnerWrapper::GetParameter, "Get the parameter of the learner")
    .method ("predict",               &BaselearnerWrapper::predict, "Predict a linear baselearner")
    .method ("GetData",               &BaselearnerWrapper::GetData, "Get the data")
    .method ("RegisterFactory",       &BaselearnerWrapper::RegisterFactory, "Register a factory")
    ;
}

// Expose Compboost:
// -----------------

RCPP_MODULE(compboost_module) {

  using namespace Rcpp;

  class_<CompboostWrapper> ("CompboostWrapper")

  .constructor<arma::vec, unsigned int, double> ("Initialize CompboostWrapper (Compboost) object")

  .method ("Train", &CompboostWrapper::Train, "Get the response of the Compboost object")
  // .method ("Predict", &CompboostWrapper::Predict, "Set the response of the Compboost object")
  ;
}
