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

#ifndef BASELEARNERLIST_H_
#define BASELEARNERLIST_H_

#include <map>

#include "baselearner_factory.h"

// Define the type for the list (because we are lazy :))
typedef std::map<std::string, blearnerfactory::BaselearnerFactory*> blearner_factory_map;

namespace blearnerlist
{

// Later we will create one static object of this class. This is a workaround
// to register new factorys from R.

class BaselearnerFactoryList 
{
private:
  
  // Main list object:
  blearner_factory_map my_factory_map;
  
public:
  
  BaselearnerFactoryList ();
  
  // Functions to register a baselearner factory and print all registered
  // factories:
  void registerBaselearnerFactory (const std::string&, blearnerfactory::BaselearnerFactory*);
  void printRegisteredFactories () const;
  
  // Get the actual map:
  blearner_factory_map getMap () const;
  
  // Clear all elements wich are registered:
  void clearMap();
  
  // Get the data used for modelling:
  std::pair<std::vector<std::string>, arma::mat> getModelFrame () const;

  // Get names of registered factories:
  std::vector<std::string> getRegisteredFactoryNames () const;
  
  // ~BaselearnerFactoryList () {Rcpp::Rcout << "Destroy BaselearnerFactoryList!" << std::endl; }
};

} // namespace blearnerlist

#endif // BASELEARNERLIST_H_

