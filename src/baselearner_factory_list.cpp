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
//   Implementation of the "BaselearnerFactoryList".
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
// ========================================================================== //

#include "baselearner_factory_list.h"

namespace blearnerlist 
{

// Just an empty constructor:
BaselearnerFactoryList::BaselearnerFactoryList () {}

// Register a factory:
void BaselearnerFactoryList::registerBaselearnerFactory (const std::string& factory_id, 
  blearnerfactory::BaselearnerFactory *blearner_factory)
{
  // Create iterator and check if learner is already registered:
  std::map<std::string, blearnerfactory::BaselearnerFactory*>::iterator it = my_factory_map.find(factory_id);
  
  if (it == my_factory_map.end()) {
    my_factory_map.insert(std::pair<std::string, blearnerfactory::BaselearnerFactory*>(factory_id, blearner_factory));
  } else {
    my_factory_map[ factory_id ] = blearner_factory;
  }
}

// Print all registered factorys:
void BaselearnerFactoryList::printRegisteredFactorys () const
{
  // Check if any factory is registered:
  if (my_factory_map.size() >= 1) {
    Rcpp::Rcout << "Registered Factorys:\n";
  } else {
    Rcpp::Rcout << "No registered Factorys!";
  }
  
  // Iterate over all registered factorys and print the factory identifier:
  for (auto& it : my_factory_map) {
    Rcpp::Rcout << "\t- " << it.first << std::endl;
  }
}

// Getter for the map object:
blearner_factory_map BaselearnerFactoryList::getMap () const
{
  return my_factory_map;
}

// Remove all registered factorys:
void BaselearnerFactoryList::clearMap ()
{
  // Just delete the pointer, so we have a new empty map. The factories which
  // are behind the pointers should delete themselfe
  my_factory_map.clear();
}

std::pair<std::vector<std::string>, arma::mat> BaselearnerFactoryList::getModelFrame () const
{
  arma::mat out_matrix;
  std::vector<std::string> rownames;
  
  for (auto& it : my_factory_map) {
    // This step converts the matrix from get data to an arma::mat, especially
    // if getData returns an arma::sp_mat:
    arma::mat data_temp(it.second->getData());
    out_matrix = arma::join_rows(out_matrix, data_temp);
    
    if (data_temp.n_cols > 1) {
      for (unsigned int i = 0; i < data_temp.n_cols; i++) {
        rownames.push_back(it.first + "x1" + std::to_string(i + 1));
      }
    } else {
      rownames.push_back(it.first);
    }
  }
  return std::pair<std::vector<std::string>, arma::mat>(rownames, out_matrix);
}

std::map<std::string, arma::mat> BaselearnerFactoryList::getDataMap () const
{
  std::map<std::string, arma::mat> out_map;
  
  for (auto& it : my_factory_map) {
    out_map[it.first] = it.second->getData();
  }
  return out_map;
}

} // namespace blearnerlist
