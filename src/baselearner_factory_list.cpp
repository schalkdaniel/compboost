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

#include "baselearner_factory_list.h"

namespace blearnerlist
{

BaselearnerFactoryList::BaselearnerFactoryList () {}

void BaselearnerFactoryList::registerBaselearnerFactory (const std::string factory_id,
  const std::shared_ptr<blearnerfactory::BaselearnerFactory> sh_ptr_blearner_factory)
{
  // Create iterator and check if learner is already registered:
  std::map<std::string, std::shared_ptr<blearnerfactory::BaselearnerFactory>>::iterator it = _factory_map.find(factory_id);

  if (it == _factory_map.end()) {
    _factory_map.insert(std::pair<std::string, std::shared_ptr<blearnerfactory::BaselearnerFactory>>(factory_id, sh_ptr_blearner_factory));
  } else {
    _factory_map[ factory_id ] = sh_ptr_blearner_factory;
  }
}

void BaselearnerFactoryList::rmBaselearnerFactory (const std::string factory_id)
{
  // Create iterator and check if learner is registered:
  std::map<std::string, std::shared_ptr<blearnerfactory::BaselearnerFactory>>::iterator it = _factory_map.find(factory_id);

  if (it == _factory_map.end()) {
    throw std::out_of_range(factory_id + " is not registered in map");
  } else {
    _factory_map.erase(factory_id);
  }
}

void BaselearnerFactoryList::printRegisteredFactories () const
{
  // Check if any factory is registered:
  if (_factory_map.size() >= 1) {
    Rcpp::Rcout << "Registered base-learner:\n";
  } else {
    Rcpp::Rcout << "No registered base-learner.";
  }

  // Iterate over all registered factories and print the factory identifier:
  for (auto& it : _factory_map) {
    Rcpp::Rcout << "\t- " << it.first << std::endl;
  }
}

blearner_factory_map BaselearnerFactoryList::getFactoryMap () const
{
  return _factory_map;
}

void BaselearnerFactoryList::clearMap ()
{
  // Just delete the pointer, so we have a new empty map. The factories which
  // are behind the pointers should delete themself:
  _factory_map.clear();
}

std::pair<std::vector<std::string>, arma::mat> BaselearnerFactoryList::getModelFrame () const
{
  arma::mat                out_matrix;
  std::vector<std::string> rownames;

  for (auto& it : _factory_map) {

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

std::vector<std::string> BaselearnerFactoryList::getRegisteredFactoryNames () const
{
  std::vector<std::string> factory_names;

  for (auto& it : _factory_map) {
    factory_names.push_back(it.first);
  }
  return factory_names;
}

std::vector<std::string> BaselearnerFactoryList::getDataNames () const
{
  std::vector<std::string> dnames;
  std::string fname;
  for (auto& it : _factory_map) {
    dnames.push_back(it.second->getDataIdentifier());
  }
  return dnames;

}

} // namespace blearnerlist
