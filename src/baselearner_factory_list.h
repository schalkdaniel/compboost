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

#ifndef BASELEARNERLIST_H_
#define BASELEARNERLIST_H_

#include <map>

#include "baselearner_factory.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

typedef std::map<std::string, std::shared_ptr<blearnerfactory::BaselearnerFactory>> blearner_factory_map;

namespace blearnerlist
{

class BaselearnerFactoryList
{
private:
  blearner_factory_map _factory_map;

public:
  BaselearnerFactoryList ();

  // Getter/Setter
  blearner_factory_map                            getFactoryMap ()             const;
  std::pair<std::vector<std::string>, arma::mat>  getModelFrame ()             const;
  std::vector<std::string>                        getRegisteredFactoryNames () const;
  std::vector<std::string>                        getDataNames ()              const;

  // Other member functions
  void registerBaselearnerFactory (const std::string, const std::shared_ptr<blearnerfactory::BaselearnerFactory>);
  void rmBaselearnerFactory       (const std::string);
  void printRegisteredFactories   () const;
  void clearMap                   ();

  json factoryDataToJson (const bool = false) const;
};

} // namespace blearnerlist

#endif // BASELEARNERLIST_H_

