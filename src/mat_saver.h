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

#ifndef MATSAVER_H_
#define MATSAVER_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <string>

#include <fstream>
#include <sstream>
#include "single_include/nlohmann/json.hpp"

using json = nlohmann::json;

//#include <functional>

namespace saver {

json jsonLoader     (const std::string);
void checkMatInJson (const json&, const std::string);

json armaMatToJson   (const arma::mat&);
json armaSpMatToJson (const arma::sp_mat&);
json mapMatToJson    (const std::map<std::string, arma::mat>&);

arma::mat    jsonToArmaMat   (const json&);
arma::sp_mat jsonToArmaSpMat (const json&);

std::map<std::string, arma::mat> jsonToMapMat (const json&);

} // namespace saver

#endif // MATSAVER_H_
