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

#include "class_loader.h"

namespace cloader {

std::shared_ptr<response::Response> jsonToResponse (const json& j)
{
  std::shared_ptr<response::Response> r;

  if (j["Class"] == "ResponseRegr") {
    r = std::make_shared<response::ResponseRegr>(j);
  }
  if (j["Class"] == "ResponseBinaryClassif") {
    r = std::make_shared<response::ResponseBinaryClassif>(j);
  }
  if (r == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return r;
}

} // namespace cloader
