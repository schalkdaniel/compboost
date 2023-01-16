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

std::shared_ptr<loss::Loss> jsonToLoss (const json& j)
{
  std::shared_ptr<loss::Loss> l;

  if (j["Class"] == "LossQuadratic") {
    l = std::make_shared<loss::LossQuadratic>(j);
  }
  if (j["Class"] == "LossAbsolute") {
    l = std::make_shared<loss::LossAbsolute>(j);
  }
  if (j["Class"] == "LossQuantile") {
    l = std::make_shared<loss::LossQuantile>(j);
  }
  if (j["Class"] == "LossHuber") {
    l = std::make_shared<loss::LossHuber>(j);
  }
  if (j["Class"] == "LossBinomial") {
    l = std::make_shared<loss::LossBinomial>(j);
  }
  if (l == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return l;
}

std::shared_ptr<data::Data> jsonToData (const json& j)
{
  std::shared_ptr<data::Data> d;

  if (j["Class"] == "InMemoryData") {
    d = std::make_shared<data::InMemoryData>(j);
  }
  if (j["Class"] == "BinnedData") {
    d = std::make_shared<data::BinnedData>(j);
  }
  if (j["Class"] == "CategoricalDataRaw") {
    d = std::make_shared<data::CategoricalDataRaw>(j);
  }
  if (d == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return d;
}



} // namespace cloader
