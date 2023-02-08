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

#include "saver.h"

namespace saver {

json jsonLoader (const std::string file)
{
  std::ifstream is(file);
  json j;
  is >> j;

  return j;
}

json armaMatToJson (const arma::mat& X)
{
  std::stringstream ssout;
  X.save(ssout, arma::arma_ascii);

  // TO LOAD:
  //arma::sp_mat z;
  //std::stringstream ssin;
  //ssin << ystr;
  //z.load(ssin, arma::arma_binary);

  json j = {
    {"type", "arma::mat"},
    {"mat", ssout.str()}
  };

  return j;
}

json armaSpMatToJson (const arma::sp_mat& X)
{
  std::stringstream ssout;
  //X.save(ssout, arma::arma_binary);
  X.save(ssout, arma::coord_ascii);

  // TO LOAD:
  //arma::sp_mat z;
  //std::stringstream ssin;
  //ssin << ystr;
  //z.load(ssin, arma::arma_binary);

  json j = {
    {"type", "arma::sp_mat"},
    {"mat", ssout.str()}
  };

  return j;
}

json armaUvecToJson (const arma::uvec& u)
{
  std::stringstream ssout;
  u.save(ssout, arma::arma_ascii);

  json j = {
    {"type", "arma::uvec"},
    {"mat", ssout.str()}
  };

  return j;
}

void checkMatInJson (const json& j, const std::string type)
{
  if (j.find("mat") == j.end()) {
    throw std::out_of_range("No element 'mat' in the JSON object");
  }
  if (j.find("type") != j.end()) {
    if (j["type"] != type) {
      std::string jt = j["type"];
      std::string msg = "Wrong mat type " + jt;
      throw std::logic_error(msg);
    }
  } else {
    throw std::out_of_range("No element 'type' in the JSON object");
  }
}

arma::mat jsonToArmaMat (const json& j)
{
  checkMatInJson(j, "arma::mat");

  std::string mat_in = j["mat"];
  std::istringstream mat_stream(mat_in);
  arma::mat out;
  out.load(mat_stream, arma::arma_ascii);

  return out;
}


arma::sp_mat jsonToArmaSpMat (const json& j)
{
  checkMatInJson(j, "arma::sp_mat");

  std::string mat_in = j["mat"];
  std::istringstream mat_stream(mat_in);
  arma::sp_mat out;
  out.load(mat_stream, arma::coord_ascii);

  return out;
}

arma::uvec jsonToArmaUvec (const json& j)
{
  checkMatInJson(j, "arma::uvec");

  std::string vin = j["mat"];
  std::istringstream vstream(vin);
  arma::uvec out;
  out.load(vstream, arma::arma_ascii);

  return out;
}

json mapMatToJson (const std::map<std::string, arma::mat>& mmap)
{
  json j;
  for (auto& it : mmap) {
    j[it.first] = armaMatToJson(it.second);
  }
  return j;
}

std::map<std::string, arma::mat> jsonToMapMat (const json& j)
{
  std::map<std::string, arma::mat> omap;
  for (auto& it : j.items()) {
    omap[it.key()] = jsonToArmaMat(it.value());
  }
  return omap;
}

} // namespace saver

