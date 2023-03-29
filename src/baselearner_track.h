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
// it under the terms of the LGPL-3 License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// LGPL-3 License for more details. You should have received a copy of
// the license along with compboost.
//
// =========================================================================== #

#ifndef BASELEARNERTACK_H_
#define BASELEARNERTACK_H_

#include "baselearner.h"
#include "baselearner_factory_list.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

namespace blearnertrack
{

typedef std::shared_ptr<data::Data> sdata;
typedef std::shared_ptr<data::BinnedData> sbindata;
typedef std::map<std::string, sdata> mdata;

// FOR SAVING AND LOADING JSON:
json blVecToJson  (const std::vector<std::shared_ptr<blearner::Baselearner>>&);
std::vector<std::shared_ptr<blearner::Baselearner>> jsonToBlVec (const json&, const mdata&);

class BaselearnerTrack
{
private:
  double                                               _learning_rate = 1;
  std::vector<std::shared_ptr<blearner::Baselearner>>  _blearner_vector;
  std::map<std::string, arma::mat>                     _parameter_map;
  std::vector<double>                                  _step_sizes;

public:
  BaselearnerTrack ();
  BaselearnerTrack (const double);
  BaselearnerTrack (const json&, const mdata&);

  // Getter/Setter
  std::vector<std::shared_ptr<blearner::Baselearner>>  getBaselearnerVector             () const;
  std::map<std::string, arma::mat>                     getParameterMap                  () const;
  std::pair<std::vector<std::string>, arma::mat>       getParameterMatrix               () const;
  std::map<std::string, arma::mat>                     getEstimatedParameterOfIteration (const unsigned int&) const;

  void setParameterMap (std::map<std::string, arma::mat>);

  // Other member functions
  void insertBaselearner      (std::shared_ptr<blearner::Baselearner>, const double& step_size);
  void clearBaselearnerVector ();
  void setToIteration         (const unsigned int&);
  json toJson                 () const;

  // Destructor:
  ~BaselearnerTrack ();
};

} // namespace blearnertrack

#endif // BASELEARNERTRACK_H_
