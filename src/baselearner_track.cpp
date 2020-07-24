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

#include "baselearner_track.h"

namespace blearnertrack
{

BaselearnerTrack::BaselearnerTrack () { };
BaselearnerTrack::BaselearnerTrack (double learning_rate) : _learning_rate ( learning_rate ) { }

std::vector<std::shared_ptr<blearner::Baselearner>> BaselearnerTrack::getBaselearnerVector () const
{
  return _blearner_vector;
}

std::map<std::string, arma::mat> BaselearnerTrack::getParameterMap () const
{
  return _parameter_map;
}

std::map<std::string, arma::mat> BaselearnerTrack::getEstimatedParameterOfIteration (const unsigned int& k) const
{
  if (k > _blearner_vector.size()) {
    Rcpp::stop ("You can't get parameter of a state higher then the maximal iterations.");
  }

  // Create new parameter map:
  std::map<std::string, arma::mat> new_parameter_map;

  if (k <= _blearner_vector.size()) {

    for (unsigned int i = 0; i < k; i++) {
      std::string insert_id = _blearner_vector[i]->getDataIdentifier() + "_" + _blearner_vector[i]->getBaselearnerType();

      // Check if the baselearner is the first one. If so, the parameter
      // has to be instantiated with a zero matrix:
      std::map<std::string, arma::mat>::iterator it = new_parameter_map.find(insert_id);

      // Prune parameter by multiplying it with the learning rate:
      arma::mat parameter_temp = _learning_rate * _step_sizes[i] * _blearner_vector[i]->getParameter();

      // Check if this is the first parameter entry:
      if (it == new_parameter_map.end()) {
        // If this is the first entry, initialize it with zeros:
        arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
        new_parameter_map.insert(std::pair<std::string, arma::mat>(insert_id, init_parameter));
      }
      // Accumulating parameter. If there is a nan, then this will be ignored and
      // the non  nan entries are summed up:
      new_parameter_map[ insert_id ] = parameter_temp + new_parameter_map.find(insert_id)->second;
    }
  }
  return new_parameter_map;
}

std::pair<std::vector<std::string>, arma::mat> BaselearnerTrack::getParameterMatrix () const
{
  auto         new_parameter_map = _parameter_map;
  unsigned int cols = 0;

  // Set all parameter to zero in new map:
  for (auto& it : new_parameter_map) {
    arma::mat init_parameter (it.second.n_rows, it.second.n_cols, arma::fill::zeros);
    new_parameter_map[ it.first ] = init_parameter;

    // Note that parameter are stored as col vectors but in the matrix we want
    // them as row vectors. Therefore we have to use rows to count the columns
    // of the paraemter matrix.
    cols += it.second.n_rows;
  }

  // Initialize matrix:
  arma::mat parameters (_blearner_vector.size(), cols, arma::fill::zeros);

  for (unsigned int i = 0; i < _blearner_vector.size(); i++) {
    std::string insert_id = _blearner_vector[i]->getDataIdentifier() + "_" + _blearner_vector[i]->getBaselearnerType();

    // Prune parameter by multiplying it with the learning rate:
    arma::mat parameter_temp = _learning_rate * _blearner_vector[i]->getParameter();

    // Accumulating parameter. If there is a nan, then this will be ignored and
    // the non  nan entries are summed up:
    new_parameter_map[ insert_id ] = parameter_temp + new_parameter_map.find(insert_id)->second;

    arma::mat param_insert;

    // Join columns to one huge column vector:
    for (auto& it : new_parameter_map) {
      param_insert = arma::join_cols(param_insert, it.second);
    }
    // Insert this huge vector at row i, therefore transpose it:
    parameters.row(i) = param_insert.t();
  }
  std::pair<std::vector<std::string>, arma::mat> out_pair;

  // If a base-learner has more than one parameter, than we rename the parameter
  // with a corresponding number (Note: In new_parameter_map is a list
  // containing the last state of the parameter, that means a map with an
  // identifier string and parameter matrix):
  for (auto& it : new_parameter_map) {
    if (it.second.n_rows > 1) {
      for (unsigned int i = 0; i < it.second.n_rows; i++) {
        out_pair.first.push_back(it.first + "_x" + std::to_string(i + 1));
      }
    } else {
      out_pair.first.push_back(it.first);
    }
  }
  out_pair.second = parameters;

  return out_pair;
}

void BaselearnerTrack::setParameterMap (std::map<std::string, arma::mat> new_parameter_map)
{
  _parameter_map = new_parameter_map;
}


void BaselearnerTrack::insertBaselearner (std::shared_ptr<blearner::Baselearner> sh_ptr_blearner, const double& step_size)
{
  _blearner_vector.push_back(sh_ptr_blearner);
  _step_sizes.push_back(step_size);

  std::string insert_id = sh_ptr_blearner->getDataIdentifier() + "_" + sh_ptr_blearner->getBaselearnerType();

  // Check if the baselearner is the first one. If so, the parameter
  // has to be instantiated with a zero matrix:
  std::map<std::string, arma::mat>::iterator it = _parameter_map.find(insert_id);

  // Prune parameter by multiplying it with the learning rate:
  arma::mat parameter_temp = _learning_rate * step_size * sh_ptr_blearner->getParameter();

  // Check if this is the first parameter entry:
  if (it == _parameter_map.end()) {
    // If this is the first entry, initialize it with zeros:
    arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
    _parameter_map.insert(std::pair<std::string, arma::mat>(insert_id, init_parameter));

  }
  // Accumulating parameter. If there is a nan, then this will be ignored and
  // the non nan entries are summed up:
  _parameter_map[ insert_id ] = parameter_temp + _parameter_map.find(insert_id)->second;
}

void BaselearnerTrack::clearBaselearnerVector ()
{
  _blearner_vector.clear();
}

void BaselearnerTrack::setToIteration (const unsigned int& k)
{
  if (k > _blearner_vector.size()) {
    Rcpp::stop ("You can't set the crrent iteration higher then the maximal trained iterations.");
  }
  _parameter_map = getEstimatedParameterOfIteration(k);
}

BaselearnerTrack::~BaselearnerTrack ()
{
  clearBaselearnerVector();
}

} // blearnertrack
