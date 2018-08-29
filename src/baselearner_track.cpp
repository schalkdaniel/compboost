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

#include "baselearner_track.h"

namespace blearnertrack
{

// Just an empty constructor:
BaselearnerTrack::BaselearnerTrack () {}

BaselearnerTrack::BaselearnerTrack (double learning_rate) : learning_rate ( learning_rate ) {}

// Insert a baselearner to the vector. We also want to add up the parameter
// in there to get an estimator in the end:
void BaselearnerTrack::insertBaselearner (blearner::Baselearner* blearner)
{
  // Insert new baselearner:
  blearner_vector.push_back(blearner);
  
  std::string insert_id = blearner->getDataIdentifier() + "_" + blearner->getBaselearnerType();
  // std::cout << "Insert base-learner with base-learner id: " << blearner->getBaselearnerType() << std::endl;;

  // Check if the baselearner is the first one. If so, the parameter
  // has to be instantiated with a zero matrix:
  std::map<std::string, arma::mat>::iterator it = my_parameter_map.find(insert_id);
  
  // Prune parameter by multiplying it with the learning rate:
  arma::mat parameter_temp = learning_rate * blearner->getParameter();
  
  // Check if this is the first parameter entry:
  if (it == my_parameter_map.end()) {
    
    // If this is the first entry, initialize it with zeros:
    arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
    my_parameter_map.insert(std::pair<std::string, arma::mat>(insert_id, init_parameter));

  }
  
  // Accumulating parameter. If there is a nan, then this will be ignored and 
  // the non  nan entries are added up:
  // arma::mat parameter_insert = parameter_temp + my_parameter_map.find(blearner->getBaselearnerType())->second;
  // my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner->getBaselearnerType(), parameter_insert));
  my_parameter_map[ insert_id ] = parameter_temp + my_parameter_map.find(insert_id)->second;
  
}

// Get the vector of baselearner:
std::vector<blearner::Baselearner*> BaselearnerTrack::getBaselearnerVector () const
{
  return blearner_vector;
}

// Get parameter map:
std::map<std::string, arma::mat> BaselearnerTrack::getParameterMap () const
{
  return my_parameter_map;
}

// Clear baselearner vector:
void BaselearnerTrack::clearBaselearnerVector ()
{
  for (unsigned int i = 0; i < blearner_vector.size(); i++)
  {
    delete blearner_vector[i];
  } 
  blearner_vector.clear();
}

// Get estimated parameter for specific iteration:
std::map<std::string, arma::mat> BaselearnerTrack::getEstimatedParameterOfIteration (const unsigned int& k) const
{
  if (k > blearner_vector.size()) {
    Rcpp::stop ("You can't get parameter of a state higher then the maximal iterations.");
  }
  
  // Create new parameter map:
  std::map<std::string, arma::mat> my_new_parameter_map;
  
  if (k <= blearner_vector.size()) {
    
    for (unsigned int i = 0; i < k; i++) {
      std::string insert_id = blearner_vector[i]->getDataIdentifier() + "_" + blearner_vector[i]->getBaselearnerType();
      
      // Check if the baselearner is the first one. If so, the parameter
      // has to be instantiated with a zero matrix:
      std::map<std::string, arma::mat>::iterator it = my_new_parameter_map.find(insert_id);
      
      // Prune parameter by multiplying it with the learning rate:
      arma::mat parameter_temp = learning_rate * blearner_vector[i]->getParameter();
      
      // Check if this is the first parameter entry:
      if (it == my_new_parameter_map.end()) {
        
        // If this is the first entry, initialize it with zeros:
        arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
        my_new_parameter_map.insert(std::pair<std::string, arma::mat>(insert_id, init_parameter));
        
      }
      
      // Accumulating parameter. If there is a nan, then this will be ignored and 
      // the non  nan entries are added up:
      my_new_parameter_map[ insert_id ] = parameter_temp + my_new_parameter_map.find(insert_id)->second;
    }
  }
  return my_new_parameter_map;
}

// Create parameter matrix:
std::pair<std::vector<std::string>, arma::mat> BaselearnerTrack::getParameterMatrix () const
{
  // Instantiate list to iterate:
  std::map<std::string, arma::mat> my_new_parameter_map = my_parameter_map;
  
  unsigned int cols = 0;
  
  // Set all parameter to zero in new map:
  for (auto& it : my_new_parameter_map) {
    arma::mat init_parameter (it.second.n_rows, it.second.n_cols, arma::fill::zeros);
    my_new_parameter_map[ it.first ] = init_parameter;
    
    // Note that parameter are stored as col vectors but in the matrix we want
    // them as row vectors. Therefore we have to use rows to count the columns
    // of the paraemter matrix. 
    cols += it.second.n_rows;
  }

  // Initialize matrix:
  arma::mat parameters (blearner_vector.size(), cols, arma::fill::zeros);
    
  for (unsigned int i = 0; i < blearner_vector.size(); i++) {
    std::string insert_id = blearner_vector[i]->getDataIdentifier() + "_" + blearner_vector[i]->getBaselearnerType();

    // Prune parameter by multiplying it with the learning rate:
    arma::mat parameter_temp = learning_rate * blearner_vector[i]->getParameter();
    
    // Accumulating parameter. If there is a nan, then this will be ignored and 
    // the non  nan entries are added up:
    my_new_parameter_map[ insert_id ] = parameter_temp + my_new_parameter_map.find(insert_id)->second;
    
    arma::mat param_insert;
    
    // Join columns to one huge column vector:
    for (auto& it : my_new_parameter_map) {
      param_insert = arma::join_cols(param_insert, it.second);
    }
    // Insert this huge vector at row i, therefore transpose it:
    parameters.row(i) = param_insert.t();
  }
  std::pair<std::vector<std::string>, arma::mat> out_pair;
  
  // If a baselearner have more than one parameter, than we rename the parameter
  // with a corresponding number (Note: In my_new_parameter_map is a list 
  // containing the last state of the parameter, that means a map with an
  // identifier string and parameter matrix):
  for (auto& it : my_new_parameter_map) {
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

void BaselearnerTrack::setToIteration (const unsigned int& k)
{
  if (k > blearner_vector.size()) {
    Rcpp::stop ("You can't set the actual state to a higher state then the maximal iterations.");
  }
  
  my_parameter_map = getEstimatedParameterOfIteration(k);
}

// Destructor:
BaselearnerTrack::~BaselearnerTrack ()
{
  // Rcpp::Rcout << "Call BaselearnerTrack Destructor" << std::endl;
  clearBaselearnerVector();
}

} // blearnertrack
