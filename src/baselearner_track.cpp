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
//   Implementation of "BaselearnerTrack".
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
// =========================================================================== #

#include "baselearner_track.h"

namespace blearnertrack
{

// Just an empty constructor:
BaselearnerTrack::BaselearnerTrack () {};

// Insert a baselearner to the vector. We also want to add up the parameter
// in there to get an estimator in the end:
void BaselearnerTrack::InsertBaselearner (blearner::Baselearner* blearner,
  double learning_rate)
{
  // Insert new baselearner:
  blearner_vector.push_back(blearner);
  
  std::cout << "Insert new baselearner" << std::endl;
  
  // Check if the baselearner is the first one. If so, the parameter
  // has to be instantiated with a zero matrix:
  std::map<std::string, arma::mat>::iterator it = my_parameter_map.find(blearner->GetBaselearnerType());
  
  std::cout << "Check if this was the first baselearner!" << std::endl;
  
  // Prune parameter by multiplying it with the learning rate:
  arma::mat parameter_temp = learning_rate * blearner->GetParameter();
  
  std::cout << "Parameter dim: rows = " << parameter_temp.n_rows << " cols = " << parameter_temp.n_cols << std::endl;
  
  if (it == my_parameter_map.end()) {
    
    // If this is the first entry, initialize it with zeros:
    arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
    my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner->GetBaselearnerType(), init_parameter));
    
    std::cout << "The answer is YES!" << std::endl;
  }
  
  // Accumulating parameter. If there is a nan, then this will be ignored and 
  // the non  nan entries are added up:
  arma::mat parameter_insert = parameter_temp + my_parameter_map.find(blearner->GetBaselearnerType())->second;
  my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner->GetBaselearnerType(), parameter_insert));
  
  std::cout << "Know I have insert the new accumulated parameter:" << std::endl;
  for (unsigned int i = 0; i < parameter_insert.size(); i++) {
    std::cout << parameter_insert[i] << " ";
  }
  std::cout << std::endl;
  
}

// Get the vector of baselearner:
std::vector<blearner::Baselearner*> BaselearnerTrack::GetBaselearnerVector ()
{
  return blearner_vector;
}

} // blearnertrack