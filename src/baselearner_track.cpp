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

BaselearnerTrack::BaselearnerTrack () {};

void BaselearnerTrack::InsertBaselearner (blearner::Baselearner* blearner,
  double learning_rate)
{
  // blearner::Baselearner* blearner_temp = blearner->Clone();
  blearner_vector.push_back(blearner);
  
  std::cout << "Insert new baselearner" << std::endl;
  
  blearner_type_vector.push_back(blearner->GetBaselearnerType());
  
  std::cout << "Insert new baselearner type" << std::endl;
  
  // Check if the baselearner is the first one. If so, the parameter
  // has to be instantiated with a zero matrix:
  std::map<std::string, arma::mat>::iterator it = my_parameter_map.find(blearner->GetBaselearnerType());
  
  std::cout << "Check if this was the first baselearner!" << std::endl;
  
  arma::mat parameter_temp = learning_rate * blearner->GetParameter();
  
  std::cout << "Parameter dim: rows = " << parameter_temp.n_rows << " cols = " << parameter_temp.n_cols << std::endl;
  
  if (it == my_parameter_map.end()) {
    
    arma::mat z(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
    
    // double erase = 0;
    // parameter_temp = parameter_temp * erase;
    
    std::cout << "Parameter dim: rows = " << z.n_rows << " cols = " << z.n_cols << std::endl;
    
    my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner->GetBaselearnerType(), z));
    
    std::cout << "The answer is YES!" << std::endl;
  }
  
  // Accumulating parameter. If there is a nan, then this will be ignored and 
  // the non  nan entries are added up:
  arma::mat parameter_insert = parameter_temp + my_parameter_map.find(blearner->GetBaselearnerType())->second;
  my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner->GetBaselearnerType(), parameter_insert));
  
  std::cout << "Know I have insert the new accumulated parameter" << std::endl;
  
}

std::vector<blearner::Baselearner*> BaselearnerTrack::GetBaselearnerVector ()
{
  return blearner_vector;
}

} // blearnertrack