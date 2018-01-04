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

void BaselearnerTrack::InsertBaselearner (blearner::Baselearner& blearner)
{
  blearner_vector.push_back(blearner);
  blearner_type_vector.push_back(blearner.GetBaselearnerType());
  
  // Check if the baselearner is the first one. If so, the parameter
  // has to be instantiated with a zero matrix:
  std::map<std::string, arma::mat>::iterator it = my_parameter_map.find(blearner.GetBaselearnerType());
  
  arma::mat parameter_temp = blearner.GetParameter();
  
  if (it != my_parameter_map.end()) {
    my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner.GetBaselearnerType(), parameter_temp * 0));
  }
  
  // Accumulating parameter. If there is a nan, then this will be ignored and 
  // the non  nan entries are added up:
  arma::mat parameter_insert = parameter_temp + my_parameter_map.find((blearner.GetBaselearnerType()))->second;
  my_parameter_map.insert(std::pair<std::string, arma::mat>(blearner.GetBaselearnerType(), parameter_temp * 0));
  
}

std::vector<blearner::Baselearner> BaselearnerTrack::GetBaselearnerVector ()
{
  return blearner_vector;
}

} // blearnertrack