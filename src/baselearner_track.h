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

#ifndef BASELEARNERTACK_H_
#define BASELEARNERTACK_H_

#include "baselearner.h"
#include "baselearner_list.h"
#include "loggerlist.h"

namespace blearnertrack
{

class BaselearnerTrack
{
  private:
    
    // Vector of selected baselearner:
    std::vector<blearner::Baselearner*> blearner_vector;
    
    // Pointer to loggerlist:
    // loggerlist::LoggerList* blearner_logger_list;
    
    // // This vector contains when which baselearner type was selected:
    // std::vector<std::string> blearner_type_vector;
    
    // Parameter map.The first element contains the baselearner type and the
    // second element the parameter. This one will be updated in every
    // iteration:
    std::map<std::string, arma::mat> my_parameter_map;
    
  public: 
    
    BaselearnerTrack ();
    
    // ???
    // blearner::Baselearner GetBaselearnerNumber (unsigned int);
    
    // Insert a baselearner into vector and update parameter:
    void InsertBaselearner (blearner::Baselearner*, double);
    
    std::vector<blearner::Baselearner*> GetBaselearnerVector ();
    
    // 
    // arma::mat GetParameterEstimator ();
    // arma::mat GetParameterEstimator (unsigned int);
    // 
    // arma::mat GetParameterMatrix ();
};

} // namespace blearnertrack

#endif // BASELEARNERTRACK_H_