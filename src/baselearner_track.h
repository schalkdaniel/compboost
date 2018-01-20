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
//   Implementation of tracking all baselearners within the main algorithm.
//   This is more convenient, multifunctional and (most important) it keeps
//   the main algorithm cleaner.
//
//   The idea is, that every returned baselearner from the optimizer ìs 
//   registered in a vector of baselearner (the track). Later on we can
//   predict by using this vector.
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

namespace blearnertrack
{

class BaselearnerTrack
{
  private:
    
    // Vector of selected baselearner:
    std::vector<blearner::Baselearner*> blearner_vector;
    
    // Parameter map. The first element contains the baselearner type and the
    // second element the parameter. This one will be updated in every
    // iteration:
    std::map<std::string, arma::mat> my_parameter_map;
    
    double learning_rate;
    
  public: 
    
    BaselearnerTrack ();
    BaselearnerTrack (double);
    
    // Insert a baselearner into vector and update parameter:
    void InsertBaselearner (blearner::Baselearner*);
    
    // Return the vector of baselearner:
    std::vector<blearner::Baselearner*> GetBaselearnerVector ();
    
    // Return so far estimated parameter map:
    std::map<std::string, arma::mat> GetParameterMap ();
    
    // Clear the vector without deleting the data in the factory:
    void ClearBaselearnerVector ();
    
    // Estimate parameter for specific iteration:
    std::map<std::string, arma::mat> GetEstimatedParameterForIteration (unsigned int);
    
    // Returns a matrix of parameters for every iteration:
    std::pair<std::vector<std::string>, arma::mat> GetParameterMatrix ();
    
    // Destructor:
    ~BaselearnerTrack ();
};

} // namespace blearnertrack

#endif // BASELEARNERTRACK_H_