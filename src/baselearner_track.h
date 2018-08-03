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

#ifndef BASELEARNERTACK_H_
#define BASELEARNERTACK_H_

#include "baselearner.h"
#include "baselearner_factory_list.h"

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
    void insertBaselearner (blearner::Baselearner*);
    
    // Return the vector of baselearner:
    std::vector<blearner::Baselearner*> getBaselearnerVector () const;
    
    // Return so far estimated parameter map:
    std::map<std::string, arma::mat> getParameterMap () const;
    
    // Clear the vector of baselearner:
    void clearBaselearnerVector ();
    
    // Estimate parameter for specific iteration:
    std::map<std::string, arma::mat> getEstimatedParameterOfIteration (const unsigned int&) const;
    
    // Returns a matrix of parameters for every iteration:
    std::pair<std::vector<std::string>, arma::mat> getParameterMatrix () const;
    
    // Set parameter map to a given iteration:
    void setToIteration (const unsigned int&);
    
    // Destructor:
    ~BaselearnerTrack ();
};

} // namespace blearnertrack

#endif // BASELEARNERTRACK_H_
