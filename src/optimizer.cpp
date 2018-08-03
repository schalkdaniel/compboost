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

#include "optimizer.h"

namespace optimizer {

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

// Destructor:
Optimizer::~Optimizer () {
  // Rcpp::Rcout << "Call Optimizer Destructor" << std::endl;
}

// -------------------------------------------------------------------------- //
// Optimizer implementations:
// -------------------------------------------------------------------------- //

// OptimizerCoordinateDescent:
// -----------------------

OptimizerCoordinateDescent::OptimizerCoordinateDescent () {}

blearner::Baselearner* OptimizerCoordinateDescent::findBestBaselearner (const std::string& iteration_id, 
  const arma::vec& pseudo_residuals, const blearner_factory_map& my_blearner_factory_map) const
{
  double ssq_temp;
  double ssq_best = std::numeric_limits<double>::infinity();
  
  blearner::Baselearner* blearner_temp;
  blearner::Baselearner* blearner_best;
  
  for (auto& it : my_blearner_factory_map) {

    // Paste string identifier for new base-learner:
    std::string id = "(" + iteration_id + ") " + it.second->getBaselearnerType();
    
    // Create new baselearner aut of the actual factory (just the 
    // pointer is overwritten):
    blearner_temp = it.second->createBaselearner(id);
    
    // Train that base learner on the pseudo residuals:
    blearner_temp->train(pseudo_residuals);
    
    // Calculate SSE:
    ssq_temp = arma::mean(arma::pow(pseudo_residuals - blearner_temp->predict(), 2));
    
    // Check if SSE of new temporary baselearner is smaller then SSE of the best
    // baselearner. If so, assign the temporary base-learner with the best 
    // base-learner (This is always triggered within the first iteration since
    // ssq_best is declared as infinity):
    if (ssq_temp < ssq_best) {
      ssq_best = ssq_temp;   
      // Deep copy since the temporary baselearner is deleted every time which
      // will also deletes the data for the best base-learner if we don't copy
      // the whole data of the object:
      blearner_best = blearner_temp->clone();
    }
    
    // Completely remove the temporary base-learner. This one isn't needed anymore:
    delete blearner_temp;
  }
  // Remove pointer of the temporary base-learner.
  blearner_temp = NULL;
  
  return blearner_best;
}

} // namespace optimizer
