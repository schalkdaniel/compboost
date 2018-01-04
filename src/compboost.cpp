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
//   Constructors and member function implementations for the class
//   "Compboost".
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

#include "compboost.h"

#include <iostream>

namespace cboost {

// --------------------------------------------------------------------------- #
// Constructors:
// --------------------------------------------------------------------------- #

Compboost::Compboost (arma::vec& response, optimizer::Optimizer& used_optimizer, 
  loss::Loss& used_loss, bool use_global_stop_criteria)
  : response ( & response ),
    used_optimizer ( & used_optimizer ),
    used_loss ( & used_loss ),
    use_global_stop_criteria ( use_global_stop_criteria ) {}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::TrainCompboost ()
{
  // Initialize pseudo residuals and the zero model:
  initialization = used_loss->ConstantInitializer(*response);
  arma::vec pseudo_residuals_init (response->size());
  pseudo_residuals_init.fill(initialization);
    
  pseudo_residuals = used_loss->DefinedGradient(*response, pseudo_residuals_init);
  
  bool stop_the_algorithm = false;
  unsigned int k = 1;
  
  while (! stop_the_algorithm) {
    
    std::string temp_string = std::to_string(k);
    blearner::Baselearner* selected_blearner = used_optimizer->FindBestBaselearner(temp_string, pseudo_residuals);
    
    blearner_track->InsertBaselearner(*selected_blearner);
    
    // This has to be done better!!!
    //
    // For instance, calculating the prediction in each step and just add the
    // new prediction for the baselearner on top!
    arma::vec predict_temp = PredictEnsemble();
    
    pseudo_residuals = used_loss->DefinedGradient(*response, predict_temp);
    
    // The last term has to be the prediction or anything like that. This is
    // important to track the risk (inbag or oob)!!!!
    
    std::chrono::system_clock::time_point current_time;
    current_time = std::chrono::high_resolution_clock::now();
    
    used_logger->LogCurrent(k, current_time, 6);
    
    stop_the_algorithm = used_logger->GetStopperStatus(use_global_stop_criteria);
    k += 1;
  }
}

arma::vec Compboost::PredictEnsemble ()
{
  arma::vec prediction(response->size());
  prediction.fill(initialization);
  
  for (std::vector<blearner::Baselearner>::iterator it = blearner_track->GetBaselearnerVector().begin(); it != blearner_track->GetBaselearnerVector().end(); ++it) {
    prediction += it->predict();
  }
  return prediction;
}

} // namespace cboost