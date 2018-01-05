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
//   Implementation of the "Compboost" class.
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



// THIS ONE IS UNDER PROGRESS!


#include "compboost.h"

#include <iostream>

namespace cboost {

// --------------------------------------------------------------------------- #
// Constructor:
// --------------------------------------------------------------------------- #

// todo: response as call by reference!

Compboost::Compboost (arma::vec response, double learning_rate, 
  bool use_global_stop_criteria, optimizer::Optimizer* used_optimizer, 
  loss::Loss* used_loss, loggerlist::LoggerList* used_logger)
  : response ( response ), 
    learning_rate ( learning_rate ),
    use_global_stop_criteria ( use_global_stop_criteria ),
    used_optimizer ( used_optimizer ),
    used_loss ( used_loss ),
    used_logger ( used_logger )
{
  // Declare the vector of selected baselearner:
  blearner_track = new blearnertrack::BaselearnerTrack();
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::TrainCompboost ()
{
  // Initialize pseudo residuals and the zero model:
  initialization = used_loss->ConstantInitializer(response);
  std::cout << "Instantiate constant initializer" << std::endl;
  
  arma::vec pseudo_residuals_init (response.size());
  pseudo_residuals_init.fill(initialization);
  pseudo_residuals = used_loss->DefinedGradient(response, pseudo_residuals_init);
  std::cout << "First pseudo residuals are defined" << std::endl;
  
  arma::vec prediction(response.size());
  prediction.fill(initialization);
  
  std::cout << "The new prediction was done and is:" << std::endl;
  for (unsigned int i = 0; i < prediction.size(); i++) {
    std::cout << prediction[i] << " ";
  }
  std::cout << std::endl;
  
  bool stop_the_algorithm = false;
  unsigned int k = 1;
  
  // Main Algorithm. While the stop criteria isn't fullfilled, run the 
  // algorithm:
  while (! stop_the_algorithm) {
    
    std::cout << std::endl;
    std::cout << "--- " << k << "th iteration of the algorithm!" << std::endl;
    
    std::string temp_string = std::to_string(k);
    blearner::Baselearner* selected_blearner = used_optimizer->FindBestBaselearner(temp_string, pseudo_residuals);
    
    std::cout << "Select: " << selected_blearner->GetBaselearnerType() << selected_blearner->GetIdentifier() << std::endl;
    
    blearner_track->InsertBaselearner(selected_blearner, learning_rate);
    
    std::cout << "Insert best baselearner to list" << std::endl;
    
    // This has to be done better!!!
    //
    // For instance, calculating the prediction in each step and just add the
    // new prediction for the baselearner on top!
    // arma::vec predict_temp = PredictEnsemble();
    
    prediction += learning_rate * selected_blearner->predict();
    
    std::cout << "The new prediction was done and is:" << std::endl;
    for (unsigned int i = 0; i < prediction.size(); i++) {
      std::cout << prediction[i] << " ";
    }
    std::cout << std::endl;
    
    
    pseudo_residuals = -used_loss->DefinedGradient(response, prediction);
    
    std::cout << "The new pseudo residuals were calculated!" << std::endl;
    
    // The last term has to be the prediction or anything like that. This is
    // important to track the risk (inbag or oob)!!!!
    
    std::chrono::system_clock::time_point current_time;
    current_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "The current time was stored" << std::endl;
    
    used_logger->LogCurrent(k, current_time, 6);
    
    std::cout << "The iteration was logged" << std::endl;
    
    stop_the_algorithm = ! used_logger->GetStopperStatus(use_global_stop_criteria);
    
    std::cout << "The actual stopper status is: " << stop_the_algorithm << std::endl;
    k += 1;
  }
}

// arma::vec Compboost::PredictEnsemble ()
// {
//   arma::vec prediction(response.size());
//   prediction.fill(initialization);
//   
//   std::cout << "Initialize the b0 model with: " << initialization << std::endl;
//   std::cout << "Rows = " << prediction.n_rows << " Cols = " << prediction.n_cols << std::endl;
//   
//   for (std::vector<blearner::Baselearner*>::iterator it = blearner_track->GetBaselearnerVector().begin(); it != blearner_track->GetBaselearnerVector().end(); ++it) {
//     std::cout << "Now iterating over baselearner!" << std::endl;
//     prediction += (*it)->predict();
//   }
//   return prediction;
// }

} // namespace cboost