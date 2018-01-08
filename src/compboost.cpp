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
// ========================================================================== //



// THIS ONE IS UNDER PROGRESS!


#include "compboost.h"

namespace cboost {

// --------------------------------------------------------------------------- #
// Constructor:
// --------------------------------------------------------------------------- #

// todo: response as call by reference!

Compboost::Compboost () {}

Compboost::Compboost (arma::vec response, double learning_rate, 
  bool use_global_stop_criteria, optimizer::Optimizer* used_optimizer, 
  loss::Loss* used_loss, loggerlist::LoggerList used_logger,
  blearnerlist::BaselearnerList used_baselearner_list)
  : response ( response ), 
    learning_rate ( learning_rate ),
    use_global_stop_criteria ( use_global_stop_criteria ),
    used_optimizer ( used_optimizer ),
    used_loss ( used_loss ),
    used_baselearner_list ( used_baselearner_list ),
    used_logger ( used_logger )

{
  // Declare the vector of selected baselearner:
  blearner_track = blearnertrack::BaselearnerTrack();
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::TrainCompboost ()
{
  // Initialize zero model and pseudo residuals:
  initialization = used_loss->ConstantInitializer(response);
  arma::vec pseudo_residuals_init (response.size());
  // std::cout << "<<Compboost>> Initialize zero model and pseudo residuals" << std::endl;
  
  // Initialize prediction and fill with zero model:
  arma::vec prediction(response.size());
  prediction.fill(initialization);
  // std::cout << "<<Compboost>> Initialize prediction and fill with zero model" << std::endl;
  
  // Declare variables to stop the algorithm:
  bool stop_the_algorithm = false;
  unsigned int k = 1;
  
  // Main Algorithm. While the stop criteria isn't fullfilled, run the 
  // algorithm:
  while (! stop_the_algorithm) {
    
    // Define pseudo residuals as negative gradient:
    pseudo_residuals = -used_loss->DefinedGradient(response, prediction);
    // std::cout << "\n<<Compboost>> Define pseudo residuals as negative gradient" << std::endl;
    
    // Cast integer k to string for baselearner identifier:
    std::string temp_string = std::to_string(k);
    blearner::Baselearner* selected_blearner = used_optimizer->FindBestBaselearner(temp_string, pseudo_residuals, used_baselearner_list.GetMap());
    // std::cout << "<<Compboost>> Cast integer k to string for baselearner identifier" << std::endl;
    
    // Insert new baselearner to vector of selected baselearner:    
    blearner_track.InsertBaselearner(selected_blearner, learning_rate);
    // std::cout << "<<Compboost>> Insert new baselearner to vector of selected baselearner" << std::endl;
    
    // Update model (prediction) and shrink by learning rate:
    prediction += learning_rate * selected_blearner->predict();
    // std::cout << "<<Compboost>> Update model (prediction) and shrink by learning rate" << std::endl;
    
    // Log the current step:
    
    // The last term has to be the prediction or anything like that. This is
    // important to track the risk (inbag or oob)!!!!
    std::chrono::system_clock::time_point current_time;
    current_time = std::chrono::high_resolution_clock::now();
    
    used_logger.LogCurrent(k, current_time, 6);
    // std::cout << "<<Compboost>> Log the current step" << std::endl;
    
    // Get status of the algorithm (is stopping criteria reached):
    stop_the_algorithm = ! used_logger.GetStopperStatus(use_global_stop_criteria);
    
    // Increment k:
    k += 1;
  }
  
  // Set model prediction:
  model_prediction = prediction;
}

arma::vec Compboost::GetPrediction ()
{
  return model_prediction;
}

std::map<std::string, arma::mat> Compboost::GetParameter ()
{
  return blearner_track.GetParameterMap();
}

std::vector<std::string> Compboost::GetSelectedBaselearner ()
{
  std::vector<std::string> selected_blearner;
  
  // Issue: https://github.com/schalkdaniel/compboost/issues/62
  
  // Doesn't work:
  // for (std::vector<blearner::Baselearner*>::iterator it = blearner_track->GetBaselearnerVector().begin(); it != blearner_track->GetBaselearnerVector().end(); ++it) {
  //   selected_blearner.push_back((*it)->GetBaselearnerType());
  // }
  
  // Does work:
  for (unsigned int i = 0; i < blearner_track.GetBaselearnerVector().size(); i++) {
    selected_blearner.push_back(blearner_track.GetBaselearnerVector()[i]->GetBaselearnerType());
  }
  return selected_blearner;
}

std::pair<std::vector<std::string>, arma::mat> Compboost::GetModelFrame ()
{
  arma::mat out_matrix;
  std::vector<std::string> rownames;
  
  // Issue: https://github.com/schalkdaniel/compboost/issues/62
  
  // Doesn't work:
  for (blearner_factory_map::iterator it = used_baselearner_list.GetMap().begin(); it != used_baselearner_list.GetMap().end(); ++it) {
    arma::mat data_temp = it->second->GetData();
    out_matrix = arma::join_rows(out_matrix, data_temp);
    
    if (data_temp.n_cols > 1) {
      for (unsigned int i = 0; i < data_temp.n_cols; i++) {
        rownames.push_back(it->first + std::to_string(i + 1));
      }
    } else {
      rownames.push_back(it->first);
    }
  }
  return std::pair<std::vector<std::string>, arma::mat>(rownames, out_matrix);
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

// Destructor:
Compboost::~Compboost ()
{
  std::cout << "Call Compboost Destructor" << std::endl;
  delete used_optimizer;
  delete used_loss;
}

} // namespace cboost