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

#include "compboost.h"

namespace cboost {

// --------------------------------------------------------------------------- #
// Constructor:
// --------------------------------------------------------------------------- #

// todo: response as call by reference!

Compboost::Compboost () {}

Compboost::Compboost (arma::vec response, double learning_rate, 
  bool stop_if_all_stopper_fulfilled, optimizer::Optimizer* used_optimizer, 
  loss::Loss* used_loss, loggerlist::LoggerList* used_logger,
  blearnerlist::BaselearnerList used_baselearner_list)
  : response ( response ), 
    learning_rate ( learning_rate ),
    stop_if_all_stopper_fulfilled ( stop_if_all_stopper_fulfilled ),
    used_optimizer ( used_optimizer ),
    used_loss ( used_loss ),
    used_baselearner_list ( used_baselearner_list ),
    used_logger ( used_logger ) 
{
  blearner_track = blearnertrack::BaselearnerTrack(learning_rate);
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::TrainCompboost (bool trace)
{
  // Make sure, that the selected baselearner and logger data is empty:
  blearner_track.ClearBaselearnerVector();
  used_logger->ClearLoggerData();
  
  
  // Initialize zero model and pseudo residuals:
  initialization = used_loss->ConstantInitializer(response);
  arma::vec pseudo_residuals_init (response.size());
  // std::cout << "<<Compboost>> Initialize zero model and pseudo residuals" << std::endl;
  
  // Initialize prediction and fill with zero model:
  arma::vec prediction(response.size());
  prediction.fill(initialization);
  // std::cout << "<<Compboost>> Initialize prediction and fill with zero model" << std::endl;
  
  // Initialize trace:
  if (trace) {
    std::cout << std::endl;
    used_logger->InitializeLoggerPrinter(); 
  }
  
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
    blearner_track.InsertBaselearner(selected_blearner);
    // std::cout << "<<Compboost>> Insert new baselearner to vector of selected baselearner" << std::endl;
    
    // Update model (prediction) and shrink by learning rate:
    prediction += learning_rate * selected_blearner->predict();
    // std::cout << "<<Compboost>> Update model (prediction) and shrink by learning rate" << std::endl;
    
    // Log the current step:
    
    // The last term has to be the prediction or anything like that. This is
    // important to track the risk (inbag or oob)!!!!
    
    used_logger->LogCurrent(k, response, prediction, selected_blearner, 
      initialization, learning_rate);
    // std::cout << "<<Compboost>> Log the current step" << std::endl;
    
    // Get status of the algorithm (is stopping criteria reached):
    stop_the_algorithm = ! used_logger->GetStopperStatus(stop_if_all_stopper_fulfilled);
    
    // Print trace:
    if (trace) {
      used_logger->PrintLoggerStatus(); 
    }
    
    // Increment k:
    k += 1;
  }
  
  if (trace) {
    std::cout << std::endl;
    std::cout << std::endl; 
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
    selected_blearner.push_back(blearner_track.GetBaselearnerVector()[i]->GetDataIdentifier() + ": " + blearner_track.GetBaselearnerVector()[i]->GetBaselearnerType());
  }
  return selected_blearner;
}

std::map<std::string, arma::mat> Compboost::GetParameterOfIteration (unsigned int k) 
{
  return blearner_track.GetEstimatedParameterForIteration(k);
}

std::pair<std::vector<std::string>, arma::mat> Compboost::GetParameterMatrix ()
{
  return blearner_track.GetParameterMatrix();
}

arma::vec Compboost::Predict (std::map<std::string, arma::mat> data_map)
{
  // std::cout << "Get into Compboost::Predict" << std::endl;
  
  std::map<std::string, arma::mat> parameter_map = blearner_track.GetParameterMap();

  arma::vec pred(data_map.begin()->second.n_rows);
  pred.fill(initialization);
  
  // std::cout << "initialize pred vec" << std::endl;
  
  for (auto& it : parameter_map) {
    
    std::string sel_factory = it.first;
    
    // std::cout << "Fatory id of parameter map: " << sel_factory << std::endl;
    
    blearnerfactory::BaselearnerFactory* sel_factory_obj = used_baselearner_list.GetMap().find(sel_factory)->second;
    
    // std::cout << "Data of selected factory: " << sel_factory_obj->GetDataIdentifier() << std::endl;
    
    arma::mat data_trafo = sel_factory_obj->InstantiateData((data_map.find(sel_factory_obj->GetDataIdentifier())->second));
    pred += data_trafo * it.second;
    
  }
  return pred;
}

void Compboost::SummarizeCompboost ()
{
  std::cout << "Compboost object with:" << std::endl;
  std::cout << "\t- Learning Rate: " << learning_rate << std::endl;
  std::cout << "\t- Are all logger used as stopper: " << stop_if_all_stopper_fulfilled << std::endl;
  
  if (blearner_track.GetBaselearnerVector().size() > 0) {
    std::cout << "\t- Model is already trained with " << blearner_track.GetBaselearnerVector().size() << " iterations/fitted baselearner" << std::endl;
  }
  std::cout << std::endl;
  std::cout << "To get more information check the other objects!" << std::endl;
}

arma::vec Compboost::PredictionOfIteration (std::map<std::string, arma::mat> data_map, unsigned int k)
{
  // std::cout << "Get into Compboost::Predict" << std::endl;
  
  std::map<std::string, arma::mat> parameter_map = blearner_track.GetEstimatedParameterForIteration(k);
  
  arma::vec pred(data_map.begin()->second.n_rows);
  pred.fill(initialization);
  
  // std::cout << "initialize pred vec" << std::endl;
  
  for (auto& it : parameter_map) {
    
    std::string sel_factory = it.first;
    
    // std::cout << "Fatory id of parameter map: " << sel_factory << std::endl;
    
    blearnerfactory::BaselearnerFactory* sel_factory_obj = used_baselearner_list.GetMap().find(sel_factory)->second;
    
    // std::cout << "Data of selected factory: " << sel_factory_obj->GetDataIdentifier() << std::endl;
    
    arma::mat data_trafo = sel_factory_obj->InstantiateData((data_map.find(sel_factory_obj->GetDataIdentifier())->second));
    pred += data_trafo * it.second;
    
  }
  return pred;
}

// Destructor:
Compboost::~Compboost ()
{
  // std::cout << "Call Compboost Destructor" << std::endl;
  // delete used_optimizer;
  // delete used_loss;
  // delete used_logger;
}

} // namespace cboost