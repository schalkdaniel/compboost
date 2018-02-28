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
  loss::Loss* used_loss, loggerlist::LoggerList* used_logger0,
  blearnerlist::BaselearnerFactoryList used_baselearner_list)
  : response ( response ), 
    learning_rate ( learning_rate ),
    stop_if_all_stopper_fulfilled ( stop_if_all_stopper_fulfilled ),
    used_optimizer ( used_optimizer ),
    used_loss ( used_loss ),
    used_baselearner_list ( used_baselearner_list )
{
  blearner_track = blearnertrack::BaselearnerTrack(learning_rate);
  used_logger["initial.training"] = used_logger0;
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::Train (const bool& trace, const arma::vec& prediction, loggerlist::LoggerList* logger)
{
  arma::vec pred_temp = prediction;
  
  // Initialize trace:
  if (trace) {
    Rcpp::Rcout << std::endl;
    logger->InitializeLoggerPrinter(); 
  }
  
  // Declare variables to stop the algorithm:
  bool stop_the_algorithm = false;
  unsigned int k = 1;
  
  // Main Algorithm. While the stop criteria isn't fullfilled, run the 
  // algorithm:
  while (! stop_the_algorithm) {
    
    // Define pseudo residuals as negative gradient:
    pseudo_residuals = -used_loss->DefinedGradient(response, pred_temp);
    // Rcpp::Rcout << "\n<<Compboost>> Define pseudo residuals as negative gradient" << std::endl;
    
    // Cast integer k to string for baselearner identifier:
    std::string temp_string = std::to_string(k);
    blearner::Baselearner* selected_blearner = used_optimizer->FindBestBaselearner(temp_string, pseudo_residuals, used_baselearner_list.GetMap());
    // Rcpp::Rcout << "<<Compboost>> Cast integer k to string for baselearner identifier" << std::endl;
    
    // Insert new baselearner to vector of selected baselearner:    
    blearner_track.InsertBaselearner(selected_blearner);
    // Rcpp::Rcout << "<<Compboost>> Insert new baselearner to vector of selected baselearner" << std::endl;
    
    // Update model (prediction) and shrink by learning rate:
    pred_temp += learning_rate * selected_blearner->predict();
    // Rcpp::Rcout << "<<Compboost>> Update model (prediction) and shrink by learning rate" << std::endl;
    
    // Log the current step:
    
    // The last term has to be the prediction or anything like that. This is
    // important to track the risk (inbag or oob)!!!!
    
    logger->LogCurrent(k, response, pred_temp, selected_blearner, 
      initialization, learning_rate);
    // Rcpp::Rcout << "<<Compboost>> Log the current step" << std::endl;
    
    // Get status of the algorithm (is stopping criteria reached):
    stop_the_algorithm = ! logger->GetStopperStatus(stop_if_all_stopper_fulfilled);
    
    // Print trace:
    if (trace) {
      logger->PrintLoggerStatus(); 
    }
    
    // Increment k:
    k += 1;
  }
  
  // Just for console appearance
  if (trace) {
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << std::endl; 
  }
  
  // Set model prediction:
  model_prediction = pred_temp;
  
  // Set actual state to the latest iteration:
  actual_state = blearner_track.GetBaselearnerVector().size();
}

void Compboost::TrainCompboost (const bool& trace)
{
  // Make sure, that the selected baselearner and logger data is empty:
  blearner_track.ClearBaselearnerVector();
  for (auto& it : used_logger) {
    it.second->ClearLoggerData();
  }
  
  // Initialize zero model and pseudo residuals:
  initialization = used_loss->ConstantInitializer(response);
  arma::vec pseudo_residuals_init (response.size());
  // Rcpp::Rcout << "<<Compboost>> Initialize zero model and pseudo residuals" << std::endl;
  
  // Initialize prediction and fill with zero model:
  arma::vec prediction(response.size());
  prediction.fill(initialization);
  // Rcpp::Rcout << "<<Compboost>> Initialize prediction and fill with zero model" << std::endl;
  
  // Initial training:
  Train(trace, prediction, used_logger["initial.training"]);
  
  // Set flag if model is trained:
  model_is_trained = true;
}

void Compboost::ContinueTraining (loggerlist::LoggerList* logger, const bool& trace)
{
  if (! model_is_trained) {
    Rcpp::stop("Initial training hasn't been done. Use 'train()' first.");
  }
  if (actual_state != blearner_track.GetBaselearnerVector().size()) {
    Rcpp::stop("To avoid unexpected behaviour, compboost wants to start retraining on the maximal iterations. You have called 'setToIteration' prior. Please set iterations to the maximal value.");
  }
  
  // Continue training:
  Train(trace, model_prediction, logger);
  
  // Register logger:
  std::string logger_id = "retraining" + std::to_string(used_logger.size());
  used_logger[logger_id] = logger;
  
  // Update actual state:
  actual_state = blearner_track.GetBaselearnerVector().size();
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
  for (unsigned int i = 0; i < actual_state; i++) {
    selected_blearner.push_back(blearner_track.GetBaselearnerVector()[i]->GetDataIdentifier() + ": " + blearner_track.GetBaselearnerVector()[i]->GetBaselearnerType());
  }
  return selected_blearner;
}

std::map<std::string, loggerlist::LoggerList*> Compboost::GetLoggerList () const
{
  return used_logger;
}

std::map<std::string, arma::mat> Compboost::GetParameterOfIteration (unsigned int k) 
{
  return blearner_track.GetEstimatedParameterOfIteration(k);
}

std::pair<std::vector<std::string>, arma::mat> Compboost::GetParameterMatrix ()
{
  return blearner_track.GetParameterMatrix();
}

arma::vec Compboost::Predict ()
{
  std::map<std::string, arma::mat> parameter_map  = blearner_track.GetParameterMap();
  std::map<std::string, arma::mat> train_data_map = used_baselearner_list.GetDataMap();
  
  arma::vec pred(train_data_map.begin()->second.n_rows);
  pred.fill(initialization);
  
  for (auto& it : parameter_map) {
    
    std::string sel_factory = it.first;
    pred += train_data_map.find(sel_factory)->second * it.second;
    
  }
  return pred;
}

// Predict for new data. Note: The data_map contains the raw columns of the used data.
// Those columns are then transformed by the corresponding transform dat function of the
// specific factory. After the transformation, the transformed data is multiplied by the
// corresponding parameter.
arma::vec Compboost::Predict (std::map<std::string, arma::mat> data_map)
{
  // Rcpp::Rcout << "Get into Compboost::Predict" << std::endl;
  
  std::map<std::string, arma::mat> parameter_map = blearner_track.GetParameterMap();

  arma::vec pred(data_map.begin()->second.n_rows);
  pred.fill(initialization);
  
  // Rcpp::Rcout << "initialize pred vec" << std::endl;
  
  for (auto& it : parameter_map) {
    
    std::string sel_factory = it.first;
    
    // Rcpp::Rcout << "Fatory id of parameter map: " << sel_factory << std::endl;
    
    blearnerfactory::BaselearnerFactory* sel_factory_obj = used_baselearner_list.GetMap().find(sel_factory)->second;
    
    // Rcpp::Rcout << "Data of selected factory: " << sel_factory_obj->GetDataIdentifier() << std::endl;
    
    arma::mat data_trafo = sel_factory_obj->InstantiateData((data_map.find(sel_factory_obj->GetDataIdentifier())->second));
    pred += data_trafo * it.second;
    
  }
  return pred;
}

void Compboost::SummarizeCompboost ()
{
  Rcpp::Rcout << "Compboost object with:" << std::endl;
  Rcpp::Rcout << "\t- Learning Rate: " << learning_rate << std::endl;
  Rcpp::Rcout << "\t- Are all logger used as stopper: " << stop_if_all_stopper_fulfilled << std::endl;
  
  if (model_is_trained) {
    Rcpp::Rcout << "\t- Model is already trained with " << blearner_track.GetBaselearnerVector().size() << " iterations/fitted baselearner" << std::endl;
    Rcpp::Rcout << "\t- Actual state is at iteration " << actual_state << std::endl;
    Rcpp::Rcout << "\t- Loss optimal initialization: " << std::fixed << std::setprecision(2) << initialization << std::endl;
  }
  Rcpp::Rcout << std::endl;
  Rcpp::Rcout << "To get more information check the other objects!" << std::endl;
}

arma::vec Compboost::PredictionOfIteration (std::map<std::string, arma::mat> data_map, unsigned int k)
{
  // Rcpp::Rcout << "Get into Compboost::Predict" << std::endl;
  
  std::map<std::string, arma::mat> parameter_map = blearner_track.GetEstimatedParameterOfIteration(k);
  
  arma::vec pred(data_map.begin()->second.n_rows);
  pred.fill(initialization);
  
  // Rcpp::Rcout << "initialize pred vec" << std::endl;
  
  for (auto& it : parameter_map) {
    
    std::string sel_factory = it.first;
    
    // Rcpp::Rcout << "Fatory id of parameter map: " << sel_factory << std::endl;
    
    blearnerfactory::BaselearnerFactory* sel_factory_obj = used_baselearner_list.GetMap().find(sel_factory)->second;
    
    // Rcpp::Rcout << "Data of selected factory: " << sel_factory_obj->GetDataIdentifier() << std::endl;
    
    arma::mat data_trafo = sel_factory_obj->InstantiateData((data_map.find(sel_factory_obj->GetDataIdentifier())->second));
    pred += data_trafo * it.second;
    
  }
  return pred;
}

// Set model to an given iteration. The predictions and everything is then done at this iteration:
void Compboost::SetToIteration (const unsigned int& k) 
{
  // Set parameter:
  blearner_track.setToIteration(k);
  
  // Set prediction:
  model_prediction = Predict();
  
  // Set actual state:
  actual_state = k;
}

// Destructor:
Compboost::~Compboost ()
{
  // Rcpp::Rcout << "Call Compboost Destructor" << std::endl;
  // delete used_optimizer;
  // delete used_loss;
  // delete used_logger;
}

} // namespace cboost
