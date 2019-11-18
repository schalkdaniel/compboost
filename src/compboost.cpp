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
// =========================================================================== #

#include "compboost.h"

namespace cboost {

// --------------------------------------------------------------------------- #
// Constructor:
// --------------------------------------------------------------------------- #

// todo: response as call by reference!

Compboost::Compboost () {}

Compboost::Compboost (std::shared_ptr<response::Response> sh_ptr_response, const double& learning_rate,
  const bool& stop_if_all_stopper_fulfilled, std::shared_ptr<optimizer::Optimizer> sh_ptr_optimizer, std::shared_ptr<loss::Loss> sh_ptr_loss,
  std::shared_ptr<loggerlist::LoggerList> sh_ptr_loggerlist0, blearnerlist::BaselearnerFactoryList used_baselearner_list)
  : sh_ptr_response ( sh_ptr_response ),
    learning_rate ( learning_rate ),
    stop_if_all_stopper_fulfilled ( stop_if_all_stopper_fulfilled ),
    sh_ptr_optimizer ( sh_ptr_optimizer ),
    sh_ptr_loss ( sh_ptr_loss ),
    used_baselearner_list ( used_baselearner_list )
{
  sh_ptr_response->constantInitialization(sh_ptr_loss);
  sh_ptr_response->initializePrediction();
  blearner_track = blearnertrack::BaselearnerTrack(learning_rate);
  sh_ptr_loggerlist = sh_ptr_loggerlist0;
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::train (const unsigned int& trace, std::shared_ptr<loggerlist::LoggerList> logger_list)
{

  if (used_baselearner_list.getMap().size() == 0) {
    Rcpp::stop("Could not train without any registered base-learner.");
  }

  // Bool to indicate whether the stop criteria (stopc) is reached or not:
  bool is_stopc_reached = false;
  unsigned int k = 1;

  // Main Algorithm. While the stop criteria isn't fulfilled, run the
  // algorithm:
  while (! is_stopc_reached) {

    actual_iteration = blearner_track.getBaselearnerVector().size() + 1;

    sh_ptr_response->setActualIteration(actual_iteration);
    sh_ptr_response->updatePseudoResiduals(sh_ptr_loss);

    sh_ptr_optimizer->optimize(actual_iteration, learning_rate, sh_ptr_loss, sh_ptr_response, blearner_track,
      used_baselearner_list);

    logger_list->logCurrent(actual_iteration, sh_ptr_response, blearner_track.getBaselearnerVector().back(),
      learning_rate, sh_ptr_optimizer->getStepSize(actual_iteration), sh_ptr_optimizer);

    // Calculate and log risk:
    risk.push_back(sh_ptr_response->calculateEmpiricalRisk(sh_ptr_loss));

    // Get status of the algorithm (is the stopping criteria reached?). The negation here
    // seems a bit weird, but it makes the while loop easier to read:
    is_stopc_reached = ! logger_list->getStopperStatus(stop_if_all_stopper_fulfilled);

    if (helper::checkTracePrinter(actual_iteration, trace)) logger_list->printLoggerStatus(risk.back());
    k += 1;
  }

  if (trace) {
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << std::endl;
  }
}

void Compboost::trainCompboost (const unsigned int& trace)
{
  // Make sure, that the selected baselearner and logger data is empty:
  blearner_track.clearBaselearnerVector();
  sh_ptr_loggerlist->clearLoggerData();

  // Calculate risk for initial model:
  risk.push_back(sh_ptr_response->calculateEmpiricalRisk(sh_ptr_loss));

  // track time:
  auto t1 = std::chrono::high_resolution_clock::now();

  // Initial training:
  train(trace, sh_ptr_loggerlist);

  // track time:
  auto t2 = std::chrono::high_resolution_clock::now();

  // After training call printer for a status:
  Rcpp::Rcout << "Train " << std::to_string(actual_iteration) << " iterations in "
              << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " Seconds." << std::endl;
  Rcpp::Rcout << "Final risk based on the train set: " << std::setprecision(2)
              << risk.back() << std::endl << std::endl;

  // Set flag if model is trained:
  model_is_trained = true;
}

void Compboost::continueTraining (const unsigned int& trace)
{
  if (! model_is_trained) {
    Rcpp::stop("Initial training hasn't been done yet. Use 'train()' first.");
  }
  if (actual_iteration != blearner_track.getBaselearnerVector().size()) {
    unsigned int max_iteration = blearner_track.getBaselearnerVector().size();
    setToIteration(max_iteration, -1);
  }
  train(trace, sh_ptr_loggerlist);

  // Update actual state:
  actual_iteration = blearner_track.getBaselearnerVector().size();
}

arma::vec Compboost::getPrediction (const bool& as_response) const
{
  arma::vec pred;
  if (as_response) {
    return sh_ptr_response->getPredictionTransform();
  } else {
    return sh_ptr_response->getPredictionScores();
  }
}

std::map<std::string, arma::mat> Compboost::getParameter () const
{
  return blearner_track.getParameterMap();
}

std::vector<std::string> Compboost::getSelectedBaselearner () const
{
  std::vector<std::string> selected_blearner_names;

  for (unsigned int i = 0; i < actual_iteration; i++) {
    selected_blearner_names.push_back(blearner_track.getBaselearnerVector()[i]->getDataIdentifier() + "_" + blearner_track.getBaselearnerVector()[i]->getBaselearnerType());
  }
  return selected_blearner_names;
}

std::shared_ptr<loggerlist::LoggerList> Compboost::getLoggerList () const
{
  return sh_ptr_loggerlist;
}

std::map<std::string, arma::mat> Compboost::getParameterOfIteration (const unsigned int& k) const
{
  // Check is done in function GetEstimatedParameterOfIteration in baselearner_track.cpp
  return blearner_track.getEstimatedParameterOfIteration(k);
}

std::pair<std::vector<std::string>, arma::mat> Compboost::getParameterMatrix () const
{
  return blearner_track.getParameterMatrix();
}

arma::vec Compboost::predict () const
{
  std::map<std::string, arma::mat> parameter_map  = blearner_track.getParameterMap();
  arma::mat pred = sh_ptr_response->calculateInitialPrediction(sh_ptr_response->getResponse());

  // Calculate vector - matrix product for each selected base-learner:
  for (auto& it : parameter_map) {
    std::string sel_factory = it.first;
    pred += used_baselearner_list.getMap().find(sel_factory)->second->getData() * it.second;
    // pred += train_data_map.find(sel_factory)->second * it.second;
  }
  return pred;
}

// Predict for new data. Note: The data_map contains the raw columns of the used data.
// Those columns are then transformed by the corresponding transform data function of the
// specific factory. After the transformation, the transformed data is multiplied by the
// corresponding parameter.
arma::vec Compboost::predict (std::map<std::string, std::shared_ptr<data::Data>> data_map, const bool& as_response) const
{
  // IMPROVE THIS FUNCTION!!! See:
  // https://github.com/schalkdaniel/compboost/issues/206

  std::map<std::string, arma::mat> parameter_map = blearner_track.getParameterMap();

  arma::mat pred(data_map.begin()->second->getData().n_rows, sh_ptr_response->getResponse().n_cols, arma::fill::zeros);
  pred = sh_ptr_response->calculateInitialPrediction(pred);

  // Idea is simply to calculate the vector matrix product of parameter and
  // newdata. The problem here is that the newdata comes as raw data and has
  // to be transformed first:
  for (auto& it : parameter_map) {

    // Name of current feature:
    std::string sel_factory = it.first;

    // Find the element with key 'hat'
    std::shared_ptr<blearnerfactory::BaselearnerFactory> sel_factory_obj = used_baselearner_list.getMap().find(sel_factory)->second;

    // Select newdata corresponding to selected facotry object:
    std::map<std::string, std::shared_ptr<data::Data>>::iterator it_newdata;
    it_newdata = data_map.find(sel_factory_obj->getDataIdentifier());

    // Calculate prediction by accumulating the design matrices multiplied by the estimated parameter:
    if (it_newdata != data_map.end()) {
      arma::mat data_trafo = sel_factory_obj->instantiateData(it_newdata->second->getData());
      pred += data_trafo * it.second;
    }
  }
  if (as_response) {
    pred = sh_ptr_response->getPredictionTransform(pred);
  }
  return pred;
}

// Set model to an given iteration. The predictions and everything is then done at this iteration:
void Compboost::setToIteration (const unsigned int& k, const unsigned int& trace)
{
  unsigned int max_iteration = blearner_track.getBaselearnerVector().size();

  // Set parameter:
  if (k > max_iteration) {
    unsigned int iteration_diff = k - max_iteration;
    Rcpp::Rcout << "\nYou have already trained " << std::to_string(max_iteration) << " iterations.\n"
                <<"Train " << std::to_string(iteration_diff) << " additional iterations."
                << std::endl << std::endl;

    sh_ptr_loggerlist->prepareForRetraining(k);
    continueTraining(trace);
  }

  blearner_track.setToIteration(k);
  sh_ptr_response->setActualPredictionScores(predict(), k);
  actual_iteration = k;
}

arma::mat Compboost::getOffset() const
{
  return sh_ptr_response->getInitialization();
}

std::vector<double> Compboost::getRiskVector () const
{
  return risk;
}

void Compboost::summarizeCompboost () const
{
  Rcpp::Rcout << "Compboost object with:" << std::endl;
  Rcpp::Rcout << "\t- Learning Rate: " << learning_rate << std::endl;
  Rcpp::Rcout << "\t- Are all logger used as stopper: " << stop_if_all_stopper_fulfilled << std::endl;

  if (model_is_trained) {
    Rcpp::Rcout << "\t- Model is already trained with " << blearner_track.getBaselearnerVector().size() << " iterations/fitted baselearner" << std::endl;
    Rcpp::Rcout << "\t- Actual state is at iteration " << actual_iteration << std::endl;
    // Rcpp::Rcout << "\t- Loss optimal initialization: " << std::fixed << std::setprecision(2) << initialization << std::endl;
  }
  Rcpp::Rcout << std::endl;
}

// Destructor:
Compboost::~Compboost () {}

} // namespace cboost
