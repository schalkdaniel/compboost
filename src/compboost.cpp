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

Compboost::Compboost (std::shared_ptr<response::Response> sh_ptr_response, const double& learning_rate,
  const bool& is_global_stopper, std::shared_ptr<optimizer::Optimizer> sh_ptr_optimizer, std::shared_ptr<loss::Loss> sh_ptr_loss,
  std::shared_ptr<loggerlist::LoggerList> sh_ptr_loggerlist, blearnerlist::BaselearnerFactoryList factory_list)
  : _learning_rate      ( learning_rate ),
    _is_global_stopper  ( is_global_stopper ),
    _sh_ptr_response    ( sh_ptr_response ),
    _sh_ptr_optimizer   ( sh_ptr_optimizer ),
    _sh_ptr_loss        ( sh_ptr_loss ),
    _factory_list       ( factory_list ),
    _sh_ptr_loggerlist  ( sh_ptr_loggerlist ),
    _blearner_track     ( blearnertrack::BaselearnerTrack(learning_rate) )
{
  _sh_ptr_response->constantInitialization(sh_ptr_loss);
  _sh_ptr_response->initializePrediction();
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::train (const unsigned int trace, const std::shared_ptr<loggerlist::LoggerList> sh_ptr_loggerlist)
{

  if (_factory_list.getFactoryMap().size() == 0) {
    Rcpp::stop("Could not train without any registered base-learner.");
  }

  bool is_stopc_reached = false;
  unsigned int k = 1;

  // Main Algorithm. While the stop criteria isn't fulfilled, run the
  // algorithm:
  while (! is_stopc_reached) {

    _current_iter = _blearner_track.getBaselearnerVector().size() + 1;

    _sh_ptr_response->setIteration(_current_iter);
    _sh_ptr_response->updatePseudoResiduals(_sh_ptr_loss);
    _sh_ptr_optimizer->optimize(_current_iter, _learning_rate, _sh_ptr_loss, _sh_ptr_response,
      _blearner_track, _factory_list);

    sh_ptr_loggerlist->logCurrent(_current_iter, _sh_ptr_response, _blearner_track.getBaselearnerVector().back(),
      _learning_rate, _sh_ptr_optimizer->getStepSize(_current_iter), _sh_ptr_optimizer);

    // Calculate and log risk:
    _risk.push_back(_sh_ptr_response->calculateEmpiricalRisk(_sh_ptr_loss));

    // Get status of the algorithm (is the stopping criteria reached?). The negation here
    // seems a bit weird, but it makes the while loop easier to read:
    is_stopc_reached = ! sh_ptr_loggerlist->getStopperStatus(_is_global_stopper);

    if (helper::checkTracePrinter(_current_iter, trace)) sh_ptr_loggerlist->printLoggerStatus(_risk.back());
    k += 1;
  }

  if (trace) {
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << std::endl;
  }
}

void Compboost::trainCompboost (const unsigned int trace)
{
  // Make sure, that the selected baselearner and logger data is empty:
  _blearner_track.clearBaselearnerVector();
  _sh_ptr_loggerlist->clearLoggerData();

  // Calculate risk for initial model:
  _risk.push_back(_sh_ptr_response->calculateEmpiricalRisk(_sh_ptr_loss));

  // track time:
  auto t1 = std::chrono::high_resolution_clock::now();

  // Initial training:
  train(trace, _sh_ptr_loggerlist);

  // track time:
  auto t2 = std::chrono::high_resolution_clock::now();

  // After training call printer for a status:
  Rcpp::Rcout << "Train " << std::to_string(_current_iter) << " iterations in "
              << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " Seconds." << std::endl;
  Rcpp::Rcout << "Final risk based on the train set: " << std::setprecision(2)
              << _risk.back() << std::endl << std::endl;

  // Set flag if model is trained:
  _is_trained = true;
}

void Compboost::continueTraining (const unsigned int trace)
{
  helper::debugPrint("From 'Compboost::continueTraining':");
  if (! _is_trained) {
    Rcpp::stop("Initial training hasn't been done yet. Use 'train()' first.");
  }
  helper::debugPrint("| > Check if new base-learner needs to be trained");
  if (_current_iter != _blearner_track.getBaselearnerVector().size()) {
    unsigned int iter_max = _blearner_track.getBaselearnerVector().size();
    setToIteration(iter_max, -1);
  }
  helper::debugPrint("| > Train new base-learner");
  train(trace, _sh_ptr_loggerlist);

  // Update actual state:
  _current_iter = _blearner_track.getBaselearnerVector().size();
  helper::debugPrint("| > Update iteration");
  helper::debugPrint("Finished 'Compboost::continueTraining'");
}

arma::vec Compboost::getPrediction (const bool& as_response) const
{
  arma::vec pred;
  if (as_response) {
    return _sh_ptr_response->getPredictionTransform();
  } else {
    return _sh_ptr_response->getPredictionScores();
  }
}

std::map<std::string, arma::mat> Compboost::getParameter () const
{
  return _blearner_track.getParameterMap();
}

std::vector<std::string> Compboost::getSelectedBaselearner () const
{
  std::vector<std::string> selected_blearner_names;

  for (unsigned int i = 0; i < _current_iter; i++) {
    selected_blearner_names.push_back(_blearner_track.getBaselearnerVector()[i]->getDataIdentifier() + "_" + _blearner_track.getBaselearnerVector()[i]->getBaselearnerType());
  }
  return selected_blearner_names;
}

std::shared_ptr<loggerlist::LoggerList> Compboost::getLoggerList () const
{
  return _sh_ptr_loggerlist;
}

std::map<std::string, arma::mat> Compboost::getParameterOfIteration (const unsigned int& k) const
{
  return _blearner_track.getEstimatedParameterOfIteration(k);
}

std::pair<std::vector<std::string>, arma::mat> Compboost::getParameterMatrix () const
{
  return _blearner_track.getParameterMatrix();
}

arma::vec Compboost::predict () const
{
  helper::debugPrint("From 'Compboost::predict()'");

  std::map<std::string, arma::mat> parameter_map = _blearner_track.getParameterMap();
  helper::debugPrint("| > Calculate initial prediction");
  arma::mat pred = _sh_ptr_response->calculateInitialPrediction(_sh_ptr_response->getResponse());

  // Calculate vector - matrix product for each selected base-learner:
  for (auto& it : parameter_map) {
    std::string sel_factory = it.first;
    helper::debugPrint("| > data_trafo:" + helper::getMatStatus(_factory_list.getFactoryMap().find(sel_factory)->second->getData()));
    helper::debugPrint("| > parameter iterator:" + helper::getMatStatus(it.second));
    pred += _factory_list.getFactoryMap().find(sel_factory)->second->calculateLinearPredictor(it.second);
  }
  helper::debugPrint("Finished 'Compboost::predict()'");
  return pred;
}

// Predict for new data. Note: The data_map contains the raw columns of the used data.
// Those columns are then transformed by the corresponding transform data function of the
// specific factory. After the transformation, the transformed data is multiplied by the
// corresponding parameter.
arma::vec Compboost::predict (std::map<std::string, std::shared_ptr<data::Data>> data_map, const bool& as_response) const
{
  helper::debugPrint("From 'Compboost::predict(std::map, bool)'");
  // IMPROVE THIS FUNCTION!!! See:
  // https://github.com/schalkdaniel/compboost/issues/206

  std::map<std::string, arma::mat> parameter_map = _blearner_track.getParameterMap();

  arma::mat pred(data_map.begin()->second->getData().n_rows, _sh_ptr_response->getResponse().n_cols, arma::fill::zeros);
  helper::debugPrint("| > Calculate initial prediction");
  pred = _sh_ptr_response->calculateInitialPrediction(pred);

  // Idea is simply to calculate the vector matrix product of parameter and
  // newdata. The problem here is that the newdata comes as raw data and has
  // to be transformed first:
  for (auto& it_pair_param : parameter_map) {

    // Name of current feature:
    std::string factory_id = it_pair_param.first;

    // Find the element with key 'hat'
    std::shared_ptr<blearnerfactory::BaselearnerFactory> blearner_factory = _factory_list.getFactoryMap().find(factory_id)->second;

    // Select newdata corresponding to selected factory object:
    std::map<std::string, std::shared_ptr<data::Data>>::iterator it_newdata;
    it_newdata = data_map.find(blearner_factory->getDataIdentifier());

    // Calculate prediction by accumulating the design matrices multiplied by the estimated parameter:
    if (it_newdata != data_map.end()) {
      pred += blearner_factory->calculateLinearPredictor(it_pair_param.second, it_newdata->second);
    }
  }
  if (as_response) {
    pred = _sh_ptr_response->getPredictionTransform(pred);
  }
  return pred;
}

void Compboost::setToIteration (const unsigned int& k, const unsigned int& trace)
{
  helper::debugPrint("From 'Compboost::setToIteration'");
  unsigned int iter_max = _blearner_track.getBaselearnerVector().size();

  helper::debugPrint("| > Check if new base-learner needs to be trained");
  if (k > iter_max) {
    unsigned int iter_diff = k - iter_max;
    Rcpp::Rcout << "\nYou have already trained " << std::to_string(iter_max) << " iterations.\n"
                <<"Train " << std::to_string(iter_diff) << " additional iterations."
                << std::endl << std::endl;

    _sh_ptr_loggerlist->prepareForRetraining(k);
    continueTraining(trace);
  }
  helper::debugPrint("| > Set base-learner track to the new iteration");
  _blearner_track.setToIteration(k);
  helper::debugPrint("| > Set new prediction scores by calling predict()");
  _sh_ptr_response->setPredictionScores(predict(), k);
  _current_iter = k;
  helper::debugPrint("Finished 'Compboost::setToIteration'");
}

arma::mat Compboost::getOffset() const { return _sh_ptr_response->getInitialization(); }
std::vector<double> Compboost::getRiskVector () const { return _risk; }

void Compboost::summarizeCompboost () const
{
  Rcpp::Rcout << "Compboost object with:" << std::endl;
  Rcpp::Rcout << "\t- Learning Rate: " << _learning_rate << std::endl;
  Rcpp::Rcout << "\t- Are all logger used as stopper: " << _is_global_stopper << std::endl;

  if (_is_trained) {
    Rcpp::Rcout << "\t- Model is already trained with " << _blearner_track.getBaselearnerVector().size() << " iterations/fitted baselearner" << std::endl;
    Rcpp::Rcout << "\t- Actual state is at iteration " << _current_iter << std::endl;
  }
  Rcpp::Rcout << std::endl;
}

// Destructor:
Compboost::~Compboost () {}

} // namespace cboost
