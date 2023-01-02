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
    _sh_ptr_loggerlist  ( sh_ptr_loggerlist ),
    _factory_list       ( factory_list ),
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
      _learning_rate, _sh_ptr_optimizer->getStepSize(_current_iter), _sh_ptr_optimizer, _factory_list);

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

  auto bl_track = _blearner_track.getBaselearnerVector();
  for (unsigned int i = 0; i < _current_iter; i++) {
    selected_blearner_names.push_back(bl_track[i]->getDataIdentifier() + "_" + bl_track[i]->getBaselearnerType());
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

arma::mat Compboost::predictFactory (const std::string& factory_id) const
{
  auto parameter_map = _blearner_track.getParameterMap();
  auto it_par_map    = parameter_map.find(factory_id);
  if (it_par_map == parameter_map.end())
    throw std::range_error("Cannot find factory in parameter map.");

  auto fac_map = _factory_list.getFactoryMap();
  auto it_fac  = fac_map.find(factory_id);
  if (it_fac == fac_map.end())
    throw std::range_error("Cannot find factory in factory map.");

  return it_fac->second->calculateLinearPredictor(it_par_map->second);
}

std::map<std::string, arma::mat> Compboost::predictIndividual () const
{
  auto parameter_map = _blearner_track.getParameterMap();
  std::map<std::string, arma::mat> out;

  for (auto& it : parameter_map) {
    arma::mat tmp = predictFactory(it.first);
    out.insert(std::pair<std::string, arma::mat>(it.first, tmp));
  }
  return out;
}

arma::vec Compboost::predict () const
{
  arma::mat pred = _sh_ptr_response->calculateInitialPrediction(_sh_ptr_response->getResponse());

  auto ind_preds = predictIndividual();
  for (auto& it : ind_preds)
    pred += it.second;

  helper::debugPrint("Finished 'Compboost::predict()'");
  return pred;
}

arma::mat Compboost::predictFactory(const std::string& factory_id, const std::map<std::string, std::shared_ptr<data::Data>>& data_map) const
{
  auto parameter_map = _blearner_track.getParameterMap();
  auto it_par_map    = parameter_map.find(factory_id);
  if (it_par_map == parameter_map.end())
    throw std::range_error("Cannot find factory in parameter map.");

  auto fac_map = _factory_list.getFactoryMap();
  auto it_fac  = fac_map.find(factory_id);
  if (it_fac == fac_map.end())
    throw std::range_error("Cannot find factory in factory map.");

  return it_fac->second->calculateLinearPredictor(it_par_map->second, data_map);
}

std::map<std::string, arma::mat> Compboost::predictIndividual (const std::map<std::string, std::shared_ptr<data::Data>>& data_map) const
{
  auto parameter_map = _blearner_track.getParameterMap();
  std::map<std::string, arma::mat> out;

  for (auto& it : parameter_map) {
    arma::mat tmp = predictFactory(it.first, data_map);
    out.insert(std::pair<std::string, arma::mat>(it.first, tmp));
  }
  return out;
}


arma::vec Compboost::predict (const std::map<std::string, std::shared_ptr<data::Data>>& data_map, const bool& as_response) const
{
  arma::mat pred(data_map.begin()->second->getNObs(), _sh_ptr_response->getResponse().n_cols, arma::fill::zeros);

  if (_sh_ptr_response->getInitialization().n_rows == 1)
    pred = _sh_ptr_response->calculateInitialPrediction(pred);

  auto ind_preds = predictIndividual(data_map);

  for (auto& it : ind_preds)
    pred += it.second;

   if (as_response)
    pred = _sh_ptr_response->getPredictionTransform(pred);

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
                << "Train " << std::to_string(iter_diff) << " additional iterations."
                << std::endl << std::endl;

    _sh_ptr_loggerlist->prepareForRetraining(k);
    continueTraining(trace);
  }
  helper::debugPrint("| > Set base-learner track to the new iteration");
  auto tmp_param_map = _sh_ptr_optimizer->getParameterAtIteration(k, _learning_rate, _blearner_track);
  _blearner_track.setParameterMap(tmp_param_map);
  //
  //_blearner_track.setToIteration(k);
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

void Compboost::saveJson (std::string file) const
{
  json j = {
    {"learning_rate",     _learning_rate},
    {"is_global_stopper", _is_global_stopper},
    {"is_trained",        _is_trained},
    {"current_iter",      _current_iter},
    {"risk",              _risk},

    {"response",     _sh_ptr_response->toJson() },
    {"optimizer",    nullptr}, //std::shared_ptr<optimizer::Optimizer>.toJson()
    {"loss",         nullptr}, //std::shared_ptr<loss::Loss>.toJson()
    {"loggerlist",   nullptr}, //std::shared_ptr<loggerlist::LoggerList>.toJson()
    {"factory_list", nullptr}, //blearnerlist::BaselearnerFactoryList.toJson()
    {"baselearner",  nullptr}  //blearnertrack::BaselearnerTrack.toJson()
  };

  std::ofstream o(file);
  o << j.dump(2) << std::endl;
}

// Destructor:
Compboost::~Compboost () {}

} // namespace cboost
