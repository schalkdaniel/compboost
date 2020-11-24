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
// ========================================================================== //

#include "optimizer.h"

namespace optimizer {

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

Optimizer::Optimizer () { };
Optimizer::Optimizer (const unsigned int num_threads) : _num_threads ( num_threads ) { }

// Destructor:
Optimizer::~Optimizer () { }

// -------------------------------------------------------------------------- //
// Optimizer implementations:
// -------------------------------------------------------------------------- //

// OptimizerCoordinateDescent:
// ---------------------------------------------------

OptimizerCoordinateDescent::OptimizerCoordinateDescent ()
{
  _step_sizes.assign(1, 1.0);
}

OptimizerCoordinateDescent::OptimizerCoordinateDescent (const unsigned int num_threads)
  : Optimizer::Optimizer ( num_threads )
{
  _step_sizes.assign(1, 1.0);
}

std::shared_ptr<blearner::Baselearner> OptimizerCoordinateDescent::findBestBaselearner (std::string iteration_id,
  const std::shared_ptr<response::Response>& sh_ptr_response, const blearner_factory_map& factory_map) const
{
  std::map<double, std::shared_ptr<blearner::Baselearner>> best_blearner_map;

  /* ****************************************************************************************
   * OLD SEQUENTIAL LOOP:
   */

  // Use ordinary sequential loop if just one thread should be used. This saves the costs
  // of distributing data etc. and results in a significant speed up:
  if (_num_threads == 1) {
    double ssq_temp;
    double ssq_best = std::numeric_limits<double>::infinity();

    std::shared_ptr<blearner::Baselearner> blearner_temp;
    std::shared_ptr<blearner::Baselearner> blearner_best;

    for (auto& it : factory_map) {

      // Paste string identifier for new base-learner:
      std::string id = "(" + iteration_id + ") " + it.second->getBaselearnerType();

      // Create new base-learner out of the actual factory (just the
      // pointer is overwritten):
      blearner_temp = it.second->createBaselearner();
      blearner_temp->train(sh_ptr_response->getPseudoResiduals());
      ssq_temp = helper::calculateSumOfSquaredError(sh_ptr_response->getPseudoResiduals(), blearner_temp->predict());


      // Check if SSE of new temporary base-learner is smaller then SSE of the best
      // base-learner. If so, assign the temporary base-learner with the best
      // base-learner (This is always triggered within the first iteration since
      // ssq_best is declared as infinity):
      if (ssq_temp < ssq_best) {
        ssq_best = ssq_temp;
        // // Deep copy since the temporary base-learner is deleted every time which
        // // will also deletes the data for the best base-learner if we don't copy
        // // the whole data of the object:
        // blearner_best = blearner_temp->clone();
        blearner_best = blearner_temp;
      }
    }
    return blearner_best;
  }
  /* **************************************************************************************** */

  #pragma omp parallel num_threads(_num_threads) default(none) shared(iteration_id, sh_ptr_response, factory_map, best_blearner_map)
  {
    // private per core:
    double ssq_temp;
    double ssq_best = std::numeric_limits<double>::infinity();

    std::shared_ptr<blearner::Baselearner> blearner_temp;
    std::shared_ptr<blearner::Baselearner> blearner_best;

    #pragma omp for schedule(dynamic)
    for (unsigned int i = 0; i < factory_map.size(); i++) {

      // increment iterator to "index map elements by index" (https://stackoverflow.com/questions/8848870/use-openmp-in-iterating-over-a-map):
      auto it_factory_pair = factory_map.begin();
      std::advance(it_factory_pair, i);

      // Paste string identifier for new base-learner:
      std::string id = "(" + iteration_id + ") " + it_factory_pair->second->getBaselearnerType();

      // Create new base-learner out of the actual factory (just the
      // pointer is overwritten):
      blearner_temp = it_factory_pair->second->createBaselearner();
      blearner_temp->train(sh_ptr_response->getPseudoResiduals());
      ssq_temp = helper::calculateSumOfSquaredError(sh_ptr_response->getPseudoResiduals(), blearner_temp->predict());

      // Check if SSE of new temporary base-learner is smaller then SSE of the best
      // base-learner. If so, assign the temporary base-learner with the best
      // base-learner (This is always triggered within the first iteration since
      // ssq_best is declared as infinity):
      if (ssq_temp < ssq_best) {
        ssq_best = ssq_temp;
        // // Deep copy since the temporary base-learner is deleted every time which
        // // will also deletes the data for the best base-learner if we don't copy
        // // the whole data of the object:
        // blearner_best = blearner_temp->clone();
        blearner_best = blearner_temp;
      }
    }

    #pragma omp critical
    {
      // return best blearner per core with corresponding ssq to master:
      best_blearner_map[ssq_best] = blearner_best;
    }
  }
  if (best_blearner_map.size() == 1) {
    return best_blearner_map.begin()->second;
  } else {
    auto it = min_element(best_blearner_map.begin(), best_blearner_map.end(),
      [](decltype(best_blearner_map)::value_type& l, decltype(best_blearner_map)::value_type& r) -> bool { return l.first < r.first; });
    return it->second;
  }
}

arma::mat OptimizerCoordinateDescent::calculateUpdate (const double learning_rate, const double step_size,
  const arma::mat& blearner_pred, const std::map<std::string, std::shared_ptr<data::Data>>& oob_data, const std::shared_ptr<response::Response>& sh_ptr_oob_response) const
{
  return learning_rate * step_size * blearner_pred;
}

void OptimizerCoordinateDescent::optimize (const unsigned int actual_iteration, const double learning_rate, const std::shared_ptr<loss::Loss>& sh_ptr_loss, const std::shared_ptr<response::Response>& sh_ptr_response,
  blearnertrack::BaselearnerTrack& blearner_track, const blearnerlist::BaselearnerFactoryList& factory_list)
{
  std::string temp_string = std::to_string(actual_iteration);
  auto sh_ptr_blearner_selected = findBestBaselearner(temp_string, sh_ptr_response, factory_list.getFactoryMap());

  // Prediction is needed more often, use a temp vector to avoid multiple computations:
  arma::mat blearner_pred_temp = sh_ptr_blearner_selected->predict();
  calculateStepSize(sh_ptr_loss, sh_ptr_response, blearner_pred_temp);

  // Insert new base-learner to vector of selected base-learner. The parameter are estimated here, hence
  // the contribution to the old parameter is the estimated parameter times the learning rate times
  // the step size. Therefore we have to pass the step size which changes in each iteration:
  blearner_track.insertBaselearner(sh_ptr_blearner_selected, getStepSize(actual_iteration));
  sh_ptr_response->updatePrediction(learning_rate * getStepSize(actual_iteration) * blearner_pred_temp);
}

void OptimizerCoordinateDescent::calculateStepSize (const std::shared_ptr<loss::Loss>& sh_ptr_loss, const std::shared_ptr<response::Response>& sh_ptr_response,
  const arma::vec& baselearner_prediction)
{
  // This function does literally nothing!
}

std::vector<double> OptimizerCoordinateDescent::getStepSize () const
{
  return _step_sizes;
}

double OptimizerCoordinateDescent::getStepSize (const unsigned int actual_iteration) const
{
  return 1;
}


// OptimizerCoordinateDescentLineSearch:
// ---------------------------------------------------

OptimizerCoordinateDescentLineSearch::OptimizerCoordinateDescentLineSearch ()
{
  _step_sizes.clear();
}

OptimizerCoordinateDescentLineSearch::OptimizerCoordinateDescentLineSearch  (const unsigned int num_threads)
  : OptimizerCoordinateDescent::OptimizerCoordinateDescent ( num_threads )
{
  _step_sizes.clear();
}

void OptimizerCoordinateDescentLineSearch::calculateStepSize (const std::shared_ptr<loss::Loss>& sh_ptr_loss, const std::shared_ptr<response::Response>& sh_ptr_response,
  const arma::vec& baselearner_prediction)
{
  _step_sizes.push_back(linesearch::findOptimalStepSize(sh_ptr_loss, sh_ptr_response->getResponse(), sh_ptr_response->getPredictionScores(), baselearner_prediction));
}

std::vector<double> OptimizerCoordinateDescentLineSearch::getStepSize () const
{
  return _step_sizes;
}

double OptimizerCoordinateDescentLineSearch::getStepSize (const unsigned int iteration) const
{
  if (_step_sizes.size() < iteration) {
    std::string msg = "Requested iteration " + std::to_string(iteration) + " is greater than the already trained iterations " + std::to_string(_step_sizes.size()) +".";
    Rcpp::stop(msg);
  }
  // Subtract 1 since the actual iteration starts counting with 1 and the step sizes with 0:
  return _step_sizes[iteration - 1];
}


// OptimizerAGBM:
// ---------------------------------------------------

OptimizerAGBM::OptimizerAGBM () : _momentum ( .0 ) { }

OptimizerAGBM::OptimizerAGBM (const double momentum) : _momentum ( momentum )
{
  _step_sizes.assign(1, 1.0);
}

OptimizerAGBM::OptimizerAGBM (const double momentum, const unsigned int num_threads)
  : Optimizer::Optimizer ( num_threads ),
    _momentum            ( momentum )
{
  _step_sizes.assign(1, 1.0);
}

OptimizerAGBM::OptimizerAGBM (const double momentum, const unsigned int acc_iters, const unsigned int num_threads)
  : Optimizer::Optimizer ( num_threads ),
    _momentum            ( momentum ),
    _acc_iters           ( acc_iters )
{
  _step_sizes.assign(1, 1.0);
}

std::shared_ptr<blearner::Baselearner> OptimizerAGBM::findBestBaselearner (std::string iteration_id,
    const std::shared_ptr<response::Response>& sh_ptr_response, const blearner_factory_map& factory_map) const
{
  throw std::logic_error("The use of 'findBestBaselearner' is just allowed with pseudo residuals for the AGBM optimizer!");

  return std::shared_ptr<blearner::Baselearner>();
}

std::shared_ptr<blearner::Baselearner> OptimizerAGBM::findBestBaselearner (std::string iteration_id,
    const arma::mat& pr, const blearner_factory_map& factory_map) const
{
  std::map<double, std::shared_ptr<blearner::Baselearner>> best_blearner_map;

  /* ****************************************************************************************
   * OLD SEQUENTIAL LOOP:
   */

  // Use ordinary sequential loop if just one thread should be used. This saves the costs
  // of distributing data etc. and results in a significant speed up:
  if (_num_threads == 1) {
    double ssq_temp;
    double ssq_best = std::numeric_limits<double>::infinity();

    std::shared_ptr<blearner::Baselearner> blearner_temp;
    std::shared_ptr<blearner::Baselearner> blearner_best;

    for (auto& it : factory_map) {

      // Paste string identifier for new base-learner:
      std::string id = "(" + iteration_id + ") " + it.second->getBaselearnerType();
      // Create new base-learner out of the actual factory (just the
      // pointer is overwritten):
      blearner_temp = it.second->createBaselearner();
      blearner_temp->train(pr);
      ssq_temp = helper::calculateSumOfSquaredError(pr, blearner_temp->predict());

      // Check if SSE of new temporary base-learner is smaller then SSE of the best
      // base-learner. If so, assign the temporary base-learner with the best
      // base-learner (This is always triggered within the first iteration since
      // ssq_best is declared as infinity):
      if (ssq_temp < ssq_best) {
        ssq_best = ssq_temp;
        // // Deep copy since the temporary base-learner is deleted every time which
        // // will also deletes the data for the best base-learner if we don't copy
        // // the whole data of the object:
        // blearner_best = blearner_temp->clone();
        blearner_best = blearner_temp;
      }
    }
    return blearner_best;
  }
  /* **************************************************************************************** */
  #pragma omp parallel num_threads(_num_threads) default(none) shared(iteration_id, pr, factory_map, best_blearner_map)
  {
    // private per core:
    double ssq_temp;
    double ssq_best = std::numeric_limits<double>::infinity();

    std::shared_ptr<blearner::Baselearner> blearner_temp;
    std::shared_ptr<blearner::Baselearner> blearner_best;

    #pragma omp for schedule(dynamic)
    for (unsigned int i = 0; i < factory_map.size(); i++) {

      // increment iterator to "index map elements by index" (https://stackoverflow.com/questions/8848870/use-openmp-in-iterating-over-a-map):
      auto it_factory_pair = factory_map.begin();
      std::advance(it_factory_pair, i);

      // Paste string identifier for new base-learner:
      std::string id = "(" + iteration_id + ") " + it_factory_pair->second->getBaselearnerType();

      // Create new base-learner out of the actual factory (just the
      // pointer is overwritten):
      blearner_temp = it_factory_pair->second->createBaselearner();
      blearner_temp->train(pr);
      ssq_temp = helper::calculateSumOfSquaredError(pr, blearner_temp->predict());

      // Check if SSE of new temporary base-learner is smaller then SSE of the best
      // base-learner. If so, assign the temporary base-learner with the best
      // base-learner (This is always triggered within the first iteration since
      // ssq_best is declared as infinity):
      if (ssq_temp < ssq_best) {
        ssq_best = ssq_temp;
        // // Deep copy since the temporary base-learner is deleted every time which
        // // will also deletes the data for the best base-learner if we don't copy
        // // the whole data of the object:
        // blearner_best = blearner_temp->clone();
        blearner_best = blearner_temp;
      }
    }

    #pragma omp critical
    {
      // return best blearner per core with corresponding ssq to master:
      best_blearner_map[ssq_best] = blearner_best;
    }
  }
  if (best_blearner_map.size() == 1) {
    return best_blearner_map.begin()->second;
  } else {
    auto it = min_element(best_blearner_map.begin(), best_blearner_map.end(),
      [](decltype(best_blearner_map)::value_type& l, decltype(best_blearner_map)::value_type& r) -> bool { return l.first < r.first; });
    return it->second;
  }
}

void OptimizerAGBM::optimize (const unsigned int actual_iteration, const double learning_rate, const std::shared_ptr<loss::Loss>& sh_ptr_loss, const std::shared_ptr<response::Response>& sh_ptr_response,
  blearnertrack::BaselearnerTrack& blearner_track, const blearnerlist::BaselearnerFactoryList& factory_list)
{
  arma::mat prediction_scores = sh_ptr_response->getPredictionScores();
  double weight_param = 2.0 / ((double)actual_iteration + 1.0);
  if (actual_iteration == 1) {
    _pred_momentum = prediction_scores;
    _pred_aggr     = _pred_momentum;
  } else {
    _pred_aggr = (1 - weight_param) * sh_ptr_response->getPredictionScores() + weight_param * _pred_momentum;

  }

  std::string temp_string;
  arma::mat pr_aggr = sh_ptr_loss->calculatePseudoResiduals(sh_ptr_response->getResponse(), _pred_aggr);

  // Find best base-learner w.r.t. pr_aggr
  temp_string = std::to_string(actual_iteration);
  auto sh_ptr_blearner_selected = findBestBaselearner(temp_string, pr_aggr, factory_list.getFactoryMap());
  blearner_track.insertBaselearner(sh_ptr_blearner_selected, getStepSize(actual_iteration));
  sh_ptr_response->updatePrediction(-prediction_scores + _pred_aggr + learning_rate * sh_ptr_blearner_selected->predict());

      //Do the same for corrected pseudo residuals
  if (actual_iteration == 1) {
    _pr_corr = pr_aggr;
  } else {
    _pr_corr = pr_aggr + (double)actual_iteration / ((double)actual_iteration + 1) * (_pr_corr - _momentum_blearnertrack.getBaselearnerVector().at(actual_iteration - 2)->predict());
;
  }

  // Find best base-learner w.r.t. _pr_corr
  auto sh_ptr_blearner_mom = findBestBaselearner(temp_string, _pr_corr, factory_list.getFactoryMap());
  //_momentum_blearner.push_back(sh_ptr_blearner_mom);

    std::string insert_id = sh_ptr_blearner_mom->getDataIdentifier() + "_" + sh_ptr_blearner_mom->getBaselearnerType();

  // Update momentum model
  double lr_mom  = _momentum * learning_rate / weight_param;
  if (actual_iteration > _acc_iters) lr_mom = 0;

  //std::cout << "Momentum param from optimize: " << sh_ptr_blearner_mom->getDataIdentifier() << lr_mom * sh_ptr_blearner_mom->getParameter();
  _pred_momentum = _pred_momentum + lr_mom * sh_ptr_blearner_mom->predict();
  //std::cout << "Insert momentum blearner";
  _momentum_blearnertrack.insertBaselearner(sh_ptr_blearner_mom, lr_mom);

  // Aggregate:
  //_pred_aggr = (1 - weight_param) * sh_ptr_response->getPredictionScores() + weight_param * _pred_momentum;

  updateAggrParameter(sh_ptr_blearner_selected, learning_rate, weight_param, blearner_track);
  //for (auto ita = _aggr_parameter_map.begin(); ita != _aggr_parameter_map.end(); ++ita) {
    //std::cout << ita->first << ita->second << std::endl;
  //}
}



arma::mat OptimizerAGBM::calculateUpdate (const double learning_rate, const double step_size,
  const arma::mat& blearner_pred, const std::map<std::string, std::shared_ptr<data::Data>>& oob_data, const std::shared_ptr<response::Response>& sh_ptr_oob_response) const
{
  unsigned int m = _momentum_blearnertrack.getBaselearnerVector().size();
  std::shared_ptr<blearner::Baselearner> momentum_blearner = _momentum_blearnertrack.getBaselearnerVector().at(m-1);

  arma::mat pred_scores = sh_ptr_oob_response->getPredictionScores();
  if (m == 1) {
    sh_ptr_oob_response->setPredictionScoresTemp2(pred_scores); // oob momentum sequence
  }

  double weight_param = 2.0 / ((double)m + 1.0);
  double lr_momentum  = _momentum * learning_rate / weight_param;

  // Predict test data for last momentum base-learner. Check, whether the data object is present or not:
  std::string data_id = momentum_blearner->getDataIdentifier();
  auto  it_oob_data = oob_data.find(data_id);
  if (it_oob_data != oob_data.end()) {
    std::shared_ptr<data::Data> oob_blearner_data = it_oob_data->second;

    // Predict this data using the selected baselearner:
    arma::mat temp_oob_prediction = momentum_blearner->predict(oob_blearner_data);
    sh_ptr_oob_response->setPredictionScoresTemp2(sh_ptr_oob_response->getPredictionScoresTemp2() + lr_momentum * temp_oob_prediction);
  }

  arma::mat aggr_pred = (1 - weight_param) * pred_scores + weight_param * sh_ptr_oob_response->getPredictionScoresTemp2();

  return -sh_ptr_oob_response->getPredictionScores() + aggr_pred + learning_rate * blearner_pred;
}

void OptimizerAGBM::calculateStepSize (const std::shared_ptr<loss::Loss>& sh_ptr_loss, const std::shared_ptr<response::Response>& sh_ptr_response,
  const arma::vec& baselearner_prediction)
{
  // This function does literally nothing!
}

double OptimizerAGBM::getStepSize (const unsigned int actual_iteration) const { return 1; }
std::vector<double> OptimizerAGBM::getStepSize () const {
  return _step_sizes;
}

std::map<std::string, arma::mat> OptimizerAGBM::getMomentumParameter () const
{
  //return _momentum_blearnertrack.getParameterMap();
  return _aggr_parameter_map;
}

std::vector<std::string> OptimizerAGBM::getSelectedMomentumBaselearner () const
{
  std::vector<std::string> out;
  std::vector<std::shared_ptr<blearner::Baselearner>> mom_blearnertrack = _momentum_blearnertrack.getBaselearnerVector();
  std::string id;
  for (unsigned int i = 0; i < mom_blearnertrack.size(); i++) {
    id = mom_blearnertrack.at(i)->getDataIdentifier() + "_" + mom_blearnertrack.at(i)->getBaselearnerType();
    out.push_back(id);
  }
  return out;
}


std::pair<std::vector<std::string>, arma::mat> OptimizerAGBM::getParameterMatrix () const { return _momentum_blearnertrack.getParameterMatrix(); }


void OptimizerAGBM::updateAggrParameter (std::shared_ptr<blearner::Baselearner>& sh_ptr_bl_new, double learning_rate, double weight_parameter, blearnertrack::BaselearnerTrack& blearner_track)
{
  // Add new base-learner to g to get f:
  // ------------------------------------------------
  std::string insert_id = sh_ptr_bl_new->getDataIdentifier() + "_" + sh_ptr_bl_new->getBaselearnerType();

  // Check if the baselearner is the first one. If so, the parameter
  // has to be instantiated with a zero matrix:
  std::map<std::string, arma::mat>::iterator it = _aggr_parameter_map.find(insert_id);

  // Prune parameter by multiplying it with the learning rate:
  arma::mat parameter_temp = learning_rate * sh_ptr_bl_new->getParameter();

  // Check if this is the first parameter entry:
  if (it == _aggr_parameter_map.end()) {

    // If this is the first entry, initialize it with zeros:
    arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
    _aggr_parameter_map.insert(std::pair<std::string, arma::mat>(insert_id, init_parameter));

  }
  // Accumulating parameter. If there is a nan, then this will be ignored and
  // the non nan entries are summed up:
  _aggr_parameter_map[ insert_id ] = parameter_temp + _aggr_parameter_map.find(insert_id)->second;

  blearner_track.setParameterMap(_aggr_parameter_map);

  // Shrink h and add to f:
  // ------------------------------------------------
  auto _mom_parameter_map = _momentum_blearnertrack.getParameterMap();
  for (it = _mom_parameter_map.begin(); it != _mom_parameter_map.end(); ++it) {
    //it->second = it->second * weight_parameter;
    insert_id  = it->first;

    // Check if the baselearner in h is the first one. If so, the parameter
    // has to be instantiated with a zero matrix:
    std::map<std::string, arma::mat>::iterator it_s = _aggr_parameter_map.find(insert_id);

    // Prune parameter by multiplying it with the learning rate:
    //arma::mat parameter_temp = it->second * weight_parameter;
    arma::mat parameter_temp = it->second;

    // Check if this is the first parameter entry:
    if (it_s == _aggr_parameter_map.end()) {

      // If this is the first entry, initialize it with zeros:
      arma::mat init_parameter(parameter_temp.n_rows, parameter_temp.n_cols, arma::fill::zeros);
      _aggr_parameter_map.insert(std::pair<std::string, arma::mat>(insert_id, init_parameter));

    }
    // Accumulating parameter. If there is a nan, then this will be ignored and
    // the non nan entries are summed up:
    _aggr_parameter_map[ insert_id ] = weight_parameter * parameter_temp + (1 - weight_parameter) * _aggr_parameter_map.find(insert_id)->second;
  }
}


} // namespace optimizer
