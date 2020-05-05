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

std::shared_ptr<blearner::Baselearner> OptimizerCoordinateDescent::findBestBaselearner (const std::string iteration_id,
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
  const arma::mat& blearner_pred) const
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
  sh_ptr_response->updatePrediction(calculateUpdate(learning_rate, getStepSize(actual_iteration), blearner_pred_temp));
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

} // namespace optimizer
