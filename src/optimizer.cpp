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

#include "optimizer.h"

namespace optimizer {

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

// Destructor:
Optimizer::~Optimizer () {
  // Rcpp::Rcout << "Call Optimizer Destructor" << std::endl;
}

// -------------------------------------------------------------------------- //
// Optimizer implementations:
// -------------------------------------------------------------------------- //

// OptimizerCoordinateDescent:
// ---------------------------------------------------

OptimizerCoordinateDescent::OptimizerCoordinateDescent () {
  // Initialize step size vector as scalar:
  step_sizes.assign(1, 1.0);
}

std::shared_ptr<blearner::Baselearner> OptimizerCoordinateDescent::findBestBaselearner (const std::string& iteration_id,
  std::shared_ptr<response::Response> sh_ptr_response, const blearner_factory_map& my_blearner_factory_map) const
{
  double ssq_temp;
  double ssq_best = std::numeric_limits<double>::infinity();

  std::shared_ptr<blearner::Baselearner> blearner_temp;
  std::shared_ptr<blearner::Baselearner> blearner_best;

  for (auto& it : my_blearner_factory_map) {

    // Paste string identifier for new base-learner:
    std::string id = "(" + iteration_id + ") " + it.second->getBaselearnerType();

    // Create new base-learner out of the actual factory (just the
    // pointer is overwritten):
    blearner_temp = it.second->createBaselearner(id);
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

void OptimizerCoordinateDescent::calculateStepSize (std::shared_ptr<loss::Loss> sh_ptr_loss, std::shared_ptr<response::Response> sh_ptr_response,
  const arma::vec& baselearner_prediction)
{
  // This function does literally nothing!
}

std::vector<double> OptimizerCoordinateDescent::getStepSize () const
{
  return step_sizes;
}

double OptimizerCoordinateDescent::getStepSize (const unsigned int& actual_iteration) const
{
  return 1;
}


// OptimizerCoordinateDescentLineSearch:
// ---------------------------------------------------

OptimizerCoordinateDescentLineSearch::OptimizerCoordinateDescentLineSearch () { }


std::shared_ptr<blearner::Baselearner> OptimizerCoordinateDescentLineSearch::findBestBaselearner (const std::string& iteration_id,
  std::shared_ptr<response::Response> sh_ptr_response, const blearner_factory_map& my_blearner_factory_map) const
{
  double ssq_temp;
  double ssq_best = std::numeric_limits<double>::infinity();

  std::shared_ptr<blearner::Baselearner> blearner_temp;
  std::shared_ptr<blearner::Baselearner> blearner_best;

  for (auto& it : my_blearner_factory_map) {

    // Paste string identifier for new base-learner:
    std::string id = "(" + iteration_id + ") " + it.second->getBaselearnerType();

    // Create new base-learner out of the actual factory (just the
    // pointer is overwritten):
    blearner_temp = it.second->createBaselearner(id);
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

    // // Completely remove the temporary base-learner. This one isn't needed anymore:
    // delete blearner_temp;
  }
  return blearner_best;
}

void OptimizerCoordinateDescentLineSearch::calculateStepSize (std::shared_ptr<loss::Loss> sh_ptr_loss, std::shared_ptr<response::Response> sh_ptr_response,
  const arma::vec& baselearner_prediction)
{
  step_sizes.push_back(linesearch::findOptimalStepSize(sh_ptr_loss, sh_ptr_response->getResponse(), sh_ptr_response->getPredictionScores(), baselearner_prediction));
}

std::vector<double> OptimizerCoordinateDescentLineSearch::getStepSize () const
{
  return step_sizes;
}

double OptimizerCoordinateDescentLineSearch::getStepSize (const unsigned int& actual_iteration) const
{
  if (step_sizes.size() < actual_iteration) {
    Rcpp::stop("You cannot select a step size which is not trained!");
  }
  // Subtract 1 since the actual iteration starts counting with 1 and the step sizes with 0:
  return step_sizes[actual_iteration - 1];
}

} // namespace optimizer
