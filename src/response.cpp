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

#include "response.h"

namespace response
{

// -------------------------------------------------------------------------- //
// Abstract 'Response' class:
// -------------------------------------------------------------------------- //

Response::Response () {}


void Response::setActualIteration (const unsigned int& actual_iter) { actual_iteration = actual_iter; }
void Response::setActualPredictionScores (const arma::mat& new_prediction_scores, const unsigned int& actual_iter)
{
  prediction_scores = new_prediction_scores;
  actual_iteration = actual_iter;
}

std::vector<std::string> Response::getTargetName () { return target_name; }
std::string Response::getTaskIdentifier () const { return task_id; }
arma::mat Response::getResponse () const { return response; }
arma::mat Response::getWeights () const { return weights; }
arma::mat Response::getInitialization () const { return initialization; }
arma::mat Response::getPseudoResiduals () const { return pseudo_residuals; }
arma::mat Response::getPredictionScores () const { return prediction_scores; }


void Response::checkLossCompatibility (std::shared_ptr<loss::Loss> sh_ptr_loss) const
{
  if ((task_id != sh_ptr_loss->getTaskId()) && (sh_ptr_loss->getTaskId() != "custom")) {
    std::string error_msg = "Loss task '" + sh_ptr_loss->getTaskId() + "' is not compatible with the response class task '" + task_id + "'.";
    Rcpp::stop(error_msg);
  }
}


void Response::updatePseudoResiduals (std::shared_ptr<loss::Loss> sh_ptr_loss)
{
  checkLossCompatibility(sh_ptr_loss);
  if (use_weights) {
    pseudo_residuals = sh_ptr_loss->calculateWeightedPseudoResiduals(response, prediction_scores, weights);
  } else {
    pseudo_residuals = sh_ptr_loss->calculatePseudoResiduals(response, prediction_scores);
  }
}

void Response::updatePrediction (const double& learning_rate, const double& step_size, const arma::mat& update)
{
  prediction_scores += learning_rate * step_size * update;
}


void Response::constantInitialization (std::shared_ptr<loss::Loss> sh_ptr_loss)
{
  checkLossCompatibility(sh_ptr_loss);

  if (! is_initialization_initialized) {
    if (use_weights) {
      initialization = sh_ptr_loss->weightedConstantInitializer(response, weights);
    } else {
      initialization = sh_ptr_loss->constantInitializer(response);
    }
    is_initialization_initialized = true;
  } else {
    Rcpp::stop("Constant initialization is already initialized.");
  }
}

void Response::constantInitialization (const arma::mat& init_mat)
{
  if (! is_initialization_initialized) {
    initialization = init_mat;
    is_initialization_initialized = true;
  } else {
    Rcpp::stop("Constant initialization is already initialized.");
  }
}


double Response::calculateEmpiricalRisk (std::shared_ptr<loss::Loss> sh_ptr_loss) const
{
  checkLossCompatibility(sh_ptr_loss);
  if (use_weights) {
    return sh_ptr_loss->calculateWeightedEmpiricalRisk(response, getPredictionTransform(), weights);
  } else {
    return sh_ptr_loss->calculateEmpiricalRisk(response, getPredictionTransform());
  }
}

arma::mat Response::getPredictionTransform () const
{
  return getPredictionTransform(prediction_scores);
}

arma::mat Response::getPredictionResponse () const
{
  return getPredictionResponse(prediction_scores);
}

// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

// Regression

ResponseRegr::ResponseRegr (std::vector<std::string>& target_name0, const arma::mat& response0)
{
  target_name = target_name0;
  response = response0;
  task_id = "regression"; // set parent
  arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
  prediction_scores = temp_mat; // set parent
  pseudo_residuals = temp_mat;  // set parent
}

ResponseRegr::ResponseRegr (std::vector<std::string>& target_name0, const arma::mat& response0, const arma::mat& weights0)
{
  helper::checkMatrixDim(response0, weights0);
  target_name = target_name0;
  response = response0;
  weights = weights0;
  use_weights = true;
  task_id = "regression"; // set parent
  arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
  prediction_scores = temp_mat; // set parent
  pseudo_residuals = temp_mat;  // set parent
}

arma::mat ResponseRegr::calculateInitialPrediction (const arma::mat& response) const
{
  arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);

  if (! is_initialization_initialized) {
     Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  }
  // Use just first element to correctly use .fill:
  init.fill(initialization[0]);
  return init;
}

void ResponseRegr::initializePrediction ()
{
  if (is_initialization_initialized) {
    if (! is_model_initialized) {
      prediction_scores = calculateInitialPrediction(response);
      is_model_initialized = true;
    } else {
      Rcpp::stop("Prediction is already initialized.");
    }
  } else {
    Rcpp::stop("Initialize constant initialization first by calling 'constantInitialization()'.");
  }
}


arma::mat ResponseRegr::getPredictionTransform (const arma::mat& pred_scores) const
{
  // No transformation is done in regression
  return pred_scores;
}

arma::mat ResponseRegr::getPredictionResponse (const arma::mat& pred_scores) const
{
  return pred_scores;
}

void ResponseRegr::filter (const arma::uvec& idx)
{
  response = response.elem(idx);
  if (use_weights) {
    weights = weights.elem(idx);
  }
  pseudo_residuals = pseudo_residuals.elem(idx);
  prediction_scores = prediction_scores.elem(idx);
}


// Binary Classification

ResponseBinaryClassif::ResponseBinaryClassif (std::vector<std::string>& target_name0, const arma::mat& response0)
{
  helper::checkForBinaryClassif(response0, -1, 1);
  target_name = target_name0;
  response = response0;
  task_id = "binary_classif"; // set parent
  arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
  prediction_scores = temp_mat; // set parent
  pseudo_residuals = temp_mat;  // set parent
}

ResponseBinaryClassif::ResponseBinaryClassif (std::vector<std::string>& target_name0, const arma::mat& response0, const arma::mat& weights0)
{
  helper::checkForBinaryClassif(response0, -1, 1);
  helper::checkMatrixDim(response0, weights0);
  target_name = target_name0;
  response = response0;
  weights = weights0;
  use_weights = true;
  task_id = "binary_classif"; // set parent
  arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
  prediction_scores = temp_mat; // set parent
  pseudo_residuals = temp_mat;  // set parent
}

arma::mat ResponseBinaryClassif::calculateInitialPrediction (const arma::mat& response) const
{
  arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);

  if (! is_initialization_initialized) {
     Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  }
  // Use just first element to correctly use .fill:
  init.fill(initialization[0]);
  return init;
}

void ResponseBinaryClassif::initializePrediction ()
{
  if (is_initialization_initialized) {
    if (! is_model_initialized) {
      prediction_scores = calculateInitialPrediction(response);
      is_model_initialized = true;
    } else {
      Rcpp::stop("Prediction is already initialized.");
    }
  } else {
    Rcpp::stop("Initialize constant initialization first by calling 'constantInitialization()'.");
  }
}

arma::mat ResponseBinaryClassif::getPredictionTransform (const arma::mat& pred_scores) const
{
  return helper::sigmoid(pred_scores);
}

arma::mat ResponseBinaryClassif::getPredictionResponse (const arma::mat& pred_scores) const
{
  return helper::transformToBinaryResponse(getPredictionTransform(pred_scores), threshold, 1, -1);
}

void ResponseBinaryClassif::filter (const arma::uvec& idx)
{
  response = response.elem(idx);
  if (use_weights) {
    weights = weights.elem(idx);
  }
  pseudo_residuals = pseudo_residuals.elem(idx);
  prediction_scores = prediction_scores.elem(idx);
}

void ResponseBinaryClassif::setThreshold (const double& new_thresh)
{
  if ((new_thresh < 0) || (new_thresh > 1)) {
    Rcpp::stop("Threshold must be element of [0,1]");
  }
  threshold = new_thresh;
}


// Functional Data Response

ResponseFDA::ResponseFDA (std::vector<std::string>& target_name0, const arma::mat& response0, const arma::mat& grid0)
{
  target_name = target_name0;
  response = response0;
  task_id = "regression"; // set parent
  arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
  prediction_scores = temp_mat; // set parent
  pseudo_residuals = temp_mat;  // set parent
  // FDA specifics
  grid = grid0;
  arma::mat temp_mat_1(response.n_rows, response.n_cols, arma::fill::ones);
  weights = temp_mat_1; 
  trapez_weights = tensors::trapezWeights(grid0);
  // vectorize
  // FIXME inefficient because feature are copied length(grid0)-times
  response = response.t();
  response.reshape(response.n_cols*response.n_rows,1);
  prediction_scores = response.t();
  prediction_scores.reshape(prediction_scores.n_cols*prediction_scores.n_rows,1);
  pseudo_residuals = pseudo_residuals.t();
  pseudo_residuals.reshape(pseudo_residuals.n_cols*pseudo_residuals.n_rows,1);
  weights = weights.t();
  weights.reshape(weights.n_cols*weights.n_rows,1);
  trapez_weights = trapez_weights.t();
  trapez_weights.reshape(trapez_weights.n_cols*trapez_weights.n_rows,1);
}

ResponseFDA::ResponseFDA (std::vector<std::string>& target_name0, const arma::mat& response0, const arma::mat& weights0, const arma::mat& grid0)
{
  helper::checkMatrixDim(response0, weights0);
  target_name = target_name0;
  response = response0;
  weights = weights0;
  task_id = "regression"; // set parent
  arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
  prediction_scores = temp_mat; // set parent
  pseudo_residuals = temp_mat;  // set parent
  // FDA specifics
  grid = grid0;
  trapez_weights = tensors::trapezWeights(grid0);
  // FIXME inefficient because feature are copied length(grid0)-times
  response = response.t();
  response.reshape(response.n_cols*response.n_rows,1);
  prediction_scores = response.t();
  prediction_scores.reshape(prediction_scores.n_cols*prediction_scores.n_rows,1);
  pseudo_residuals = pseudo_residuals.t();
  pseudo_residuals.reshape(pseudo_residuals.n_cols*pseudo_residuals.n_rows,1);
  weights = weights.t();
  weights.reshape(weights.n_cols*weights.n_rows,1);
  trapez_weights = trapez_weights.t();
  trapez_weights.reshape(trapez_weights.n_cols*trapez_weights.n_rows,1);
}

arma::mat ResponseFDA::calculateInitialPrediction (const arma::mat& response) const
{
  arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);
  
  if (! is_initialization_initialized) {
    Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  }
  // Use just first element to correctly use .fill:
  init.fill(initialization[0]);
  return init;
}

void ResponseFDA::initializePrediction ()
{
  if (is_initialization_initialized) {
    if (! is_model_initialized) {
      prediction_scores = calculateInitialPrediction(response);
      is_model_initialized = true;
    } else {
      Rcpp::stop("Prediction is already initialized.");
    }
  } else {
    Rcpp::stop("Initialize constant initialization first by calling 'constantInitialization()'.");
  }
}

void ResponseFDA::updatePseudoResiduals (std::shared_ptr<loss::Loss> sh_ptr_loss)
{
  checkLossCompatibility(sh_ptr_loss);
  weights = weights.each_row() % trapez_weights.t();
  pseudo_residuals = sh_ptr_loss->calculateWeightedPseudoResiduals(response, prediction_scores, weights);
}


arma::mat ResponseFDA::getPredictionTransform (const arma::mat& pred_scores) const
{
  // No transformation is done in regression
  return pred_scores;
}

arma::mat ResponseFDA::getPredictionResponse (const arma::mat& pred_scores) const
{
  return pred_scores;
}

arma::mat ResponseFDA::getGrid (const arma::mat& grid) const
{
  return grid;
}

void ResponseFDA::filter (const arma::uvec& idx)
{
  response = response.elem(idx);
  if (use_weights) {
    weights = weights.elem(idx);
  }
  pseudo_residuals = pseudo_residuals.elem(idx);
  prediction_scores = prediction_scores.elem(idx);
}

arma::mat ResponseFDA::getGrid () const { return grid; }


} // namespace response