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

Response::Response (const std::string target_name, const std::string task_id,
  const arma::mat& response)
  : _target_name       ( target_name ),
    _task_id           ( task_id ),
    _response          ( response ),
    _pseudo_residuals  ( arma::mat(response.n_rows, response.n_cols, arma::fill::zeros) ),
    _prediction_scores ( arma::mat(response.n_rows, response.n_cols, arma::fill::zeros) )
{ }

Response::Response (const std::string target_name, const std::string task_id,
  const arma::mat& response, const arma::mat& weights)
  : _target_name       ( target_name ),
    _task_id           ( task_id ),
    _use_weights       ( true ),
    _response          ( response ),
    _weights           ( weights ),
    _pseudo_residuals  ( arma::mat(response.n_rows, response.n_cols, arma::fill::zeros) ),
    _prediction_scores ( arma::mat(response.n_rows, response.n_cols, arma::fill::zeros) )
{ }

void Response::setIteration (const unsigned int iter) { _iteration = iter; }

void Response::setPredictionScores (const arma::mat& scores, const unsigned int iter)
{
  _prediction_scores = scores;
  _iteration = iter;
}

void Response::setPredictionScoresTemp1 (const arma::mat& new_temp_scores)
{
  _prediction_scores_temp1 = new_temp_scores;
}

void Response::setPredictionScoresTemp2 (const arma::mat& new_temp_scores)
{
  _prediction_scores_temp2 = new_temp_scores;
}

std::string Response::getTargetName            () const { return _target_name; }
std::string Response::getTaskIdentifier        () const { return _task_id; }
arma::mat   Response::getResponse              () const { return _response; }
arma::mat   Response::getWeights               () const { return _weights; }
arma::mat   Response::getInitialization        () const { return _initialization; }
arma::mat   Response::getPseudoResiduals       () const { return _pseudo_residuals; }
arma::mat   Response::getPredictionScores      () const { return _prediction_scores; }
arma::mat   Response::getPredictionScoresTemp1 () const { return _prediction_scores_temp1; }
arma::mat   Response::getPredictionScoresTemp2 () const { return _prediction_scores_temp2; }


void Response::checkLossCompatibility (const std::shared_ptr<loss::Loss>& sh_ptr_loss) const
{
  if ((_task_id != sh_ptr_loss->getTaskId()) && (sh_ptr_loss->getTaskId() != "custom")) {
    std::string error_msg = "Loss task '" + sh_ptr_loss->getTaskId() + "' is not compatible with the response class task '" + _task_id + "'.";
    Rcpp::stop(error_msg);
  }
}


void Response::updatePseudoResiduals (const std::shared_ptr<loss::Loss>& sh_ptr_loss)
{
  checkLossCompatibility(sh_ptr_loss);
  if (_use_weights) {
    _pseudo_residuals = sh_ptr_loss->calculateWeightedPseudoResiduals(_response, _prediction_scores, _weights);
  } else {
    _pseudo_residuals = sh_ptr_loss->calculatePseudoResiduals(_response, _prediction_scores);
  }
}

void Response::updatePrediction (const arma::mat& update)
{
  _prediction_scores += update;
}

void Response::updatePrediction (const double learning_rate, const double step_size, const arma::mat& update)
{
  _prediction_scores += learning_rate * step_size * update;
}


void Response::constantInitialization (const std::shared_ptr<loss::Loss>& sh_ptr_loss)
{
  checkLossCompatibility(sh_ptr_loss);

  if (! _is_initialized) {
    if (_use_weights) {
      _initialization = sh_ptr_loss->weightedConstantInitializer(_response, _weights);
    } else {
      _initialization = sh_ptr_loss->constantInitializer(_response);
    }
    _is_initialized = true;
  } else {
    Rcpp::stop("Response is already initialized");
  }
}

void Response::constantInitialization (const arma::mat& init_mat)
{
  if (! _is_initialized) {
    _initialization = init_mat;
    _is_initialized = true;
  } else {
    Rcpp::stop("Response is already initialized");
  }
}


double Response::calculateEmpiricalRisk (const std::shared_ptr<loss::Loss>& sh_ptr_loss) const
{
  checkLossCompatibility(sh_ptr_loss);
  if (_use_weights) {
    //return sh_ptr_loss->calculateWeightedEmpiricalRisk(_response, getPredictionTransform(), _weights);
    return sh_ptr_loss->calculateWeightedEmpiricalRisk(_response, _prediction_scores, _weights);
  } else {
    return sh_ptr_loss->calculateEmpiricalRisk(_response, _prediction_scores);
  }
}

arma::mat Response::getPredictionTransform () const { return getPredictionTransform(_prediction_scores); }
arma::mat Response::getPredictionResponse  () const { return getPredictionResponse(_prediction_scores); }

// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

// ResponseRegression
// ------------------------------------

ResponseRegr::ResponseRegr (const std::string target_name, const arma::mat& response)
  : Response::Response ( target_name, "regression", response )
{ }

ResponseRegr::ResponseRegr (const std::string target_name, const arma::mat& response, const arma::mat& weights)
  : Response::Response ( target_name, "regression", response, weights )
{
  helper::checkMatrixDim(response, weights);
}

arma::mat ResponseRegr::calculateInitialPrediction (const arma::mat& response) const
{
  arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);

  if (! _is_initialized) {
     Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  }
  if (_initialization.n_rows > 1) {
    init = _initialization;
  } else {
    // Use just first element to correctly use .fill:
    init.fill(_initialization[0]);
  }
  return init;
}
//{
  //arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);

  //if (! _is_initialized) {
     //Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  //}
   ////Use just first element to correctly use .fill:
  //init.fill(_initialization[0]);
  //return init;
//}


void ResponseRegr::initializePrediction ()
{
  if (_is_initialized) {
    if (! _is_model_initialized) {
      _prediction_scores = calculateInitialPrediction(_response);
      _is_model_initialized = true;
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
  _response = _response.elem(idx);
  if (_use_weights) {
    _weights = _weights.elem(idx);
  }
  _pseudo_residuals  = _pseudo_residuals.elem(idx);
  _prediction_scores = _prediction_scores.elem(idx);
}


// ResponseBinaryClassif
// ------------------------------------

ResponseBinaryClassif::ResponseBinaryClassif (const std::string target_name, const std::string pos_class, const std::vector<std::string>& response)
  : Response::Response ( target_name, "binary_classif", helper::stringVecToBinaryVec(response, pos_class) ),
    _pos_class   ( pos_class ),
    _class_table ( helper::tableResponse(response) )
{
  helper::checkForBinaryClassif(response);
}

ResponseBinaryClassif::ResponseBinaryClassif (const std::string target_name, const std::string pos_class, const std::vector<std::string>& response, const arma::mat& weights)
  : Response::Response ( target_name, "binary_classif", helper::stringVecToBinaryVec(response, pos_class), weights ),
    _pos_class   ( pos_class ),
    _class_table ( helper::tableResponse(response) )
{
  helper::checkForBinaryClassif(response);
  helper::checkMatrixDim(_response, weights);
}

arma::mat ResponseBinaryClassif::calculateInitialPrediction (const arma::mat& response) const
{
  arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);

  if (! _is_initialized) {
     Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  }
  if (_initialization.n_rows > 1) {
    init = _initialization;
  } else {
    // Use just first element to correctly use .fill:
    init.fill(_initialization[0]);
  }
  return init;
}
//{
  //arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);

  //if (! _is_initialized) {
     //Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  //}
   ////Use just first element to correctly use .fill:
  //init.fill(_initialization[0]);
  //return init;
//}

void ResponseBinaryClassif::initializePrediction ()
{
  if (_is_initialized) {
    if (! _is_model_initialized) {
      _prediction_scores = calculateInitialPrediction(_response);
      _is_model_initialized = true;
    } else {
      Rcpp::stop("Prediction is already initialized.");
    }
  } else {
    Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
  }
}

arma::mat ResponseBinaryClassif::getPredictionTransform (const arma::mat& pred_scores) const
{
  return helper::sigmoid(pred_scores);
}

arma::mat ResponseBinaryClassif::getPredictionResponse (const arma::mat& pred_scores) const
{
  return helper::transformToBinaryResponse(getPredictionTransform(pred_scores), _threshold, 1, -1);
}

std::string ResponseBinaryClassif::getPositiveClass () const { return _pos_class; }
std::map<std::string, unsigned int> ResponseBinaryClassif::getClassTable () const { return _class_table; }

void ResponseBinaryClassif::filter (const arma::uvec& idx)
{
  _response = _response.elem(idx);
  if (_use_weights) {
    _weights = _weights.elem(idx);
  }
  _pseudo_residuals  = _pseudo_residuals.elem(idx);
  _prediction_scores = _prediction_scores.elem(idx);
}

void ResponseBinaryClassif::setThreshold (const double new_thresh)
{
  if ((new_thresh < 0) || (new_thresh > 1)) {
    Rcpp::stop("Threshold must be in [0,1]");
  }
  _threshold = new_thresh;
}

double ResponseBinaryClassif::getThreshold () const { return _threshold; }

// Functional Data Response
// ------------------------------------

// ResponseFDA::ResponseFDA (const std::string& target_name0, const arma::mat& response0, const arma::mat& grid0)
// {
//   target_name = target_name0;
//   response = response0;
//   task_id = "regression"; // set parent
//   arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
//   prediction_scores = temp_mat; // set parent
//   pseudo_residuals = temp_mat;  // set parent
//   // FDA specifics
//   grid = grid0;
//   arma::mat temp_mat_1(response.n_rows, response.n_cols, arma::fill::ones);
//   weights = temp_mat_1;
//   trapez_weights = tensors::trapezWeights(grid0);
// }
//
// ResponseFDA::ResponseFDA (const std::string& target_name0, const arma::mat& response0, const arma::mat& weights0, const arma::mat& grid0)
// {
//   helper::checkMatrixDim(response0, weights0);
//   target_name = target_name0;
//   response = response0;
//   weights = weights0;
//   task_id = "regression"; // set parent
//   arma::mat temp_mat(response.n_rows, response.n_cols, arma::fill::zeros);
//   prediction_scores = temp_mat; // set parent
//   pseudo_residuals = temp_mat;  // set parent
//   // FDA specifics
//   grid = grid0;
//   trapez_weights = tensors::trapezWeights(grid0);
// }
//
// arma::mat ResponseFDA::calculateInitialPrediction (const arma::mat& response) const
// {
//   arma::mat init(response.n_rows, response.n_cols, arma::fill::zeros);
//
//   if (! _is_initialized) {
//     Rcpp::stop("Response is not initialized, call 'constantInitialization()' first.");
//   }
//   // Use just first element to correctly use .fill:
//   init.fill(initialization[0]);
//   return init;
// }
//
// void ResponseFDA::initializePrediction ()
// {
//   if (_is_initialized) {
//     if (! _is_model_initialized) {
//       prediction_scores = calculateInitialPrediction(response);
//       _is_model_initialized = true;
//     } else {
//       Rcpp::stop("Prediction is already initialized.");
//     }
//   } else {
//     Rcpp::stop("Initialize constant initialization first by calling 'constantInitialization()'.");
//   }
// }
//
// void ResponseFDA::updatePseudoResiduals (std::shared_ptr<loss::Loss> sh_ptr_loss)
// {
//   checkLossCompatibility(sh_ptr_loss);
//   weights = weights.each_row() % trapez_weights.t();
//   pseudo_residuals = sh_ptr_loss->calculateWeightedPseudoResiduals(response, prediction_scores, weights);
// }
//
//
// arma::mat ResponseFDA::getPredictionTransform (const arma::mat& pred_scores) const
// {
//   // No transformation is done in regression
//   return pred_scores;
// }
//
// arma::mat ResponseFDA::getPredictionResponse (const arma::mat& pred_scores) const
// {
//   return pred_scores;
// }
//
// void ResponseFDA::filter (const arma::uvec& idx)
// {
//   response = response.elem(idx);
//   if (_use_weights) {
//     weights = weights.elem(idx);
//   }
//   pseudo_residuals = pseudo_residuals.elem(idx);
//   prediction_scores = prediction_scores.elem(idx);
// }

} // namespace response
