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

std::string Response::getTaskIdentifier () const
{
  return task_id;
}

void Response::setActualIteration(const unsigned int& actual_iter)
{
  actual_iteration = actual_iter;
}

double Response::getInitialization ()
{
  return initialization;
}

arma::mat Response::getResponse () const { return response; }
arma::mat Response::getWeights () const { return weights; }
arma::mat Response::getPrediction () const { return prediction; }

double Response::getEmpiricalRisk ()
{
  return arma::accu(used_loss->definedLoss(response, prediction)) / response.size();
}

// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

ResponseRegr::ResponseRegr (const arma::mat& response, loss::Loss* used_loss0) : response ( response )
{
  task_id = "regression";
  used_loss = used_loss0;

  initialization = used_loss->constantInitializer(response);

  arma::mat temp(response.n_rows, response.n_cols, arma::fill::zeros);
  
  prediction = temp;
  prediction.fill(initialization);
  
  pseudo_residuals = temp;
}

arma::mat ResponseRegr::getPseudoResiduals () const { return pseudo_residuals; }
void updatePseudoResiduals ()
{
  pseudo_residuals = -used_loss->definedGradient(response, prediction);
}

void ResponseRegr::updatePrediction (const double& learning_rate, const double& step_size, const arma::mat& update) 
{
  prediction += learning_rate * step_size * update;
}

arma::mat ResponseRegr::responseTransformation (const arma::mat& prediction) const {}

arma::mat ResponseRegr::getPrediction (const bool& as_response) const
{
  return getPrediction();
}

} // namespace response