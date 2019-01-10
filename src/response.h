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

#ifndef RESPONSE_H_
#define RESPONSE_H_

#include "RcppArmadillo.h"
#include "loss.h"
#include "helper.h"

namespace response
{

// -------------------------------------------------------------------------- //
// Abstract 'Response' class:
// -------------------------------------------------------------------------- //

class Response
{
protected:
  std::string target_name;
  std::string task_id;
  arma::mat response;
  arma::mat weights;
  arma::mat initialization;
  arma::mat pseudo_residuals;
  arma::mat prediction_scores;
  arma::mat actual_prediction_scores;

  unsigned int actual_iteration = 0;
  bool is_initialization_initialized = false;
  bool is_model_initialized = false;

public:

  Response ();

  void setActualIteration (const unsigned int&);
  void setActualPredictionScores (const arma::mat&, const unsigned int&);

  std::string getTargetName () const;
  std::string getTaskIdentifier () const;
  arma::mat getResponse () const;
  arma::mat getWeights () const;
  arma::mat getInitialization () const;
  arma::mat getPseudoResiduals () const;
  arma::mat getPredictionScores () const;

  void checkLossCompatibility (loss::Loss*) const;

  void updatePseudoResiduals (loss::Loss*);
  void updatePrediction (const double&, const double&, const arma::mat&);

  void constantInitialization (loss::Loss*);
  virtual arma::mat calculateInitialPrediction (loss::Loss*, const arma::mat&) const = 0;
  virtual void initializePrediction (loss::Loss*) = 0;
  arma::mat getPredictionTransform () const;
  virtual arma::mat getPredictionTransform (const arma::mat&) const = 0;
  arma::mat getPredictionResponse () const;
  virtual arma::mat getPredictionResponse (const arma::mat&) const = 0;

  double calculateEmpiricalRisk (loss::Loss*) const;

  virtual ~Response () { };
};


// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

class ResponseRegr : public Response
{

public:
  ResponseRegr (const std::string&, const arma::mat&);
  ResponseRegr (const std::string&, const arma::mat&, const arma::mat&);

  arma::mat calculateInitialPrediction (loss::Loss*, const arma::mat&) const;
  void initializePrediction (loss::Loss*);
  arma::mat getPredictionTransform (const arma::mat&) const;
  arma::mat getPredictionResponse (const arma::mat&) const;
};

class ResponseBinaryClassif : public Response
{
public:
  double threshold = 0.5;

  ResponseBinaryClassif (const std::string&, const arma::mat&);
  ResponseBinaryClassif (const std::string&, const arma::mat&, const arma::mat&);

  arma::mat calculateInitialPrediction (loss::Loss*, const arma::mat&) const;
  void initializePrediction (loss::Loss*);
  arma::mat getPredictionTransform (const arma::mat&) const;
  arma::mat getPredictionResponse (const arma::mat&) const;

  void setThreshold (const double&);
};


// class ResponseFDA : public Response
// {
// public:

// };


} // namespace response

#endif // RESPONSE_H_