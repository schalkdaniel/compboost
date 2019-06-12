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
#include "tensors.h"

namespace response
{

// -------------------------------------------------------------------------- //
// Abstract 'Response' class:
// -------------------------------------------------------------------------- //

class Response
{
protected:
  std::vector<std::string> target_name;
  std::string task_id;
  arma::mat response;
  arma::mat weights;
  arma::mat initialization;
  arma::mat pseudo_residuals;
  arma::mat prediction_scores;

  unsigned int actual_iteration = 0;
  bool is_initialization_initialized = false;
  bool is_model_initialized = false;

public:

  Response ();

  void setActualIteration (const unsigned int&);
  void setActualPredictionScores (const arma::mat&, const unsigned int&);

  std::vector<std::string> getTargetName ();
  std::string getTaskIdentifier () const;
  arma::mat getResponse () const;
  arma::mat getWeights () const;
  arma::mat getInitialization () const;
  arma::mat getPseudoResiduals () const;
  arma::mat getPredictionScores () const;

  bool use_weights = false;

  void checkLossCompatibility (std::shared_ptr<loss::Loss>) const;

  void updatePseudoResiduals (std::shared_ptr<loss::Loss>);
  void updatePrediction (const double&, const double&, const arma::mat&);

  void constantInitialization (std::shared_ptr<loss::Loss>);
  void constantInitialization (const arma::mat&);
  virtual arma::mat calculateInitialPrediction (const arma::mat&) const = 0;
  virtual void initializePrediction () = 0;
  arma::mat getPredictionTransform () const;
  virtual arma::mat getPredictionTransform (const arma::mat&) const = 0;
  arma::mat getPredictionResponse () const;
  virtual arma::mat getPredictionResponse (const arma::mat&) const = 0;

  double calculateEmpiricalRisk (std::shared_ptr<loss::Loss>) const;

  virtual void filter (const arma::uvec&) = 0;

  virtual ~Response () { };
};


// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

class ResponseRegr : public Response
{

public:
  ResponseRegr (std::vector<std::string>&, const arma::mat&);
  ResponseRegr (std::vector<std::string>&, const arma::mat&, const arma::mat&);

  arma::mat calculateInitialPrediction (const arma::mat&) const;
  void initializePrediction ();
  arma::mat getPredictionTransform (const arma::mat&) const;
  arma::mat getPredictionResponse (const arma::mat&) const;
  void filter (const arma::uvec&);
};

class ResponseBinaryClassif : public Response
{
public:
  double threshold = 0.5;

  ResponseBinaryClassif (std::vector<std::string>&, const arma::mat&);
  ResponseBinaryClassif (std::vector<std::string>&, const arma::mat&, const arma::mat&);

  arma::mat calculateInitialPrediction (const arma::mat&) const;
  void initializePrediction ();
  arma::mat getPredictionTransform (const arma::mat&) const;
  arma::mat getPredictionResponse (const arma::mat&) const;

  void filter (const arma::uvec&);

  void setThreshold (const double&);
};

// -----------------------------------------------------------------------------------------------------------------
class ResponseFDA : public Response
{
  
public:
  arma::mat grid;
  arma::mat trapez_weights;
  arma::mat mean_offset;
  arma::mat median_offset;
  
  ResponseFDA (std::vector<std::string>&, const arma::mat&, const arma::mat&);
  ResponseFDA (std::vector<std::string>&, const arma::mat&, const arma::mat&, const arma::mat&);
  
  arma::mat calculateInitialPrediction (const arma::mat&) const;
  void initializePrediction ();
  void updatePseudoResiduals (std::shared_ptr<loss::Loss>);
  arma::mat getPredictionTransform (const arma::mat&) const;
  void constantInitialization (std::shared_ptr<loss::Loss>);
  arma::mat getPredictionResponse (const arma::mat&) const;
  arma::mat getGrid (const arma::mat&) const;

  void filter (const arma::uvec&);
  arma::mat getGrid () const;
};

// -----------------------------------------------------------------------------------------------------------------
class ResponseFDALong : public Response
{
  
public:
  arma::mat grid;
  arma::field<arma::mat> grid_field;
  arma::mat trapez_weights;
  
  ResponseFDALong (std::vector<std::string>&, arma::field<arma::mat>&, arma::field<arma::mat>&);
  ResponseFDALong (std::vector<std::string>&, arma::field<arma::mat>&, arma::field<arma::mat>&, arma::field<arma::mat>&);
  
  arma::mat calculateInitialPrediction (const arma::mat&) const;
  void initializePrediction ();
  void updatePseudoResiduals (std::shared_ptr<loss::Loss>);
  arma::mat getPredictionTransform (const arma::mat&) const;
  arma::mat getPredictionResponse (const arma::mat&) const;
  
  void filter (const arma::uvec&);
  arma::field<arma::mat> getGrid_field () const;
};


} // namespace response

#endif // RESPONSE_H_