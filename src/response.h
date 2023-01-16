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
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

namespace response
{

// -------------------------------------------------------------------------- //
// Abstract 'Response' class:
// -------------------------------------------------------------------------- //

class Response
{
protected:
  const std::string _target_name;
  const std::string _task_id;
  const bool        _use_weights = false;

  arma::mat _response;
  arma::mat _weights;
  arma::mat _initialization;

  arma::mat _pseudo_residuals;
  arma::mat _prediction_scores;
  // Buffer for additional stuff like for AGBM:
  arma::mat _prediction_scores_temp1;
  arma::mat _prediction_scores_temp2;

  unsigned int _iteration = 0;
  bool         _is_initialized = false;
  bool         _is_model_initialized = false;

  Response (const std::string, const std::string, const arma::mat&);
  Response (const std::string, const std::string, const arma::mat&, const arma::mat&);
  Response (const json&);

public:
  Response ();

  // Virtual methods
  virtual void      initializePrediction       ()                       = 0;
  virtual void      filter                     (const arma::uvec&)      = 0;
  virtual arma::mat calculateInitialPrediction (const arma::mat&) const = 0;
  virtual arma::mat getPredictionTransform     (const arma::mat&) const = 0;
  virtual arma::mat getPredictionResponse      (const arma::mat&) const = 0;
  virtual json      toJson                     ()                 const = 0;

  // Setter/Getter
  void setIteration             (const unsigned int);
  void setPredictionScores      (const arma::mat&, const unsigned int);
  void setPredictionScoresTemp1 (const arma::mat&);
  void setPredictionScoresTemp2 (const arma::mat&);

  std::string getTargetName            () const;
  std::string getTaskIdentifier        () const;
  arma::mat   getResponse              () const;
  arma::mat   getWeights               () const;
  arma::mat   getInitialization        () const;
  arma::mat   getPseudoResiduals       () const;
  arma::mat   getPredictionScores      () const;
  arma::mat   getPredictionTransform   () const;
  arma::mat   getPredictionResponse    () const;
  arma::mat   getPredictionScoresTemp1 () const;
  arma::mat   getPredictionScoresTemp2 () const;
  // Other methods
  void checkLossCompatibility (const std::shared_ptr<loss::Loss>&) const;
  void updatePseudoResiduals  (const std::shared_ptr<loss::Loss>&);
  void updatePrediction       (const arma::mat&);
  void updatePrediction       (const double, const double, const arma::mat&);
  void constantInitialization (const std::shared_ptr<loss::Loss>&);
  void constantInitialization (const arma::mat&);

  json baseToJson (const std::string) const;

  double calculateEmpiricalRisk (const std::shared_ptr<loss::Loss>&) const;

  // Destructor
  virtual ~Response () { };
};

std::shared_ptr<Response> jsonToResponse (const json&);


// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

// ResponseRegression
// ------------------------------------

class ResponseRegr : public Response
{
public:
  ResponseRegr (const std::string, const arma::mat&);
  ResponseRegr (const std::string, const arma::mat&, const arma::mat&);
  ResponseRegr (const json&);

  void      initializePrediction       ();
  void      filter                     (const arma::uvec&);
  arma::mat calculateInitialPrediction (const arma::mat&) const;
  arma::mat getPredictionTransform     (const arma::mat&) const;
  arma::mat getPredictionResponse      (const arma::mat&) const;
  json      toJson                     ()                 const;
};


// ResponseBinaryClassif
// ------------------------------------

class ResponseBinaryClassif : public Response
{
private:
  double                                    _threshold = 0.5;
  const std::string                         _pos_class;
  const std::map<std::string, unsigned int> _class_table;

public:
  ResponseBinaryClassif (const std::string, const std::string, const std::vector<std::string>&);
  ResponseBinaryClassif (const std::string, const std::string, const std::vector<std::string>&, const arma::mat&);
  ResponseBinaryClassif (const json&);

  void      initializePrediction       ();
  void      filter                     (const arma::uvec&);
  arma::mat calculateInitialPrediction (const arma::mat&) const;
  arma::mat getPredictionTransform     (const arma::mat&) const;
  arma::mat getPredictionResponse      (const arma::mat&) const;
  json      toJson                     ()                 const;

  void                                setThreshold     (const double);
  double                              getThreshold     () const;
  std::string                         getPositiveClass () const;
  std::map<std::string, unsigned int> getClassTable    () const;
};

// -----------------------------------------------------------------------------------------------------------------
// class ResponseFDA : public Response
// {
//
// public:
//   arma::mat grid;
//   arma::mat trapez_weights;
//
//   ResponseFDA (const std::string&, const arma::mat&, const arma::mat&);
//   ResponseFDA (const std::string&, const arma::mat&, const arma::mat&, const arma::mat&);
//
//   arma::mat calculateInitialPrediction (const arma::mat&) const;
//   void initializePrediction ();
//   void updatePseudoResiduals (std::shared_ptr<loss::Loss>);
//   arma::mat getPredictionTransform (const arma::mat&) const;
//   arma::mat getPredictionResponse (const arma::mat&) const;
//   void filter (const arma::uvec&);
// };


} // namespace response

#endif // RESPONSE_H_
