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

namespace response 
{

// -------------------------------------------------------------------------- //
// Abstract 'Response' class:
// -------------------------------------------------------------------------- //

class Response
{
protected:
  // Response and weights as matrix
  arma::mat response;
  arma::mat weights;
  arma::mat pseudo_residuals;
  arma::mat prediction;

  std::string task_id;
  unsigned int actual_iteration = 0;
  loss::Loss* used_loss;
  double initialization;
  
public:
  
  Response ();
  
  std::string getTaskIdentifier () const = 0;
  void setActualIteration (const unsigned int&);
  double getInitialization ();
  
  arma::mat getResponse () const; 
  virtual arma::mat getWeights () const = 0;

  arma::mat getPseudoResiduals () const;
  virtual void updatePseudoResiduals () = 0; 

  arma::mat getPrediction () const;
  virtual void updatePrediction (const double&, const double&, const arma::mat&) = 0;
  virtual arma::mat getPrediction (bool) const = 0;
  
  virtual arma::mat responseTransformation (const arma::mat&) const = 0;

  double getEmpiricalRisk ();
    
  virtual ~Response () { };
};


// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

class ResponseRegr : public Response
{

public:

  ResponseRegr ();

  arma::mat getWeights () const;

  void updatePseudoResiduals (); 

  void updatePrediction (const double&, const double&, const arma::mat&);
  arma::mat responseTransformation (const arma::mat&) const;
  arma::mat getPrediction (const bool&) const;


};

// class ResponseBinaryClassif : public Response
// {
// public:
    
// };

// class ResponseFDA : public Response
// {
// public:
    
// };


} // namespace response

#endif // RESPONSE_H_