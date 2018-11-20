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

#ifndef RESPONSE_H_
#define RESPONSE_H_

#include "RcppArmadillo.h"

namespace response 
{

// -------------------------------------------------------------------------- //
// Abstract 'Response' class:
// -------------------------------------------------------------------------- //

class Response
{
protected:
  
  std::string task_id;
  unsigned int actual_iteration = 0;
  
public:
  
  Response ();
  
  std::string getTaskIdentifier () const = 0;
  
  virtual arma::mat getResponse () const = 0; 
  virtual arma::mat getWeights () const = 0;
  virtual arma::mat getPseudoResiduals () const = 0;
  virtual arma::mat getPrediction () const = 0;
    
  virtual ~Response () { };
};


// -------------------------------------------------------------------------- //
// Response implementations:
// -------------------------------------------------------------------------- //

class ResponseRegr : public Response
{
private:

  // Response and weights as matrix
  arma::mat response;
  arma::mat weights;
  arma::mat pseudo_residuals;
  arma::mat response;

public:

  ResponseRegr ();

  arma::mat getResponse () const;
  arma::mat getWeights () const;
  arma::mat getPseudoResiduals () const;
  arma::mat getPrediction () const;

};

class ResponseBinaryClassif : public Response
{
public:
    
};

class ResponseFDA : public Response
{
public:
    
};


} // namespace response

#endif // RESPONSE_H_