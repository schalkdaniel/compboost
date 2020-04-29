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

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <iostream>
#include <memory>
#include <map>
#include <limits>

#include <RcppArmadillo.h>

#include "baselearner.h"
#include "baselearner_factory_list.h"
#include "baselearner_track.h"
#include "loss.h"
#include "line_search.h"
#include "helper.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace optimizer {

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

class Optimizer
{
protected:
  // blearner_factory_map my_blearner_factory_map;
  std::vector<double> _step_sizes;
  const unsigned int  _num_threads = 1;

  Optimizer ();
  Optimizer (const unsigned int);

public:
  // Virtual methods
  virtual double              getStepSize  (const unsigned int) const = 0;
  virtual std::vector<double> getStepSize  ()                   const = 0;

  virtual std::shared_ptr<blearner::Baselearner> findBestBaselearner (const std::string,
    const std::shared_ptr<response::Response>&, const blearner_factory_map&) const = 0;

  virtual void optimize (const unsigned int, const double, const std::shared_ptr<loss::Loss>&,
    const std::shared_ptr<response::Response>&, blearnertrack::BaselearnerTrack&,
    const blearnerlist::BaselearnerFactoryList&) = 0;

  virtual arma::mat calculateUpdate   (const double, const double, const arma::mat&) const = 0;
  virtual void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&) = 0;


  // Destructor
  virtual ~Optimizer ();
};

// -------------------------------------------------------------------------- //
// Optimizer implementations:
// -------------------------------------------------------------------------- //

// Coordinate Descent:
// -------------------------------------------

class OptimizerCoordinateDescent : public Optimizer
{
public:
  OptimizerCoordinateDescent ();
  OptimizerCoordinateDescent (const unsigned int);

  double              getStepSize  (const unsigned int) const;
  std::vector<double> getStepSize  ()                   const;

  std::shared_ptr<blearner::Baselearner> findBestBaselearner (const std::string,
    const std::shared_ptr<response::Response>&, const blearner_factory_map&) const;

  void optimize (const unsigned int, const double, const std::shared_ptr<loss::Loss>&,
    const std::shared_ptr<response::Response>&, blearnertrack::BaselearnerTrack&,
    const blearnerlist::BaselearnerFactoryList&);

  arma::mat calculateUpdate   (const double, const double, const arma::mat&) const;
  void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&);
};

class OptimizerCoordinateDescentLineSearch : public OptimizerCoordinateDescent
{
public:
  OptimizerCoordinateDescentLineSearch ();
  OptimizerCoordinateDescentLineSearch (const unsigned int);

  void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&);

  double              getStepSize (const unsigned int) const;
  std::vector<double> getStepSize ()                   const;

  void calculateStepSize (std::shared_ptr<loss::Loss>, std::shared_ptr<response::Response>, const arma::vec&);
};

} // namespace optimizer

#endif // OPTIMIZER_H_
