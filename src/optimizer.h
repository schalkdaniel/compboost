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
#include <math.h>

#include <RcppArmadillo.h>

#include "baselearner.h"
#include "baselearner_factory_list.h"
#include "baselearner_track.h"
#include "loss.h"
#include "line_search.h"
#include "helper.h"
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

#ifdef _OPENMP
#include <omp.h>
#endif

namespace optimizer {

typedef std::shared_ptr<data::Data> sdata;
typedef std::map<std::string, sdata> mdata;

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

class Optimizer
{
protected:
  std::vector<double> _step_sizes;
  const unsigned int  _num_threads = 1;

  Optimizer ();
  Optimizer (const unsigned int);
  Optimizer (const json&);

public:
  // Virtual methods
  virtual double              getStepSize  (const unsigned int) const = 0;
  virtual std::vector<double> getStepSize  ()                   const = 0;

  virtual std::shared_ptr<blearner::Baselearner> findBestBaselearner (std::string,
    const std::shared_ptr<response::Response>&, const blearner_factory_map&) const = 0;

  virtual void optimize (const unsigned int, const double, const std::shared_ptr<loss::Loss>&,
    const std::shared_ptr<response::Response>&, blearnertrack::BaselearnerTrack&,
    const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&) = 0;

  virtual arma::mat calculateUpdate   (const double, const double, const arma::mat&,
    const std::map<std::string, std::shared_ptr<data::Data>>&, const std::shared_ptr<response::Response>&) const = 0;

  virtual void calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&) = 0;

  virtual std::map<std::string, arma::mat> getParameterAtIteration (const unsigned int, const double, blearnertrack::BaselearnerTrack&) const;

  json baseToJson (const std::string) const;
  virtual json toJson () const = 0;

  // Destructor
  virtual ~Optimizer ();
};

std::shared_ptr<Optimizer> jsonToOptimizer (const json&, const mdata&);


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
  OptimizerCoordinateDescent (const json&);

  double              getStepSize  (const unsigned int) const;
  std::vector<double> getStepSize  ()                   const;

  std::shared_ptr<blearner::Baselearner> findBestBaselearner (const std::string,
    const std::shared_ptr<response::Response>&, const blearner_factory_map&) const;

  void optimize (const unsigned int, const double, const std::shared_ptr<loss::Loss>&,
    const std::shared_ptr<response::Response>&, blearnertrack::BaselearnerTrack&,
    const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&);

  arma::mat calculateUpdate   (const double, const double, const arma::mat&,
    const std::map<std::string, std::shared_ptr<data::Data>>&, const std::shared_ptr<response::Response>&) const;
  void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&);

  json toJson () const;
};

class OptimizerCoordinateDescentLineSearch : public OptimizerCoordinateDescent
{
public:
  OptimizerCoordinateDescentLineSearch ();
  OptimizerCoordinateDescentLineSearch (const unsigned int);
  OptimizerCoordinateDescentLineSearch (const json&);

  void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&);

  double              getStepSize (const unsigned int) const;
  std::vector<double> getStepSize ()                   const;

  json toJson () const;
};


class OptimizerCosineAnnealing : public OptimizerCoordinateDescent
{
private:
  const double       _nu_min;
  const double       _nu_max;
  const unsigned int _cycles;
  const unsigned int _anneal_iter_max;

  const unsigned int _iters_per_cycle;
  unsigned int       _current_iter;
  unsigned int       _cycle_iter;

public:
  OptimizerCosineAnnealing ();
  OptimizerCosineAnnealing (unsigned int);
  OptimizerCosineAnnealing (const double, const double, const unsigned int, const unsigned int, const unsigned int);
  OptimizerCosineAnnealing (const json&);

  void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&);

  double              getStepSize (const unsigned int) const;
  std::vector<double> getStepSize ()                   const;

  json toJson () const;
};


// Accelerated Gradient Boosting:
// -----------------------------------------------------------

class OptimizerAGBM: public Optimizer
{
private:
  const double             _momentum;
  arma::mat                _pred_momentum;
  arma::mat                _pred_aggr;
  arma::mat                _pr_corr;
  const unsigned int       _acc_iters = std::numeric_limits<unsigned int>::max();
  std::vector<std::string> _bl_unique_id;

  std::map<std::string, arma::mat> _aggr_parameter_map;
  blearnertrack::BaselearnerTrack  _momentum_blearnertrack = blearnertrack::BaselearnerTrack(1.0);

public:
  OptimizerAGBM ();
  OptimizerAGBM (const double);
  OptimizerAGBM (const double, const unsigned int);
  OptimizerAGBM (const double, const unsigned int, const unsigned int);
  OptimizerAGBM (const json&, const mdata&);

  std::shared_ptr<blearner::Baselearner> findBestBaselearner (const std::string,
    const std::shared_ptr<response::Response>&, const blearner_factory_map&) const;
  std::shared_ptr<blearner::Baselearner> findBestBaselearner (const std::string,
    const arma::mat&, const blearner_factory_map&) const;

  void optimize (const unsigned int, const double, const std::shared_ptr<loss::Loss>&,
    const std::shared_ptr<response::Response>&, blearnertrack::BaselearnerTrack&,
    const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&);

  arma::mat calculateUpdate   (const double, const double, const arma::mat&,
    const std::map<std::string, std::shared_ptr<data::Data>>&, const std::shared_ptr<response::Response>&) const;

  void      calculateStepSize (const std::shared_ptr<loss::Loss>&, const std::shared_ptr<response::Response>&, const arma::vec&);

  double                                         getStepSize (const unsigned int)  const;
  std::vector<double>                            getStepSize ()                    const;
  std::map<std::string, arma::mat>               getMomentumParameter ()           const;
  std::vector<std::string>                       getSelectedMomentumBaselearner () const;
  std::pair<std::vector<std::string>, arma::mat> getParameterMatrix ()             const;

  //void updateAggrParameter (std::shared_ptr<blearner::Baselearner>&, double, double, blearnertrack::BaselearnerTrack&);
  std::map<std::string, arma::mat> addParamMaps (const std::map<std::string, arma::mat>&, const std::map<std::string, arma::mat>&, const double) const;
  void updateAggrParameter (double, blearnertrack::BaselearnerTrack&);
   std::map<std::string, arma::mat> getParameterAtIteration (const unsigned int, const double, blearnertrack::BaselearnerTrack&) const;

   json toJson () const;
};






} // namespace optimizer

#endif // OPTIMIZER_H_
