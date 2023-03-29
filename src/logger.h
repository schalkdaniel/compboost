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
// it under the terms of the LGPL-3 License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// LGPL-3 License for more details. You should have received a copy of
// the license along with compboost.
//
// =========================================================================== #

/**
 *  @file    logger.h
 *  @author  Daniel Schalk (github: schalkdaniel)
 *
 *  @brief Logger class definition
 *
 *  @section DESCRIPTION
 *
 *  This file contains all the available logger which also can be used for
 *  early stopping. The idea is not just about to use multiple stopping
 *  criteria (e.g. maximal number of iterations + a given amount of time),
 *  but also to log while training (e.g. the inbag or oob risk).
 *
 *  The logger are collected by the `LoggerList` class. The `LoggerList` class
 *  takes an arbitrary number of logger to indicate when to stop. This can be used,
 *  for example, to log different risks for different loss functions. Just create two
 *  inbag or oob risk logger with a different loss.
 *
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#include <vector>
#include <chrono>
#include <memory>
#include <iomanip> // ::setw
#include <sstream> // ::stringstream

#include "loss.h"
#include "baselearner.h"
#include "response.h"
#include "optimizer.h"
#include "baselearner_factory_list.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

#include "single_include/date.h"
using namespace date;

namespace logger
{

std::chrono::system_clock::time_point stringToChrono (const std::string);

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

/**
 * \class Logger
 *
 * \brief Abstract logger with minimal functionality each logger has
 *
 * This class defines the minimal functionality each logger must
 * have! The key of the logger is nut only the logging of the process, but also
 * to be used as stopper for early stopping if one or all of the used logger
 * have reached the stopping criteria.
 *
 * **Note** this minimal functionality mentioned above differs for every
 * class and is explained within the specific class documentation.
 *
 */
class Logger
{
private:
  std::string _logger_type;
  std::string _logger_id;

protected:
  bool _is_stopper = false;
  Logger (const bool, const std::string, const std::string);
  Logger (const json&);

public:
  // Virtual functions:
  virtual void logStep (const unsigned int, const std::shared_ptr<response::Response>&,
    const std::shared_ptr<blearner::Baselearner>&, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>&, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&) = 0;

  virtual bool         reachedStopCriteria ()       = 0;
  virtual arma::vec    getLoggedData       () const = 0;
  virtual void         clearLoggerData     ()       = 0;
  virtual std::string  printLoggerStatus   () const = 0;

  virtual json toJson (const bool = false) const = 0;

  // Setter/Getter
  void setIsStopper (const bool);

  std::string getLoggerId   () const;
  std::string getLoggerType () const;
  bool        isStopper     () const;
  json        baseToJson    (const std::string) const;

  // Destructor
  virtual ~Logger ();
};

std::shared_ptr<logger::Logger> jsonToLogger (const json&);

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

/**
 * \class LoggerIteration
 *
 * \brief Logger to log the current iteration
 *
 * The `LoggerIteration` tracks the current iteration and can be used to stop the
 * algorithm after a pre-defined number of iteration is reached.
 *
 */
class LoggerIteration : public Logger
{
private:
  unsigned int              _max_iterations;
  std::vector<unsigned int> _iterations;

public:
  LoggerIteration (const std::string, const bool, const unsigned int);
  LoggerIteration (const json&);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>&,
    const std::shared_ptr<blearner::Baselearner>&, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>&, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;

  json toJson (const bool = false) const;

  void updateMaxIterations (const unsigned int&);
};


// InbagRisk:
// -----------------------

/**
 * \class LoggerInbagRisk
 *
 * \brief Logger to log the inbag risk
 *
 * This class tracks the inbag risk for a specific loss function. It is possible
 * to define more than one risk logger (e.g. for 2 different loss
 * functions). For details about logging and stopping see the description of the
 * `logStep()` function.
 *
 */
class LoggerInbagRisk : public Logger
{
private:
  const std::shared_ptr<loss::Loss> _sh_ptr_loss;
  std::vector<double>               _inbag_risk;

  const double       _eps_for_break;
  const unsigned int _patience = 5;
  unsigned int       _count_patience = 0;

public:
  LoggerInbagRisk (const std::string, const bool, const std::shared_ptr<loss::Loss>, const double, const unsigned int);
  LoggerInbagRisk (const json&);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>&,
    const std::shared_ptr<blearner::Baselearner>&, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>&, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;

  json toJson (const bool = false) const;
};


// OobRisk:
// -----------------------

/**
 * \class LoggerOobRisk
 *
 * \brief Logger to log the out of bag risk
 *
 * This class tracks the out of bag risk for a specific loss function and a map
 * of new data. It is possible to define more than one risk logger
 * (e.g. for 2 different loss functions). For details about logging and
 * stopping see the description of the `logStep()` function.
 *
 */
class LoggerOobRisk : public Logger
{
private:
  const std::shared_ptr<loss::Loss> _sh_ptr_loss;
  std::vector<double>               _oob_risk;

  const double       _eps_for_break;
  const unsigned int _patience = 5;
  unsigned int       _count_patience = 0;

  arma::mat _oob_prediction;

  std::map<std::string, std::shared_ptr<data::Data>>  _oob_data_map;
  std::map<std::string, std::shared_ptr<data::Data>>  _oob_data_map_inst;
  const std::shared_ptr<response::Response>           _sh_ptr_oob_response;

public:
  LoggerOobRisk (const std::string, const bool, const std::shared_ptr<loss::Loss>, const double, const unsigned int,
    const std::map<std::string, std::shared_ptr<data::Data>>, const std::shared_ptr<response::Response>);
  LoggerOobRisk (const json&);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>&,
    const std::shared_ptr<blearner::Baselearner>&, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>&, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;

  json toJson (const bool = false) const;
};


// LoggerTime:
// -----------------------

/**
 * \class LoggerTime
 *
 * \brief Logger to log the elapsed time
 *
 * This class tracks the elapsed time. This is very handy since
 * it allows to stop the algorithm, e.g., after one hour of training.
 * There are three time units available for logging:
 *   - minutes
 *   - seconds
 *   - microseconds
 */
class LoggerTime : public Logger
{
private:
  std::chrono::system_clock::time_point _init_time;
  std::vector<unsigned int>             _current_time;
  unsigned int                          _retrain_drift = 0;

  const unsigned int _max_time;
  const std::string  _time_unit;

public:
  LoggerTime (const std::string, const bool, const unsigned int, const std::string);
  LoggerTime (const json&);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>&,
    const std::shared_ptr<blearner::Baselearner>&, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>&, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>&);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;

  json toJson (const bool = false) const;

  void reInitializeTime();

};

} // namespace logger

#endif // LOGGER_H_
