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
 *  The logger are collected by the `LoggerList` class. Basicall, that class
 *  takes as much logger as you like and logs every step. This can be used
 *  to log different risks for different loss functions. Just create two
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

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

/**
 * \class Logger
 *
 * \brief Abstract logger class with minimal requirements to all logger
 *
 * This class is meant to define some minimal functionality any logger must
 * have! The key of the logger is nut only the logging of the process, but also
 * to be able to define a logger as stopper to force an early stopping if one
 * or all of the used logger have reached a stopping criteria. This is more
 * explained within the child classes.
 *
 * **Note** that this minimal functionality mentioned above differs for every
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

public:
  // Virtual functions:
  virtual void logStep (const unsigned int, const std::shared_ptr<response::Response>,
    const std::shared_ptr<blearner::Baselearner>, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>) = 0;

  virtual bool         reachedStopCriteria ()       = 0;
  virtual arma::vec    getLoggedData       () const = 0;
  virtual void         clearLoggerData     ()       = 0;
  virtual std::string  printLoggerStatus   () const = 0;

  // Setter/Getter
  void setIsStopper (const bool);

  std::string getLoggerId   () const;
  std::string getLoggerType () const;
  bool        isStopper     () const;

  // Destructor
  virtual ~Logger ();

};

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

/**
 * \class LoggerIteration
 *
 * \brief Logger class to log the current iteration
 *
 * This class seems to be useless, but it gives more control about the algorithm
 * and doesn't violate the idea of object programming here. Additionally, it is
 * quite convenient to have this class instead of tracking the iteration at any
 * stage of the fitting within the compboost object as another vector.
 *
 */

class LoggerIteration : public Logger
{
private:
  unsigned int              _max_iterations;
  std::vector<unsigned int> _iterations;

public:
  LoggerIteration (const std::string, const bool, const unsigned int);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>,
    const std::shared_ptr<blearner::Baselearner>, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;

  void updateMaxIterations (const unsigned int&);
};

// InbagRisk:
// -----------------------

/**
 * \class LoggerInbagRisk
 *
 * \brief Logger class to log the inbag risk
 *
 * This class loggs the inbag risk for a specific loss function. It is possible
 * to define more than one inbag risk logger (e.g. for 2 different loss
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


  void logStep (const unsigned int, const std::shared_ptr<response::Response>,
    const std::shared_ptr<blearner::Baselearner>, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;
};

// OobRisk:
// -----------------------

/**
 * \class LoggerOobRisk
 *
 * \brief Logger class to log the out of bag risk
 *
 * This class loggs the out of bag risk for a specific loss function and a map
 * of new data. It is possible to define more than one inbag risk logger
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

  std::map<std::string, std::shared_ptr<data::Data>>  _oob_data;
  const std::shared_ptr<response::Response>           _sh_ptr_oob_response;

public:
  LoggerOobRisk (const std::string, const bool, const std::shared_ptr<loss::Loss>, const double, const unsigned int,
    const std::map<std::string, std::shared_ptr<data::Data>>, const std::shared_ptr<response::Response>);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>,
    const std::shared_ptr<blearner::Baselearner>, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;
};

// LoggerTime:
// -----------------------

/**
 * \class LoggerTime
 *
 * \brief Logger class to log the ellapsed time
 *
 * This class just loggs the ellapsed time. This sould be very handy if one
 * wants to run the algorithm for just 2 hours and see how far he comes within
 * that time. There are three time units available for logging:
 *   - minutes
 *   - seconds
 *   - microseconds
 *
 */
class LoggerTime : public Logger
{
private:
  std::chrono::steady_clock::time_point _init_time;
  std::vector<unsigned int>             _current_time;
  unsigned int                          _retrain_drift = 0;

  const unsigned int _max_time;
  const std::string  _time_unit;

public:
  LoggerTime (const std::string, const bool, const unsigned int, const std::string);

  void logStep (const unsigned int, const std::shared_ptr<response::Response>,
    const std::shared_ptr<blearner::Baselearner>, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>);

  bool         reachedStopCriteria ();
  arma::vec    getLoggedData       () const;
  void         clearLoggerData     ();
  std::string  printLoggerStatus   () const;

  void reInitializeTime();

};

} // namespace logger

#endif // LOGGER_H_
