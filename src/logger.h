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
public:
  /// Tag if the logger is used as stopper
  bool is_a_stopper;

  /// Log current step of compboost iteration dependent on the child class
  virtual void logStep (const unsigned int&, std::shared_ptr<response::Response>,
    std::shared_ptr<blearner::Baselearner>, const double&, const double&) = 0;

  /// Class dependent check if the stopping criteria is fulfilled
  virtual bool reachedStopCriteria () = 0;

  /// Return the data stored within the logger
  virtual arma::vec getLoggedData () const = 0;

  /// Clear the logger data
  virtual void clearLoggerData () = 0;

  /// Print status of current iteration into the console
  virtual std::string printLoggerStatus () const = 0;

  /// Get logger identifier:
  std::string getLoggerId () const;
  std::string getLoggerType () const;

  /// Just a getter if the logger is also used as stopper
  bool getIfLoggerIsStopper () const;

  virtual
    ~Logger ();

protected:
  std::string logger_type;
  std::string logger_id;
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

  /// Maximal number of iterations (only interesting if used as stopper)
  unsigned int max_iterations;

  /// Vector to log the iterations
  std::vector<unsigned int> iterations;


public:

  /// Default constructor of class `LoggerIteration`
  LoggerIteration (const std::string&, const bool&, const unsigned int&);

  /// Log current step of compboost iteration of class `LoggerIteration`
  void logStep (const unsigned int&, std::shared_ptr<response::Response>,
    std::shared_ptr<blearner::Baselearner>, const double&, const double&);

  /// Stop criteria is fulfilled if the current iteration exceed `max_iteration`
  bool reachedStopCriteria ();

  /// Return the data stored within the iteration logger
  arma::vec getLoggedData () const;

  /// Clear the logger data
  void clearLoggerData ();

  /// Print status of current iteration into the console
  std::string printLoggerStatus () const;

  /// Update maximal iteration:
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

  /// Used loss. **Note** that you can specify a different loss than the loss used for training
  std::shared_ptr<loss::Loss> sh_ptr_loss;

  /// Vector of inbag risk for every iteration
  std::vector<double> tracked_inbag_risk;

  /// Stopping criteria, stop if \f$(\mathrm{risk}_{i-1} - \mathrm{risk}_i) / \mathrm{risk}_{i-1} < \mathrm{eps\_for\_break}\f$
  double eps_for_break;


public:

  /// Default constructor
  LoggerInbagRisk (const std::string&, const bool&, std::shared_ptr<loss::Loss>, const double&);

  /// Log current step of compboost iteration for class `LoggerInbagRisk`
  void logStep (const unsigned int&, std::shared_ptr<response::Response>,
    std::shared_ptr<blearner::Baselearner>, const double&, const double&);

  /// Stop criteria is fulfilled if the relative improvement falls below `eps_for_break`
  bool reachedStopCriteria ();

  /// Return the data stored within the logger
  arma::vec getLoggedData () const;

  /// Clear the logger data
  void clearLoggerData ();

  /// Print status of current iteration into the console
  std::string printLoggerStatus () const;

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

  /// Used loss. **Note** that you can specify a different loss than the loss used for training
  std::shared_ptr<loss::Loss> sh_ptr_loss;

  /// Vector of OOB risk for every iteration
  std::vector<double> tracked_oob_risk;

  /// Stopping criteria, stop if \f$(\mathrm{risk}_{i-1} - \mathrm{risk}_i) / \mathrm{risk}_{i-1} < \mathrm{eps\_for\_break}\f$
  double eps_for_break;
  unsigned int patience = 5;
  unsigned int count_patience = 0;

  /// OOB prediction which is internally done in every iteration
  arma::mat oob_prediction;

  /// The OOB data provided by the user
  std::map<std::string, std::shared_ptr<data::Data>> oob_data;

  /* This is part of the memory saving version (see logger.cpp)
   * /// Transformed oob data for predicting on the oob set:
   * std::map<std::string, arma::mat> oob_data_transformed;
   */

  /// The response variable which corresponds to the given OOB data
  std::shared_ptr<response::Response> sh_ptr_oob_response;


public:

  /// Default constructor
  LoggerOobRisk (const std::string&, const bool&, std::shared_ptr<loss::Loss>, const double&, const unsigned int&,
    std::map<std::string, std::shared_ptr<data::Data>>, std::shared_ptr<response::Response>);

  /// Log current step of compboost iteration for class `LoggerOobRisk`
  void logStep (const unsigned int&, std::shared_ptr<response::Response>,
    std::shared_ptr<blearner::Baselearner>, const double&, const double&);

  /// Stop criteria is fulfilled if the relative improvement falls below `eps_for_break`
  bool reachedStopCriteria ();

  /// Return the data stored within the logger
  arma::vec getLoggedData () const;

  /// Clear the logger data
  void clearLoggerData ();

  /// Print status of current iteration into the console
  std::string printLoggerStatus () const;

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

  /// Initial time, important to get the actual elapsed time
  std::chrono::steady_clock::time_point init_time;

  /// Vector of elapsed time at each iteration
  std::vector<unsigned int> current_time;

  /// Stopping criteria, stop if \f$\mathrm{current_time} > \mathrm{max_time}\f$
  unsigned int max_time;

  /// The unit for time measuring, allowed are `minutes`, `seconds` and `microseconds`
  std::string time_unit;

  /// Drift that is added to runtime to be able to proper stop time for retraining
  unsigned int retrain_drift = 0;


public:

  /// Default constructor of class `LoggerTime`
  LoggerTime (const std::string&, const bool&, const unsigned int&, const std::string&);

  /// Log current step of compboost iteration for class `LoggerTime`
  void logStep (const unsigned int&, std::shared_ptr<response::Response>,
    std::shared_ptr<blearner::Baselearner>, const double&, const double&);

  /// Stop criteria is fulfilled if the passed time exceeds `max_time`
  bool reachedStopCriteria ();

  /// Return the data stored within the logger
  arma::vec getLoggedData () const;

  /// Clear the logger data
  void clearLoggerData();

  /// Print status of current iteration into the console
  std::string printLoggerStatus () const;

  void reInitializeTime();

};

} // namespace logger

#endif // LOGGER_H_
