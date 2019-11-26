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

#ifndef LOGGERLIST_H_
#define LOGGERLIST_H_

#include <chrono>
#include <string>
#include <memory>

#include "logger.h"
#include "response.h"
#include "optimizer.h"

typedef std::map<std::string, std::shared_ptr<logger::Logger>> logger_map;

namespace loggerlist
{

class LoggerList
{
private:

  logger_map log_list;
  unsigned int sum_of_stopper = 0;

public:

  LoggerList ();
  // LoggerList (arma::mat&, std::chrono::system_clock::time_point, double);

  // String for logger and the logger itselfe:
  void registerLogger (std::shared_ptr<logger::Logger>);
  void printRegisteredLogger () const;

  logger_map getMap () const;
  void clearMap ();

  // This function should iterate over all registered logger, check if it is
  // a stopper and returns just one bool, aggregated over a vector of bools
  // from the single logger. This could be e.g. one is fullfilled or an all
  // check (all stopper has to be fullfilled). The priority comes with the
  // map identifier since it sorts the entrys after name.

  // If the argument is 'true', than all stopper has to be fullfilled.
  bool getStopperStatus (const bool&) const;

  // Get a matrix of tracked logger (iterator over all logger and paste
  // all columns of the private member). The return is a pair with a
  // string vector containing the logger type and a matrix with corresponging
  // columns for each logger type:
  std::pair<std::vector<std::string>, arma::mat> getLoggerData () const;

  // Log the current step (structure <iteration, actual time, actual risk>).
  // This is given to the instantiated logger:
  void logCurrent (const unsigned int&, std::shared_ptr<response::Response>,
    std::shared_ptr<blearner::Baselearner>, const double&, const double&,
    std::shared_ptr<optimizer::Optimizer>);

  // Print the logger status:
  void printLoggerStatus (const double&) const;

  // Prepare logger data for retraining:
  void prepareForRetraining (const unsigned int&);

  // Clear the logger data (should be used in front of every compboost training):
  void clearLoggerData ();

  // Destructor:
  ~LoggerList ();
};

} // namespace loggerlist

#endif // LOGGERLIST_H_
