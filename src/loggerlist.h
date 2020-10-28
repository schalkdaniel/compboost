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

#ifndef LOGGERLIST_H_
#define LOGGERLIST_H_

#include <chrono>
#include <string>
#include <memory>
#include <numeric>

#include "logger.h"
#include "response.h"
#include "optimizer.h"

typedef std::map<std::string, std::shared_ptr<logger::Logger>>  logger_map;
typedef std::pair<std::string, std::shared_ptr<logger::Logger>> logger_pair;
typedef std::pair<std::vector<std::string>, arma::mat>          logger_data;
namespace loggerlist
{

class LoggerList
{
private:
  logger_map   _logger_list;
  unsigned int _sum_of_stopper = 0;

public:
  LoggerList ();

  // Setter/Getter
  bool        getStopperStatus (const bool) const;
  logger_map  getLoggerMap     ()           const;
  logger_data getLoggerData    ()           const;


  // Other member functions
  void logCurrent (const unsigned int, const std::shared_ptr<response::Response>&,
    const std::shared_ptr<blearner::Baselearner>&, const double, const double,
    const std::shared_ptr<optimizer::Optimizer>&);

  void printLoggerStatus     (const double) const;
  void printRegisteredLogger ()             const;

  void registerLogger        (const std::shared_ptr<logger::Logger>);
  void prepareForRetraining  (const unsigned int);
  void clearMap              ();
  void clearLoggerData       ();

  // Destructor
  ~LoggerList ();
};

} // namespace loggerlist

#endif // LOGGERLIST_H_
