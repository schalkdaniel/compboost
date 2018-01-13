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
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with Compboost. If not, see <http://www.gnu.org/licenses/>.
//
// This file contains:
// -------------------
//
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
// =========================================================================== #

#include <numeric>

#include "loggerlist.h"

namespace loggerlist 
{

LoggerList::LoggerList () {};

// LoggerList::LoggerList (arma::mat &evaluation_data, std::chrono::system_clock::time_point init_time, 
//   double init_risk)
//   : evaluation_data_ptr ( &evaluation_data ),
//     init_time ( init_time ),
//     init_risk ( init_risk ) {};

void LoggerList::RegisterLogger (std::string logger_id, logger::Logger *which_logger)
{
  log_list.insert(std::pair<std::string, logger::Logger *>(logger_id, which_logger));
  if (which_logger->GetIfLoggerIsStopper()) {
    sum_of_stopper += 1;
  }
}

void LoggerList::PrintRegisteredLogger ()
{
  std::cout << "Registered Logger:\n";
  for (logger_map::iterator it = log_list.begin(); it != log_list.end(); ++it) {
    std::cout << "\t>>" << it->first << "<< Logger" << std::endl;
  }
}

logger_map LoggerList::GetMap ()
{
  return log_list;
}

void LoggerList::ClearMap ()
{
  log_list.clear();
}

bool LoggerList::GetStopperStatus (bool use_global_stop)
{
  // Define variables to get the status of the algorithm:
  
  // Should the algorithm be returned?
  bool return_algorithm = true;
  // Get status for every registered logger:
  std::vector<bool> status;
  
  // Iterate over logger and get stopper status:
  for (logger_map::iterator it = log_list.begin(); it != log_list.end(); ++it) {
    status.push_back(it->second->ReachedStopCriteria());
  }
  // Sum over status vector to decide if the stop criteria is fullfilled:
  unsigned int status_sum = std::accumulate(status.begin(), status.end(), 0);
  
  // Check if global stop (all stopper has to be true) or local stop (it is
  // sufficient to have just one stopper saying true):
  if (use_global_stop) {
    if (status_sum == sum_of_stopper) {
      return_algorithm = false;
    }
  } else {
    if (status_sum >= 1) {
      return_algorithm = false;
    }
  }
  return return_algorithm;
}

std::pair<std::vector<std::string>, arma::mat> LoggerList::GetLoggerData ()
{
  arma::mat out_matrix;
  std::vector<std::string> logger_names;
  
  for (logger_map::iterator it = log_list.begin(); it != log_list.end(); ++it) {
    out_matrix = arma::join_rows(out_matrix, it->second->GetLoggedData());
    logger_names.push_back(it->first);
  }
  return std::pair<std::vector<std::string>, arma::mat>(logger_names, out_matrix);
}

void LoggerList::LogCurrent (unsigned int current_iteration, double current_risk)
{
  // Think about how to implement this the best way. I think the computations 
  // e.g. for the risk should be done within the logger object. If so, the
  // computation is just done if one would really use the logger!
  
  // Maybe the current risk should be replaced by the map of baselearner and
  // the initial response. Then for the risk it is necessary to call:
  
  // used_loss.DefinedLoss(initial_response, selected_baselearner.predict())
  
  // This can be easily extended to an oob risk by just using the evaluation
  // data specified by initializing the logger list.
  for (logger_map::iterator it = log_list.begin(); it != log_list.end(); ++it) {
    it->second->LogStep(current_iteration, current_risk);
  }
}

// Initialize logger printer:
void LoggerList::InitializeLoggerPrinter ()
{
  std::string printer;
  for (logger_map::iterator it = log_list.begin(); it != log_list.end(); ++it) {
    printer += it->second->InitializeLoggerPrinter() + " |";
  }
  std::cout << printer << std::endl;
  std::cout << std::string(printer.size(), '-') << std::endl;
}

// Print logger:
void LoggerList::PrintLoggerStatus ()
{
  std::string printer;
  for (logger_map::iterator it = log_list.begin(); it != log_list.end(); ++it) {
    printer += it->second->PrintLoggerStatus() + " |";
  }
  std::cout << printer << std::endl;
}

// Destructor:
LoggerList::~LoggerList ()
{
  std::cout << "Call LoggerList Destructor" << std::endl;
  // delete evaluation_data_ptr;
}

} // namespace loggerlist
