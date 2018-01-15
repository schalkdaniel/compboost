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
//   Implementation of "Logger" class.
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

#include "logger.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

bool Logger::GetIfLoggerIsStopper ()
{
  return is_a_stopper;
}

// Destructor:
Logger::~Logger ()
{
  // Shouldn't be deleted. This are pointers needed in other contextes to!
  // delete used_loss;
  // delete evaluation_data;
}

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// LogIteration:
// -----------------------

LogIteration::LogIteration (bool is_a_stopper0, unsigned int max_iterations) 
  : max_iterations ( max_iterations ) 
{
  is_a_stopper = is_a_stopper0;
};

void LogIteration::LogStep (unsigned int current_iteration, double current_risk)
{
  iterations.push_back(current_iteration);
}

bool LogIteration::ReachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (max_iterations <= iterations.back()) {
      stop_criteria_is_reached = true;
    }
  }
  return stop_criteria_is_reached;
}

arma::vec LogIteration::GetLoggedData ()
{
  // Cast integer vector to double:
  std::vector<double> iterations_double (iterations.begin(), iterations.end());
  
  arma::vec out (iterations_double);
  return out;
}

void LogIteration::ClearLoggerData ()
{
  iterations.clear();
}

std::string LogIteration::InitializeLoggerPrinter ()
{
  // 15 characters:
  return "      Iteration";
}

std::string LogIteration::PrintLoggerStatus ()
{
  std::stringstream ss;
  ss << std::setw(15) << std::to_string(iterations.back()) + "/" + std::to_string(max_iterations);
  
  return ss.str();
}

// LogRisk:
// -----------------------

// LogTime:
// -----------------------

LogTime::LogTime (bool is_a_stopper0, unsigned int max_time, std::string time_unit)
  : max_time ( max_time ),
    time_unit ( time_unit )
{
  is_a_stopper = is_a_stopper0;
}

void LogTime::LogStep (unsigned int current_iteration, double current_risk)
{
  if (times_seconds.size() == 0) {
    init_time = std::chrono::steady_clock::now();
  }
  if (time_unit == "minutes") {
    times_seconds.push_back(std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - init_time).count());
  } 
  if (time_unit == "seconds") {
    times_seconds.push_back(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - init_time).count());
  } 
  if (time_unit == "microseconds") {
    times_seconds.push_back(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - init_time).count());
  }
}

bool LogTime::ReachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (times_seconds.back() >= max_time) {
      stop_criteria_is_reached = true;
    }
  }
  return stop_criteria_is_reached;
}

arma::vec LogTime::GetLoggedData ()
{
  // Cast integer vector to double:
  std::vector<double> seconds_double (times_seconds.begin(), times_seconds.end());
  
  arma::vec out (seconds_double);
  return out;
}

void LogTime::ClearLoggerData ()
{
  times_seconds.clear();
}

std::string LogTime::InitializeLoggerPrinter ()
{
  std::stringstream ss;
  ss << std::setw(17) << time_unit;

  return ss.str();
}

std::string LogTime::PrintLoggerStatus ()
{
  std::stringstream ss;
  ss << std::setw(17) << std::setprecision(2) << times_seconds.back();
  
  return ss.str();
}

} // namespace logger