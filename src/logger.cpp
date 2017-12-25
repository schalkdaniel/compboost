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

#include "logger.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

void Logger::InitializeLogger(loss::Loss &used_loss0, arma::mat &evaluation_data0, 
  bool is_a_stopper0, std::chrono::system_clock::time_point &init_time0, 
  double &init_risk0)
{
  used_loss       = &used_loss0;
  evaluation_data = &evaluation_data0;
  is_a_stopper    = is_a_stopper0;
  init_time       = &init_time0;
  init_risk       = &init_risk0;
}

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// LogIteration:
// -----------------------

LogIteration::LogIteration (unsigned int max_iterations) : max_iterations ( max_iterations ) {};

void LogIteration::LogStep (unsigned int current_iteration, 
  std::chrono::system_clock::time_point current_time, double &current_risk)
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

// LogRisk:
// -----------------------

// LogTime:
// -----------------------

} // namespace logger