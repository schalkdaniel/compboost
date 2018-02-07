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

void LogIteration::LogStep (unsigned int current_iteration, arma::vec& response,
  arma::vec& prediction, blearner::Baselearner* used_blearner, double& offset,
  double& learning_rate)
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

// InbagRisk:
// -----------------------

LogInbagRisk::LogInbagRisk (bool is_a_stopper0, loss::Loss* used_loss, double eps_for_break)
  : used_loss ( used_loss ),
    eps_for_break ( eps_for_break )
{
  is_a_stopper = is_a_stopper0;
}

void LogInbagRisk::LogStep (unsigned int current_iteration, arma::vec& response,
  arma::vec& prediction, blearner::Baselearner* used_blearner, double& offset,
  double& learning_rate)
{
  double temp_risk = arma::accu(used_loss->DefinedLoss(response, prediction)) / response.size();
  
  tracked_inbag_risk.push_back(temp_risk);
}

bool LogInbagRisk::ReachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (tracked_inbag_risk.size() > 1) {
      double inbag_eps = tracked_inbag_risk[tracked_inbag_risk.size() - 1] - tracked_inbag_risk[tracked_inbag_risk.size()];
      inbag_eps = inbag_eps / tracked_inbag_risk[tracked_inbag_risk.size() - 1];
      
      if (inbag_eps <= eps_for_break) {
        stop_criteria_is_reached = true;
      }
    }
  }
  return stop_criteria_is_reached;
}

arma::vec LogInbagRisk::GetLoggedData ()
{
  arma::vec out (tracked_inbag_risk);
  return out;
}

void LogInbagRisk::ClearLoggerData ()
{
  tracked_inbag_risk.clear();
}

std::string LogInbagRisk::InitializeLoggerPrinter ()
{  
  std::stringstream ss;
  ss << std::setw(17) << "Inbag Risk";
  
  return ss.str();
}

std::string LogInbagRisk::PrintLoggerStatus ()
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << tracked_inbag_risk.back();
  
  return ss.str();
}

// OobRisk:
// -----------------------

LogOobRisk::LogOobRisk (bool is_a_stopper0, loss::Loss* used_loss, double eps_for_break,
  std::map<std::string, arma::mat> oob_data, arma::vec& oob_response)
  : used_loss ( used_loss ),
    eps_for_break ( eps_for_break ),
    oob_data ( oob_data ),
    oob_response ( oob_response )
{
  is_a_stopper = is_a_stopper0;
  
  arma::vec temp (oob_response.size());
  oob_prediction = temp;
}

void LogOobRisk::LogStep (unsigned int current_iteration, arma::vec& response,
  arma::vec& prediction, blearner::Baselearner* used_blearner, double& offset,
  double& learning_rate)
{
  if (current_iteration == 1) {
    oob_prediction.fill(offset);
  }
  
  arma::mat oob_blearner_data = oob_data.find(used_blearner->GetDataIdentifier())->second;
  
  arma::vec temp_oob_prediction = used_blearner->predict(oob_blearner_data);
  
  oob_prediction += learning_rate * temp_oob_prediction;
  double temp_risk = arma::accu(used_loss->DefinedLoss(oob_response, oob_prediction)) / response.size();
  
  tracked_oob_risk.push_back(temp_risk);
}

bool LogOobRisk::ReachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (tracked_oob_risk.size() > 1) {
      double oob_eps = tracked_oob_risk[tracked_oob_risk.size() - 1] - tracked_oob_risk[tracked_oob_risk.size()];
      oob_eps = oob_eps / tracked_oob_risk[tracked_oob_risk.size() - 1];
      
      if (oob_eps <= eps_for_break) {
        stop_criteria_is_reached = true;
      }
    }
  }
  return stop_criteria_is_reached;
}

arma::vec LogOobRisk::GetLoggedData ()
{
  arma::vec out (tracked_oob_risk);
  return out;
}

void LogOobRisk::ClearLoggerData ()
{
  tracked_oob_risk.clear();
}

std::string LogOobRisk::InitializeLoggerPrinter ()
{  
  std::stringstream ss;
  ss << std::setw(17) << "Out of Bag Risk";
  
  return ss.str();
}

std::string LogOobRisk::PrintLoggerStatus ()
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << tracked_oob_risk.back();
  
  return ss.str();
}

// LogTime:
// -----------------------

LogTime::LogTime (bool is_a_stopper0, unsigned int max_time, std::string time_unit)
  : max_time ( max_time ),
    time_unit ( time_unit )
{
  is_a_stopper = is_a_stopper0;
}

void LogTime::LogStep (unsigned int current_iteration, arma::vec& response,
  arma::vec& prediction, blearner::Baselearner* used_blearner, double& offset,
  double& learning_rate)
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
  ss << std::setw(17) << std::fixed << std::setprecision(2) << times_seconds.back();
  
  return ss.str();
}

} // namespace logger
