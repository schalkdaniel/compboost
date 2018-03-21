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
// ========================================================================== //

#include "logger.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

bool Logger::GetIfLoggerIsStopper () const
{
  return is_a_stopper;
}

// Destructor:
Logger::~Logger () { }

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// IterationLogger:
// -----------------------

IterationLogger::IterationLogger (const bool& is_a_stopper0, 
  const unsigned int& max_iterations) 
  : max_iterations ( max_iterations ) 
{
  is_a_stopper = is_a_stopper0;
};

void IterationLogger::LogStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  iterations.push_back(current_iteration);
}

bool IterationLogger::ReachedStopCriteria () const
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (max_iterations <= iterations.back()) {
      stop_criteria_is_reached = true;
    }
  }
  return stop_criteria_is_reached;
}

arma::vec IterationLogger::GetLoggedData () const
{
  // Cast integer vector to double:
  std::vector<double> iterations_double (iterations.begin(), iterations.end());
  
  arma::vec out (iterations_double);
  return out;
}

void IterationLogger::ClearLoggerData ()
{
  iterations.clear();
}

std::string IterationLogger::InitializeLoggerPrinter () const
{
  // 15 characters:
  return "      Iteration";
}

std::string IterationLogger::PrintLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(15) << std::to_string(iterations.back()) + "/" + std::to_string(max_iterations);
  
  return ss.str();
}

// InbagRisk:
// -----------------------

InbagRiskLogger::InbagRiskLogger (const bool& is_a_stopper0, loss::Loss* used_loss, 
  const double& eps_for_break)
  : used_loss ( used_loss ),
    eps_for_break ( eps_for_break )
{
  is_a_stopper = is_a_stopper0;
}

void InbagRiskLogger::LogStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  double temp_risk = arma::accu(used_loss->DefinedLoss(response, prediction)) / response.size();
  
  tracked_inbag_risk.push_back(temp_risk);
}

bool InbagRiskLogger::ReachedStopCriteria () const
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

arma::vec InbagRiskLogger::GetLoggedData () const
{
  arma::vec out (tracked_inbag_risk);
  return out;
}

void InbagRiskLogger::ClearLoggerData ()
{
  tracked_inbag_risk.clear();
}

std::string InbagRiskLogger::InitializeLoggerPrinter () const
{  
  std::stringstream ss;
  ss << std::setw(17) << "Inbag Risk";
  
  return ss.str();
}

std::string InbagRiskLogger::PrintLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << tracked_inbag_risk.back();
  
  return ss.str();
}

// OobRisk:
// -----------------------

OobRiskLogger::OobRiskLogger (const bool& is_a_stopper0, loss::Loss* used_loss, 
  const double& eps_for_break, std::map<std::string, data::Data*> oob_data, 
  const arma::vec& oob_response)
  : used_loss ( used_loss ),
    eps_for_break ( eps_for_break ),
    oob_data ( oob_data ),
    oob_response ( oob_response )
{
  is_a_stopper = is_a_stopper0;
  
  arma::vec temp (oob_response.size());
  oob_prediction = temp;
}

void OobRiskLogger::LogStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  if (current_iteration == 1) {
    oob_prediction.fill(offset);
  }
  
  // Get data of corresponding selected baselearner. E.g. iteration 100 linear 
  // baselearner of feature x_7, then get the data of feature x_7:
  data::Data* oob_blearner_data = oob_data.find(used_blearner->GetDataIdentifier())->second;
  
  // Predict this data using the selected baselearner:
  arma::vec temp_oob_prediction = used_blearner->predict(oob_blearner_data);
  
  // Cumulate prediction and shrink by learning rate:
  oob_prediction += learning_rate * temp_oob_prediction;
  
  // Calculate empirical risk:
  double temp_risk = arma::accu(used_loss->DefinedLoss(oob_response, oob_prediction)) / response.size();
  
  // Track empirical risk:
  tracked_oob_risk.push_back(temp_risk);
}

bool OobRiskLogger::ReachedStopCriteria () const
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

arma::vec OobRiskLogger::GetLoggedData () const
{
  arma::vec out (tracked_oob_risk);
  return out;
}

void OobRiskLogger::ClearLoggerData ()
{
  tracked_oob_risk.clear();
}

std::string OobRiskLogger::InitializeLoggerPrinter () const
{  
  std::stringstream ss;
  ss << std::setw(17) << "Out of Bag Risk";
  
  return ss.str();
}

std::string OobRiskLogger::PrintLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << tracked_oob_risk.back();
  
  return ss.str();
}

// TimeLogger:
// -----------------------

TimeLogger::TimeLogger (const bool& is_a_stopper0, const unsigned int& max_time, 
  const std::string& time_unit)
  : max_time ( max_time ),
    time_unit ( time_unit )
{
  is_a_stopper = is_a_stopper0;
}

void TimeLogger::LogStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
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

bool TimeLogger::ReachedStopCriteria () const
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (times_seconds.back() >= max_time) {
      stop_criteria_is_reached = true;
    }
  }
  return stop_criteria_is_reached;
}

arma::vec TimeLogger::GetLoggedData () const
{
  // Cast integer vector to double:
  std::vector<double> seconds_double (times_seconds.begin(), times_seconds.end());
  
  arma::vec out (seconds_double);
  return out;
}

void TimeLogger::ClearLoggerData ()
{
  times_seconds.clear();
}

std::string TimeLogger::InitializeLoggerPrinter () const
{
  std::stringstream ss;
  ss << std::setw(17) << time_unit;
  
  return ss.str();
}

std::string TimeLogger::PrintLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << times_seconds.back();
  
  return ss.str();
}

} // namespace logger
