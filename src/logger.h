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
//   Logger implementations. This file also includes if a logger is used as
//   stopper and a function to determine if the algorithm should stop or not.
//   The logger childs can just use some basic objects which are given from
//   the main algorithm:
//
//     - Current Iteration (unsigned int): Classic way to stop the algorithm
//     - Current time point (chrono::system_clock::time_point): With that it
//       is possible to run the algorithm just for 2 hours and then stop it
//     - https://github.com/schalkdaniel/compboost/issues/56
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

#ifndef LOGGER_H_
#define LOGGER_H_

#include <vector>
#include <chrono>
#include <iomanip> // ::setw
#include <sstream> // ::stringstream

#include "loss.h"
#include "baselearner.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

class Logger
{
public:
  
  virtual void LogStep (const unsigned int&, const arma::vec&, const arma::vec&, 
    blearner::Baselearner*, const double&, const double&) = 0;
  
  // This one should check if the stop criteria is reached. If not it should
  // return 'true' otherwise 'false'. Every function should have this 
  // structure:
  
  // bool ReachedStopCriteria ()
  // {
  //   bool stop_criteria_is_reached;
  //
  //   if (is_a_stopper) {
  //     if (CHECK IF STOP CRITERIA IS FULLFILLED!) {
  //       stop_criteria_is_reached = true;
  //     }      
  //   } else {
  //     stop_criteria_is_reached = false;
  //   }
  //   return stop_criteria_is_reached;
  // }
  virtual bool ReachedStopCriteria () const = 0;
  
  virtual arma::vec GetLoggedData () const = 0;
  
  virtual std::string InitializeLoggerPrinter () const = 0;
  virtual std::string PrintLoggerStatus () const = 0;
  
  virtual void ClearLoggerData() = 0;
  
  bool GetIfLoggerIsStopper () const;
  
  virtual 
    ~Logger ();
  
protected:
  
  loss::Loss* used_loss;
  
  bool is_a_stopper;
  
  // Pointer to the publics of the loggerlist. The child classes then change
  // the value of the pointed values to update the steps.
  double init_risk;
  
};

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// IterationLogger:
// -----------------------

// This one is the default one:

class IterationLogger : public Logger 
{
private:
  
  unsigned int max_iterations;
  std::vector<unsigned int> iterations;
  
public:
  
  IterationLogger (const bool&, const unsigned int&);
  
  // This just loggs the iteration (unsigned int):
  void LogStep (const unsigned int&, const arma::vec&, const arma::vec&, 
    blearner::Baselearner*, const double&, const double&);
  
  bool ReachedStopCriteria () const;
  arma::vec GetLoggedData () const;
  void ClearLoggerData();
  
  std::string InitializeLoggerPrinter () const;
  std::string PrintLoggerStatus () const;
  
};

// InbagRisk:
// -----------------------

class InbagRiskLogger : public Logger
{
private:
  loss::Loss* used_loss;
  std::vector<double> tracked_inbag_risk;
  double eps_for_break;
  
public:
  InbagRiskLogger (const bool&, loss::Loss*, const double&);
  
  void LogStep (const unsigned int&, const arma::vec&, const arma::vec&, 
    blearner::Baselearner*, const double&, const double&);
  
  bool ReachedStopCriteria () const;
  arma::vec GetLoggedData () const;
  void ClearLoggerData();
  
  std::string InitializeLoggerPrinter () const;
  std::string PrintLoggerStatus () const;
  
};

// OobRisk:
// -----------------------

class OobRiskLogger : public Logger
{
private:
  loss::Loss* used_loss;
  std::vector<double> tracked_oob_risk;
  double eps_for_break;
  arma::vec oob_prediction;
  std::map<std::string, data::Data*> oob_data;
  arma::vec oob_response;
  
public:
  OobRiskLogger (const bool&, loss::Loss*, const double&, 
    std::map<std::string, data::Data*>, const arma::vec&);
  
  void LogStep (const unsigned int&, const arma::vec&, const arma::vec&, 
    blearner::Baselearner*, const double&, const double&);
  
  bool ReachedStopCriteria () const;
  arma::vec GetLoggedData () const;
  void ClearLoggerData();
  
  std::string InitializeLoggerPrinter () const;
  std::string PrintLoggerStatus () const;
  
};

// TimeLogger:
// -----------------------

class TimeLogger : public Logger
{
private:
  
  std::chrono::steady_clock::time_point init_time;
  std::vector<unsigned int> times_seconds;
  unsigned int max_time;
  std::string time_unit;
  
public:
  
  TimeLogger (const bool&, const unsigned int&, const std::string&);
  
  void LogStep (const unsigned int&, const arma::vec&, const arma::vec&, 
    blearner::Baselearner*, const double&, const double&);
  
  bool ReachedStopCriteria () const;
  arma::vec GetLoggedData () const;
  void ClearLoggerData();
  
  std::string InitializeLoggerPrinter () const;
  std::string PrintLoggerStatus () const;
  
};

} // namespace logger

#endif // LOGGER_H_
