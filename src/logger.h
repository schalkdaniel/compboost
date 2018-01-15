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
// =========================================================================== #

#ifndef LOGGER_H_
#define LOGGER_H_

#include <vector>
#include <chrono>
#include <iomanip> // ::setw
#include <sstream> // ::stringstream

#include "loss.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

class Logger
{

  public:
    
    virtual void LogStep (unsigned int, double) = 0;
    
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
    virtual bool ReachedStopCriteria () = 0;
    
    virtual arma::vec GetLoggedData () = 0;
    
    virtual std::string InitializeLoggerPrinter () = 0;
    virtual std::string PrintLoggerStatus () = 0;
    
    virtual void ClearLoggerData() = 0;
    
    bool GetIfLoggerIsStopper ();
    
    virtual ~Logger ();
    
  protected:
    
    loss::Loss* used_loss;
    arma::mat* evaluation_data;
    
    bool is_a_stopper;
    
    // Pointer to the publics of the loggerlist. The child classes then change
    // the value of the pointed values to update the steps.
    double init_risk;
     
};

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// LogIteration:
// -----------------------

// This one is the default one:

class LogIteration : public Logger 
{
  private:
  
    unsigned int max_iterations;
    std::vector<unsigned int> iterations;
    
  public:
    
    LogIteration (bool, unsigned int);
    
    // This just loggs the iteration (unsigned int):
    void LogStep (unsigned int, double);
    bool ReachedStopCriteria ();
    arma::vec GetLoggedData ();
    void ClearLoggerData();
    
    std::string InitializeLoggerPrinter ();
    std::string PrintLoggerStatus ();
    
};

// LogRisk:
// -----------------------

// LogTime:
// -----------------------

class LogTime : public Logger
{
  private:
    
    std::chrono::steady_clock::time_point init_time;
    std::vector<unsigned int> times_seconds;
    unsigned int max_time;
    std::string time_unit;
    
  public:
    
    LogTime (bool, unsigned int, std::string);
    
    void LogStep (unsigned int, double);
    bool ReachedStopCriteria ();
    arma::vec GetLoggedData ();
    void ClearLoggerData();
    
    std::string InitializeLoggerPrinter ();
    std::string PrintLoggerStatus ();
    
};

} // namespace logger

#endif // LOGGER_H_