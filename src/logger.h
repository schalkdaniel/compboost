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

#ifndef LOGGER_H_
#define LOGGER_H_

#include "loss.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

class Logger
{
  private:
  
    loss::Loss *used_loss;
    arma::mat evaluation_data;
    
    bool is_a_stopper;

  public:
    
    virtual void LogStep () = 0;
    
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
     
};

// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// LogIteration:
// -----------------------

// LogRisk:
// -----------------------

// LogTime:
// -----------------------



} // namespace logger

#endif // LOGGER_H_