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

#ifndef LOGGERLIST_H_
#define LOGGERLIST_H_

#include "logger.h"

typedef std::map<std::string, logger::Logger *> logger_map;

namespace loggerlist
{

class LoggerList
{

  private:
    
    logger_map log_list;
    
    // Pointer to the data which should be used for evaluation. If the pointer
    // is a null pointer, than the training data should be used.
    // if (! evaluation_data_ptr) { USE TRAINING DATA } ele { USE GIVEN DATA }
    arma::mat *evaluation_data_ptr;
    
  public:
  
    LoggerList (arma::mat &);
    
    // String for logger, the logger itselfe and a bool if this one should be
    // used as stopper:
    void RegisterLogger (std::string, logger::Logger *, bool);
    void PrintRegisteredLogger ();
    
    logger_map GetMap ();
    void ClearMap ();
    
    // This function should iterate over all registered logger, check if it is
    // a stopper and returns just one bool, aggregated over a vector of bools
    // from the single logger. This could be e.g. a priority check (risk is
    // more important than iterations) or an all check (all stopper has to be
    // fullfilled). The priority comes with the map identifier since it sorts
    // the entrys after name.
    
    // If the argument is 'true', than all stopper has to be fullfilled.
    bool GetStopperStatus (bool);
    
};

} // namespace loggerlist

#endif // LOGGERLIST_H_