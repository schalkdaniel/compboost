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

#ifndef TACKBASELEARNER_H_
#define TACKBASELEARNER_H_

#include <map>
#include <RcppArmadillo.h>

#include "baselearner.h"
#include "baselearner_list.h"
#include "loggerlist.h"

typedef std::map<unsigned int, blearner::Baselearner *> selected_blearner_map;

namespace trackblearner
{

class TrackBaselearner
{
  private:
    
    selected_blearner_map blearner_map;
    
    loggerlist::LoggerList blearner_log;
    
  public: 
    
    TrackBaselearner (blearnerlist::BaselearnerList);
    
    blearner::Baselearner GetBaselearnerNumber (unsigned int);
    
    arma::mat GetParameterEstimator ();
    arma::mat GetParameterEstimator (unsigned int);
    
    arma::mat GetParameterMatrix ();
    
    arma::mat PredictEnsemble ();
    arma::mat PredictEnsemble (arma::mat &);
};

} // namespace trackblearner

#endif // TRACKBASELEARNER_H_