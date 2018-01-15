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
//   The main "Compboost" class collects all used elements of the algorithm
//   like loss, optimizer and logger and runs the main algorithm by calling
//   the "Train" member function.
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




// THIS ONE IS UNDER PROGRESS!



#ifndef COMPBOOST_H_
#define COMPBOOST_H_

#include "baselearner_track.h"
#include "optimizer.h"
#include "loss.h"
#include "loggerlist.h"

namespace cboost {

// Main class:

class Compboost
{

  private:
    
    arma::vec response;
    arma::vec pseudo_residuals;
    arma::vec model_prediction;
    
    double learning_rate;
    double initialization;
    
    bool stop_if_all_stopper_fulfilled;
    
    // Pieces to run the algorithm:
    blearnertrack::BaselearnerTrack blearner_track = blearnertrack::BaselearnerTrack();
    optimizer::Optimizer* used_optimizer;
    loss::Loss* used_loss;
    blearnerlist::BaselearnerList used_baselearner_list;
    loggerlist::LoggerList* used_logger;
  
  public:

    Compboost ();
    
    Compboost (arma::vec, double, bool, optimizer::Optimizer*, loss::Loss*, 
      loggerlist::LoggerList*, blearnerlist::BaselearnerList);
    
    void TrainCompboost (bool);
    
    arma::vec GetPrediction ();
    
    std::map<std::string, arma::mat> GetParameter ();
    std::vector<std::string> GetSelectedBaselearner ();

    // arma::vec PredictEnsemble ();
    // arma::vec PredictEnsemble (arma::mat &);
    
    // Destructor:
    ~Compboost ();
    
};

} // namespace cboost

#endif // COMPBOOST_H_