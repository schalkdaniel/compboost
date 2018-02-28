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
    bool model_is_trained = false;
    
    unsigned int actual_state;
    
    // Pieces to run the algorithm:
    blearnertrack::BaselearnerTrack blearner_track;
    optimizer::Optimizer* used_optimizer;
    loss::Loss* used_loss;
    blearnerlist::BaselearnerFactoryList used_baselearner_list;
    
    // Vector of loggerlists, needed if one want to continue training:
    std::map<std::string, loggerlist::LoggerList*> used_logger;
    // loggerlist::LoggerList* used_logger;
  
  public:

    Compboost ();
    
    Compboost (arma::vec, double, bool, optimizer::Optimizer*, loss::Loss*, 
      loggerlist::LoggerList*, blearnerlist::BaselearnerFactoryList);
    
    // Basic train function used by TrainCompbost and ContinueTraining:
    void Train (const bool&, const arma::vec&, loggerlist::LoggerList*);
    
    // Initial training:
    void TrainCompboost (const bool&);
    
    // Retraining after initial training:
    void ContinueTraining (loggerlist::LoggerList*, const bool&);
    
    arma::vec GetPrediction ();
    
    std::map<std::string, arma::mat> GetParameter ();
    std::vector<std::string> GetSelectedBaselearner ();
    
    std::map<std::string, loggerlist::LoggerList*> GetLoggerList () const;
    std::map<std::string, arma::mat> GetParameterOfIteration (unsigned int);
    
    std::pair<std::vector<std::string>, arma::mat> GetParameterMatrix ();
    
    arma::vec Predict ();
    arma::vec Predict (std::map<std::string, arma::mat>);
    arma::vec PredictionOfIteration (std::map<std::string, arma::mat>, unsigned int);
    
    void SetToIteration (const unsigned int&);
    
    void SummarizeCompboost ();
    
    // Destructor:
    ~Compboost ();
    
};

} // namespace cboost

#endif // COMPBOOST_H_
