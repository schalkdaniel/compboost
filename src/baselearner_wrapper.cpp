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
//   The "LinearWrapper" class which is exported to R by using the Rcpp
//   modules. For a tutorial see
//   <http://dirk.eddelbuettel.com/code/rcpp/Rcpp-modules.pdf>.
//
//   This one is just to play around with! It doesn't have any real usecase!!!
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

#include "baselearner.h"

class LinearWrapper
{
private:

  blearner::LinearFactory *obj;

public:
  // Constructors
  // ------------
  
  LinearWrapper (arma::mat X, std::string id)
  {
    obj = new blearner::LinearFactory(X, id);
  }

  // Member functions
  // ----------------
  
  arma::vec train (arma::vec &response)
  {
    blearner::Linear *newobj = obj->TrainBaselearner(response);
    return newobj->GetParameter();
  }
  
  // Very stupid at the moment since it first calculate the parameter again
  // and then makes the prediction
  arma::mat predict (arma::vec &response)
  {
    blearner::Linear *newobj = obj->TrainBaselearner(response);
    return newobj->predict();
  }
  
  std::string GetId () 
  {
    return obj->GetIdentifier();
  }
  
  arma::mat GetData () 
  {
    return obj->GetData();
  }
};

// --------------------------------------------------------------------------- #
// Rcpp module:
// --------------------------------------------------------------------------- #

RCPP_MODULE(baselearner_module) {

  using namespace Rcpp;

  class_<LinearWrapper> ("LinearWrapper")

    .constructor <arma::mat, std::string> ("Initialize LinearWrapper (Baselearner) object")

    .method ("train", &LinearWrapper::train, "Train a linear baselearner")
    .method ("predict", &LinearWrapper::predict, "Predict a linear baselearner")
    .method ("GetId", &LinearWrapper::GetId, "Get the identifier")
    .method ("GetData", &LinearWrapper::GetData, "Get the data")
    ;
}
