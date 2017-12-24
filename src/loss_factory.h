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
//   "LossFactory" class
//
//     - Default loss is defined in empty constructor as quadratic loss
//     - Other losses can be accessed via a string as constructor parameter.
//       Implemented losses are:
//         + quadratic
//         + absolute
//     - The loss and gradient function are defined in 'loss_definition.h'. The
//       Strategy is to define a parent class 'LossDefinition' which calls the
//       child member functions of the child classes 'Quadratic', 'Absolute'
//       etc.
//     - The private member 'loss_type' is just for controlling which loss is
//       selected.
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

#ifndef LOSSFACTORY_H_
#define LOSSFACTORY_H_

#include "loss.h"

#include <RcppArmadillo.h>

#include <iostream>
#include <string>

namespace lossfactory {

class LossFactory
{
  private:
    
    loss::Loss *loss_obj;
    std::string loss_type;
    
  public:
    
    LossFactory (); // Default is quadratic loss
    LossFactory (std::string); // Constructor returns different loss classes e.g. quadratic
    LossFactory (std::string, Rcpp::Function, Rcpp::Function, Rcpp::Function); // Constructor for own R losses
    
    arma::vec CalcLoss (arma::vec &, arma::vec &);
    arma::vec CalcGradient (arma::vec &, arma::vec &);
    double ConstantInitializer (arma::vec &);
    
    std::string GetLossType ();
};

} // namespace lossfactory

#endif // LOSSFACTORY_H_