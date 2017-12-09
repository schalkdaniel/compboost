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
//   Implementation for the 'Loss' class.
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

#include "loss.h"

namespace loss {

// --------------------------------------------------------------------------- #
// Constructors:
// --------------------------------------------------------------------------- #

Loss::Loss ()
{
  // for debugging:
  std::cout << "Create new Loss (default: quadratic): "
            << &loss_obj
            << std::endl;

  loss_obj = new lossdef::Quadratic;
  loss_type = "quadratic";
}

Loss::Loss (std::string loss_type0)
{
  // for debugging:
  std::cout << "Create new Loss with a specific type: "
            << loss_type0
            << ": "
            << &loss_obj
            << std::endl;
  
  loss_type = loss_type0;

  // if statements to dynamically declare loss
  if (loss_type0 == "quadratic") { loss_obj  = new lossdef::Quadratic; }
  if (loss_type0 == "absolute")  { loss_obj  = new lossdef::Absolute;  }
  
  // loss_obj  = LossFactory::LossFactory.CreateInstance(loss_type0)
  // loss_type = loss_type0;
}

Loss::Loss (std::string loss_type0, Rcpp::Function lossFun, Rcpp::Function gradientFun, Rcpp::Function initFun)
{
  loss_obj = new lossdef::CustomLoss(lossFun, gradientFun, initFun);
  loss_type = loss_type0;
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

arma::vec Loss::CalcLoss (arma::vec &true_value, arma::vec &prediction)
{
  // Call DefinedLoss function of the child class:
  return loss_obj->DefinedLoss(true_value, prediction);
}

arma::vec Loss::CalcGradient (arma::vec &true_value, arma::vec &prediction)
{
  // Call DefinedGradient of the child class:
  return loss_obj->DefinedGradient(true_value, prediction);
}

arma::vec Loss::ConstantInitializer (arma::vec &true_value)
{
  // Call ConstantInitializer of the child class:
  return loss_obj->ConstantInitializer(true_value);
}

std::string Loss::GetLossType ()
{
  return loss_type;
}

} // namespace loss