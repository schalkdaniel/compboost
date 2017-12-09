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
//   "Baselearner" class
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München

//   https://www.compstat.statistik.uni-muenchen.de
//
// =========================================================================== #


#include "baselearner.h"
#include <iostream>

namespace blearner {

// --------------------------------------------------------------------------- #
// Linear:
// --------------------------------------------------------------------------- #

Linear::Linear (arma::vec &response, arma::mat data)
{
  data_ptr  = &data;
  parameter = arma::solve(data, response);
}

arma::vec Linear::GetParameter ()
{
  return parameter;
}

arma::mat Linear::predict ()
{
  return *data_ptr * parameter;
}

arma::mat Linear::predict (arma::mat &newdata)
{
  return newdata * parameter;
}



// --------------------------------------------------------------------------- #
// LinearFactory:
// --------------------------------------------------------------------------- #

LinearFactory::LinearFactory (arma::mat &data0, std::string &blearner_identifier0)
{
  data = data0;
  blearner_identifier = blearner_identifier0;
}

arma::mat LinearFactory::GetData ()
{
  return data;
}

std::string LinearFactory::GetIdentifier ()
{
  return blearner_identifier;
}

Linear *LinearFactory::TrainBaselearner (arma::vec &response)
{
  return new Linear (response, data);
}

} // namespace blearner