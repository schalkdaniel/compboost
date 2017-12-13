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

namespace blearner {

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

void Baselearner::SetData (arma::mat &data)
{
  data_ptr = &data;
}

arma::mat Baselearner::GetData ()
{
  return *data_ptr;
}

arma::mat Baselearner::GetParameter () 
{
  return parameter;
}

void Baselearner::SetIdentifier (std::string id0)
{
  blearner_identifier = id0;
}

std::string Baselearner::GetIdentifier ()
{
  return blearner_identifier;
}

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// Linear:
// -----------------------

Linear::Linear (arma::mat &data, std::string &identifier)
{
  // Called from parent class 'Baselearner':
  Baselearner::SetData(data);
  Baselearner::SetIdentifier(identifier);
}

void Linear::train (arma::vec &response)
{
  parameter = arma::solve(*data_ptr, response);
}

arma::mat Linear::predict ()
{
  return *data_ptr * parameter;
}

arma::mat Linear::predict (arma::mat &newdata)
{
  return newdata * parameter;
}

} // namespace blearner