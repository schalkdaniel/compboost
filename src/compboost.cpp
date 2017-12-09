// ========================================================================== \\
//                                 ___.                          __           \\
//        ____  ____   _____ ______\_ |__   ____   ____  _______/  |_         \\
//      _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\        \\
//      \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |          \\
//       \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|          \\
//           \/            \/|__|       \/                  \/                \\
//                                                                            \\
// ========================================================================== \\
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
//   Constructors and member function implementations for the class
//   "Compboost".
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

#include "compboost.h"

#include <iostream>

namespace cboost {

// --------------------------------------------------------------------------- #
// Constructors:
// --------------------------------------------------------------------------- #

Compboost::Compboost ()
{
  std::cout << "A new Compboost object has ben created!" << std::endl;
}

// --------------------------------------------------------------------------- #
// Member functions:
// --------------------------------------------------------------------------- #

void Compboost::SetResponse (arma::vec response0)
{
  response = response0;
}

arma::vec Compboost::GetResponse ()
{
  return response;
}

} // namespace cboost