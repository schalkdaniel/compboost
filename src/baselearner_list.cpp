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

#include "baselearner_list.h"

namespace blearnerlist 
{

BaselearnerList::BaselearnerList ()
{
  // std::cout << "Initialize Baselearnerlist!" << std::endl;
}

void BaselearnerList::RegisterBaselearnerFactory (std::string factory_id, blearnerfactory::BaselearnerFactory blearner_factory)
{
  my_factory_map.insert(std::pair<std::string, blearnerfactory::BaselearnerFactory>(factory_id, blearner_factory));
}

void BaselearnerList::PrintRegisteredFactorys ()
{
  std::cout << "Registered Factorys:\n";
  for (blearner_factory_map::iterator it = my_factory_map.begin(); it != my_factory_map.end(); ++it) {
    std::cout << "\t>>" << it->first << "<< Factory" << std::endl;
  }
}

blearner_factory_map BaselearnerList::GetMap ()
{
  return my_factory_map;
}

void BaselearnerList::ClearMap ()
{
  my_factory_map.clear();
}

} // namespace blearnerlist
