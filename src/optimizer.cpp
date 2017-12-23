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
// ========================================================================== //

#include "optimizer.h"

namespace optimizer {

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

void Optimizer::SetFactoryMap (blearnerlist::BaselearnerList & factory_list)
{
  my_blearner_factory_map = factory_list.GetMap();
}

// -------------------------------------------------------------------------- //
// Optimizer implementations:
// -------------------------------------------------------------------------- //

// Greedy:
// -----------------------

Greedy::Greedy (blearnerlist::BaselearnerList & factory_list)
{
  SetFactoryMap(factory_list);
}

blearner::Baselearner *Greedy::FindBestBaselearner (std::string &iteration_id, arma::vec &pseudo_residuals)
{
  double ssq_temp;
  double ssq_best = 0;
  
  blearner::Baselearner *blearner_temp;
  blearner::Baselearner *blearner_best;
  
  // The use of k crashes the system???? 
  // unsigned int k = 0;
  // arma::vec ssq(my_blearner_factory_map.size());

  for (blearner_factory_map::iterator it = my_blearner_factory_map.begin(); it != my_blearner_factory_map.end(); ++it) {

    std::string id = "(" + iteration_id + ") " + it->second->GetBaselearnerType();
    
    blearner_temp = it->second->CreateBaselearner(id);
    blearner_temp->train(pseudo_residuals);
    
    ssq_temp = arma::accu(arma::pow(blearner_temp->predict() - pseudo_residuals, 2)) / pseudo_residuals.size();
    // ssq[k] = arma::accu(arma::pow(blearner_temp->predict() - pseudo_residuals, 2)) / pseudo_residuals.size();
    
    if (ssq_best == 0) {
      ssq_best = ssq_temp;
      blearner_best = blearner_temp->Clone();
    }
    
    if (ssq_temp < ssq_best) {
      ssq_best = ssq_temp;
      blearner_best = blearner_temp->Clone();
    }
    
    // if (k > 0) {
    //   if (ssq[k] < ssq[k - 1]) {
    //     blearner_best = blearner_temp->Clone();
    //   }
    // }
    // k += 1;
  }
  return blearner_best;
}

} // namespace optimizer