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
// it under the terms of the MIT License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// MIT License for more details. You should have received a copy of 
// the MIT License along with compboost. 
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Department of Statistics
//   Ludwig-Maximilians-University Munich
//   Ludwigstrasse 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
//   Contact
//   e: contact@danielschalk.com
//   w: danielschalk.com
//
// =========================================================================== #

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <iostream>
#include <map>
#include <limits>

#include <RcppArmadillo.h>

#include "baselearner.h"
#include "baselearner_factory_list.h"

namespace optimizer {

// -------------------------------------------------------------------------- //
// Abstract 'Optimizer' class:
// -------------------------------------------------------------------------- //

class Optimizer
{
  public:
    
    virtual blearner::Baselearner* findBestBaselearner (const std::string&, 
      const arma::vec&, const blearner_factory_map&) const = 0;
    
    virtual ~Optimizer ();

  protected:
    
    blearner_factory_map my_blearner_factory_map;

};

// -------------------------------------------------------------------------- //
// Optimizer implementations:
// -------------------------------------------------------------------------- //

// Greedy:
// -----------------------

class OptimizerCoordinateDescent : public Optimizer
{
  public:
    
    // No special initialization necessary:
    OptimizerCoordinateDescent ();

    blearner::Baselearner* findBestBaselearner (const std::string&, 
      const arma::vec&, const blearner_factory_map&) const;
};


} // namespace optimizer

#endif // OPTIMIZER_H_
