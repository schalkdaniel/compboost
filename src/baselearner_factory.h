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
//   "BaselearnerFactory" class. This file implements the factorys as well as  
//   the classes which are made in those factorys. The reason behind that pattern
//   is that every baselearner have its own data. This data is stored within
//   the factory. Every baselearner which is created in the factory points to
//   that data (if possible).
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

#ifndef BASELEARNERFACTORY_H_
#define BASELEARNERFACTORY_H_

#include <RcppArmadillo.h>

#include <iostream>
#include <string>

#include "baselearner.h"

namespace blearnerfactory {

// linear baselearner:
// -----------------------

class BaselearnerFactory
{

  public:
    
    
    BaselearnerFactory (std::string, arma::mat);
    
    // Ordinary definition:
    blearner::Baselearner *CreateBaselearner (std::string);
    // Definition of custom baselearner:
    blearner::Baselearner *CreateBaselearner (std::string, Rcpp::Function, 
      Rcpp::Function, Rcpp::Function, Rcpp::Function);
    
    bool GetCheck ();
    
    std::string GetBaselearnerType ();
    
  private:
    
    std::string blearner_type;
    arma::mat data;
    bool data_check = false;
  
};

} // namespace blearnerfactory

#endif // BASELEARNERFACTORY_H_