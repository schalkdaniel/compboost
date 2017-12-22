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
//   "BaselearnerFactory" class
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

#include <RcppArmadillo.h>
#include "baselearner_factory.h"

namespace blearnerfactory {

BaselearnerFactory::BaselearnerFactory (std::string blearner_type0, arma::mat data0)
{
  blearner_type = blearner_type0;
  
  // Initialize data with the pointer to the original ones. The data will be
  // transformed to the data used for modelling within the function 
  // 'CreateBaselearner'.
  data = data0;
}

blearner::Baselearner * BaselearnerFactory::CreateBaselearner (std::string identifier)
{
  std::cout << "New " 
            << blearner_type 
            << " baselearner "   
            << identifier 
            << " will be created." 
            << std::endl;
  
  blearner::Baselearner *blearner_obj;
  
  // Create new baselearner. This one will be returned by the factory:
  if (blearner_type == "linear") {
    blearner_obj = new blearner::Linear(data, identifier);
  }
  if (blearner_type == "quadratic") {
    blearner_obj = new blearner::Quadratic(data, identifier);
  }
  // Check if the data is already set. If not, run 'TransformData' from the
  // baselearner:
  if (! data_check) {
    std::cout << "Transform data as specified in 'TransformData'." << std::endl;
    data = blearner_obj->TransformData();
    
    data_check = true;
  }
  return blearner_obj;
}

blearner::Baselearner * BaselearnerFactory::CreateBaselearner (std::string identifier,
  Rcpp::Function transformDataFun, Rcpp::Function trainFun, Rcpp::Function predictFun, 
  Rcpp::Function extractParameter)
{
  std::cout << "New custom baselearner "   
            << identifier 
            << " will be created." 
            << std::endl;
  
  blearner::Baselearner *blearner_obj;
  
  blearner_obj = new blearner::Custom(data, identifier, transformDataFun, 
    trainFun, predictFun, extractParameter);
  
  // Check if the data is already set. If not, run 'TransformData' from the
  // baselearner:
  if (! data_check) {
    std::cout << "Transform data as specified in 'TransformData'." << std::endl;
    data = blearner_obj->TransformData();
    
    data_check = true;
  }
  return blearner_obj;
}

bool BaselearnerFactory::GetCheck ()
{
  return data_check;
}

std::string BaselearnerFactory::GetBaselearnerType()
{
  return blearner_type;
}

} // namespace blearnerfactory