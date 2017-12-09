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
//   "Baselearner" classes. This file implements the factorys as well as the 
//   classes which are made in those factorys. The reason behind that pattern
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

#ifndef BASELEARNER_H_
#define BASELEARNER_H_

#include <RcppArmadillo.h>
#include <string>

namespace blearner {

// linear baselearner:
// -----------------------

class Linear
{
  private:
    
    arma::vec parameter;
    arma::mat *data_ptr;
    
  public:
    
    Linear (arma::vec &, arma::mat );
    
    arma::vec GetParameter ();
    
    arma::mat predict ();
    arma::mat predict (arma::mat &);
  
};

// Factory class:
// -----------------------

// The parent class includes the minimal functionality every baselearner
// must have.

class LinearFactory
{
  private:
  
    // The data member should be stored as pointer (to save memory)
    arma::mat data;
    std::string blearner_identifier;
  
  public:
  
    LinearFactory (arma::mat &, std::string &);
    
    arma::mat GetData ();
    std::string GetIdentifier ();
    
    // Factory method which instantly train the baselearner
    Linear *TrainBaselearner (arma::vec &);
    
};




} // namespace cboost

#endif // BASELEARNER_H_