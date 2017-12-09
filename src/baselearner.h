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
//
//   https://www.compstat.statistik.uni-muenchen.de
//
// =========================================================================== #

#ifndef BASELEARNER_H_
#define BASELEARNER_H_

#include <RcppArmadillo.h>
#include <string>

namespace blearner {

// Parent class:
// -----------------------

// The parent class includes the minimal functionality every baselearner
// must have.

class Baselearner
{
  private:
  
    arma::mat data;
    std::string blearner_identifier;
  
  public:
  
    Baselearner (std::string);
    
    void SetData (arma::mat);
    arma::mat GetData ();
    
    // Functions which will be overloaded by the child functions. 
    virtual void train () = 0;
    virtual arma::mat predict () = 0;
    virtual arma::mat predict (arma::mat newdata) = 0;
    
};

// --------------------------------------------------------------------------- #
// Child classes, acutal the real baselearner:
// --------------------------------------------------------------------------- #

// linear baselearner:
// -----------------------

class Linear: public Baselearner
{
  private:
    
    arma::vec parameter;
    
  public:
    
    void train ();
    arma::mat predict ();
    arma::mat predict (arma::mat newdata);
    
};

} // namespace cboost

#endif // BASELEARNER_H_