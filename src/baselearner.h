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

// -------------------------------------------------------------------------- //
// Abstract 'Baselearner' class:
// -------------------------------------------------------------------------- //

class Baselearner
{
  public:
    
    virtual void train (arma::vec &) = 0;
    virtual arma::mat predict (arma::mat &) = 0;
    virtual arma::mat TransformData() = 0;
    
    // Within 'SetData' the pointer will be setted, while 'TransformData'
    // overwrite the object on which 'data_ptr' points. This guarantes that 
    // the data is just stored once in the factory and then called by reference
    // within the baselearner:
    void SetData (arma::mat &);
    arma::mat GetData ();
    
    arma::mat GetParameter ();
    arma::mat predict ();
    
    void SetIdentifier (std::string);
    std::string GetIdentifier ();
    
  protected:
    arma::mat parameter;
    std::string blearner_identifier;
    arma::mat *data_ptr;
    
};

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// Linear:
// -----------------------

// The parent class includes the minimal functionality every baselearner
// must have.

class Linear : public Baselearner
{

  public:
  
    Linear (arma::mat &, std::string &);
    
    arma::mat TransformData ();
    
    void train (arma::vec &);
    arma::mat predict (arma::mat &);
};

// Quadratic:
// -----------------------

// The parent class includes the minimal functionality every baselearner
// must have.

class Quadratic : public Baselearner
{
  
  public:
    
    Quadratic (arma::mat &, std::string &);
    
    arma::mat TransformData ();
    
    void train (arma::vec &);
    arma::mat predict (arma::mat &);
};

// Custom Baselearner:
// -----------------------
// The parent class includes the minimal functionality every baselearner
// must have.

class Custom : public Baselearner
{
  private:
    
    SEXP model_frame;
    
    Rcpp::Function transformDataFun;
    Rcpp::Function trainFun;
    Rcpp::Function predictFun;
    Rcpp::Function extractParameter;
    
  public:
    
    Custom (arma::mat &, std::string &, Rcpp::Function, Rcpp::Function, 
      Rcpp::Function, Rcpp::Function);
    
    arma::mat TransformData ();
    
    void train (arma::vec &);
    arma::mat predict (arma::mat &);
};


} // namespace blearner

#endif // BASELEARNER_H_