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
    virtual arma::mat InstantiateData() = 0;
    
    // Clone function (needed for the main algorithm)
    virtual Baselearner *Clone () = 0;
    
    // Copy function to set all members within the copy:
    void CopyMembers (arma::mat, std::string, arma::mat &, std::string &);
    
    // Within 'SetData' the pointer will be setted, while 'InstantiateData'
    // overwrite the object on which 'data_ptr' points. This guarantes that 
    // the data is just stored once in the factory and then called by reference
    // within the baselearner:
    void SetData (arma::mat &);
    arma::mat GetData ();
    
    void SetDataIdentifier (std::string &);
    std::string GetDataIdentifier ();
    
    arma::mat GetParameter ();
    arma::mat predict ();
    
    void SetIdentifier (std::string);
    std::string GetIdentifier ();
    
  protected:
    arma::mat parameter;
    std::string blearner_identifier;
    arma::mat *data_ptr;
    std::string *data_identifier_ptr;
    
};

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// Polynomial:
// -----------------------

// The parent class includes the minimal functionality every baselearner
// must have.

class Polynomial : public Baselearner
{
  private:
    unsigned int degree;
    
  public:
  
    Polynomial (arma::mat &, std::string &, std::string &, unsigned int &);
    
    Baselearner *Clone ();
    
    arma::mat InstantiateData ();
    
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
    
    Rcpp::Function instantiateDataFun;
    Rcpp::Function trainFun;
    Rcpp::Function predictFun;
    Rcpp::Function extractParameter;
    
  public:
    
    Custom (arma::mat &, std::string &, std::string &, Rcpp::Function, Rcpp::Function, 
      Rcpp::Function, Rcpp::Function);
    
    Baselearner *Clone ();
    
    arma::mat InstantiateData ();
    
    void train (arma::vec &);
    arma::mat predict (arma::mat &);
	
};


} // namespace blearner

#endif // BASELEARNER_H_