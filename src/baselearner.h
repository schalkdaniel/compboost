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
//   Implementation of the "Baselearner" class. The abstract parent class
//   has the following virtual functions which has to be declared in the 
//   child classes:
//
//      - virtual void train (arma::vec&) = 0;
//      - virtual arma::mat predict (arma::mat&) = 0;
//      - virtual arma::mat InstantiateData () = 0;
//      - virtual arma::mat InstantiateData (arma::mat&) = 0;
//      - virtual Baselearner *Clone () = 0;
//      - void CopyMembers (arma::mat, std::string, arma::mat &, std::string &);
//
//   The "Baselearner" class contains, as expected, the information about the
//   data transformation and the way the training and prediction is done.
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
    
    virtual void train (arma::vec&) = 0;
    virtual arma::mat predict (arma::mat&) = 0;
    
    // // Specify how the data has to be transformed. E. g. for splines a mapping
    // // to the higher dimension space. The overloading function with the 
    // // arma mat as parameter is used for newdata:
    // virtual arma::mat InstantiateData () = 0;
    virtual arma::mat InstantiateData (arma::mat&) = 0;
    
    // Clone function (in some places needed e. g. "optimizer.cpp"):
    virtual Baselearner *Clone () = 0;
    
    // Copy function to set all members within the copy. It is more convenient
    // to do this once instead of copy and pasting every element every time:
    void CopyMembers (arma::mat, std::string, arma::mat &);
    
    // Within 'SetData' the pointer will be setted, while 'InstantiateData'
    // overwrite the object on which 'data_ptr' points. This guarantees that 
    // the data is just stored once in the factory and then called by reference
    // within the baselearner:
    void SetData (arma::mat&);
    arma::mat GetData ();
    
    // Set an identifier for the data. This is important for fitting to new
    // data:
    void SetDataIdentifier (std::string&);
    std::string GetDataIdentifier ();
    
    arma::mat GetParameter ();
    
    // This function just calls the virtual one with the data pointer. This is
    // done to avoid duplicates within the child classes:
    virtual arma::mat predict () = 0;
    
    // Set and get identifier of a specific baselearner (this is unique):
    void SetIdentifier (std::string);
    std::string GetIdentifier ();
    
    // Set and get baselearner type (this can be the same for multiple 
    // baselearner e. g. linear baselearner for variable x1 and x2).
    // This one is setted by the factory which later creates the objects:
    void SetBaselearnerType (std::string&);
    std::string GetBaselearnerType ();
    
    // Destructor:
    virtual ~Baselearner ();
    
  protected:
    
    // Members which should be directly accessible through the child classes:
    arma::mat parameter;
    std::string blearner_identifier;
    std::string* blearner_type;
    arma::mat* data_ptr;
    std::string data_identifier;
    
};

// -------------------------------------------------------------------------- //
// Baselearner implementations:
// -------------------------------------------------------------------------- //

// PolynomialBlearner:
// -----------------------


// This baselearner trains a linear model without intercept and covariable
// x^degree:

class PolynomialBlearner : public Baselearner
{
  private:
    
    unsigned int degree;
    
  public:
  
    // (data pointer, data identifier, baselearner identifier, degree) 
    PolynomialBlearner (arma::mat&, std::string&, std::string&, unsigned int&);
    
    Baselearner* Clone ();
    
    // arma::mat InstantiateData ();
    arma::mat InstantiateData (arma::mat&);
    
    void train (arma::vec&);
    arma::mat predict (arma::mat&);
    arma::mat predict ();
    
    ~PolynomialBlearner ();
	
};

// Custom Baselearner:
// -----------------------

// This class can be used to define custom baselearner in R and expose thi
// to the c++ class:

class Custom : public Baselearner
{
  private:
    
    SEXP model;
    
    // R functions for a custom baselearner:
    Rcpp::Function instantiateDataFun;
    Rcpp::Function trainFun;
    Rcpp::Function predictFun;
    Rcpp::Function extractParameter;
    
  public:
    
    // (data pointer, data identifier, baselearner identifier, R function for
    // data instantiation, R function for training, R function for prediction,
    // R function to extract parameter):
    Custom (arma::mat&, std::string&, std::string&, Rcpp::Function, Rcpp::Function,
      Rcpp::Function, Rcpp::Function);
    
    // Copy constructor:
    Baselearner* Clone ();
    
    // Function to delete parent members. This is called by the child 
    // destructor:
    void CleanUp ();
    
    // arma::mat InstantiateData ();
    arma::mat InstantiateData (arma::mat&);
    
    void train (arma::vec&);
    arma::mat predict (arma::mat&);
    arma::mat predict ();
    
    ~Custom ();
	
};

// Custom Cpp Baselearner:
// -----------------------

// This is a  bit tricky. The key is that we store the cpp functions as 
// pointer. Therefore we can go with R and use the XPtr class of Rcpp to
// give the pointer as SEXP. To try a working example see 
// "tutorial/stages_of_custom_learner.html".

// Please note, that the result of the train function should be a matrix
// containing the estimated parameter.

typedef arma::mat (*instantiateDataFunPtr) (arma::mat& X);
typedef arma::mat (*trainFunPtr) (arma::vec& y, arma::mat& X);
typedef arma::mat (*predictFunPtr) (arma::mat& newdata, arma::mat& parameter);

class CustomCpp : public Baselearner
{
private:
  
  // Cpp functions for a custom baselearner:
  instantiateDataFunPtr instantiateDataFun;
  trainFunPtr trainFun;
  predictFunPtr predictFun;
  
public:
  
  // (data pointer, data identifier, baselearner identifier, R function for
  // data instantiation, R function for training, R function for prediction,
  // R function to extract parameter):
  CustomCpp (arma::mat&, std::string&, std::string&, SEXP, SEXP, SEXP);
  
  // Copy constructor:
  Baselearner* Clone ();
  
  // Function to delete parent members. This is called by the child 
  // destructor:
  void CleanUp ();
  
  // arma::mat InstantiateData ();
  arma::mat InstantiateData (arma::mat&);
  
  void train (arma::vec&);
  arma::mat predict (arma::mat&);
  arma::mat predict ();
  
  ~CustomCpp ();
  
};

} // namespace blearner

#endif // BASELEARNER_H_
