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
//   This file contains the different loss implementations. The structure here
//   is:
//     - Parent class 'LossDefinition' which are virtual member functions. This
//       functions are later overwritten by the child member functions which
//       contains the concrete implementation of the loss.
//
//     - Child classes have the structure:
//
//         class SpecificLoss: public LossDefinition
//         {
//           arma::vec DefinedLoss      { IMPLEMENTATION };
//           arma::vec DefinedGradient  { IMPLEMENTATION };
//           double ConstantInitializer { IMPLEMENTATION };
//         }
//
//     - There is one special child class, the 'CustomLoss' which allows to
//       define custom loss functions out of R.
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

#ifndef LOSS_H_
#define LOSS_H_

#include <RcppArmadillo.h>

#include <iostream>

namespace loss
{

// Parent class:
// -----------------------

class Loss
{
  public:

    virtual arma::vec DefinedLoss (arma::vec&, arma::vec&) = 0;
    virtual arma::vec DefinedGradient (arma::vec&, arma::vec&) = 0;
    virtual double ConstantInitializer (arma::vec&) = 0;
    
    virtual ~Loss ();
};

// -------------------------------------------------------------------------- //
// Loss implementations as child classes:
// -------------------------------------------------------------------------- //

// Quadratic loss:
// -----------------------

class Quadratic : public Loss
{
  public:

    arma::vec DefinedLoss (arma::vec&, arma::vec&);

    arma::vec DefinedGradient (arma::vec&, arma::vec&);

    double ConstantInitializer (arma::vec&);
};

// Absolute loss:
// -----------------------

class Absolute : public Loss
{
  public:

    arma::vec DefinedLoss (arma::vec&, arma::vec&);

    arma::vec DefinedGradient (arma::vec&, arma::vec&);

    double ConstantInitializer (arma::vec&);
};

// Custom loss:
// -----------------------

// This one is a special one. It allows to use a custom loss predefined in R.
// The convenience here comes from the 'Rcpp::Function' class and the use of
// a special constructor which defines the three needed functions!

// Note that there is one conversion step. There is no predefined conversion
// from 'Rcpp::Function' (which acts as SEXP) to 'double'. But it is possible
// to go the step above 'Rcpp::NumericVector'. Therefore the custom functions
// returns a 'Rcpp::NumericVector' which then is able to be converted to
// 'double' by just selecting one element.

class CustomLoss : public Loss
{
  private:

    Rcpp::Function lossFun;
    Rcpp::Function gradientFun;
    Rcpp::Function initFun;

  public:

    CustomLoss (Rcpp::Function, Rcpp::Function, Rcpp::Function);

    arma::vec DefinedLoss (arma::vec&, arma::vec&);

    arma::vec DefinedGradient (arma::vec&, arma::vec&);

    // Conversion step from 'SEXP' to double via 'Rcpp::NumericVector' which 
    // knows how to convert a 'SEXP':
    double ConstantInitializer (arma::vec&);
};

} // namespace loss

#endif // LOSS_H_
