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

#include "tensors.h"


/**
 * \calculating rowwise Kronecker Product
 * 
 * 
 * \returns `arma::mat` returns rowwise kronecker product of two matrices
 */


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat rowWiseKronecker (const arma::mat& A, const arma::mat& B)
{
  // Variables
  arma::mat out;
  arma::rowvec vecA = arma::rowvec(A.n_rows, arma::fill::ones);
  arma::rowvec vecB = arma::rowvec(B.n_rows, arma::fill::ones);
  
  
  // rowWiseKronecker
  out = arma::kron(A,vecA) * arma::kron(vecB, B);
  
  return out;
}

// [[Rcpp::export]]
arma::mat penaltySumKronecker (const arma::mat& Pa, const arma::mat& Pb)
{
  // Variables
  arma::mat out;
  // Create Diagonal matrices
  arma::mat eyePa = arma::diagmat( arma::vec(Pa.n_cols, arma::fill::ones) );
  arma::mat eyePb = arma::diagmat( arma::vec(Pb.n_cols, arma::fill::ones) );
  
  
  // sum of Kroneckers with diagonal marices
  out = arma::kron(Pa,eyePa) + arma::kron(eyePb, Pb);
  
  return out;
}

// [[Rcpp::export]]
arma::mat centerEffects (const arma::mat& X1, const arma::mat& X2)
{
  // Variables
  arma::mat out;
  arma::mat QR;
  arma::mat R;
  arma::mat X;
  arma::mat Z;
  
  // QR decomp of cross Product X1 and X2
  QR = arma::cross(X1,X2);
  arma::qr(QR,R,X);
  int rankR = arma::rank(R);
  
  // construct Z
  // FIXME
  // Z = Q[,R+1:ncol(Q)]
  Z = arma::kron(X1,Z);
  
  // Construct Penalty Matrxie
  out = arma::kron( arma::kron(Z.t(), X1) , Z); 
  
  return out;
}



