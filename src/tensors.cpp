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
    
  // Multiply both kronecker products element-wise 
  out = arma::kron(A,vecA) % arma::kron(vecB, B);
  
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
std::map<std::string, arma::mat>  centerDesignMatrix (const arma::mat& X1, const arma::mat& P1, const arma::mat& X2)
{

  // Cross Product X1 and X2
  arma::mat cross = X1.t() * X2 ;
  
  // QR decomp 
  // We require and orthogonal matrix Q
  arma::mat R;
  arma::mat Q;
  arma::qr(Q,R,cross);
  
  // get rank of R and add 1
  int rankR = arma::rank(R);
  
  // construct Z from rows 0 to last row and column R+1 to last column
  arma::mat Z = Q( arma::span(0, Q.n_rows-1), arma::span(rankR, Q.n_cols-1) );
  
  // Construct the rotated X1
  arma::mat X1_out = X1 * Z; 
  
  // Construct the rotated Penalty Matrix
  arma::mat P1_out = Z.t() * P1 * Z;
  
  // Construct out
  std::map<std::string, arma::mat> out;
  out["X1"] = X1_out;
  out["P1"] = P1_out;

  return out;
  
  /// return X1_out;
}

// 




