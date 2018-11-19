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


arma::mat penaltySumKronecker (const arma::mat& Pa, const arma::mat& Pb)
{
  // Variables
  arma::mat out;
  arma::rowvec vecPa = arma::rowvec(Pa.n_rows, arma::fill::ones);
  arma::rowvec vecPb = arma::rowvec(Pb.n_rows, arma::fill::ones);
  
  
  // sum of rowwise Kroneckers
  out = arma::kron(Pa,vecPa) + arma::kron(vecPb, Pb);
  
  return out;
}


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


// #include<iostream>
// 
// int main() {
// arma::mat UnityMatrix = arma::mat(2,2, arma::fill::eye);
// arma::mat OneMatrix = arma::mat(3,3, arma::fill::ones);
// 
// arma::mat test = rowWiseKronecker(UnityMatrix,OneMatrix);
// 
// 
// int i;
// int j;
// 
// std::ofstream output("inputmatriks.txt");
// for (i=0;i<2;i++)
// {
//   for (j=0;j<3;j++)
//   {
//     Rcpp::Rcout << test(i,j) << " "; // behaves like cout - cout is also a stream
//   }
//   Rcpp::Rcout << "\n";
// }
// return 0;
// }



