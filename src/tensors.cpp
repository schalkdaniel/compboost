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
// =========================================================================== #

#include "tensors.h"

namespace tensors
{
/**
 * \brief Calculate the rowwise kronecker product of two matrices
 *
 * This function calculates the rowwise kronecker product of matrices.
 * Sparse matrices are allowed as inputs
 *
 * \param A `arma::mat` or `arma::sp_mat` a matrix.
 * \param B `arma::mat` or `arma::sp_mat` a matrix.
 * \returns `arma::mat` or `arma::sp_mat`.
 */
arma::mat rowWiseKronecker (const arma::mat& A, const arma::mat& B)
{
  // Variables
  arma::mat out;
  arma::rowvec vecA = arma::rowvec(A.n_cols, arma::fill::ones);
  arma::rowvec vecB = arma::rowvec(B.n_cols, arma::fill::ones);

  // Multiply both kronecker products element-wise
  out = arma::kron(A,vecB) % arma::kron(vecA, B);

  return out;
}

/**
 * \brief Calculate the rowwise kronecker product of two matrices
 *
 * This function calculates the rowwise kronecker product of matrices.
 * Sparse matrices are allowed as inputs
 *
 * \param A `arma::mat` or `arma::sp_mat` a matrix.
 * \param B `arma::mat` or `arma::sp_mat` a matrix.
 * \returns `arma::mat` or `arma::sp_mat`.
 */
arma::sp_mat rowWiseKroneckerSparse (const arma::sp_mat A, const arma::sp_mat B)
{
  // Variables
  arma::rowvec vecA = arma::rowvec(A.n_cols, arma::fill::ones);
  arma::rowvec vecB = arma::rowvec(B.n_cols, arma::fill::ones);

  arma::sp_mat vecAsparse = arma::sp_mat(vecA);
  arma::sp_mat vecBsparse = arma::sp_mat(vecB);

  // Multiply both kronecker products element-wise
  arma::sp_mat out = arma::kron(A,vecBsparse) % arma::kron(vecAsparse, B);

  return out;
}

/**
 * \brief Calculate the penalty matrix of a combined baselearner
 *
 * This function takes two penalty matrices of two baselearners
 * and calculates the correct kroneckered sum for a new baselearner
 * combining the two inputs.
 *
 * \param A `arma::mat` or `arma::sp_mat` a penalty matrix.
 * \param B `arma::mat` or `arma::sp_mat` a penalty matrix.
 * \returns `arma::mat` or `arma::sp_mat`.
 */
arma::mat penaltySumKronecker (const arma::mat& Pa, const arma::mat& Pb)
{
  // Variables
  arma::mat out;
  // Create Diagonal matrices
  arma::mat eyePa = arma::diagmat( arma::vec(Pa.n_cols, arma::fill::ones) );
  arma::mat eyePb = arma::diagmat( arma::vec(Pb.n_cols, arma::fill::ones) );


  // sum of Kroneckers with diagonal marices
  out = arma::kron(Pb,eyePa) + arma::kron(eyePb, Pa);

  return out;
}


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
  arma::mat Z = Q.cols(rankR,Q.n_cols-1);

  // Construct the rotated X1
  arma::mat X1_out = X1 * Z;

  // Construct the rotated Penalty Matrix
  arma::mat P1_out = Z.t() * P1 * Z;

  // Construct out
  std::map<std::string, arma::mat> out;
  out["X"] = X1_out;
  out["P"] = P1_out;
  out["Z"] = Z;

  return out;
}

arma::mat trapezWeights (const arma::mat& time_points)
{
  /// Get Differences of the current function
  arma::mat t_diffs = arma::diff( time_points, 1, 1) ;
  arma::mat weights = time_points;


  /// change the border values
  weights.col(0) = t_diffs.col(0);
  weights.col(weights.n_cols-1) = t_diffs.col(t_diffs.n_cols-1);

  /// divide all by half except for beginning and end, add to get mean per value
  arma::mat t_diff_halfs = t_diffs / 2;

  weights.cols(1,(weights.n_cols-2)) =  t_diff_halfs.cols(1,t_diff_halfs.n_cols-1) +
    t_diff_halfs.cols(0,t_diff_halfs.n_cols-2);

  return weights;
}

} // namespace tensors

