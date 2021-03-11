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

#include "binning.h"

namespace binning
{

/**
 * \brief Calculate vector of bins of specific size
 *
 * This function returns a vector of quantiles of length n_bins.
 *
 * \param x `arma::vec` Vector that should be discretized.
 *
 * \param n_bins `unsigned int` Number of unique points for binning the vector x.
 *
 * \returns `arma::vec` Vector of discretized x.
 */
arma::vec binVectorCustom (const arma::vec& x, const unsigned int bin_root)
{
  // TODO: Check if n_bins is set correctly

  // Old equal spacing:
  //const unsigned int n_bins = std::floor(std::pow(x.size(), 1.0/bin_root));
  //return arma::linspace(arma::min(x), arma::max(x), n_bins);

  // New quantile spacing:
  const unsigned int n_bins = std::floor(std::pow(x.size(), 1.0/bin_root));
  const arma::vec quants = arma::linspace(0, 1, n_bins);
  return arma::quantile(x, quants);
}



/**
 * \brief Calculate vector of bins
 *
 * This function returns a vector of equally spaced points of length the square root of the size of the vector.
 *
 * \param x `arma::vec` Vector that should be discretized.
 *
 * \param n_bins `unsigned int` Number of unique points for binning the vector x.
 *
 * \returns `arma::vec` Vector of discretized x.
 */
arma::vec binVector (const arma::vec& x)
{
  return binVectorCustom(x, 2);
}


/**
 * \brief Calculate index vector for binned vector
 *
 * This function returns the indexes of the unique values to the complete binned vector.
 *
 * \param x `arma::vec` Vector that should be discretized.
 *
 * \param x_bins `arma::vec` Vector of unique values for binning.
 *
 * \returns `arma::uvec' Index vector.
 */
arma::uvec calculateIndexVector (const arma::vec& x, const arma::vec& x_bins)
{
  arma::uvec idx(x.size(), arma::fill::zeros);
  const double delta = (x_bins(1) - x_bins(0)) / 2;

  for (unsigned int i = 0; i < x.size(); i++) {
    unsigned int j = 0;
    while ((x_bins(j) + delta) < x(i)) { j += 1; }
    idx(i) = j;
  }
  return idx;
}


/**
 * \brief Calculating binned matrix product
 *
 * This function calculates the matrix product using Algorithm 3 of Zheyuan Li, Simon N. Wood: "Faster
 * model matrix crossproducts for large generalized linear models with discretized covariates". The idea
 * is to compute just on the unique rows of X by also using an index vector to map to the original matrix.
 * The algorithm implemented here is a small adaption of the original algorithm. Instead of calculating $XW$ which
 * which again, needs to be transposed, we directly calculate $X^TW$ to avoid another transposing step.
 *
 * \param X `arma::mat` Matrix X.
 *
 * \param k `arma::uvec` Index vector for mapping to original matrix $X_o(i,) = X(k(i),.)$.
 *
 * \param w `arma::vec` Vector of weights that are accumulated.
 *
 * \returns `arma::mat` Matrix Product $X^TWX$.
 */
arma::mat binnedMatMult (const arma::mat& X, const arma::uvec& k, const arma::vec& w)
{
  unsigned int n = k.size();
  unsigned int ind;

  arma::colvec wcum(X.n_rows, arma::fill::zeros);
  if ( (w.size() == 1) && (w(0) == 1) ) {
    for (unsigned int i = 0; i < n; i++) {
      ind = k(i);
      wcum(ind) += 1;
    }
  } else {
    for (unsigned int i = 0; i < n; i++) {
      ind = k(i);
      wcum(ind) += w(i);
    }
  }
  return arma::trans(X.each_col() % wcum) * X;
}


/**
 * Calculating binned matrix product for response term
 *
 * This function calculates the matrix product using Algorithm 3 of Zheyuan Li, Simon N. Wood: "Faster
 * model matrix crossproducts for large generalized linear models with discretized covariates". The idea
 * is to compute just on the unique rows of X by also using an index vector to map to the original matrix.
 * The algorithm implemented here is a small adaption of the original algorithm. Instead of calculating $XW$
 * which again, needs to be transposed, we directly calculate $X^TW$ to avoid another transposing step. In addition
 * to the original algorithm the algorithm here directly calculates the crossproduct with the response.
 *
 * \param X `arma::mat` Matrix X.
 *
 * \param y `arma::vec` Response vector y.
 *
 * \param k `arma::uvec` Index vector for mapping to original matrix $X_o(i,) = X(k(i),.)$.
 *
 * \param w `arma::vec` Vector of weights that are accumulated.
 *
 * \return `arma::mat` Matrix Product $X^TWX$.
 */

arma::mat binnedMatMultResponse (const arma::mat& X, const arma::vec& y,  const arma::uvec& k, const arma::vec& w)
{
  unsigned int n = k.size();
  unsigned int ind;

  arma::rowvec wcum(X.n_rows, arma::fill::zeros);

  if ( (w.size() == 1) && (w(0) == 1) ) {
    for (unsigned int i = 0; i < n; i++) {
       ind = k(i);
       wcum(ind) += y(i);
    }
  } else {
    for (unsigned int i = 0; i < n; i++) {
      ind = k(i);
      wcum(ind) += w(i) * y(i);
    }
  }
  return wcum * X;
}


/**
 * \brief Calculating sparse binned matrix product
 *
 * This function calculates the matrix product (with sparse matrices) using Algorithm 3 of Zheyuan Li, Simon N. Wood: "Faster
 * model matrix crossproducts for large generalized linear models with discretized covariates". The idea
 * is to compute just on the unique rows of X by also using an index vector to map to the original matrix.
 * The algorithm implemented here is a small adaption of the original algorithm. Instead of calculating $XW$ which
 * which again, needs to be transposed, we directly calculate $X^TW$ to avoid another transposing step.
 *
 * \param X `arma::sp_mat` Sparse matrix X.
 *
 * \param k `arma::uvec` Index vector for mapping to original matrix $X_o(i,) = X(k(i),.)$.
 *
 * \param w `arma::vec` Vector of weights that are accumulated.
 *
 * \returns `arma::mat` Matrix Product $X^TWX$.
 */
arma::mat binnedSparseMatMult (const arma::sp_mat& X, const arma::uvec& k, const arma::vec& w)
{
  const unsigned int n = k.size();
  const unsigned int n_unique = X.n_cols;
  unsigned int ind;

  arma::sp_mat sp_out(X);
  arma::colvec wcum(n_unique, arma::fill::zeros);

  if ( (w.size() == 1) && (w(0) == 1) ) {
    for (unsigned int i = 0; i < n; i++) {
      ind = k(i);
      wcum(ind) += 1;
    }
  } else {
    for (unsigned int i = 0; i < n; i++) {
      ind = k(i);
      wcum(ind) += w(i);
    }
  }
  for (unsigned int i = 0; i < n_unique; i++) {
    sp_out.col(i) *= wcum(i);
  }
  arma::mat out(X * arma::trans(sp_out));
  return out;
}


/**
 * Calculating sparse binned matrix product for response term
 *
 * This function calculates the matrix product (with sparse matrices) using Algorithm 3 of Zheyuan Li, Simon N. Wood: "Faster
 * model matrix crossproducts for large generalized linear models with discretized covariates". The idea
 * is to compute just on the unique rows of X by also using an index vector to map to the original matrix.
 * The algorithm implemented here is a small adaption of the original algorithm. Instead of calculating $XW$
 * which again, needs to be transposed, we directly calculate $X^TW$ to avoid another transposing step. In addition
 * to the original algorithm the algorithm here directly calculates the crossproduct with the response.
 *
 * \param X `arma::sp_mat` Matrix X.
 *
 * \param y `arma::vec` Response vector y.
 *
 * \param k `arma::uvec` Index vector for mapping to original matrix $X_o(i,) = X(k(i),.)$.
 *
 * \param w `arma::vec` Vector of weights that are accumulated.
 *
 * \return `arma::mat` Matrix Product $X^TWX$.
 */
arma::mat binnedSparseMatMultResponse (const arma::sp_mat& X, const arma::vec& y,  const arma::uvec& k, const arma::vec& w)
{
  unsigned int n = k.size();
  unsigned int ind;

  arma::colvec wcum(X.n_cols, arma::fill::zeros);

  if ( (w.size() == 1) && (w(0) == 1) ) {
    for (unsigned int i = 0; i < n; i++) {
       ind = k(i);
       wcum(ind) += y(i);
    }
  } else {
    for (unsigned int i = 0; i < n; i++) {
      ind = k(i);
      wcum(ind) += w(i) * y(i);
    }
  }
  return X * wcum;
}

arma::mat binnedSparsePrediction (const arma::sp_mat& X, const arma::mat& param, const arma::uvec& k)
{
  unsigned int n = k.size();
  unsigned int ind;

  arma::colvec out(n, arma::fill::zeros);
  arma::mat param_temp = arma::trans(param);

  arma::mat temp = param_temp * X;

  for (unsigned int i = 0; i < n; i++) {
     ind = k(i);
     out(i) = temp(ind);
  }
  return out;
}

} // namespace binning
