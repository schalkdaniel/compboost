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

#include "coordinate_descent.h"

namespace coodesc
{

/**
 * \brief
 *
 * \param X `arma::sp_mat`
 *
 * \param colnr `colnr`
 *
 * \returns `arma::sp_mat`
 */
arma::sp_mat subsetSparseCols (const arma::sp_mat& X, const unsigned int& colnr)
{
  arma::sp_mat Y = X;
  Y.shed_col(colnr);
  return Y;
}

/**
 * \brief
 *
 * \param X `arma::sp_mat`
 *
 * \param colnr `rownr`
 *
 * \returns `arma::mat`
 */
arma::colvec subsetRows (const arma::colvec& X, const unsigned int& rownr)
{
  arma::colvec Y = X;
  Y.shed_row(rownr);
  return Y;
}



/**
 * \brief
 *
 * \param X `arma::sp_mat`
 *
 * \param colnr `colnr`
 *
 * \returns `arma::sp_mat`
 */
arma::colvec coordinateDescent (const arma::mat& yr, const arma::sp_mat& Xr, const double& lambda,
  const double& alpha, const unsigned int& itern)
{
  arma::uword n = Xr.n_rows;
  arma::uword p = Xr.n_cols;
  arma::sp_mat X = Xr; // reuses memory and avoids extra copy  n*p
  arma::mat y = yr; // (yr.begin(), n, 1, false);   //  n*1
  arma::colvec w = arma::zeros(p);  //starting parameter wi=0, i=1,2,...,26   p*1
  arma::colvec objectives = arma::zeros(itern);    // itern*1

  arma::colvec resid = yr - X * w;
  double r = arma::as_scalar(arma::trans(resid) * resid);
  double r_prime;

  for (unsigned int i = 0; i < itern; i++){
    objectives[i] = r;

    for (unsigned int k = 0; k < p; k++){
      double z_k = arma::as_scalar(arma::trans(X.col(k)) * X.col(k));
      double w_k = 0.0;
      double p_k = arma::as_scalar( arma::trans(X.col(k)) * (y -  subsetSparseCols(X,k) * subsetRows(w,k)) );
      if (p_k < -lambda/2) {
        w_k = (p_k + lambda/2)/z_k;
      }else{
        if(p_k > lambda/2) {
          w_k = (p_k - lambda/2)/z_k;
        }
      }
      w[k] = w[k] + w_k * alpha;
    }
    resid = yr - X * w;
    double r = arma::as_scalar(arma::trans(resid) * resid);
    double delta = r - r_prime;
    r = r_prime;

    if (delta < 0) break;
  }
  return w;
}

} // namespace cooddesc
