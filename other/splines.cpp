// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

/*
 * \brief Calculating penalty matrix
 * 
 * This function calculates the penalty matrix for a given number of 
 * parameters (`nparams`) and a given number of differences (`differences`).
 * 
 * \param nparams `unsigned int` Number of params which should be penalized.
 *   This also pretend the number of rows and columns.
 *   
 * \param differences `unsigned int` Number of penalized differences.
 * 
 * \returns `arma::sp_mat` Sparse penalty matrix used for p splines. 
 */

// [[Rcpp::export]]
arma::sp_mat penaltyMat (const unsigned int& nparams, const unsigned int& differences)
{
  // Create frame for sparse difference matrix:
  arma::sp_mat diffs(0, nparams);
  for (unsigned int i = 0; i < nparams - 1; i++) {
    arma::sp_mat insert(1, nparams);
    insert[i] = -1;
    insert[i + 1] = 1;
    diffs = join_cols(diffs, insert);
  }
  
  // Calculate the difference matrix for higher orders:
  if (differences > 1) {
    arma::sp_mat diffs_reduced = diffs;
    for (unsigned int k = 0; k < differences - 1; k++) {
      diffs_reduced = diffs_reduced(arma::span(1, diffs_reduced.n_rows - 1), arma::span(1, diffs_reduced.n_cols - 1));
      diffs = diffs_reduced * diffs;
    }
  }
  
  arma::sp_mat K = diffs.t() * diffs;
  
  return K;
}

/*
 * \brief Binary search to find index of given point within knots
 * 
 * This small functions search for the position of `x` within the
 * `knots` and returns the smalles index for which x >= knots[i].
 * 
 * Note that this function returns the `C++` index which starts 
 * with `0` and ends with `n-1`.
 * 
 * \param x `double` Point to search for position in knots.
 * \param knots `arma::vec` Vector of knots. It's the users responsibility to
 *   pass a **SORTED** vector.
 *   
 * \returns `unsigned int` of position of `x` in `knots`.
 */

// [[Rcpp::export]]
unsigned int findSpan (const double& x, const arma::vec& knots)
{
  // Special case which the algorithm can't handle:
  if (x < knots[1]) { return 0; }
  if (x == knots[knots.size() - 1]) { return knots.size() - 1; }
  
  unsigned int low = 0;
  unsigned int high = knots.size() - 1;
  unsigned int mid = std::round( (low + high) / 2 );
  
  while (x < knots[mid] || x >= knots[mid + 1]) {
    if (x < knots[mid]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = std::round( (low + high) / 2 );
  }
  return mid;
}

/*
 * \brief De Boors algorithm to find basis functions
 * 
 * \param x `double` Point to search for position in knots.
 * \param degree `unsigned int` Degree of the polynomial between the knots.
 * \param knots `arma::vec` Vector of knots. It's the users responsibility to
 *   pass a **SORTED** vector.
 *   
 * \returns `arma::sp_mat` Sparse matrix containing the basis functions.
 */

// [[Rcpp::export]]
arma::rowvec basisFuns (const double& x, const unsigned int& degree, 
  const arma::vec& knots)
{
  // Sparse output vector of bases:
  arma::rowvec full_base(knots.size() - (degree + 1), arma::fill::zeros);
  
  // Index of x within the konts:
  unsigned int idx = findSpan(x, knots);
  
  // A problem occurs if x = max(knots), then idx is bigger than
  // length(full_base) which couses problems. Catch that:
  if (idx > (full_base.n_cols - 1)) { idx = full_base.n_cols - 1; }
  
  // Output for basis functions. Here we have the non-zero entries:
  arma::rowvec N(degree + 1, arma::fill::zeros);
  N[0] = 1.0;
  
  arma::vec left(degree + 1, arma::fill::zeros);
  arma::vec right(degree + 1, arma::fill::zeros);
  
  double saved;
  double temp;
  
  // De Boors algorithm to recursive find base in a triangle scheme:
  for (unsigned int j = 1; j <= degree; j++) {

    left[j]  = x - knots[idx + 1 - j];
    right[j] = knots[idx + j] - x;

    saved = 0;

    for (unsigned int r = 0; r < j; r++) {
      temp  = N[r] / (right[r + 1] + left[j - r]);
      N[r]  = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  
  full_base(arma::span(idx - degree, idx)) = N;
  
  // arma::sp_mat out(1, full_base.size());
  // out.row(0) = full_base.t();
  
  return full_base;
}

/*
 * \brief Create knots for a specific number, degree and values
 * 
 * This functions takes a vector of points and creates knots used for the
 * splines depending on the number of knots and degree. This function just
 * handles equidistant knots.
 * 
 * \param values `arma::vec` Points to create the basis matrix.
 * \param n_knots `unsigned int` Number of innter knots.
 * \param degree `unsigned int` polynomial degree of splines.
 *    
 * \returns `arma::vec` of knots.
 */

// [[Rcpp::export]]
arma::vec createKnots (const arma::vec& values, const unsigned int& n_knots,
  const unsigned int& degree)
{
  // Expand inner knots to avoid ugly issues on the edges:
  arma::vec knots(n_knots + 2 * (degree + 1), arma::fill::zeros);
  
  double inner_knot_min = values.min();
  double inner_knot_max = values.max();
  
  double knot_range = (inner_knot_max - inner_knot_min) / (n_knots + 1);
  
  // Inner knots:
  for (unsigned int i = 0; i <= n_knots + 1; i++) {
    knots[degree + i] = inner_knot_min + i * knot_range;
  }
  // Lower and upper 'boundary knots'
  for (unsigned int i = 0; i < degree; i++) {
    // Lower:
    knots[i] = inner_knot_min - (degree - i) * knot_range;
    // Upper:
    knots[degree + n_knots + i + 2] = inner_knot_max + (i + 1) * knot_range;
  }
  
  return knots;
}

/*
 * \brief Transformation from a vector of input points to sparse matrix of basis
 * 
 * This functions takes a vector of points and create a sparse matrix of
 * basis functions. Each row contains the basis of the corresponding value 
 * in `values`.
 * 
 * \param values `arma::vec` Points to create the basis matrix.
 * \param n_knots `unsigned int` Number of innter knots.
 * \param degree `unsigned int` polynomial degree of splines.
 *    
 * \returns `sp_mat` sparse matrix of base functions.
 */

// [[Rcpp::export]]
arma::mat createBasis (const arma::vec& values, const unsigned int& degree, 
  const arma::vec& knots)
{
  // Frame for output:
  arma::mat spline_basis(values.size(), knots.size() - (degree + 1), arma::fill::zeros);
  
  // Fill frame with functions:
  for (unsigned int i = 0; i < values.size(); i++) {
    spline_basis.row(i) = basisFuns(values[i], degree, knots);
  }
  
  return spline_basis;
}

// [[Rcpp::export]]
arma::mat estimateSplines (const arma::vec& response, const arma::vec& x,
  const unsigned int& degree, const unsigned int& n_knots, 
  const unsigned int& differences, const double& penalty)
{
  arma::vec knots = createKnots(x, n_knots, degree);
  arma::mat basis = createBasis(x, degree, knots);
  arma::sp_mat pen = penaltyMat(basis.n_cols, differences);
  
  arma::mat params = arma::inv(basis.t() * basis + penalty * pen) * basis.t() * response;
  
  return params;
}

// [[Rcpp::export]]
arma::mat testSparse (arma::sp_mat X, arma::vec y)
{
  return arma::spsolve(X, y, "lapack");
}

// [[Rcpp::export]]
arma::mat testDense (arma::mat X, arma::vec y)
{
  return arma::solve(X, y);
}