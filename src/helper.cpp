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

#include "helper.h"

bool _DEBUG_PRINT = false;

namespace helper
{

void debugPrint (const std::string msg)
{
  if (_DEBUG_PRINT) { std::cout << msg << std::endl; }
}

/**
 * \brief Check if a string occurs within a string vector
 *
 * This function just takes a string, iterates over a given vector of strings,
 * and returns true if the string occurs within the vector. This function is
 * used to check the argument name match up.
 *
 * \param str `str::string` String for the lookup.
 *
 * \param differences `std::vector<std::string>` Vector of strings which we
 '   want to check if str occurs..
 *
 * \returns `bool` boolean if the string occurs within the vector.
 */
bool stringInNames (std::string str, std::vector<std::string> names)
{
  bool string_in_names = false;
  for (unsigned int i = 0; i < names.size(); i++) {
    if (str == names[i]) {
      string_in_names = true;
      return string_in_names;
    }
  }
  return string_in_names;
}


void assertChoice (const std::string x, const std::vector<std::string>& choices)
{
  if (! stringInNames(x, choices)) {
    std::string msg = "Must be element of set {";
    for (unsigned int i = 0; i < choices.size(); i++) {
      if (i+1 < choices.size()) {
        msg += "'" + choices.at(i) + "',";
      } else {
        msg += "'" + choices.at(i) + "'";
      }
    }
    throw std::out_of_range(msg + "}, but is '" + x + "'.'");
  }
}

/**
 * \brief Check and set list arguments
 *
 * This function matches to lists to update an internal list with a new
 * match-up list. This function checks which elements are available,
 * if they occurs in the internal list, and if the underlying data types
 * matches. If so, this function replaces
 * the values of the internal list with the new values. If the new list
 * contains unused elements, this function also throws a warning and prints
 * the unused ones.
 *
 * \param internal_list `Rcpp::List` Internal list with default values.
 * \param matching_list `Rcpp::List` New list to replace default values.
 * \returns `Rcpp::List` of updated default values.
 */
Rcpp::List argHandler (Rcpp::List internal_list, Rcpp::List matching_list, bool type_check = TRUE)
{
  // First of all, we want to check if the matching list contain anything. If not, we just
  // return the default setting:
  if (matching_list.size() == 0) {
    return internal_list;
  }

  // Next, check if the matching list has names. If not, stop instantly.
  // Otherwise this could cause segfoults:
  if (! matching_list.hasAttribute("names")) {
    // This is necessary to prevent the program from segfolds... whyever???
    // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
    try {
      Rcpp::stop("Be sure to specify names within your argument list.");
    } catch ( std::exception &ex ) {
      forward_exception_to_r( ex );
    } catch (...) {
      ::Rf_error( "c++ exception (unknown reason)" );
    }
  }

  std::vector<std::string> internal_list_names = internal_list.names();
  std::vector<std::string> matching_list_names = matching_list.names();
  std::vector<std::string> unused_args;

  int arg_type_internal;
  int arg_type_match;

  for (unsigned int i = 0; i < matching_list_names.size(); i++) {
    if (stringInNames(matching_list_names[i], internal_list_names)) {

      if (type_check) {
        // Check element type:
        arg_type_internal = TYPEOF(internal_list[matching_list_names[i]]);
        arg_type_match    = TYPEOF(matching_list[matching_list_names[i]]);

        // Int as 13 and double as 14 are both ok, so set both to 13 (int) if they are 14 (double). The
        // as conversion of Rcpp handles to set them correctly later:
        if (arg_type_internal == 14) { arg_type_internal -= 1; }
        if (arg_type_match == 14) { arg_type_match -= 1; }

        if (arg_type_internal == arg_type_match) {
          internal_list[matching_list_names[i]] = matching_list[matching_list_names[i]];
        } else {
          // This is necessary to prevent the program from segfolds... whyever???
          // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
          try {
            Rcpp::stop("Argument types for \"" + matching_list_names[i] + "\" does not match. Maybe you should take a look at the documentation.");
          } catch ( std::exception &ex ) {
            forward_exception_to_r( ex );
          } catch (...) {
            ::Rf_error( "c++ exception (unknown reason)" );
          }
        }
      } else {
        internal_list[matching_list_names[i]] = matching_list[matching_list_names[i]];
      }
    } else {
      unused_args.push_back(matching_list_names[i]);
    }
  }
  if (unused_args.size() > 0) {
    std::stringstream message;
    message << "Unused arguments ";

    for (unsigned int i = 0; i < unused_args.size(); i++) {
      message << "\"" + unused_args[i] + "\"";
      if (i < unused_args.size() - 1) {
        message << ", ";
      }
    }
    message << " in list.";
    Rcpp::warning(message.str());
  }
  return internal_list;
}

double calculateSumOfSquaredError (const arma::mat& response, const arma::mat& prediction)
{
  return arma::accu(arma::pow(response - prediction, 2));
}

arma::mat sigmoid (const arma::mat& scores)
{
  return 1 / (1 + arma::exp(-scores));
}

arma::mat transformToBinaryResponse (const arma::mat& score_mat, const double& threshold, const double& pos, const double& neg)
{
  arma::mat out = score_mat;

  arma::umat ids_pos = find(score_mat >= threshold);
  arma::umat ids_neg = find(score_mat < threshold);

  out.elem(ids_pos).fill(pos);
  out.elem(ids_neg).fill(neg);

  return out;
}

std::map<std::string, unsigned int> tableResponse (const std::vector<std::string>& response)
{
  std::map<std::string, unsigned int> out;
  for (unsigned int i = 0; i < response.size(); i++) {
    out[response[i]] += 1;
  }
  return out;
}

arma::vec stringVecToBinaryVec(const std::vector<std::string>& response, const std::string& pos_class)
{
  arma::mat out(response.size(), 1, arma::fill::ones);
  for (unsigned int i = 0; i < response.size(); i++) {
    if (response[i] != pos_class) out(i, 0) = -1;
  }
  return out;
}

void checkForBinaryClassif (const std::vector<std::string>& response)
{
  std::map<std::string, unsigned int> class_table = helper::tableResponse(response);
  try {
    if (class_table.size() > 2) {
      Rcpp::stop("Multiple classes detected.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }
}

void checkMatrixDim (const arma::mat& X, const arma::mat& Y)
{
  try {
    if (X.n_rows != Y.n_rows || X.n_cols != Y.n_cols) {
      std::string error_msg = "Dimension does not match " + std::to_string(X.n_rows) + "x" + std::to_string(X.n_cols) + " and " + std::to_string(Y.n_rows) + "x" + std::to_string(Y.n_cols) + ".";
      Rcpp::stop(error_msg);
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }
}

bool checkTracePrinter (const unsigned int& k, const unsigned int& trace)
{
  bool print_trace = false;
  if (trace > 0) {
    if ((k == 1) || ((k % trace) == 0)) {
      print_trace = true;
    }
  }
  return print_trace;
}

double matrixQuantile (const arma::mat& X, const double& quantile)
{
  double nq = X.size() * quantile;
  int nq_int = int(nq);

  arma::mat X_sort = arma::sort(arma::reshape(X, X.n_rows * X.n_cols, 1));

  if (nq == nq_int) {
    return arma::accu(X_sort( arma::span(nq_int - 1, nq_int), arma::span(0, 0) )) / 2;
  } else {
    return X_sort(nq_int - 1, 0);
  }
}

arma::SpMat<unsigned int> binaryMat (const arma::Row<unsigned int>& classes)
{
  const unsigned int n_levels = arma::max(classes);
  const unsigned int n_obs = classes.size();

  const arma::Row<unsigned int> ones(n_obs, arma::fill::ones);

  arma::umat indices = arma::join_cols(arma::cumsum(ones) - 1, classes);
  arma::SpMat<unsigned int> sp_out(indices, ones);

  return sp_out;
}

arma::uvec binaryToIndex (const arma::mat& x)
{
  std::vector<unsigned int> idx;
  unsigned int k = 0;
  for (unsigned int i = 0; i < x.size(); i++) {
    if (x(k) != 0) idx.push_back(k);
    k += 1;
  }
  idx.push_back(x.size());

  return arma::conv_to<arma::uvec>::from(idx);
}

arma::uvec binaryToIndex (const arma::sp_mat& x)
{
  arma::uvec idx(x.n_nonzero + 1, arma::fill::zeros);
  for (unsigned int i = 0; i < x.n_nonzero; i++) {
    idx(i) = x.row_indices[i];
  }
  idx(x.n_nonzero) = x.size();

  return idx;
}

arma::mat predictBinaryIndex (const arma::uvec& idx, const double value)
{
  unsigned int n_idx = idx.size() - 1;
  arma::mat out(idx(n_idx), 1, arma::fill::zeros);
  for (unsigned int i = 0; i < n_idx; i++) {
    out(idx(i)) = value;
  }
  return out;
}

std::string getMatStatus (const arma::mat& X)
{
  std::string msg = "ncol = " + std::to_string(X.n_cols) + " nrows = " + std::to_string(X.n_rows);
  return msg;
}

void printMatStatus (const arma::mat& X, const std::string message)
{
  std::cout << message << getMatStatus(X) << std::endl;
}

arma::mat solveCholesky (const arma::mat& U, const arma::mat& y)
{
  arma::mat out =  arma::solve(arma::trimatl(U.t()), y, arma::solve_opts::fast);
  return arma::solve(arma::trimatu(U), out, arma::solve_opts::fast);
}

arma::mat cboostSolver (const std::pair<std::string, arma::mat>& mat_cache, const arma::mat& y)
{
  if (mat_cache.first == "cholesky") return solveCholesky(mat_cache.second, y);
  if (mat_cache.first == "inverse") return mat_cache.second * y;
}

template<typename SH_PTR>
inline unsigned int countSharedPointer (const SH_PTR& sh_ptr)
{
  return sh_ptr.use_count();
}

} // namespace helper
